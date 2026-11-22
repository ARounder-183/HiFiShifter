#!/usr/bin/env python3
"""Dependency vulnerability scan gate: feed both lockfiles to OSV package by
package, and only let through advisories that have already been reviewed.

[Why a separate script] `npm audit` only covers npm, and `cargo audit` requires
installing a toolchain, while this repo's dependency surface is "two lockfiles" -
`backend/Cargo.lock` and `frontend/package-lock.json`. OSV indexes both
crates.io and npm, so a single batch query scans both sides and needs nothing
installed locally.

[Why an allowlist exists] This repo carries 5 advisories that "cannot be fixed"
(upstream pins the version / the project is unmaintained). Without an allowlist
the gate stays red forever; with one, a **newly added** advisory becomes a real
alarm. Every entry must state a reason (see `advisory-allowlist.json`) and be
reviewed periodically.

[Exit codes]
- 0: no advisories outside the allowlist (network failure also lands here, but
  prints a loud WARN - a scheduled job must not go red because OSV hiccuped;
  what really matters is "something new was found").
- 1: advisories exist outside the allowlist.
- 2: argument/file error.

Usage:
    python scripts/checks/check-dependency-advisories.py
    python scripts/checks/check-dependency-advisories.py --verbose
"""

from __future__ import annotations

import argparse
import json
from datetime import date
import pathlib
import sys
import time
import urllib.error
import urllib.request

# On Windows, stdout follows the ANSI code page (cp1252 on English systems, GBK on
# Chinese ones); printing Chinese raises UnicodeEncodeError and kills the process -
# not a single real verdict makes it into the log, and what turns red is the
# encoding, not the rule.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

OSV_BATCH = "https://api.osv.dev/v1/querybatch"
ROOT = pathlib.Path(__file__).resolve().parents[2]
CARGO_LOCK = ROOT / "backend" / "Cargo.lock"
NPM_LOCK = ROOT / "frontend" / "package-lock.json"
ALLOWLIST = pathlib.Path(__file__).resolve().parent / "advisory-allowlist.json"
CHUNK = 200


def cargo_packages() -> list[tuple[str, str]]:
    """Extract (name, version) pairs from Cargo.lock.

    Cargo.lock is TOML, but the first two lines of every `[[package]]` block are
    always name/version; pulling in a TOML parser for that little information is
    not worth it, and the gate should not depend on extra packages anyway.
    """
    text = CARGO_LOCK.read_text(encoding="utf-8")
    packages: list[tuple[str, str]] = []
    for block in text.split("[[package]]")[1:]:
        name = version = None
        for line in block.splitlines():
            line = line.strip()
            if line.startswith("name = ") and name is None:
                name = line[len("name = ") :].strip('"')
            elif line.startswith("version = ") and version is None:
                version = line[len("version = ") :].strip('"')
            elif name and version:
                break
        if name and version:
            packages.append((name, version))
    return packages


def npm_packages() -> list[tuple[str, str]]:
    """Extract (name, version) pairs from the `packages` table of package-lock.json.

    Nested installs produce keys like `node_modules/a/node_modules/b`: the package
    name is the part after the last `node_modules/`.
    """
    lock = json.loads(NPM_LOCK.read_text(encoding="utf-8"))
    packages: list[tuple[str, str]] = []
    for key, meta in lock.get("packages", {}).items():
        if not key or "node_modules/" not in key:
            continue
        name = key.rsplit("node_modules/", 1)[-1]
        version = meta.get("version")
        if name and version:
            packages.append((name, version))
    return packages


def query_osv(ecosystem: str, packages: list[tuple[str, str]]) -> dict[tuple[str, str], list[dict]]:
    """Batch query; returns {(name, version): [vuln, ...]}, keeping only entries with hits."""
    hits: dict[tuple[str, str], list[dict]] = {}
    for start in range(0, len(packages), CHUNK):
        chunk = packages[start : start + CHUNK]
        payload = json.dumps(
            {
                "queries": [
                    {"package": {"name": name, "ecosystem": ecosystem}, "version": version}
                    for name, version in chunk
                ]
            }
        ).encode("utf-8")
        body = _post_with_retry(payload)
        results = body.get("results", [])
        if len(results) != len(chunk):
            raise RuntimeError(
                f"OSV returned {len(results)} results for {len(chunk)} queries"
            )
        for (name, version), result in zip(chunk, results):
            vulns = result.get("vulns") or []
            if vulns:
                hits[(name, version)] = vulns
    return hits


def _post_with_retry(payload: bytes) -> dict:
    last_error: Exception | None = None
    for attempt in range(3):
        request = urllib.request.Request(
            OSV_BATCH,
            data=payload,
            headers={"content-type": "application/json", "user-agent": "hifishifter-advisory-check"},
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            last_error = error
            time.sleep(1 + attempt)
    raise RuntimeError(f"OSV query failed after 3 attempts: {last_error}")


def load_allowlist() -> list[dict]:
    entries = json.loads(ALLOWLIST.read_text(encoding="utf-8"))
    today = date.today().isoformat()
    for entry in entries:
        for field in ("package", "id", "reason", "expires", "tracking"):
            if not entry.get(field):
                raise SystemExit(f"advisory-allowlist.json: entry missing '{field}': {entry}")
        if entry["expires"] < today:
            raise SystemExit(
                f"advisory-allowlist.json: waiver for {entry['package']} / {entry['id']} "
                f"expired on {entry['expires']}; {entry['tracking']}"
            )
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true", help="print accepted advisories too")
    args = parser.parse_args()

    for path in (CARGO_LOCK, NPM_LOCK, ALLOWLIST):
        if not path.is_file():
            print(f"ERROR: missing {path}", file=sys.stderr)
            return 2

    allowlist = load_allowlist()
    allowed = {(entry["package"], entry["id"]) for entry in allowlist}

    crates = cargo_packages()
    modules = npm_packages()
    print(f"scanning {len(crates)} crates + {len(modules)} npm packages against OSV ...")

    try:
        hits = query_osv("crates.io", crates)
        hits.update(query_osv("npm", modules))
    except RuntimeError as error:
        # A network problem must not turn the scheduled job red (see the file-header
        # comment), but it must leave a trace.
        print(f"WARN: {error}")
        print("WARN: advisory scan skipped -- re-run when the network is back")
        return 0

    accepted: list[tuple[str, str, list[str]]] = []
    unexpected: list[tuple[str, str, list[str]]] = []
    for (name, version), vulns in sorted(hits.items()):
        ids = [vuln.get("id", "?") for vuln in vulns]
        # [Adjudicate per ID; never green-light a whole package in one stroke] The
        # allowlist key is (package, id). The old form, `any(one id allowed ->
        # whole package accepted)`, let a newly disclosed advisory on the same
        # package be swallowed by that older waiver - the security gate stayed
        # green for ages, while a new vulnerability that actually needed review
        # never even appeared in the log. Collect separately: allowed ids go into
        # accepted, the rest into unexpected.
        allowed_ids = [vuln_id for vuln_id in ids if (name, vuln_id) in allowed]
        unallowed_ids = [vuln_id for vuln_id in ids if (name, vuln_id) not in allowed]
        if allowed_ids:
            accepted.append((name, version, allowed_ids))
        if unallowed_ids:
            unexpected.append((name, version, unallowed_ids))

    for name, version, ids in accepted:
        if args.verbose:
            print(f"  accepted  {name} {version}: {', '.join(ids)}")
    print(f"accepted (allow-listed): {len(accepted)}")
    print(f"unexpected: {len(unexpected)}")

    if unexpected:
        print("\nUnallowlisted advisories -- either upgrade the dependency or add a reviewed")
        print("entry to scripts/checks/advisory-allowlist.json with a reason:\n")
        for name, version, ids in unexpected:
            print(f"  {name} {version}: {', '.join(ids)}")
        return 1

    print("OK: no unallowlisted advisories")
    return 0


if __name__ == "__main__":
    sys.exit(main())
