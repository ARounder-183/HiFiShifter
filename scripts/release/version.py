#!/usr/bin/env python3
"""Sync / verify the product version: one input, written to every file listed in the manifest
(the manifest is version-sources.json in this directory).

[Why .py instead of staying .ps1] Bumping the version is "read text → regex replace → write
back", with no platform API involved; keeping it in PowerShell would only mean macOS/Linux
contributors must first install a 100 MB pwsh to change a version number (this repo's
install_deps_* scripts do not install it, and the static gate only runs on windows-latest, so
nobody ever hit that gap). Python is already a hard three-platform dependency here:
fetch-models.py, check-*.py and vst3/unix/build.sh all need it, and the checker
(check-product-consistency.ps1) invokes it anyway.

[Why writing and checking live in the same file] Both sides must interpret, occurrence by
occurrence, "which spots are the product version" per version-sources.json. Writing it in two
languages means implementing the same rule twice — and if the rule misses one spot, the symptom
is merely some UI (or "Programs and Features") displaying a different version number.

Usage:
    python scripts/release/version.py --write 0.2.0-beta   # full version
    python scripts/release/version.py --write beta16       # shorthand: current major.minor.patch + -beta.16
    python scripts/release/version.py --write beta         # shorthand: current major.minor.patch + -beta
    python scripts/release/version.py --check              # verify all sources agree (exit code 1 on mismatch)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Windows stdout follows the ANSI code page (cp1252 on English systems, GBK on Chinese ones);
# printing non-ASCII text there can kill the process with UnicodeEncodeError — none of the real
# conclusions reach the log, and what turns red is the encoding, not the rule.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCES_PATH = Path(__file__).resolve().parent / "version-sources.json"


def load_sources() -> dict:
    if not SOURCES_PATH.is_file():
        raise SystemExit(f"Version manifest missing: {SOURCES_PATH}")
    with SOURCES_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_text(relative: str) -> str:
    # [Why newline=""] Keep each file's original line endings: writing CRLF on Windows when the
    # repo stores LF turns one version bump into a whole-file diff that shows a reviewer nothing.
    with (REPO_ROOT / relative).open("r", encoding="utf-8", newline="") as handle:
        return handle.read()


def write_text(relative: str, text: str) -> None:
    # Consistent with the rest of the repo's text files: UTF-8 without BOM.
    with (REPO_ROOT / relative).open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)


def source_label(source: dict, index: int = 0) -> str:
    name = source.get("id") or source["path"]
    return name if int(source["occurrences"]) == 1 else f"{name} #{index + 1}"


def canonical_version(manifest: dict) -> str:
    for source in manifest["sources"]:
        if source["path"] == manifest["canonical"]:
            match = re.search(source["pattern"], read_text(source["path"]))
            if not match:
                raise SystemExit(f"No version field found in {source['path']}")
            return match.group(2)
    raise SystemExit(f"version-sources.json canonical is not among sources: {manifest['canonical']}")


def resolve_target(requested: str, current: str) -> str:
    """Expand shorthands like beta / beta16 into a full version.

    [Why the prefix is captured separately] The earlier PowerShell implementation, after one
    match, went on to read the second group of the automatic match variables — a position that
    was empty — so -Version beta15 produced "0.1.0-beta.". Here the ordinal is taken first and
    the prefix matched independently, so the two steps cannot interfere.
    """
    match = re.fullmatch(r"(?i)beta(?:[.\-]?(\d+))?", requested)
    if not match:
        return requested
    prefix_match = re.match(r"^(\d+\.\d+\.\d+)-", current)
    if not prefix_match:
        raise SystemExit(f"Cannot derive a beta target version from current version {current}")
    ordinal = match.group(1)
    return f"{prefix_match.group(1)}-beta" + (f".{ordinal}" if ordinal else "")


def collect_sources(manifest: dict) -> list[tuple[dict, str, list[re.Match[str]]]]:
    """Read all sources and verify that the **first N** product-version fields exist; returns (source, text, matches).

    [Why "at least occurrences" rather than "exactly occurrences"] In the manifest, occurrences
    means "the first N regex hits are the product version", not that the file may contain only N
    matches: the `frontend/package-lock.json` pattern also hits hundreds of dependency version
    fields, and `re.sub(..., count=N)` only rewrites the first two (top level + packages[""]).
    The check must follow the same rule, or legitimate dependency versions get treated as drift.
    """
    collected: list[tuple[dict, str, list[re.Match[str]]]] = []
    for source in manifest["sources"]:
        text = read_text(source["path"])
        expected = int(source["occurrences"])
        matches = list(re.finditer(source["pattern"], text))
        if len(matches) < expected:
            raise SystemExit(
                f"Found only {len(matches)} version field(s) in {source['path']}, "
                f"version-sources.json requires at least {expected}"
            )
        collected.append((source, text, matches))
    return collected


def write_version(requested: str) -> int:
    manifest = load_sources()
    current = canonical_version(manifest)
    version = resolve_target(requested, current)

    # [Two-phase] Validate and compute every write first, then write all at once. The old
    # implementation wrote while validating: if the 5th source suddenly lost one version field,
    # the first 4 files were already rewritten to the new version and the repo was stuck in a
    # half-upgraded state — which --check of the time could not detect as this "one extra/missing"
    # drift.
    pending: list[tuple[str, str]] = []
    for source, text, _matches in collect_sources(manifest):
        occurrences = int(source["occurrences"])
        updated = re.sub(
            source["pattern"],
            lambda match, v=version: match.group(1) + v + match.group(3),
            text,
            count=occurrences,
        )
        pending.append((source["path"], updated))

    for path, updated in pending:
        write_text(path, updated)

    print(f"Version: {current} -> {version}")
    return 0


def check_version() -> int:
    manifest = load_sources()
    try:
        collected = collect_sources(manifest)
    except SystemExit as error:
        print(str(error), file=sys.stderr)
        return 1

    found: list[tuple[str, str]] = []
    for source, _text, matches in collected:
        # Same rule as the writer: compare only the first occurrences hits (the rest are dependency versions).
        for index in range(int(source["occurrences"])):
            found.append((source_label(source, index), matches[index].group(2)))

    distinct = sorted({value for _, value in found})
    if len(distinct) != 1:
        report = " | ".join(f"{label} = {value}" for label, value in found)
        print(f"Version mismatch (run python scripts/release/version.py --write <version>): {report}", file=sys.stderr)
        return 1
    print(f"All sources = {distinct[0]} ({len(found)} occurrence(s))")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync / verify the HiFiShifter product version")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", metavar="VERSION", help="write the version (beta / beta16 shorthand supported)")
    group.add_argument("--check", action="store_true", help="only verify that all sources agree")
    args = parser.parse_args()

    if args.check:
        return check_version()
    return write_version(args.write)


if __name__ == "__main__":
    raise SystemExit(main())
