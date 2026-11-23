#!/usr/bin/env python3
"""Keep `cfg(target_os = ...)` out of the app crate's business code.

The mobile adaptation introduced exactly one place that is allowed to ask "what
machine is this": `backend/src-tauri/src/platform/`. Everywhere else the
question must go through it (`platform::form()`, `platform::profile()`, one of
the facades).

Why a gate rather than a convention: a stray `cfg(target_os = "android")` in
business code raises no error on the developer's machine. It only shows up when
somebody builds for a target they are not on — and then as an unrelated link or
compile failure. The rule is cheap to check and expensive to discover.

Baseline: the crate already had platform switches before the convergence layer
existed (the macOS menu, the Linux desktop preparation, ...). They are recorded
in `platform-boundary-allowlist.txt` with a reason each, the same shape
`check-ffi-guards.py` uses for its SCAN_ROOTS. A new site fails the gate unless
it is added there — and adding it requires writing down why it cannot move.

Exit code 0 when every site is either inside `platform/` or allowlisted.
"""

from __future__ import annotations

import pathlib
import re
import sys

# On Windows, stdout follows the ANSI code page; printing characters outside it
# raises UnicodeEncodeError and kills the process before a single finding shows.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCAN_ROOT = ROOT / "backend" / "src-tauri" / "src"
ALLOWED_DIR = SCAN_ROOT / "platform"
ALLOWLIST = pathlib.Path(__file__).with_name("platform-boundary-allowlist.txt")

# `cfg!(target_os = ...)`, `#[cfg(target_os = ...)]`, and the `all(...)` /
# `any(...)` forms that embed it. Matching the attribute opener plus the token
# is enough: every real occurrence contains "target_os" somewhere in the same
# attribute, and a false positive here costs one allowlist line.
PATTERN = re.compile(r"(?:cfg!?\s*\(|#\[cfg\s*\()[^\n]*target_os")

# Test modules may assert platform-specific behaviour; they never ship.
TEST_PATH = re.compile(r"#\[cfg\(test\)\]|/tests?/")


def allowlisted() -> set[tuple[str, str]]:
    """(relative path, trimmed line) pairs recorded as intentional."""
    if not ALLOWLIST.is_file():
        return set()
    entries: set[tuple[str, str]] = set()
    for raw in ALLOWLIST.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # Format: <relative path> :: <trimmed source line>
        if "::" not in line:
            raise SystemExit(
                f"check-platform-boundary: malformed allowlist entry: {raw!r}"
            )
        path, snippet = line.split("::", 1)
        entries.add((path.strip(), snippet.strip()))
    return entries


def main() -> int:
    if not SCAN_ROOT.is_dir():
        raise SystemExit(f"check-platform-boundary: missing {SCAN_ROOT}")

    allowed = allowlisted()
    findings: list[tuple[str, int, str]] = []
    seen: set[tuple[str, str]] = set()

    for path in sorted(SCAN_ROOT.rglob("*.rs")):
        if ALLOWED_DIR in path.parents:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        relative = path.relative_to(ROOT).as_posix()
        for number, line in enumerate(text.splitlines(), start=1):
            if "target_os" not in line:
                continue
            if not PATTERN.search(line):
                continue
            stripped = line.strip()
            # Comments discuss this very API (the convergence layer's own docs do),
            # so only code counts.
            if stripped.startswith("//") or stripped.startswith("*") or stripped.startswith("/*"):
                continue
            if (relative, stripped) in allowed:
                seen.add((relative, stripped))
                continue
            findings.append((relative, number, stripped))

    if findings:
        print(
            "check-platform-boundary: cfg(target_os) outside platform/ "
            f"({len(findings)} site(s)):"
        )
        for relative, number, line in findings:
            print(f"  - {relative}:{number}: {line}")
        print(
            "\nMove the branch into backend/src-tauri/src/platform/ and expose a\n"
            "facade (see platform/mod.rs). If it genuinely cannot move, add the\n"
            "exact line to scripts/checks/platform-boundary-allowlist.txt with a\n"
            "reason comment above it."
        )
        return 1

    stale = allowed - seen
    if stale:
        # A stale allowlist is worse than none: it silently absorbs a future
        # regression on the same line.
        print("check-platform-boundary: stale allowlist entries (delete them):")
        for relative, line in sorted(stale):
            print(f"  - {relative} :: {line}")
        return 1

    print("check-platform-boundary: every cfg(target_os) lives in platform/ or is allowlisted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
