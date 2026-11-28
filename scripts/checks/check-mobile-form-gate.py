#!/usr/bin/env python3
"""One question, one answer: which form is this app running in?

The mobile adaptation hinges on a single predicate, `appForm()` in
`frontend/src/services/hostDetect.ts`. Everything else — the shell, the dock
profile, the control density, the input profile, the CSS token layer, the
gesture layer — reads it.

The failure this gate prevents is not "somebody wrote a bad width check". It is
that a *second* answer exists: `window.innerWidth < 600`, a `navigator.userAgent`
sniff, a `matchMedia("(pointer: coarse)")`. Each of those is right on the
developer's device and wrong on somebody else's, and the symptom is "the phone
layout sometimes does not appear" — which cannot be reproduced on demand.

Banned here (outside tests and the gate itself):
  * `innerWidth` / `innerHeight` compared against a number
  * `navigator.userAgent` / `userAgentData` used for form detection
  * `matchMedia` with pointer/hover media features

Exit code 0 when no such expression exists.
"""

from __future__ import annotations

import pathlib
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCAN_ROOT = ROOT / "frontend" / "src"

# Files allowed to answer the question by other means:
#   hostDetect.ts        - the gate itself
#   surfaceProfileStore  - holds the backend answer
#   *test* files         - they simulate forms on purpose
EXEMPT = {
    # The gate itself and the store that holds the backend's answer.
    "frontend/src/services/hostDetect.ts",
    "frontend/src/services/surfaceProfileStore.ts",
    # OS detection for modifier-key semantics (macOS Command vs Control) and for
    # hiding features that depend on Windows-only third-party software. That is a
    # different question from "which form is this" — it must stay UA-based,
    # because it is about the *keyboard convention* the user expects, not about
    # how much screen there is.
    "frontend/src/utils/platform.ts",
    # Records the UA and DPR into the diagnostics bundle. It reports, it does not
    # decide.
    "frontend/src/components/layout/renderKernel/gl/glDiagnostics.ts",
}

RULES: list[tuple[str, re.Pattern[str], str]] = [
    (
        "viewport-width form check",
        re.compile(r"(?:innerWidth|innerHeight|clientWidth)\s*[<>]=?\s*\d"),
        "use appForm() / surfaceCapability(...) instead of measuring the window",
    ),
    (
        "user-agent form check",
        re.compile(r"navigator\.(?:userAgent|userAgentData)"),
        "the backend already tells us the form; do not sniff the UA",
    ),
    (
        "pointer media-query form check",
        re.compile(r"matchMedia\(\s*[\"'`][^\"'`]*\((?:pointer|hover)"),
        "use surfaceProfile().input instead of a media query",
    ),
]

# Inline escape hatch: append this comment to a line that legitimately needs a
# measurement (e.g. a tooltip that must stay inside the viewport). It is not a
# form decision, so the gate has nothing to say about it.
ESCAPE = "form-gate: allow"


def main() -> int:
    if not SCAN_ROOT.is_dir():
        raise SystemExit(f"check-mobile-form-gate: missing {SCAN_ROOT}")

    findings: list[tuple[str, int, str, str]] = []
    for path in sorted(SCAN_ROOT.rglob("*")):
        if path.suffix not in {".ts", ".tsx"}:
            continue
        relative = path.relative_to(ROOT).as_posix()
        if relative in EXEMPT:
            continue
        if ".test." in relative or ".bench." in relative:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            if ESCAPE in line:
                continue
            # Skip pure comments: the rule is about code, and the repo's
            # comments discuss these very APIs.
            stripped = line.lstrip()
            if stripped.startswith("//") or stripped.startswith("*"):
                continue
            for label, pattern, hint in RULES:
                if pattern.search(line):
                    findings.append((relative, number, label, hint))
                    break

    if findings:
        print(f"check-mobile-form-gate: {len(findings)} form decision(s) outside appForm():")
        for relative, number, label, hint in findings:
            print(f"  - {relative}:{number}: {label} -> {hint}")
        print(
            "\nThe application form has exactly one answer (appForm in\n"
            "services/hostDetect.ts) and capabilities have one table\n"
            "(services/surfaceCapabilities.ts). If the measurement is not a form\n"
            f"decision, append \"{ESCAPE}\" to the line."
        )
        return 1

    print("check-mobile-form-gate: no shadow form detection found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
