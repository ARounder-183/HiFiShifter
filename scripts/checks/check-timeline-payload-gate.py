#!/usr/bin/env python3
"""One place builds a timeline payload, so one place stamps its revision.

The frontend arbitrates out-of-order snapshot replies by comparing the
`revision` field the backend puts on every `TimelineStatePayload`
(see `TimelineStatePayload::revision` and `AppState::timeline_payload`).
That only works if *every* payload carries a number, and if the number is
taken while the timeline lock is held.

The failure this gate prevents is the quiet one: a new command builds its
payload with `tl.to_payload()`, forgets the revision, and the frontend falls
back to the legacy single-slot `requestId` comparison — where two different
actions invalidate each other and the user sees "I split the clip and it did
not split". Nothing crashes, nothing logs, and it only reproduces under
latency. So: exactly one construction site, enforced mechanically.

Allowed: `backend/src-tauri/src/state/app.rs` (the two constructors plus the
version counter). Everything else in the Tauri crate must call
`AppState::timeline_payload` or `state::unversioned_timeline_payload`.

The kernel and the plugin crate are out of scope: they have no `AppState`
and their payloads travel over the host bridge, not the WebView IPC.

Exit code 0 when no stray construction site exists.
"""

from __future__ import annotations

import pathlib
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCAN_ROOT = ROOT / "backend" / "src-tauri" / "src"

# The single construction site. `unversioned_timeline_payload` lives here too:
# it is the deliberate escape hatch for synthetic error/placeholder payloads,
# which must NOT carry a revision (a blank timeline with the newest revision
# would win every comparison and wipe the user's project).
EXEMPT = {
    "backend/src-tauri/src/state/app.rs",
}

PATTERN = re.compile(r"\.to_payload(?:_lite)?\(\)")

# Inline escape hatch, same convention as check-mobile-form-gate.py.
ESCAPE = "payload-gate: allow"


def main() -> int:
    if not SCAN_ROOT.is_dir():
        raise SystemExit(f"check-timeline-payload-gate: missing {SCAN_ROOT}")

    findings: list[tuple[str, int, str]] = []
    for path in sorted(SCAN_ROOT.rglob("*.rs")):
        relative = path.relative_to(ROOT).as_posix()
        if relative in EXEMPT:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            if ESCAPE in line:
                continue
            stripped = line.lstrip()
            if stripped.startswith("//") or stripped.startswith("*"):
                continue
            if PATTERN.search(line):
                findings.append((relative, number, line.strip()))

    if findings:
        print(
            f"check-timeline-payload-gate: {len(findings)} payload(s) built outside "
            "AppState::timeline_payload:"
        )
        for relative, number, snippet in findings:
            print(f"  - {relative}:{number}: {snippet}")
        print(
            "\nEvery timeline payload must carry a monotonic revision, stamped while the\n"
            "timeline lock is held:\n"
            "    let mut payload = state.timeline_payload(&tl);   // instead of tl.to_payload()\n"
            "Synthetic error payloads use state::unversioned_timeline_payload(&tl) —\n"
            "they must NOT carry a revision.\n"
            f'If a line genuinely must build a raw payload, append "{ESCAPE}".'
        )
        return 1

    print("check-timeline-payload-gate: every timeline payload comes from AppState::timeline_payload")
    return 0


if __name__ == "__main__":
    sys.exit(main())
