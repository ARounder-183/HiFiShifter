#!/usr/bin/env python3
"""The two canvases must be able to answer "is this input a finger?".

Why this gate exists
--------------------
The user reported that scrolling on the phone was unusable. The investigation
found the reason in one number: **zero** occurrences of the string "touch" in
`components/layout/timeline` and `components/layout/pianoRoll`. Every touch
adaptation lived in the shell (top bar, drawer, menus); the moment you entered a
canvas the code went back to desktop semantics -- wheel to scroll, right button
for the menu, drag-on-empty to box-select.

That is not a bug you can fix once. It is a *property of the layer*: any new
gesture added to a canvas will silently be mouse-only unless somebody remembers.
So the layer has to be able to answer the question at all, and that is what this
gate checks: each canvas directory must contain a real touch branch.

What counts
-----------
A file under the canvas directory mentioning `pointerType` (the only reliable
way to tell a finger from a mouse) or `pointerType === "touch"`. Presence of the
capability bit (`two_finger_gesture`) also counts, because that is how the touch
layer is mounted.

Exit code 0 when both canvases have a touch branch.
"""

from __future__ import annotations

import pathlib
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = pathlib.Path(__file__).resolve().parents[2]
CANVASES = {
    "timeline": ROOT / "frontend" / "src" / "components" / "layout" / "timeline",
    "pianoRoll": ROOT / "frontend" / "src" / "components" / "layout" / "pianoRoll",
}

# Evidence that a canvas can distinguish a finger from a mouse.
MARKERS = (
    'pointerType === "touch"',
    'pointerType !== "touch"',
    "two_finger_gesture",
    "useViewportGestures",
    "resolveTouchIntent",
)


def main() -> int:
    missing: list[str] = []
    for name, root in CANVASES.items():
        if not root.is_dir():
            raise SystemExit(f"check-touch-coverage: missing {root}")
        found = False
        for path in sorted(root.rglob("*.ts*")):
            if ".test." in path.name or ".bench." in path.name:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if any(marker in text for marker in MARKERS):
                found = True
                break
        if not found:
            missing.append(name)

    if missing:
        print(
            "check-touch-coverage: these canvases have no touch branch at all "
            f"({len(missing)}):"
        )
        for name in missing:
            print(f"  - {name}")
        print(
            "\nA canvas that never asks \"is this a finger?\" is a canvas where every\n"
            "new gesture is silently mouse-only. Mount the touch layer\n"
            "(features/touch/useViewportGestures) or branch on pointerType."
        )
        return 1

    print("check-touch-coverage: both canvases can tell a finger from a mouse")
    return 0


if __name__ == "__main__":
    sys.exit(main())
