#!/usr/bin/env python3
"""The gain layer is named in every payload, or the two knobs swap values.

REAPER has two gain knobs per clip: the item's `D_VOL` and the active take's
`D_VOL`.  They are independent, they multiply, and they are one word apart in
every payload.  A bare `gain` therefore has no answer to "which one is this?",
and the field produced exactly that: the kernel read it as the take layer, the
UI's optimistic update wrote it as the item layer, and the completion gate then
compared the two against each other.  The user dragged the knob to +12 dB, typed
0 dB in the inline editor, and got a pair of mirrored readings --
`gain=4.0 expected 1.0` and `gain=1.0 expected 4.0` -- with both layers wrong.

Nothing about that is a crash, a warning or a failed compile *at the moment it
is written*: `{ gain }` and `{ itemGain }` are one word apart, and a map typed
`Map<string, Record<string, unknown>>` accepts both.  That is why this is a
mechanical gate and not a code review habit.

The rules, and what each one protects:

  * the bulk-change type declares `itemGain` / `takeGain` and no bare `gain`
    (the name is the whole fix);
  * the change map is typed `Map<string, BulkClipRemoteChange>` (this is what
    turns a future bare `gain` into a compile error instead of a field bug);
  * both layers have exactly one constructor (`itemGainChange` /
    `takeGainChange`), so no call site writes the literal by hand;
  * the workspace projection writes the payload's `gain` from `g.take_gain`
    (writing it from `item_gain` is the "the projection put the item gain into
    the take layer" defect);
  * `ClipStatePatch::take_gain` keeps `alias = "gain"` (old projects and old
    frontends still send the bare key; dropping the alias silently ignores them);
  * the `[host] gain layers` diagnostic prints both layers by name (the next
    incident must not need a source read to answer "which one is which").

Exit code 0 when every rule holds.
"""

from __future__ import annotations

import pathlib
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = pathlib.Path(__file__).resolve().parents[2]

PAYLOAD_TYPE = "frontend/src/components/layout/timeline/hooks/bulkClipRemotePayloads.ts"
WORKSPACE = "backend/hifishifter-plugin/src/editor/workspace.rs"
PATCH = "backend/hifishifter-kernel/src/state/model.rs"
HOST_EDIT = "backend/hifishifter-plugin/src/host/reaper/host_edit.rs"

# Inline escape hatch, same convention as check-mobile-form-gate.py.
ESCAPE = "gain-gate: allow"

# (file, regex, why) -- every one of these must still be true.
REQUIRED = (
    (
        PAYLOAD_TYPE,
        r"itemGain\?\s*:\s*number\s*;",
        "the bulk-change type must name the item layer as itemGain",
    ),
    (
        PAYLOAD_TYPE,
        r"takeGain\?\s*:\s*number\s*;",
        "the bulk-change type must name the take layer as takeGain",
    ),
    (
        PAYLOAD_TYPE,
        r"export function itemGainChange\(",
        "the item layer needs one constructor, so call sites cannot write the literal",
    ),
    (
        PAYLOAD_TYPE,
        r"export function takeGainChange\(",
        "the take layer needs one constructor, so call sites cannot write the literal",
    ),
    (
        PAYLOAD_TYPE,
        r"changesById:\s*Map<string,\s*BulkClipRemoteChange>",
        "the change map must be typed BulkClipRemoteChange: that typing is what makes a "
        "bare gain key a compile error",
    ),
    # Both projections, one rule each: a single "some line writes g.take_gain" rule is
    # satisfied by the take line alone, which is exactly how the item line got it
    # wrong without this gate noticing.
    (
        WORKSPACE,
        r'clip\["gain"\]\s*=\s*serde_json::json!\(\s*g\.take_gain\s*\)',
        "the payload clip.gain field is the TAKE layer: it must be written from g.take_gain",
    ),
    (
        WORKSPACE,
        r'take\["gain"\]\s*=\s*serde_json::json!\(\s*g\.take_gain\s*\)',
        "the payload takes[].gain field is the take layer too",
    ),
    (
        PATCH,
        r'#\[serde\(default,\s*alias\s*=\s*"gain"\)\]',
        'ClipStatePatch must keep alias = "gain", or old projects and old frontends are '
        "silently ignored",
    ),
    (
        HOST_EDIT,
        r"item=\{.*\}.*take=",
        "the [host] gain layers line must print both layers by name",
    ),
)

# (file, regex, why) -- every one of these must stay absent.
FORBIDDEN = (
    (
        PAYLOAD_TYPE,
        r"^\s*gain\??\s*:\s*number\s*;",
        "the bulk-change type must not have a bare gain: it names no layer",
    ),
    # The projection defect, caught by its shape: the take-layer field written from
    # an item-layer value.  This is the one that made a +12 dB knob come back as
    # -12 dB, and it compiles cleanly because both sides are f64.
    (
        WORKSPACE,
        r'\["gain"\]\s*=\s*serde_json::json!\(\s*[a-z_.]*(item_gain|clip_gain)',
        "the take-layer gain field must never be written from the item layer",
    ),
)


def read(relative: str) -> str:
    path = ROOT / relative
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise SystemExit("check-gain-layers: cannot read %s: %s" % (path, error))


def escaped_lines(text: str) -> set[int]:
    return {
        number
        for number, line in enumerate(text.splitlines(), start=1)
        if ESCAPE in line
    }


def main() -> int:
    findings: list[str] = []
    for relative, pattern, why in REQUIRED:
        text = read(relative)
        skipped = escaped_lines(text)
        matched = False
        for number, line in enumerate(text.splitlines(), start=1):
            if number in skipped or re.search(pattern, line):
                matched = True
                break
        if not matched:
            findings.append("%s: missing /%s/ -- %s" % (relative, pattern, why))
    for relative, pattern, why in FORBIDDEN:
        text = read(relative)
        skipped = escaped_lines(text)
        for number, line in enumerate(text.splitlines(), start=1):
            if number in skipped:
                continue
            if re.search(pattern, line):
                findings.append("%s:%d: %s -- %s" % (relative, number, line.strip(), why))

    if findings:
        print("check-gain-layers: %d gain-layer violation(s):" % len(findings))
        for finding in findings:
            print("  - %s" % finding)
        print(
            "\nThe item layer and the take layer are two knobs in the host and one word\n"
            "apart in every payload.  Name the layer:\n"
            "    { itemGain }  // timeline knob, inline edit, double-click reset\n"
            "    { takeGain }  // take properties panel\n"
            "and build them through itemGainChange() / takeGainChange().\n"
            'If a line genuinely must stay as it is, append "' + ESCAPE + '".'
        )
        return 1

    print("check-gain-layers: every gain payload names its layer")
    return 0


if __name__ == "__main__":
    sys.exit(main())
