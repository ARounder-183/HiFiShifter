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
    incident must not need a source read to answer "which one is which");
  * the IPC boundary names the layer too (`take_gain` on the `set_clip_state`
    command, which Tauri maps to the `takeGain` key the frontend sends);
  * no `ClipStatePatch` literal in the backend passes a bare `gain` field.

[Why the last one is a gate and not just a compile error] Renaming the kernel field
left one construction site behind in `src-tauri` -- a crate that the fast test loop
never compiles. The compiler does catch it, but only after a full app build; the
gate catches it in a tenth of a second, and it names the field that has to change.

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
# The IPC surface: the command wrapper (`#[tauri::command(rename_all = "camelCase")]`,
# so `take_gain` is the `takeGain` key the frontend sends) and its implementation.
COMMAND_SURFACE = "backend/src-tauri/src/commands.rs"
COMMAND_IMPL = "backend/src-tauri/src/commands/timeline.rs"

# Where a `ClipStatePatch` literal can be written: every workspace member's source tree.
CONSTRUCTION_SITES = ("backend/*/src/**/*.rs",)

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
    (
        COMMAND_SURFACE,
        r"^\s*take_gain:\s*Option<f32>,",
        "the set_clip_state command must name the take layer take_gain (the frontend "
        "sends it as takeGain)",
    ),
    (
        COMMAND_IMPL,
        r"^\s*take_gain:\s*Option<f32>,",
        "the command implementation must name the take layer take_gain as well",
    ),

    # ── 分层读数入口：所有消费点只许从这四个函数取 ─────────────────────
    (
        "frontend/src/components/layout/timeline/clipGain.ts",
        r"export function clipItemGain\(",
        "the item-layer reader must exist and be the only way to read clip_gain",
    ),
    (
        "frontend/src/components/layout/timeline/clipGain.ts",
        r"export function clipTakeGain\(",
        "the take-layer reader must preserve polarity",
    ),
    (
        "frontend/src/components/layout/timeline/clipGain.ts",
        r"export function clipAudibleGain\(",
        "the audible reader must be item × take",
    ),
    (
        "frontend/src/components/layout/timeline/clipGain.ts",
        r"export function clipWaveformGain\(",
        "the waveform reader must be |item × take|",
    ),
    (
        "frontend/src/components/layout/timeline/takeLanes.ts",
        r"clipWaveformGain\(clip\)",
        "single-lane waveform must include the item layer",
    ),
    (
        "frontend/src/components/layout/timeline/takeLanes.ts",
        r"clipTakeWaveformGain\(clip,\s*take\.gain\)",
        "multi-lane waveform must include item × its own take",
    ),
    (
        "frontend/src/components/layout/pianoRoll/useClipsPeaksForPianoRoll.ts",
        r"clipWaveformGain\(clip\)",
        "parameter-editor waveform must include the item layer",
    ),
    (
        "frontend/src/components/layout/timeline/hooks/useTimelineKernelFade.ts",
        r'case "gain-knob":',
        "the gain-knob tooltip branch must exist",
    ),
    (
        "frontend/src/components/layout/timeline/hooks/useTimelineKernelFade.ts",
        r"clipItemGain\(clip\)",
        "the gain-knob tooltip must read the item layer",
    ),
    (
        "backend/src-tauri/src/audio_engine/snapshot.rs",
        r"clip\.render_gain\(gain_scope\)",
        "the live engine must apply the scope-aware gain, not the take field",
    ),
    (
        "backend/hifishifter-kernel/src/pitch_analysis/dyn_analysis.rs",
        r"clip\.render_gain\(gain_scope\)",
        "dyn analysis must use the same scope as rendering",
    ),
    (
        "backend/hifishifter-kernel/src/pitch_analysis/dyn_analysis.rs",
        r"clip\.clip_gain",
        "the dyn cache key must hash the item layer",
    ),
    (
        "backend/hifishifter-kernel/src/state/model.rs",
        r"glued\.clip_gain\s*=\s*1\.0;",
        "glue must reset the item layer after baking it into the WAV",
    ),
    (
        "backend/hifishifter-kernel/src/state/model.rs",
        r"source\.item_gain\(\)",
        "packing clips into takes must fold the old item layer into each take",
    ),
    (
        "frontend/src/features/session/sessionSlice/settings.ts",
        r"MAX_TRACK_VOLUME\s*=\s*GAIN_MAX_LINEAR",
        "the track fader must share the same 16× ceiling as the kernel",
    ),
)

# (file, regex, why) -- every one of these must stay absent.
FORBIDDEN = (
    (
        "frontend/src/utils/gainDb.ts",
        r"gainDbFromKnobTravel",
        "the old dB-returning gesture is forbidden: silence must be linear 0",
    ),
    (
        "frontend/src/components/layout/timeline/hooks/useTimelineKernelGainCrossfade.ts",
        r"gainDbFromKnobTravel",
        "the gain drag must use resolveGainDrag, not the old dB gesture",
    ),
    (
        "frontend/src/components/layout/timeline/hooks/useTimelineKernelFade.ts",
        r"clip\.gain\b",
        "the hover tooltip must not read the take projection as Clip gain",
    ),
    (
        "frontend/src/components/layout/timeline/hooks/useTimelineKernelGainCrossfade.ts",
        r"clip\.gain\b",
        "the badge initial value must not read the take projection",
    ),
    (
        "frontend/src/components/layout/pianoRoll/useClipsPeaksForPianoRoll.ts",
        r"gain:\s*clip\.gain",
        "the parameter-editor waveform must not read the take projection",
    ),
    (
        "frontend/src/components/layout/timeline/takeLanes.ts",
        r"gain:\s*take\.gain",
        "take lanes must include the item layer, not take alone",
    ),
    (
        "backend/src-tauri/src/audio_engine/snapshot.rs",
        r"clip\.gain\.max",
        "the live engine must not read the raw take projection",
    ),
    (
        "backend/hifishifter-kernel/src/state/model.rs",
        r"take[^\n]*clamp_gain\(",
        "take gain must use clamp_take_gain (polarity-preserving), never clamp_gain",
    ),
    (
        "frontend/src/features/session/sessionSlice/settings.ts",
        r"MAX_TRACK_VOLUME\s*=\s*4\b",
        "the track fader ceiling must not regress to 4×",
    ),
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
    (
        COMMAND_SURFACE,
        r"^\s*gain:\s*Option<f32>,",
        "the set_clip_state command must not take a bare gain: it names no layer",
    ),
)

# A bare `gain` field inside a ClipStatePatch literal is not a style problem: the
# struct has no such field, so the line is a compile error that only surfaces when the
# crate that holds it is built.  `src-tauri` is not part of the fast test loop, which
# is exactly how a kernel rename left one behind.
BARE_GAIN_FIELD = re.compile(r"^\s*gain\s*[,:]")
PATCH_LITERAL = re.compile(r"ClipStatePatch\s*\{")


def read(relative: str) -> str:
    path = ROOT / relative
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise SystemExit("check-gain-layers: cannot read %s: %s" % (path, error))


def patch_literals(text: str) -> list[tuple[int, str]]:
    """Return (first line number, literal text) for every ClipStatePatch literal.

    Brace counting is enough: the fields of a struct literal are identifiers, so no
    literal here contains an unbalanced brace inside a string.
    """
    blocks: list[tuple[int, str]] = []
    for match in PATCH_LITERAL.finditer(text):
        depth = 0
        end = None
        for index in range(match.end() - 1, len(text)):
            char = text[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end = index + 1
                    break
        if end is None:
            continue
        blocks.append((text.count("\n", 0, match.start()) + 1, text[match.start():end]))
    return blocks


def construction_sites() -> list[str]:
    paths: set[str] = set()
    for pattern in CONSTRUCTION_SITES:
        for path in ROOT.glob(pattern):
            paths.add(str(path.relative_to(ROOT)).replace("\\", "/"))
    return sorted(paths)


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
    seen: set[str] = set()
    for relative in construction_sites():
        text = read(relative)
        skipped = escaped_lines(text)
        for start, block in patch_literals(text):
            for offset, line in enumerate(block.splitlines()):
                number = start + offset
                if number in skipped or not BARE_GAIN_FIELD.match(line):
                    continue
                finding = (
                    "%s:%d: %s -- a ClipStatePatch literal cannot carry a bare gain: "
                    "the take layer is take_gain, the item layer is item_gain"
                    % (relative, number, line.strip())
                )
                if finding not in seen:
                    seen.add(finding)
                    findings.append(finding)

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
