#!/usr/bin/env python3
"""Mechanical gate for the transient gain-audition channel.

A gain drag has two independent paths:

  * the optimistic Redux value, which only paints the UI;
  * the authoritative commit, which writes undo / the timeline / the host.

The audition channel is a third, deliberately disposable path.  If it is wired
into the commit path (or the commit path is wired into the audio callback), the
symptom is not a compile error: dragging becomes laggy, undo grows one step per
frame, or the audio thread starts allocating.  These rules keep the three paths
separate.

Exit code 0 when every rule holds.
"""

from __future__ import annotations

import pathlib
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = pathlib.Path(__file__).resolve().parents[2]

REQUIRED = (
    (
        "backend/hifishifter-kernel/src/engine_command.rs",
        r"SetGainPreview\s*\{",
        "the kernel command channel must carry SetGainPreview",
    ),
    (
        "backend/hifishifter-kernel/src/engine_command.rs",
        r"ClearGainPreview\s*\{",
        "the kernel command channel must carry ClearGainPreview",
    ),
    (
        "backend/src-tauri/src/commands/gain_preview.rs",
        r"fn set_gain_preview\(",
        "the Tauri command set_gain_preview must exist",
    ),
    (
        "backend/src-tauri/src/commands/gain_preview.rs",
        r"fn clear_gain_preview\(",
        "the Tauri command clear_gain_preview must exist",
    ),
    (
        "backend/src-tauri/src/lib.rs",
        r"commands::gain_preview::set_gain_preview",
        "set_gain_preview must be registered in generate_handler!",
    ),
    (
        "backend/src-tauri/src/audio_engine/types.rs",
        r"struct LiveGainFrame",
        "the engine needs a prebuilt LiveGainFrame",
    ),
    (
        "backend/src-tauri/src/audio_engine/types.rs",
        r"struct LiveGainRampState",
        "the engine needs a preallocated 5ms ramp state",
    ),
    (
        "backend/src-tauri/src/audio_engine/mix.rs",
        r"const LIVE_GAIN_RAMP_MS: f32 = 5\.0;",
        "the audition ramp must stay 5ms",
    ),
    (
        "backend/src-tauri/src/audio_engine/mix.rs",
        r"struct LiveGainPlayback",
        "the mixer must accept a LiveGainPlayback view",
    ),
    (
        "backend/src-tauri/src/audio_engine/mix.rs",
        r"Arc::ptr_eq\(&live_frame\.snapshot, &snap\)",
        "the mixer must reject a live frame that belongs to a previous snapshot",
    ),
    (
        "frontend/src/features/session/gainAudition.ts",
        r"export class GainAuditionController",
        "the frontend needs the rAF-coalescing audition controller",
    ),
    (
        "frontend/src/components/layout/timeline/hooks/useGainAudition.ts",
        r"export function useGainAudition",
        "the React wrapper must exist",
    ),
    (
        "frontend/src/components/layout/timeline/hooks/useTimelineKernelGainCrossfade.ts",
        r"audition\.preview\(",
        "clip gain drag must preview every frame",
    ),
    (
        "frontend/src/components/layout/timeline/hooks/useTimelineKernelGainCrossfade.ts",
        r"audition\.commit\(",
        "clip gain drag must clear its preview after the authoritative commit",
    ),
    (
        "frontend/src/components/layout/timeline/hooks/useTimelineKernelGainCrossfade.ts",
        r"audition\.cancel\(",
        "clip gain cancel must discard the preview",
    ),
    (
        "frontend/src/components/layout/timeline/TrackList.tsx",
        r"audition\?\.preview",
        "track fader drag/wheel must preview",
    ),
    (
        "frontend/src/components/layout/timeline/hooks/useTimelineKernelFade.ts",
        r"audition\.preview\(",
        "fade handles must preview",
    ),
    (
        "frontend/src/components/layout/pianoRoll/useLiveParamEditing.ts",
        r'kind: "trackCurve"',
        "volume automation strokes must preview through a track curve window",
    ),
    (
        "frontend/src/services/api/timeline.ts",
        r'"set_gain_preview"',
        "the frontend API must call set_gain_preview",
    ),
)

FORBIDDEN = (
    (
        "backend/hifishifter-plugin/src",
        r"set_gain_preview",
        "the ARA plugin must never call the App-only audition command",
    ),
    (
        "frontend/src/features/session/gainAudition.ts",
        r"dispatch\(",
        "the audition controller must not write Redux",
    ),
    (
        "frontend/src/features/session/gainAudition.ts",
        r"checkpoint",
        "the audition controller must not create undo checkpoints",
    ),
)

PLUGIN_ROOT = "backend/hifishifter-plugin/src"


def read(relative: str) -> str:
    path = ROOT / relative
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise SystemExit("check-gain-audition: cannot read %s: %s" % (path, error))


def iter_files(relative: str):
    path = ROOT / relative
    if path.is_file():
        yield path
        return
    for child in sorted(path.rglob("*.rs")):
        yield child


def main() -> int:
    findings: list[str] = []
    for relative, pattern, why in REQUIRED:
        text = read(relative)
        import re

        if re.search(pattern, text, re.MULTILINE) is None:
            findings.append("%s: missing /%s/ -- %s" % (relative, pattern, why))
    import re

    for relative, pattern, why in FORBIDDEN:
        for path in iter_files(relative):
            text = path.read_text(encoding="utf-8", errors="replace")
            for number, line in enumerate(text.splitlines(), start=1):
                if re.search(pattern, line):
                    findings.append(
                        "%s:%d: %s -- %s"
                        % (path.relative_to(ROOT).as_posix(), number, line.strip(), why)
                    )
    if findings:
        print("check-gain-audition: %d audition violation(s):" % len(findings))
        for finding in findings:
            print("  - %s" % finding)
        return 1
    print("check-gain-audition: transient audition is separate from authority")
    return 0


if __name__ == "__main__":
    sys.exit(main())
