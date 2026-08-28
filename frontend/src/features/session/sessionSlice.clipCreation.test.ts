import { test } from "vitest";

import reducer from "./sessionSlice.ts";
import { importAudioAtPosition } from "./thunks/importThunks.ts";

test("features/session/sessionSlice.clipCreation.test.ts scripted checks", async () => {
    function assertEqual(actual: unknown, expected: unknown, label: string): void {
        const actualJson = JSON.stringify(actual);
        const expectedJson = JSON.stringify(expected);
        if (actualJson !== expectedJson) {
            throw new Error(`${label}: expected ${expectedJson}, received ${actualJson}`);
        }
    }

    const baseState = {
        ...reducer(undefined, { type: "@@INIT" }),
        paramsEpoch: 7,
        clipPitchCurves: {
            "clip-a": {
                curveStartSec: 1,
                midiCurve: [60, 61, 62],
                framePeriodMs: 5,
            },
        },
    } as any;

    const next = reducer(
        baseState,
        importAudioAtPosition.fulfilled(
            {
                ok: true,
                imported: {
                    ok: true,
                    bpm: 120,
                    playhead_sec: 0,
                    project_sec: 30,
                    selected_track_id: "track_main",
                    selected_clip_id: "clip-b",
                    tracks: [
                        {
                            id: "track_main",
                            name: "Main",
                            muted: false,
                            solo: false,
                            volume: 1,
                            compose_enabled: false,
                            pitch_analysis_algo: "nsf_hifigan_onnx",
                        },
                    ],
                    clips: [
                        {
                            id: "clip-b",
                            track_id: "track_main",
                            name: "New Clip",
                            start_sec: 4,
                            length_sec: 1,
                            color: "emerald",
                            source_path: "voice.wav",
                            duration_sec: 1,
                            gain: 1,
                            muted: false,
                            source_start_sec: 0,
                            source_end_sec: 1,
                            playback_rate: 1,
                            reversed: false,
                            fade_in_sec: 0,
                            fade_out_sec: 0,
                            fade_in_shape: 5,
                            fade_out_shape: 5,
                            fade_in_dir: 0,
                            fade_out_dir: 0,
                        },
                    ],
                } as any,
                newClipIds: ["clip-b"],
            },
            "req-import",
            {
                audioPath: "voice.wav",
                trackId: "track_main",
                startSec: 4,
            },
        ),
    );

    assertEqual(next.paramsEpoch, 7, "import keeps param epoch stable");
    assertEqual(
        next.clipPitchCurves,
        baseState.clipPitchCurves,
        "import keeps detected pitch curves stable",
    );
});
