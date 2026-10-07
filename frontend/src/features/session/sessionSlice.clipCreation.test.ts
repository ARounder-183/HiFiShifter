import { test } from "vitest";

import reducer from "./sessionSlice.ts";
import { importAudioAtPosition } from "./thunks/importThunks.ts";
import type { TimelineResult } from "../../types/api";

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
            // 已从时间线删除的 clip 仍留在曲线 map 里：导入快照应用后必须被
            // 剪掉，不能把整包旧 map 原样恢复回来。
            "clip-orphan": {
                curveStartSec: 0,
                midiCurve: [40, 41, 42],
                framePeriodMs: 5,
            },
        },
    } as unknown as ReturnType<typeof reducer>;

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
                            // 快照里仍然存在的既有 clip：它的曲线必须被保留。
                            id: "clip-a",
                            track_id: "track_main",
                            name: "Existing Clip",
                            start_sec: 0,
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
                } as unknown as TimelineResult,
                newClipIds: ["clip-b"],
                playheadSec: 4,
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
        { "clip-a": baseState.clipPitchCurves["clip-a"] },
        "import keeps pitch curves of surviving clips and prunes deleted ones",
    );
});
