import { test } from "vitest";

import reducer from "./sessionSlice.ts";
import { fetchTimeline } from "./thunks/transportThunks.ts";

test("features/session/sessionSlice.formantMorph.test.ts scripted checks", async () => {
    function assertEqual(actual: unknown, expected: unknown, label: string): void {
        if (actual !== expected) {
            throw new Error(`${label}: expected ${String(expected)}, received ${String(actual)}`);
        }
    }

    const baseState = reducer(undefined, { type: "@@INIT" });

    const next = reducer(
        baseState,
        fetchTimeline.fulfilled(
            {
                ok: true,
                bpm: 120,
                playhead_sec: 0,
                project_sec: 8,
                selected_track_id: "track_main",
                selected_clip_id: "clip-1",
                tracks: [
                    {
                        id: "track_main",
                        name: "Main",
                        muted: false,
                        solo: false,
                        volume: 1,
                        compose_enabled: false,
                        pitch_analysis_algo: "nsf_hifigan_onnx",
                        color: "#ffffff",
                    },
                ],
                clips: [
                    {
                        id: "clip-1",
                        track_id: "track_main",
                        name: "clip",
                        start_sec: 0,
                        length_sec: 1,
                        color: "emerald",
                        source_path: "demo.wav",
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
                        formant_morph: {
                            enabled: true,
                            target_f1_hz: 700,
                            target_f2_hz: 1700,
                            strength: 0.6,
                        },
                    },
                ],
            } as any,
            "req-formant-morph",
            undefined,
        ),
    );

    assertEqual(next.clips[0]?.formantMorph?.enabled, true, "formantMorph enabled");
    assertEqual(next.clips[0]?.formantMorph?.targetF1Hz, 700, "formantMorph targetF1Hz");
    assertEqual(next.clips[0]?.formantMorph?.targetF2Hz, 1700, "formantMorph targetF2Hz");
    assertEqual(next.clips[0]?.formantMorph?.strength, 0.6, "formantMorph strength");

});
