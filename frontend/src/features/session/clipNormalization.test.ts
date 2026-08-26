import { test } from "vitest";

import {
    computeClipNormalizationGain,
    computeNormalizationGainFromInterleaved,
} from "./clipNormalization.ts";

test("features/session/clipNormalization.test.ts scripted checks", async () => {
    function assertNear(actual: number | null, expected: number, label: string): void {
        if (actual == null || Math.abs(actual - expected) > 1e-6) {
            throw new Error(`${label}: expected ${expected}, received ${String(actual)}`);
        }
    }

    function assertEqual(actual: unknown, expected: unknown, label: string): void {
        if (actual !== expected) {
            throw new Error(`${label}: expected ${String(expected)}, received ${String(actual)}`);
        }
    }

    assertNear(
        computeNormalizationGainFromInterleaved([0.25, -0.5, 0.4, -0.2]),
        2,
        "normalize gain follows peak",
    );

    assertNear(
        computeNormalizationGainFromInterleaved([0.001, -0.001]),
        3.9810717055349722,
        "normalize gain is clamped to +12 dB",
    );

    assertEqual(
        computeNormalizationGainFromInterleaved([0, 0, 0, 0]),
        null,
        "silent data is ignored",
    );

    let released = false;
    assertNear(
        computeClipNormalizationGain(
            {
                sourcePath: "voice.wav",
                durationSec: 2,
                lengthSec: 1,
                sourceStartSec: 0,
                sourceEndSec: 1,
                playbackRate: 1,
            },
            {
                getInterleavedSlice: () => ({ interleaved: [0.5, -0.25] }),
                releaseInterleaved: () => {
                    released = true;
                },
            },
        ),
        2,
        "clip normalization uses waveform slice data",
    );

    assertEqual(released, true, "slice buffer is released");

    console.log("clip normalization checks passed");
});
