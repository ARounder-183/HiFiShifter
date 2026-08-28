import { test } from "vitest";

import {
    getDivisionFactors,
    getSppThresholds,
    waveformMipmapStore,
    type WaveformMipmapLevel,
} from "./waveformMipmapStore.js";

test("utils/waveformMipmapStore.test.ts scripted checks", async () => {
    function assertEqual(actual: unknown, expected: unknown, label: string): void {
        const actualJson = JSON.stringify(actual);
        const expectedJson = JSON.stringify(expected);
        if (actualJson !== expectedJson) {
            throw new Error(`${label}: expected ${expectedJson}, received ${actualJson}`);
        }
    }

    assertEqual(getSppThresholds(), [512, 1024], "updated spp thresholds");
    assertEqual(getDivisionFactors(), [16, 512, 4096], "updated division factors");

    assertEqual(waveformMipmapStore.selectLevel(512), 0, "L0 covers spp <= 512");
    assertEqual(waveformMipmapStore.selectLevel(513), 1, "L1 starts above 512");
    assertEqual(waveformMipmapStore.selectLevel(1024), 1, "L1 covers spp <= 1024");
    assertEqual(waveformMipmapStore.selectLevel(1025), 2, "L2 starts above 1024");

    assertEqual(
        waveformMipmapStore.selectLevelStable(641, 0 as WaveformMipmapLevel),
        1,
        "stable selector enters L1 at updated hysteresis boundary",
    );
    assertEqual(
        waveformMipmapStore.selectLevelStable(1281, 1 as WaveformMipmapLevel),
        2,
        "stable selector enters L2 at updated hysteresis boundary",
    );

});
