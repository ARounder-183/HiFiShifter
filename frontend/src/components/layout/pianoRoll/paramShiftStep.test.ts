import { test } from "vitest";

import { getParamShiftStep } from "./paramShiftStep.js";

test("components/layout/pianoRoll/paramShiftStep.test.ts scripted checks", async () => {
    function assertEqual(actual: number, expected: number): void {
        if (Math.abs(actual - expected) > 1e-6) {
            throw new Error(`Expected ${expected}, received ${actual}`);
        }
    }

    assertEqual(getParamShiftStep("pitch"), 1);

    assertEqual(
        getParamShiftStep("breath_gain", {
            id: "breath_gain",
            display_name: "Breath Gain",
            group: "NSF-HiFiGAN",
            kind: {
                type: "automation_curve",
                unit: "x",
                default_value: 1,
                min_value: 0,
                max_value: 2,
            },
        }),
        0.05,
    );

    assertEqual(
        getParamShiftStep("hifigan_tension", {
            id: "hifigan_tension",
            display_name: "Tension",
            group: "NSF-HiFiGAN",
            kind: {
                type: "automation_curve",
                unit: "%",
                default_value: 0,
                min_value: -100,
                max_value: 100,
            },
        }),
        5,
    );

    assertEqual(getParamShiftStep("unknown_param"), 0.05);
});
