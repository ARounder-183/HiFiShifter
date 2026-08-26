import { test } from "vitest";

import { resolveTimelineFadeDisplay } from "./timelineFadeDisplay.js";

test("components/layout/timeline/runtime/timelineFadeDisplay.test.ts scripted checks", async () => {
    function assertEqual(actual: unknown, expected: unknown, label: string): void {
        const actualJson = JSON.stringify(actual);
        const expectedJson = JSON.stringify(expected);
        if (actualJson !== expectedJson) {
            throw new Error(`${label}: expected ${expectedJson}, received ${actualJson}`);
        }
    }

    assertEqual(
        resolveTimelineFadeDisplay({ hovered: false }),
        {
            showGuide: true,
            showCurve: true,
            guideOpacity: 0.5,
            curveOpacity: 0.58,
        },
        "fade stays visible without hover",
    );

    assertEqual(
        resolveTimelineFadeDisplay({ hovered: true }),
        {
            showGuide: true,
            showCurve: true,
            guideOpacity: 0.72,
            curveOpacity: 0.9,
        },
        "hover only increases fade emphasis",
    );

});
