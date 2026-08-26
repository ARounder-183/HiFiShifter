import { test } from "vitest";

import { VOWEL_GUIDE_LINES, VOWEL_POINTS } from "./vowelChartLayout";

test("components/layout/timeline/clip/vowelChartLayout.test.ts scripted checks", async () => {
    function assert(condition: unknown, message: string): void {
        if (!condition) {
            throw new Error(message);
        }
    }

    const labels = new Set(VOWEL_POINTS.map((point) => point.label));

    assert(labels.has("y"), "layout should include front rounded close vowel y");
    assert(labels.has("ɯ"), "layout should include close back unrounded vowel ɯ");
    assert(labels.has("ɐ"), "layout should include near-open central vowel ɐ");
    assert(labels.has("ɒ"), "layout should include open back rounded vowel ɒ");

    for (const line of VOWEL_GUIDE_LINES) {
        for (const label of line) {
            assert(labels.has(label), `guide line references missing vowel ${label}`);
        }
    }

});
