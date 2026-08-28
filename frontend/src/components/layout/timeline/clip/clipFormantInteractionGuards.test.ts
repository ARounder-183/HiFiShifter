import { test } from "vitest";

import {
    formantChartPointerDownShouldPreventDefault,
    shouldBlockPlaybackToggleForFormantWindow,
} from "./clipFormantInteractionGuards.ts";

test("components/layout/timeline/clip/clipFormantInteractionGuards.test.ts scripted checks", async () => {
    function assert(condition: unknown, message: string): void {
        if (!condition) {
            throw new Error(message);
        }
    }

    assert(
        shouldBlockPlaybackToggleForFormantWindow({
            key: "space",
            formantToolActive: true,
            focusWindow: "clipFormant",
        }),
        "space should be blocked while the formant tool is active",
    );

    assert(
        !shouldBlockPlaybackToggleForFormantWindow({
            key: "enter",
            formantToolActive: true,
            focusWindow: "clipFormant",
        }),
        "non-space keys should not be blocked by the formant tool guard",
    );

    assert(
        formantChartPointerDownShouldPreventDefault({ disabled: false }),
        "enabled vowel chart drags should prevent default selection behavior",
    );

    assert(
        !formantChartPointerDownShouldPreventDefault({ disabled: true }),
        "disabled vowel chart should not claim drag prevention",
    );

});
