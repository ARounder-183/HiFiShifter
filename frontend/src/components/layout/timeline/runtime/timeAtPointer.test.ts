import { test } from "vitest";

import { timeAtPointerX } from "./timeAtPointer";

test("components/layout/timeline/runtime/timeAtPointer.test.ts scripted checks", async () => {
    function near(actual: number, expected: number, label: string): void {
        if (Math.abs(actual - expected) > 1e-9) {
            throw new Error(`${label}: expected ${expected}, received ${actual}`);
        }
    }

    // 量化 state 滞后 137px 时，交互必须使用调用方给出的实时滚动位置。
    near(
        timeAtPointerX({ clientX: 400, boundsLeft: 100, scrollLeft: 1234, pxPerSec: 1000 }),
        1.534,
        "live scroll",
    );
    near(
        timeAtPointerX({ clientX: 100, boundsLeft: 100, scrollLeft: 0, pxPerSec: 0 }),
        0,
        "degenerate rate",
    );
    near(
        timeAtPointerX({ clientX: 0, boundsLeft: 100, scrollLeft: -20, pxPerSec: 100 }),
        0,
        "clamped at the project origin",
    );
});
