import { test } from "vitest";

import { createTimelinePerfProbe } from "./timelinePerfProbe.js";

test("components/layout/timeline/runtime/timelinePerfProbe.test.ts scripted checks", async () => {
    function assertNear(actual: number, expected: number, label: string): void {
        if (Math.abs(actual - expected) > 1e-6) {
            throw new Error(`${label}: expected ${expected}, received ${actual}`);
        }
    }

    const probe = createTimelinePerfProbe(4);
    probe.pushDrawMs(8);
    probe.pushDrawMs(12);
    probe.pushDrawMs(10);
    probe.pushHitTestMs(2);
    probe.pushHitTestMs(4);

    const snapshot = probe.getSnapshot();
    assertNear(snapshot.avgDrawMs, 10, "draw average");
    assertNear(snapshot.avgHitTestMs, 3, "hit-test average");

    console.log("timelinePerfProbe checks passed");
});
