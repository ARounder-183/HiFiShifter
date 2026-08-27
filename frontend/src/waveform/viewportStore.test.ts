import { test } from "vitest";

import { createWaveformViewportStore } from "./viewportStore.ts";

test("waveform/viewportStore.test.ts scripted checks", async () => {
    function assertEqual(actual: unknown, expected: unknown, label: string): void {
        const actualJson = JSON.stringify(actual);
        const expectedJson = JSON.stringify(expected);
        if (actualJson !== expectedJson) {
            throw new Error(`${label}: expected ${expectedJson}, received ${actualJson}`);
        }
    }

    const store = createWaveformViewportStore({
        revision: 0,
        scrollLeftPx: 0,
        pxPerSec: 150,
        widthPx: 1200,
        heightPx: 600,
        devicePixelRatio: 2,
    });

    let notifications = 0;
    const unsubscribe = store.subscribe(() => {
        notifications += 1;
    });

    store.set({ scrollLeftPx: 450, pxPerSec: 300, widthPx: 1000 });

    assertEqual(
        store.getSnapshot(),
        {
            revision: 1,
            scrollLeftPx: 450,
            pxPerSec: 300,
            widthPx: 1000,
            heightPx: 600,
            devicePixelRatio: 2,
        },
        "new subscribers synchronously see one complete latest snapshot",
    );
    assertEqual(notifications, 1, "one atomic patch emits one notification");

    store.set({ scrollLeftPx: 450, pxPerSec: 300, widthPx: 1000 });
    assertEqual(notifications, 1, "an identical patch emits no notification");

    store.set({
        scrollLeftPx: Number.NaN,
        pxPerSec: 0,
        widthPx: -20,
        heightPx: Number.POSITIVE_INFINITY,
        devicePixelRatio: 0,
    });
    assertEqual(
        store.getSnapshot(),
        {
            revision: 1,
            scrollLeftPx: 450,
            pxPerSec: 300,
            widthPx: 1000,
            heightPx: 600,
            devicePixelRatio: 2,
        },
        "invalid viewport values do not corrupt the current snapshot",
    );

    unsubscribe();
    store.set({ heightPx: 500 });
    assertEqual(notifications, 1, "unsubscribed listeners are not called");
    assertEqual(store.getSnapshot().revision, 2, "revision advances for a later real change");
});