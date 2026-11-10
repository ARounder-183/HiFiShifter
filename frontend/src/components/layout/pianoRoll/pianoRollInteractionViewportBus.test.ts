import { describe, expect, it, vi } from "vitest";
import {
    isSameInteractionViewport,
    pianoRollInteractionViewportBus,
    type PianoRollInteractionViewport,
} from "./pianoRollInteractionViewportBus";

const base: PianoRollInteractionViewport = {
    scrollLeftPx: 100,
    pxPerSec: 150,
    viewportWidthPx: 1200,
    viewportHeightPx: 600,
    param: "pitch",
    valueCenter: 60,
    valueSpan: 24,
    dpr: 1.5,
};

describe("pianoRollInteractionViewportBus", () => {
    it("compares every horizontal and vertical field", () => {
        expect(isSameInteractionViewport(base, { ...base })).toBe(true);
        for (const patch of [
            { scrollLeftPx: 101 },
            { pxPerSec: 151 },
            { viewportWidthPx: 1201 },
            { viewportHeightPx: 601 },
            { param: "volume" },
            { valueCenter: 61 },
            { valueSpan: 25 },
            { dpr: 2 },
        ]) {
            expect(isSameInteractionViewport(base, { ...base, ...patch })).toBe(false);
        }
    });

    it("notifies subscribers and returns the latest snapshot", () => {
        const listener = vi.fn();
        const unsubscribe = pianoRollInteractionViewportBus.subscribe(listener);
        pianoRollInteractionViewportBus.emit(base);
        expect(listener).toHaveBeenCalledWith(base);
        expect(pianoRollInteractionViewportBus.get()).toBe(base);
        unsubscribe();
        pianoRollInteractionViewportBus.emit({ ...base, scrollLeftPx: 200 });
        expect(listener).toHaveBeenCalledTimes(1);
    });
});
