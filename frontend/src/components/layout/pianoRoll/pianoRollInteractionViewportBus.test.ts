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

/** 排空当前微任务队列（投递本身就是一个微任务）。 */
const flushMicrotasks = () => Promise.resolve();

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

    it("keeps get() synchronous but defers the notification to a microtask", async () => {
        const listener = vi.fn();
        const unsubscribe = pianoRollInteractionViewportBus.subscribe(listener);
        pianoRollInteractionViewportBus.publish(base);
        // 状态立刻可见（新订阅者必须能马上读到当前值）……
        expect(pianoRollInteractionViewportBus.get()).toBe(base);
        // ……但通知还没发生：这正是「apply() 里同步重入 track()」被切断的地方。
        expect(listener).not.toHaveBeenCalled();
        await flushMicrotasks();
        expect(listener).toHaveBeenCalledTimes(1);
        expect(listener).toHaveBeenCalledWith(base);
        unsubscribe();
        pianoRollInteractionViewportBus.publish({ ...base, scrollLeftPx: 200 });
        await flushMicrotasks();
        expect(listener).toHaveBeenCalledTimes(1);
    });

    it("★ coalesces several publishes in one microtask into a single delivery", async () => {
        const listener = vi.fn();
        const unsubscribe = pianoRollInteractionViewportBus.subscribe(listener);
        pianoRollInteractionViewportBus.publish({ ...base, scrollLeftPx: 1 });
        pianoRollInteractionViewportBus.publish({ ...base, scrollLeftPx: 2 });
        pianoRollInteractionViewportBus.publish({ ...base, scrollLeftPx: 3 });
        await flushMicrotasks();
        expect(listener).toHaveBeenCalledTimes(1);
        expect(listener).toHaveBeenCalledWith({ ...base, scrollLeftPx: 3 });
        unsubscribe();
    });

    it("★ does not notify when the value did not change", async () => {
        const listener = vi.fn();
        const unsubscribe = pianoRollInteractionViewportBus.subscribe(listener);
        pianoRollInteractionViewportBus.publish({ ...base, scrollLeftPx: 400 });
        await flushMicrotasks();
        expect(listener).toHaveBeenCalledTimes(1);
        // 逐字段相等 ⇒ 不广播：同一帧里「值域提交」与「帧提交」两次发布只剩一次。
        pianoRollInteractionViewportBus.publish({ ...base, scrollLeftPx: 400 });
        await flushMicrotasks();
        expect(listener).toHaveBeenCalledTimes(1);
        unsubscribe();
    });
});
