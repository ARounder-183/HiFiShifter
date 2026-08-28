import { describe, expect, it, vi } from "vitest";
import { timelineViewportBus } from "./timelineViewportBus";

/**
 * 视口总线的核心契约：**同步派发**。
 *
 * TimelineSurface 的 sticky 画布层不随滚动容器原生移动，必须在与原生
 * DOM 内容层相同的绘制帧内拿到最新 scrollLeft（scroll 事件在 paint 前触发，
 * emit 必须在返回前完成全部监听器通知）。任何 rAF/异步延迟都会造成
 * 滚动中“Clip 与网格分离”的分层漂移 —— 该契约被破坏时本测试会失败。
 */
describe("timelineViewportBus — 同帧提交契约", () => {
    it("emit 在返回前同步通知所有订阅者，并更新快照", () => {
        const listener = vi.fn();
        const unsub = timelineViewportBus.subscribe(listener);
        try {
            timelineViewportBus.emit(120, 150, 800, 240, 80);

            expect(listener).toHaveBeenCalledTimes(1);
            expect(listener).toHaveBeenCalledWith(120, 150, 800, 240, 80);
            expect(timelineViewportBus.getSnapshot()).toMatchObject({
                scrollLeft: 120,
                pxPerSec: 150,
                viewportWidth: 800,
                scrollTopPx: 240,
                rowHeight: 80,
            });
        } finally {
            unsub();
        }
    });

    it("省略竖直轴参数时沿用上一快照值（单轴更新）", () => {
        const seen: Array<{ scrollLeft: number; scrollTopPx: number }> = [];
        const unsub = timelineViewportBus.subscribe((scrollLeft, _px, _w, scrollTopPx) => {
            seen.push({ scrollLeft, scrollTopPx });
        });
        try {
            timelineViewportBus.emit(10, 150, 800, 96, 48);
            timelineViewportBus.emit(20, 150, 800);
            expect(seen).toEqual([
                { scrollLeft: 10, scrollTopPx: 96 },
                { scrollLeft: 20, scrollTopPx: 96 },
            ]);
            expect(timelineViewportBus.getSnapshot().scrollTopPx).toBe(96);
            expect(timelineViewportBus.getSnapshot().rowHeight).toBe(48);

            timelineViewportBus.emit(20, 150, 800, 144, 48);
            expect(timelineViewportBus.getSnapshot().scrollTopPx).toBe(144);
        } finally {
            unsub();
        }
    });

    it("订阅时机早于 emit 才能收到；退订后不再接收", () => {
        const seen: number[] = [];
        const unsub = timelineViewportBus.subscribe((scrollLeft) => {
            seen.push(scrollLeft);
        });
        try {
            timelineViewportBus.emit(10, 150, 800);
            timelineViewportBus.emit(20, 150, 800);
            expect(seen).toEqual([10, 20]);

            unsub();
            timelineViewportBus.emit(30, 150, 800);
            expect(seen).toEqual([10, 20]);
            expect(timelineViewportBus.getSnapshot().scrollLeft).toBe(30);
        } finally {
            unsub();
        }
    });

    it("无订阅者时 emit 仍更新快照（新挂载的虚拟行可直接取值）", () => {
        timelineViewportBus.emit(42, 150, 800);
        expect(timelineViewportBus.getSnapshot()).toMatchObject({
            scrollLeft: 42,
            pxPerSec: 150,
            viewportWidth: 800,
        });
    });
});
