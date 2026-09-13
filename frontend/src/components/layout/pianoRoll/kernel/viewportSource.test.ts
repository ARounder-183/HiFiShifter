/**
 * 参数编辑器渲染投影视口来源单测。
 *
 * 【本测试要补的盲区】面板的 `pxPerSecRef` / `scrollLeftRef` 在渲染期就被同步成
 * React state 的新值，而并发渲染允许"渲染但尚未提交"——此时 ref 已变、内核与 DOM
 * 还是旧值。任何一次绘制若拿 refs 当投影源，就会让面板的 Canvas2D / 曲线可见段
 * 与宿主 GL（网格 / 曲线 / 播放头）落在两套视口上。时间轴侧用 `livePxPerSec`
 * 取内核真值解决了同一问题，本模块是参数编辑器的对应实现。
 */
import { describe, expect, it } from "vitest";

import { resolvePanelRenderViewport } from "./viewportSource";

describe("resolvePanelRenderViewport（渲染投影的视口来源）", () => {
    it("★ 宿存在时一律取内核视口（渲染期 ref 已提前变化也不影响）", () => {
        const view = resolvePanelRenderViewport({
            kernelView: { pxPerSec: 150, scrollLeft: 120 },
            refPxPerSec: 165,
            refScrollLeftPx: 320,
        });
        expect(view.pxPerSec).toBe(150);
        expect(view.scrollLeftPx).toBe(120);
    });

    it("宿主未创建（挂载期）时退回 refs", () => {
        const view = resolvePanelRenderViewport({
            kernelView: null,
            refPxPerSec: 165,
            refScrollLeftPx: 320,
        });
        expect(view.pxPerSec).toBe(165);
        expect(view.scrollLeftPx).toBe(320);
    });

    it("内核视口非法时退回 refs（不把 NaN 带进投影）", () => {
        const view = resolvePanelRenderViewport({
            kernelView: { pxPerSec: Number.NaN, scrollLeft: 120 },
            refPxPerSec: 165,
            refScrollLeftPx: 320,
        });
        expect(view.pxPerSec).toBe(165);
        expect(view.scrollLeftPx).toBe(320);
    });

    it("两边都非法 → 归零（投影退化但不产生 NaN）", () => {
        const view = resolvePanelRenderViewport({
            kernelView: null,
            refPxPerSec: Number.NaN,
            refScrollLeftPx: Number.POSITIVE_INFINITY,
        });
        expect(view.pxPerSec).toBe(0);
        expect(view.scrollLeftPx).toBe(0);
    });

    it("绘制坐标原样返回（同步模式下含负值，不再二次换算）", () => {
        const view = resolvePanelRenderViewport({
            kernelView: { pxPerSec: 150, scrollLeft: -200 },
            refPxPerSec: 150,
            refScrollLeftPx: 0,
        });
        expect(view.scrollLeftPx).toBe(-200);
    });
});
