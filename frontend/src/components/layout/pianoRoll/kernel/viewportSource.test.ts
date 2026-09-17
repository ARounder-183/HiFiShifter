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

    /**
     * 回归：**交互路径**（框选换算 / 命中测试 / 标尺 seek）与渲染路径必须同源。
     *
     * 【缺陷现象】交互路径原先自建轴、直接读 `scrollLeftRef`。该 ref 在渲染期会被
     * 256px 量化的 React state 回写，因此横向滚动后最多滞后内核 255px。于是画面上
     * 的选区块按内核绘制、而框选起点/终点按滞后的 ref 计算，用户划定的选区整体
     * 偏移同一距离（报告："实际产生的选区与鼠标划定的区域不一致"）。
     *
     * 复现特征也由本用例直接对应：先做一次水平缩放会 `flushSync` 原子对齐 state
     * （ref == 内核 → 正常）；随后滚动只改内核、ref 逐渐滞后（开始偏移）；再缩放
     * 一次又对齐（"恢复正常"）。
     *
     * 判据：给定内核真值与滞后的 refs（相差一个量化残差），解析结果必须是内核值
     * ——渲染与交互因此落在同一视口上。
     */
    it("★ 交互与渲染同源：内核真值优先于量化滞后的 refs", () => {
        // 内核已被滚轮推到 1000，而 state（以及被它回写的 ref）还停在 768
        // ——差值 232 正是 256px 量化步长之内的残差。
        const kernelScrollLeft = 1000;
        const laggingRefScrollLeft = 768;
        const view = resolvePanelRenderViewport({
            kernelView: { pxPerSec: 150, scrollLeft: kernelScrollLeft },
            refPxPerSec: 150,
            refScrollLeftPx: laggingRefScrollLeft,
        });
        // 交互侧若用 ref，框选会偏移 232px；用内核则与画面一致。
        expect(view.scrollLeftPx).toBe(kernelScrollLeft);
        expect(view.scrollLeftPx).not.toBe(laggingRefScrollLeft);
    });
});
