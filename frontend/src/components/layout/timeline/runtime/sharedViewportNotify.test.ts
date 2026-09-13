/**
 * 共享水平视口逐帧通知判定单测。
 *
 * 【本测试要补的盲区】时间轴内核「逐帧通知」原先只按 `scrollLeft` 去重，而共享视口
 * 是一对 `{scrollLeft, pxPerSec}`。浏览器实测（mock 工程 120s，光标 x=800）：滚轮
 * 缩小三次，时间轴 150 → 109.35 px/s，参数编辑器始终 150 px/s —— 两个面板的网格与
 * 标尺从此不同缩放。放大没事（同一锚点放大时位置会右移，`scrollLeft` 变了）。
 *
 * 因此下面这条用例是**回归锁**：位置不变、缩放变化时也必须通知。
 */
import { describe, expect, it } from "vitest";

import { shouldNotifySharedViewport } from "./sharedViewportNotify";

describe("shouldNotifySharedViewport（共享视口逐帧通知判定）", () => {
    /** 从未通知过的哨兵值（内核的初始状态）。 */
    const NEVER = Number.NaN;

    it("★ 位置不变、缩放变化 → 必须通知（缩小时间轴时的真实路径）", () => {
        // 工程起点附近缩小：锚点位置被钳回 0，与当前位置相同。
        expect(
            shouldNotifySharedViewport({
                scrollLeftPx: 0,
                pxPerSec: 135,
                lastScrollLeftPx: 0,
                lastPxPerSec: 150,
            }),
        ).toBe(true);
    });

    it("位置变化、缩放不变 → 通知（滚动 / 放大时的真实路径）", () => {
        expect(
            shouldNotifySharedViewport({
                scrollLeftPx: 55.6,
                pxPerSec: 165,
                lastScrollLeftPx: 0,
                lastPxPerSec: 165,
            }),
        ).toBe(true);
    });

    it("两者都未变化 → 不通知（避免每帧空转）", () => {
        expect(
            shouldNotifySharedViewport({
                scrollLeftPx: 120.5,
                pxPerSec: 150,
                lastScrollLeftPx: 120.5,
                lastPxPerSec: 150,
            }),
        ).toBe(false);
    });

    it("首次通知（上次值为 NaN）→ 通知（NaN 不相等，无需特判）", () => {
        expect(
            shouldNotifySharedViewport({
                scrollLeftPx: 0,
                pxPerSec: 150,
                lastScrollLeftPx: NEVER,
                lastPxPerSec: NEVER,
            }),
        ).toBe(true);
    });

    it("只有位置首次、缩放已通知过 → 仍通知", () => {
        expect(
            shouldNotifySharedViewport({
                scrollLeftPx: 0,
                pxPerSec: 150,
                lastScrollLeftPx: NEVER,
                lastPxPerSec: 150,
            }),
        ).toBe(true);
    });

    it("亚像素位置变化 → 通知（由调用方决定是否量化，这里不做容差判等）", () => {
        expect(
            shouldNotifySharedViewport({
                scrollLeftPx: 120.5,
                pxPerSec: 150,
                lastScrollLeftPx: 120.25,
                lastPxPerSec: 150,
            }),
        ).toBe(true);
    });
});
