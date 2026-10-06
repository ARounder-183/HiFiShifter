/**
 * 视口提交判据（内核 → React）。
 *
 * 【要钉死的缺陷】标尺的**刻度窗口**由 React 侧 `(pxPerSec, scrollLeft)` 生成
 * （`useTimelineState` 的 `tickAxis`），而**屏幕位置**由内核视口决定（标尺内容层
 * 的 transform 与 GL 都按它画）—— 二者必须是同一个视口对。
 *
 * 旧判据只看 `scrollLeft`：只改缩放、位置没动时**不提交**，React 便停在
 * "旧缩放 + 新位置"（或"新缩放 + 旧位置"）。同一像素值在错误缩放下对应的时间完全
 * 不同，刻度窗口的锚点因此错位 —— 标尺某段既没有刻度线也没有文本，正是用户报告的
 * "缩放时标尺文本闪烁 / 消失"。
 */

import { describe, expect, it } from "vitest";

import { shouldCommitViewport } from "./viewportCommit.js";

/** "从未提交"的基准：两字段都是 NaN。 */
const NEVER_COMMITTED = { scrollLeftPx: Number.NaN, pxPerSec: Number.NaN };

const SCROLL_STEP_PX = 256;

describe("视口提交判据（缩放必须与位置同批提交）", () => {
    it("★ 缩放变化必须提交（即使位置完全没动）", () => {
        expect(
            shouldCommitViewport(
                { scrollLeftPx: 1000, pxPerSec: 80 },
                { scrollLeftPx: 1000, pxPerSec: 40 },
                SCROLL_STEP_PX,
            ),
        ).toBe(true);
    });

    it("★ 位置变化必须提交（即使缩放没动）", () => {
        expect(
            shouldCommitViewport(
                { scrollLeftPx: 1300, pxPerSec: 40 },
                { scrollLeftPx: 1000, pxPerSec: 40 },
                SCROLL_STEP_PX,
            ),
        ).toBe(true);
    });

    it("两者都没变 ⇒ 不提交（保住「滚动帧不进 React」的既有收益）", () => {
        expect(
            shouldCommitViewport(
                { scrollLeftPx: 1000, pxPerSec: 40 },
                { scrollLeftPx: 1000, pxPerSec: 40 },
                SCROLL_STEP_PX,
            ),
        ).toBe(false);
    });

    it("位置落在死区之内且缩放不变 ⇒ 不提交", () => {
        expect(
            shouldCommitViewport(
                { scrollLeftPx: 1000 + SCROLL_STEP_PX, pxPerSec: 40 },
                { scrollLeftPx: 1000, pxPerSec: 40 },
                SCROLL_STEP_PX,
            ),
        ).toBe(false);
    });

    it("首次提交（基准为 NaN）必须提交", () => {
        expect(
            shouldCommitViewport(
                { scrollLeftPx: 0, pxPerSec: 40 },
                NEVER_COMMITTED,
                SCROLL_STEP_PX,
            ),
        ).toBe(true);
        // 位置为 NaN 时同理。
        expect(
            shouldCommitViewport(
                { scrollLeftPx: 0, pxPerSec: 40 },
                { scrollLeftPx: Number.NaN, pxPerSec: 40 },
                SCROLL_STEP_PX,
            ),
        ).toBe(true);
    });

    it("缩放的相对容差：浮点抖动不算变化", () => {
        expect(
            shouldCommitViewport(
                { scrollLeftPx: 1000, pxPerSec: 40 * (1 + 1e-12) },
                { scrollLeftPx: 1000, pxPerSec: 40 },
                SCROLL_STEP_PX,
            ),
        ).toBe(false);
        // 但真实的缩放变化（哪怕只有千分之一）必须提交。
        expect(
            shouldCommitViewport(
                { scrollLeftPx: 1000, pxPerSec: 40.04 },
                { scrollLeftPx: 1000, pxPerSec: 40 },
                SCROLL_STEP_PX,
            ),
        ).toBe(true);
    });
});
