/**
 * 刻度窗口量化的**覆盖性**不变量。
 *
 * 【背景】`useTimelineState` 不再用实时 `scrollLeft` 生成刻度，而是量化到
 * `TICK_WINDOW_STEP_PX` 的锚点上，取刻度时把视口宽加宽一个步长。这样滚动
 * 期间刻度数组与标尺子树都不必每帧重算重渲染——这是 P5 的核心收益。
 *
 * 【要锁的不变量】无论 `scrollLeft` 落在量化窗口的哪个位置，生成的刻度都
 * 必须**完整覆盖真实视口** `[scrollLeft, scrollLeft + viewportWidth]`，
 * 否则滚动时会出现"刻度突然少一截"的缺口。
 *
 * 【为什么成立】锚点 ≤ scrollLeft < 锚点 + 步长，而取刻度用的视口宽是
 * `viewportWidth + 步长`，且 `buildTimelineTicks` 自身还会在两侧各加
 * `max(320, viewportWidthPx * 0.5)` 的缓冲。三重余量叠加后覆盖必然成立。
 * 这里用参数扫描把它钉死，防止以后有人调大步长或改缓冲公式时踩坑。
 */

import { describe, expect, it } from "vitest";

import { buildTimelineTicks, TICK_WINDOW_STEP_PX } from "./buildTimelineTicks.js";
import { createTimelineAxis } from "./timelineAxis.js";

/** 复刻 `useTimelineState` 里的取刻度方式。 */
function ticksFor(args: {
    scrollLeft: number;
    pxPerSec: number;
    viewportWidth: number;
}): ReturnType<typeof buildTimelineTicks> {
    const tickAnchorPx = Math.floor(args.scrollLeft / TICK_WINDOW_STEP_PX) * TICK_WINDOW_STEP_PX;
    return buildTimelineTicks({
        axis: createTimelineAxis({
            pxPerSec: args.pxPerSec,
            scrollLeftPx: tickAnchorPx,
            viewportWidthPx: args.viewportWidth + TICK_WINDOW_STEP_PX,
        }),
        bpm: 120,
        beatsPerBar: 4,
        grid: "1/4",
        primaryUnit: "seconds",
        secondaryUnit: "none",
        minLabelSpacingPx: 56,
        minGridSpacingPx: 8,
        swingPercent: 0,
        tempoMap: null,
    });
}

describe("刻度窗口量化：始终覆盖真实视口", () => {
    it("多种缩放 × 多种视口宽 × 密集 scrollLeft 采样", () => {
        const pxPerSecList = [0.625, 4, 12, 40, 120];
        const viewportWidthList = [600, 1024, 1500, 2560];

        for (const pxPerSec of pxPerSecList) {
            for (const viewportWidth of viewportWidthList) {
                // 步长取质数，保证采样点均匀落在量化窗口的各个相位上
                // （尤其是刚好跨过窗口边界的那些）。
                for (let scrollLeft = 0; scrollLeft <= TICK_WINDOW_STEP_PX * 12; scrollLeft += 7) {
                    const ticks = ticksFor({ scrollLeft, pxPerSec, viewportWidth });
                    if (ticks.length === 0) {
                        throw new Error(
                            `无刻度: pxPerSec=${pxPerSec} vw=${viewportWidth} scrollLeft=${scrollLeft}`,
                        );
                    }
                    const minX = Math.min(...ticks.map((tick) => tick.contentPx));
                    const maxX = Math.max(...ticks.map((tick) => tick.contentPx));
                    const label = `pxPerSec=${pxPerSec} vw=${viewportWidth} scrollLeft=${scrollLeft}`;
                    expect(minX, `左边界未覆盖 (${label})`).toBeLessThanOrEqual(scrollLeft);
                    expect(maxX, `右边界未覆盖 (${label})`).toBeGreaterThanOrEqual(
                        scrollLeft + viewportWidth,
                    );
                }
            }
        }
    });

    it("量化步长远小于标尺自身的缓冲（约束未被打破）", () => {
        // `TimeRulerMarks` 的缓冲是 max(320, viewportWidth * 0.5)。步长一旦
        // 逼近或超过它，标尺的可见窗口就会漏刻度。
        const minRulerBufferPx = 320;
        expect(TICK_WINDOW_STEP_PX).toBeLessThan(minRulerBufferPx);
    });

    it("滚动在小范围内不产生新刻度数组（量化的收益确实存在）", () => {
        const args = { pxPerSec: 12, viewportWidth: 1500 };
        const at = (scrollLeft: number): string =>
            JSON.stringify(ticksFor({ ...args, scrollLeft }).map((tick) => tick.contentPx));
        // 同一量化窗口内的两个位置，刻度必须完全一致。
        expect(at(0)).toBe(at(TICK_WINDOW_STEP_PX - 1));
        expect(at(TICK_WINDOW_STEP_PX * 3)).toBe(at(TICK_WINDOW_STEP_PX * 3 + 200));
    });
});
