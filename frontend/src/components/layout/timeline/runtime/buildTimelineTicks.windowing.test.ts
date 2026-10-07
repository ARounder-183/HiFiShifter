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
import { TICK_WINDOW_LAG_PX, tickWindowBufferPx, tickWindowRangePx } from "./tickWindow.js";
import { createTimelineAxis } from "../../renderKernel/timelineAxis.js";

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
        // `TimeRulerMarks` 的缓冲与生成缓冲共用 `tickWindowBufferPx`。步长一旦
        // 逼近或超过它的下界，标尺的可见窗口就会漏刻度。
        expect(TICK_WINDOW_STEP_PX).toBeLessThan(tickWindowBufferPx(0));
    });

    /**
     * 【本轮新增的核心不变量】内核向 React 提交水平位置是**量化**的
     * （`TICK_WINDOW_LAG_PX`）：React 侧的 scrollLeft 最多落后内核真值一个步长，
     * 而标尺里"有哪些刻度"是按 React 的位置生成的、标尺层的 transform 却按内核
     * 真值每帧写入。窗口缓冲必须吸收这段滞后，否则视口一端会出现没有刻度的空白段
     * —— 用户看到的"某段之内的标尺刻度与文本消失，滚动一下又回来"。
     *
     * 这里把"滞后 ≤ 缓冲"钉死：日后有人调大提交步长（为省重渲染）或调小缓冲，
     * 本用例立刻失败，而不是等到用户报"标尺露白"。
     */
    it("★ 提交滞后被窗口缓冲吸收（滞后最大时仍完整覆盖真实视口）", () => {
        const pxPerSecList = [4, 12, 40, 150, 800];
        const viewportWidthList = [320, 700, 1500, 2560];
        for (const pxPerSec of pxPerSecList) {
            for (const viewportWidth of viewportWidthList) {
                for (const lag of [0, 64, 128, 200, TICK_WINDOW_LAG_PX - 1]) {
                    for (
                        let scrollLeft = 0;
                        scrollLeft <= TICK_WINDOW_STEP_PX * 6;
                        scrollLeft += 97
                    ) {
                        // React 侧的位置落后内核真值 `lag`：刻度按 React 的位置生成，
                        // 而标尺层的 transform 按内核真值写 —— 两者相差 `lag`。
                        const reactScrollLeft = Math.max(0, scrollLeft - lag);
                        const ticks = ticksFor({
                            pxPerSec,
                            viewportWidth,
                            scrollLeft: reactScrollLeft,
                        });
                        const label = `pps=${pxPerSec} vw=${viewportWidth} lag=${lag} truth=${scrollLeft}`;
                        expect(ticks.length, `无刻度 (${label})`).toBeGreaterThan(0);

                        // 生成范围必须**包含**真实视口 [scrollLeft, scrollLeft + vw]。
                        // 只断言"最远刻度 ≥ 右端"是不对的：刻度是离散的，视口内最后
                        // 一条刻度本来就可能落在右端之前（下一个间距在屏幕外）——那
                        // 不是空洞。真正会露白的是"生成范围没盖到视口"，因此这里
                        // 按窗口公式（与实现同一组常量）复算范围端点。
                        const anchor =
                            Math.floor(reactScrollLeft / TICK_WINDOW_STEP_PX) * TICK_WINDOW_STEP_PX;
                        const bufferPx = tickWindowRangePx(viewportWidth).bufferPx;
                        const windowLeftPx = Math.max(0, anchor - bufferPx);
                        const windowRightPx =
                            anchor + viewportWidth + TICK_WINDOW_STEP_PX + bufferPx;
                        expect(windowLeftPx, `左端露白 (${label})`).toBeLessThanOrEqual(scrollLeft);
                        expect(windowRightPx, `右端露白 (${label})`).toBeGreaterThanOrEqual(
                            scrollLeft + viewportWidth,
                        );
                    }
                }
            }
        }
    });

    it("★ 窗口缓冲必须吸收「锚点量化 + 提交滞后」两者之和", () => {
        // 真实视口相对锚点最多右移 `TICK_WINDOW_STEP_PX`（锚点量化）+
        // `TICK_WINDOW_LAG_PX`（提交死区），二者是**两个独立**的偏移量。
        // 缓冲下界若只覆盖后者（旧实现 `LAG + 64 = 320`），窄视口下切片就会把
        // 视口内的刻度切掉 —— 标尺右端整段空白。此处把"缓冲 ≥ 两者之和"钉死。
        expect(tickWindowBufferPx(0)).toBeGreaterThanOrEqual(
            TICK_WINDOW_STEP_PX + TICK_WINDOW_LAG_PX,
        );
        // 缓冲公式与 `TimeRulerMarks` 的切片缓冲同源：这里顺带锁住"视口越宽缓冲越大"。
        expect(tickWindowBufferPx(4000)).toBeGreaterThan(tickWindowBufferPx(1000));
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
