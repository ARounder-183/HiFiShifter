/**
 * 刻度窗口与屏幕位置必须来自**同一个** `(pxPerSec, scrollLeft)`。
 *
 * 【为什么单独立一份】此前所有关于标尺的测试都默认一件事：喂给生成器的视口
 * （React 的 `pxPerSec` / `scrollLeft`）就是屏幕正在显示的那一个（内核视口）。
 * 本文件把这个**假设本身**变成断言：
 *
 * 1. 两者同源（或只差提交滞后）时，真实视口必然被生成的刻度覆盖；
 * 2. 两者**不同源**——典型是"缩放刚换、位置还是旧缩放下的像素值"——时覆盖
 *    **必然失败**。同一像素值在错误缩放下对应的时间完全不同，窗口锚点因此整体
 *    错位，视口可以整段没有刻度：这正是"某段之内的标尺刻度线与文本消失"的算术
 *    根因，也是"缩放必须与位置同批提交给 React"这条要求的证明。
 *
 * 第 2 条是有意断言的"反面"：它把因果钉死，任何试图用"加大缓冲"掩盖它的改动都会
 * 在这里露馅（缓冲再大也补不回按错误缩放换算出的时间差）。
 */

import { describe, expect, it } from "vitest";

import { buildTimelineTicks } from "./buildTimelineTicks.js";
import { createTickAxis } from "./tickAxis.js";
import { TICK_WINDOW_LAG_PX, TICK_WINDOW_STEP_PX } from "./tickWindow.js";

interface Pair {
    pxPerSec: number;
    scrollLeft: number;
}

/**
 * 用 `react` 这对视口生成刻度，检查 `kernel` 这对视口对应的真实视口是否被覆盖。
 *
 * 逐段对应生产实现：`createTickAxis`（含量化锚点与宽度补偿）→ `buildTimelineTicks`
 * （含两侧缓冲）。返回的 `blankPx` 是视口内**没有任何刻度**的宽度。
 */
function coverage(react: Pair, kernel: Pair, viewportWidth: number): { blankPx: number } {
    const { axis } = createTickAxis({
        pxPerSec: react.pxPerSec,
        scrollLeftPx: react.scrollLeft,
        viewportWidthPx: viewportWidth,
    });
    const ticks = buildTimelineTicks({
        axis,
        bpm: 120,
        beatsPerBar: 4,
        grid: "1/4",
        primaryUnit: "barBeats",
        secondaryUnit: "clock",
        minLabelSpacingPx: 110,
        minGridSpacingPx: 8,
        swingPercent: 0,
        tempoMap: null,
    });
    const minPx = Math.min(...ticks.map((tick) => tick.contentPx));
    const maxPx = Math.max(...ticks.map((tick) => tick.contentPx));
    const lo = kernel.scrollLeft;
    const hi = kernel.scrollLeft + viewportWidth;
    const uncoveredLeft = Math.max(0, minPx - lo);
    const uncoveredRight = Math.max(0, hi - maxPx);
    return { blankPx: Math.max(uncoveredLeft, uncoveredRight) };
}

const VIEWPORT_WIDTH = 1500;

describe("刻度窗口与屏幕位置必须同源", () => {
    it("★ 同源时真实视口被完整覆盖", () => {
        for (const pxPerSec of [4, 40, 100, 400, 1600]) {
            for (const scrollLeft of [0, 3000, 40000]) {
                const pair = { pxPerSec, scrollLeft };
                expect(
                    coverage(pair, pair, VIEWPORT_WIDTH).blankPx,
                    `pps=${pxPerSec} scroll=${scrollLeft}`,
                ).toBe(0);
            }
        }
    });

    it("★ 只差提交滞后（缩放相同、位置在 STEP + LAG 之内）时仍被覆盖", () => {
        const pxPerSec = 60;
        const kernel = { pxPerSec, scrollLeft: 8000 };
        for (const lag of [0, 64, 256, TICK_WINDOW_STEP_PX + TICK_WINDOW_LAG_PX - 1]) {
            const react = { pxPerSec, scrollLeft: kernel.scrollLeft - lag };
            expect(coverage(react, kernel, VIEWPORT_WIDTH).blankPx, `lag=${lag}`).toBe(0);
        }
    });

    it("★ 不同源（缩放已换、位置仍是旧缩放的像素值）必然露白 —— 因此必须原子提交", () => {
        // 复刻"在参数编辑器里缩小 2 倍"时旧实现的中间状态：
        //   缩放已由 React 更新（100 → 50），但位置仍是旧缩放下的 6000px，
        //   而内核的真实视口是 3000px（同一时间位置）。
        const kernel: Pair = { pxPerSec: 50, scrollLeft: 3000 };
        const staleReact: Pair = { pxPerSec: 50, scrollLeft: 6000 };
        const blank = coverage(staleReact, kernel, VIEWPORT_WIDTH).blankPx;
        // 锚点落在 5888px 附近，整个真实视口 [3000, 4500] 都在窗口左侧 —— 整段露白。
        expect(blank, "不同源时应当整段露白").toBeGreaterThan(500);

        // 反向（放大）同样露白：缩放 50 → 100，位置仍是旧的 3000px。
        const kernelZoomIn: Pair = { pxPerSec: 100, scrollLeft: 6000 };
        const staleReactZoomIn: Pair = { pxPerSec: 100, scrollLeft: 3000 };
        expect(
            coverage(staleReactZoomIn, kernelZoomIn, VIEWPORT_WIDTH).blankPx,
            "放大方向同样应当露白",
        ).toBeGreaterThan(500);
    });
});
