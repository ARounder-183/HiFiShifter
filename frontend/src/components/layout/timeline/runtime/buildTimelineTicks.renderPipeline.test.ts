/**
 * 端到端渲染管线自检：**从刻度生成一直到屏幕坐标**，而不是只测生成器的输出分布。
 *
 * 【为什么必须有这一类测试】"水平缩放时标尺刻度与文本在某段之内消失"曾经被修过
 * 两次都没修好，原因是两次都只验证了"给定一组输入，`buildTimelineTicks` 的输出
 * 是否均匀"。但缺陷不在生成器内部，而在**喂给它的输入**：标尺的"内容"（有哪些
 * 刻度）由 React 按自己的 `scrollLeft` state 生成，而标尺的"位置"（内容层
 * transform）由内核按**实时** `scrollLeft` 每帧写入 —— 两者之间隔着一个量化提交，
 * 因此 React 的位置最多落后真值 `TICK_WINDOW_LAG_PX`。
 *
 * 纯函数测试对这个滞后完全不可见：同样的输入永远得到同样的输出。只有把
 * **切片（`TimeRulerMarks`）→ 设备像素几何 → 内容层平移（内核）** 整条链路复刻
 * 出来、并把滞后注入进去，才能观察到"视口里出现没有刻度的空白段"。
 *
 * 【本文件锁什么】在存在提交滞后的前提下，**渲染到屏幕上的**带标签刻度在视口内
 * 不得出现间距超过相邻中位数 2 倍的空白段。这是用户直接看得见的那条不变量。
 */

import { describe, expect, it } from "vitest";

import { buildTimelineTicks } from "./buildTimelineTicks.js";
import { createTickAxis } from "./tickAxis.js";
import { TICK_WINDOW_LAG_PX, tickWindowBufferPx } from "./tickWindow.js";
import { rulerLayerTranslatePx } from "../../renderKernel/timelineAxis.js";
import { verticalHairlineGeometry } from "../../../../utils/devicePixelLine.js";
import type { TempoMap } from "../../../../utils/tempoMap.ts";

interface PipelineArgs {
    pxPerSec: number;
    /** 内核真值（标尺层 transform 用它）。 */
    trueScrollLeft: number;
    /** React 侧位置（刻度窗口用它）—— 最多落后真值 TICK_WINDOW_LAG_PX。 */
    reactScrollLeft: number;
    viewportWidth: number;
    dpr: number;
    minLabelSpacingPx: number;
    grid: string;
    beatsPerBar: number;
    /** 缺省为无 Tempo Map 的均匀网格。 */
    tempoMap?: TempoMap | null;
}

/**
 * 密集 Tempo Map：8 个变化点、间隔 7.3s，BPM 在 80..159 间反复变化。
 * 与 `buildTimelineTicks.labels.test.ts` 的 `denseTempoMap` 同构（此处独立定义，
 * 避免测试文件互相 import）。
 */
function denseTempoMap(): TempoMap {
    const bpms = [80, 159, 96, 128, 80, 159, 96, 128];
    return {
        points: bpms.map((bpm, i) => ({
            id: `d${i}`,
            positionSec: i * 7.3,
            bpm,
            timeSignature: { numerator: 4, denominator: 4 },
            scale: null,
        })),
    };
}

/**
 * 复刻渲染管线，返回**屏幕坐标**（相对标尺左缘）下的带标签刻度。
 *
 * 逐段对应真实实现：
 * 1. `createTickAxis` + `buildTimelineTicks` —— React 侧生成刻度；
 * 2. 二分切片 —— `TimeRulerMarks` 的 `visibleTicks`；
 * 3. `verticalHairlineGeometry` —— 刻度竖线的设备像素几何；
 * 4. 减去 `rulerLayerTranslatePx(真值)` —— 内核写的内容层 transform。
 */
function renderedLabelXs(args: PipelineArgs): number[] {
    const { axis, anchorPx } = createTickAxis({
        pxPerSec: args.pxPerSec,
        scrollLeftPx: args.reactScrollLeft,
        viewportWidthPx: args.viewportWidth,
        dpr: args.dpr,
    });
    const ticks = buildTimelineTicks({
        axis,
        bpm: 120,
        beatsPerBar: args.beatsPerBar,
        grid: args.grid,
        primaryUnit: "barBeats",
        secondaryUnit: "clock",
        minLabelSpacingPx: args.minLabelSpacingPx,
        minGridSpacingPx: 8,
        swingPercent: 0,
        tempoMap: args.tempoMap ?? null,
    });

    // ── 2. 切片（与 TimeRulerMarks 同一缓冲公式）──
    const labeled = ticks.filter((tick) => tick.showLabel);
    const bufferPx = tickWindowBufferPx(args.viewportWidth);
    const leftPx = Math.max(0, anchorPx - bufferPx);
    const rightPx = anchorPx + args.viewportWidth + bufferPx;
    const lowerBound = (target: number): number => {
        let lo = 0;
        let hi = labeled.length;
        while (lo < hi) {
            const mid = (lo + hi) >> 1;
            if (labeled[mid].contentPx < target) lo = mid + 1;
            else hi = mid;
        }
        return lo;
    };
    const start = Math.max(0, lowerBound(leftPx) - 1);
    const end = Math.min(labeled.length, lowerBound(rightPx) + 1);

    // ── 3+4. 几何 + 平移 ──
    // `verticalHairlineGeometry` 返回的是**绝对**左缘（已含 `contentPx`），不能再加
    // 一次 `contentPx` —— 早先的写法把坐标整体翻了一倍，只因所有间距同比放大才没
    // 暴露；一旦标签栅格不均匀（Tempo Map），翻倍会把视口外的一半刻度算进来、把
    // 中位数压小，误报出"空洞"。
    const translate = rulerLayerTranslatePx(args.trueScrollLeft, args.dpr);
    return labeled
        .slice(start, end)
        .map(
            (tick) =>
                verticalHairlineGeometry(tick.contentPx, tick.isBarStart ? 2 : 1, args.dpr).left -
                translate,
        )
        .sort((a, b) => a - b);
}

function maxHoleRatio(xs: number[], viewportWidth: number): number {
    const inView = xs.filter((x) => x >= -1 && x <= viewportWidth + 1);
    if (inView.length < 3) return 0; // 刻度太少，判不出空洞（极端放大，本身稀疏）
    const gaps: number[] = [];
    for (let i = 1; i < inView.length; i += 1) gaps.push(inView[i] - inView[i - 1]);
    const sorted = [...gaps].sort((a, b) => a - b);
    const median = sorted[Math.floor(sorted.length / 2)];
    if (!(median > 0)) return 0;
    return Math.max(...gaps) / median;
}

describe("渲染管线：滞后存在时标尺仍不得露白", () => {
    it("★ 缩放 × 视口宽 × 提交滞后：屏幕上的标签间距无空洞", () => {
        let worst = { ratio: 0, detail: "" };
        let checked = 0;
        for (const grid of ["1/4", "1/8", "1/8d"]) {
            for (const viewportWidth of [320, 700, 1500, 2560]) {
                for (const dpr of [1, 2]) {
                    for (let i = 0; i < 40; i += 1) {
                        const pxPerSec = 4 * Math.pow(600 / 4, i / 39);
                        for (const lag of [0, 128, TICK_WINDOW_LAG_PX - 1]) {
                            for (const trueScrollLeft of [0, 900, 4200]) {
                                const args: PipelineArgs = {
                                    pxPerSec,
                                    trueScrollLeft,
                                    reactScrollLeft: Math.max(0, trueScrollLeft - lag),
                                    viewportWidth,
                                    dpr,
                                    minLabelSpacingPx: 110,
                                    grid,
                                    beatsPerBar: 4,
                                };
                                const xs = renderedLabelXs(args);
                                const ratio = maxHoleRatio(xs, viewportWidth);
                                checked += 1;
                                if (ratio > worst.ratio) {
                                    worst = {
                                        ratio,
                                        detail: `${JSON.stringify({ grid, viewportWidth, dpr, pxPerSec: Number(pxPerSec.toFixed(2)), lag, trueScrollLeft })} maxHoleRatio=${ratio.toFixed(2)}`,
                                    };
                                }
                            }
                        }
                    }
                }
            }
        }
        expect(checked).toBeGreaterThan(1000);
        // 2 倍是候选阶梯的固有粒度上限（相邻档位是 2 的幂）；超过即说明屏幕上
        // 出现了一段本不该空白的区域。
        expect(worst.ratio, `渲染后出现空洞：${worst.detail}`).toBeLessThanOrEqual(2.05);
    });

    it("★ 提交滞后不改变屏幕上的刻度集合（除窗口边缘外逐位一致）", () => {
        // 滞后只允许影响"窗口边缘切掉多少"，不允许影响视口**内部**的刻度。
        const viewportWidth = 1500;
        for (const pxPerSec of [12, 60, 240]) {
            for (const trueScrollLeft of [0, 1500, 5000]) {
                const base = renderedLabelXs({
                    pxPerSec,
                    trueScrollLeft,
                    reactScrollLeft: trueScrollLeft,
                    viewportWidth,
                    dpr: 1,
                    minLabelSpacingPx: 110,
                    grid: "1/4",
                    beatsPerBar: 4,
                });
                const lagged = renderedLabelXs({
                    pxPerSec,
                    trueScrollLeft,
                    reactScrollLeft: Math.max(0, trueScrollLeft - (TICK_WINDOW_LAG_PX - 1)),
                    viewportWidth,
                    dpr: 1,
                    minLabelSpacingPx: 110,
                    grid: "1/4",
                    beatsPerBar: 4,
                });
                // 取视口内部（两侧各留出缓冲之外），两组必须一致。
                const inner = (xs: number[]) =>
                    xs.filter((x) => x >= 0 && x <= viewportWidth).map((x) => Math.round(x * 100));
                expect(inner(lagged), `pps=${pxPerSec} truth=${trueScrollLeft}`).toEqual(
                    inner(base),
                );
            }
        }
    });

    /**
     * Tempo Map 路径的屏幕坐标不变量。生成器内部的标签栅格已在
     * `buildTimelineTicks.labels.test.ts` 钉死；这里补上"切片 → 设备像素几何 →
     * 内容层平移"之后的**屏幕坐标**，确保变化点密集时屏幕上也看不到空洞
     * （切片缓冲、设备像素吸附都不会把它重新暴露出来）。
     */
    it("★ Tempo Map（密集变化点）下屏幕坐标的标签间距无空洞", () => {
        const tempoMap = denseTempoMap();
        let worst = { ratio: 0, detail: "" };
        let checked = 0;
        for (const grid of ["1/4", "1/8"]) {
            // 只在视口足够宽时用 max/median 判据：视口很窄时视口内只有 3~5 个标签，
            // 中位数本身不稳定（会误报）。空洞本身与视口宽无关，另有 labels 用例
            // 用 max/nominal 在全部视口宽下钉住。
            for (const viewportWidth of [1500, 2560]) {
                for (const dpr of [1, 2]) {
                    for (let i = 0; i < 24; i += 1) {
                        const pxPerSec = 4 * Math.pow(600 / 4, i / 23);
                        for (const lag of [0, TICK_WINDOW_LAG_PX - 1]) {
                            for (const trueScrollLeft of [0, 900]) {
                                const xs = renderedLabelXs({
                                    pxPerSec,
                                    trueScrollLeft,
                                    reactScrollLeft: Math.max(0, trueScrollLeft - lag),
                                    viewportWidth,
                                    dpr,
                                    minLabelSpacingPx: 110,
                                    grid,
                                    beatsPerBar: 4,
                                    tempoMap,
                                });
                                const ratio = maxHoleRatio(xs, viewportWidth);
                                checked += 1;
                                if (ratio > worst.ratio) {
                                    worst = {
                                        ratio,
                                        detail: `${JSON.stringify({ grid, viewportWidth, dpr, pxPerSec: Number(pxPerSec.toFixed(2)), lag, trueScrollLeft })} maxHoleRatio=${ratio.toFixed(2)}`,
                                    };
                                }
                            }
                        }
                    }
                }
            }
        }
        expect(checked).toBeGreaterThan(100);
        // 上界 = 相邻段 BPM 之比（159/80 ≈ 1.99）—— 与生成器侧一致，实测最坏 1.99。
        expect(worst.ratio, `Tempo Map 屏幕空洞：${worst.detail}`).toBeLessThanOrEqual(2.05);
    });
});
