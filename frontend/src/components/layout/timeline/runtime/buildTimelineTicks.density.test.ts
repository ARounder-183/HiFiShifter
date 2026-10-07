/**
 * 标尺标签**间距上限**回归 —— "标签够不够密"。
 *
 * 【为什么单独立一份】前几轮的判据分别保证了"间距彼此接近"（max/median）与
 * "视口里有标签"（覆盖率），但都允许标签**整体变稀**：只要彼此接近、只要存在，
 * 哪怕间距是请求值的 4 倍也算通过。而 `TimeRulerMarks` 只渲染 `showLabel` 的
 * 刻度，于是"标签稀" ≡ "那一段既没有刻度线、也没有文本" —— 正是用户报告的
 * "某段之内的标尺刻度与文本消失"。
 *
 * 本文件断言的就是这个缺失的口径：**相邻标签间距不得超过请求值的 2 倍**
 * （2 是标签栅格自身的固有粒度；§5b 密度补充以此为目标）。
 *
 * 只在"网格足够密"的样本上断言 —— 视口内网格线本就只有三五条时，
 * 间距大是物理必然（没有刻度可补），不是缺陷。
 */

import { describe, expect, it } from "vitest";

import { buildTimelineTicks, RULER_LABEL_HIDDEN_GAP_PX } from "./buildTimelineTicks.js";
import { createTickAxis } from "./tickAxis.js";
import type { TempoMap } from "../../../../utils/tempoMap.ts";

/** 栅格固有粒度：请求间距的 2 倍。 */
const DENSITY_TARGET_MULT = 2;

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

function threeSegmentTempoMap(): TempoMap {
    return {
        points: [
            {
                id: "a",
                positionSec: 0,
                bpm: 120,
                timeSignature: { numerator: 4, denominator: 4 },
                scale: null,
            },
            {
                id: "b",
                positionSec: 40,
                bpm: 96,
                timeSignature: { numerator: 4, denominator: 4 },
                scale: null,
            },
            {
                id: "c",
                positionSec: 90,
                bpm: 150,
                timeSignature: { numerator: 4, denominator: 4 },
                scale: null,
            },
        ],
    };
}

function median(xs: number[]): number {
    if (xs.length === 0) return 0;
    const sorted = [...xs].sort((a, b) => a - b);
    return sorted[Math.floor(sorted.length / 2)];
}

interface Analysis {
    /** 视口内的刻度总数（网格线）。 */
    gridCount: number;
    /** 视口内网格线间距的中位数。 */
    medianGridPx: number;
    /** 视口内带标签刻度的内容坐标（升序）。 */
    labelPx: number[];
    /** 视口内带标签刻度的间距（升序相邻）。 */
    labelGaps: number[];
    /** 带标签刻度里，落在被 swing 平移的（非整数拍）线上的数量。 */
    offBeatLabels: number;
}

function analyze(args: {
    pxPerSec: number;
    scrollLeft: number;
    viewportWidth: number;
    tempoMap: TempoMap | null;
    spacingPx: number;
    grid: string;
    beatsPerBar?: number;
    swingPercent?: number;
}): Analysis {
    const beatsPerBar = args.beatsPerBar ?? 4;
    const { axis } = createTickAxis({
        pxPerSec: args.pxPerSec,
        scrollLeftPx: args.scrollLeft,
        viewportWidthPx: args.viewportWidth,
    });
    const ticks = buildTimelineTicks({
        axis,
        bpm: 120,
        beatsPerBar,
        grid: args.grid,
        primaryUnit: "barBeats",
        secondaryUnit: "clock",
        minLabelSpacingPx: args.spacingPx,
        minGridSpacingPx: 8,
        swingPercent: args.swingPercent ?? 0,
        tempoMap: args.tempoMap,
    });
    const lo = args.scrollLeft;
    const hi = args.scrollLeft + args.viewportWidth;
    const inView = ticks
        .filter((tick) => tick.contentPx >= lo && tick.contentPx <= hi)
        .sort((a, b) => a.contentPx - b.contentPx);
    const labeled = inView.filter((tick) => tick.showLabel);
    const gridGaps = inView.slice(1).map((tick, k) => tick.contentPx - inView[k].contentPx);
    const labelPx = labeled.map((tick) => tick.contentPx);
    const labelGaps = labelPx.slice(1).map((x, k) => x - labelPx[k]);
    const offBeatLabels = labeled.filter(
        (tick) => Math.abs(tick.beat - Math.round(tick.beat)) > 1e-6,
    ).length;
    return {
        gridCount: inView.length,
        medianGridPx: median(gridGaps),
        labelPx,
        labelGaps,
        offBeatLabels,
    };
}

/** 网格足够密，间距才有"上限"可言（否则本就没刻度可补）。 */
function gridDenseEnough(a: Analysis, spacingPx: number): boolean {
    return a.gridCount >= 8 && a.medianGridPx > 0 && a.medianGridPx <= spacingPx;
}

describe("标尺标签间距上限（密度）", () => {
    /**
     * 【主回归锁】视口内相邻标签间距不得超过请求值的 2 倍。
     *
     * 修复前实测最坏 2.73×（无图）/ 4.09×（密集图）；修复后应 ≤ 2.0×（+ 浮点余量）。
     */
    it("★ 网格足够密时，相邻标签间距 ≤ 2 × 请求间距", () => {
        let checked = 0;
        let worst = { ratio: 0, detail: "" };
        for (const [mapName, tempoMap] of [
            ["none", null],
            ["three", threeSegmentTempoMap()],
            ["dense", denseTempoMap()],
        ] as [string, TempoMap | null][]) {
            for (const viewportWidth of [900, 1800, 2560]) {
                for (const grid of ["1/4", "1/8"]) {
                    for (const spacingPx of [110, 320]) {
                        for (let i = 0; i < 80; i += 1) {
                            const pxPerSec = 4 * Math.pow(600 / 4, i / 79);
                            for (const scrollLeft of [0, 5000]) {
                                const a = analyze({
                                    pxPerSec,
                                    scrollLeft,
                                    viewportWidth,
                                    tempoMap,
                                    spacingPx,
                                    grid,
                                });
                                if (!gridDenseEnough(a, spacingPx)) continue;
                                if (a.labelPx.length < 2) continue;
                                checked += 1;
                                const ratio = Math.max(...a.labelGaps) / spacingPx;
                                if (ratio > worst.ratio) {
                                    worst = {
                                        ratio,
                                        detail: `${mapName} vw=${viewportWidth} ${grid} spacing=${spacingPx} pxPerSec=${pxPerSec.toFixed(1)} scroll=${scrollLeft} maxGap=${Math.max(...a.labelGaps).toFixed(0)}px`,
                                    };
                                }
                            }
                        }
                    }
                }
            }
        }
        expect(checked).toBeGreaterThan(2000);
        // 2 是栅格固有粒度（相邻档位是 2 的幂）；密度补充以此为目标。
        // 修复前实测 2.097×（同一批样本），修复后 ≤ 2.0×。
        expect(worst.ratio, `标签过稀：${worst.detail}`).toBeLessThanOrEqual(DENSITY_TARGET_MULT);
    });

    it("★ 补充不得造出过密的标签（间距仍 ≥ 让位阈值）", () => {
        let worstViolation = 0;
        let detail = "";
        let checked = 0;
        for (const tempoMap of [null, denseTempoMap()]) {
            for (const viewportWidth of [900, 1800]) {
                for (const grid of ["1/4", "1/8", "1/8d"]) {
                    for (const spacingPx of [110, 320]) {
                        for (let i = 0; i < 60; i += 1) {
                            const pxPerSec = 4 * Math.pow(600 / 4, i / 59);
                            for (const scrollLeft of [0, 5000]) {
                                const a = analyze({
                                    pxPerSec,
                                    scrollLeft,
                                    viewportWidth,
                                    tempoMap,
                                    spacingPx,
                                    grid,
                                });
                                if (a.labelPx.length < 2) continue;
                                checked += 1;
                                const minGap = Math.min(...a.labelGaps);
                                const violation = RULER_LABEL_HIDDEN_GAP_PX - minGap;
                                if (violation > worstViolation) {
                                    worstViolation = violation;
                                    detail = `vw=${viewportWidth} ${grid} spacing=${spacingPx} pxPerSec=${pxPerSec.toFixed(1)} minGap=${minGap.toFixed(2)}`;
                                }
                            }
                        }
                    }
                }
            }
        }
        expect(checked).toBeGreaterThan(500);
        expect(worstViolation, `出现过密标签：${detail}`).toBeLessThanOrEqual(1e-6);
    });

    it("★ 补充不得让标签落到被 swing 平移的线上", () => {
        // swing 把奇数索引的线整体平移半步，落在其上的标签文字会渲染成 "1.2.300"。
        // 补充的候选优先级里"整数拍"排在前面，且候选必须落在两条既有标签之间 ——
        // 这里把"带标签刻度拍值为整数"钉死。
        let checked = 0;
        for (const swingPercent of [25, 50, 100]) {
            for (const grid of ["1/4", "1/8"]) {
                for (const pxPerSec of [20, 60, 200, 800]) {
                    const a = analyze({
                        pxPerSec,
                        scrollLeft: 0,
                        viewportWidth: 1500,
                        tempoMap: null,
                        spacingPx: 110,
                        grid,
                        swingPercent,
                    });
                    if (a.labelPx.length === 0) continue;
                    checked += 1;
                    expect(
                        a.offBeatLabels,
                        `swing=${swingPercent} ${grid} pxPerSec=${pxPerSec} 有标签落在被平移的线上`,
                    ).toBe(0);
                }
            }
        }
        expect(checked).toBeGreaterThan(10);
    });

    it("★ 标签可见性不随滚动窗口变化（补充必须与窗口无关）", () => {
        // 密度补充是单趟左→右、每对相邻标签各自处理的：处理 (a, b) 只看 a、b 的
        // 绝对位置，与生成窗口无关。这里用两个不同锚点的窗口交叉验证。
        const tempoMap = denseTempoMap();
        let compared = 0;
        for (const pxPerSec of [12, 40, 120, 400]) {
            const base = analyze({
                pxPerSec,
                scrollLeft: 0,
                viewportWidth: 1200,
                tempoMap,
                spacingPx: 110,
                grid: "1/4",
            });
            const shifted = analyze({
                pxPerSec,
                scrollLeft: 900,
                viewportWidth: 1200,
                tempoMap,
                spacingPx: 110,
                grid: "1/4",
            });
            // 两个窗口都覆盖的内容区间（用交集避免边缘效应）。
            const lo = 900;
            const hi = 1200;
            const inBand = (xs: number[]) =>
                xs.filter((x) => x >= lo && x <= hi).map((x) => Math.round(x * 100));
            expect(inBand(shifted.labelPx), `pxPerSec=${pxPerSec}`).toEqual(inBand(base.labelPx));
            compared += 1;
        }
        expect(compared).toBeGreaterThan(0);
    });
});
