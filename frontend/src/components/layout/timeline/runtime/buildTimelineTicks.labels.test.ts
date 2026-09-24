/**
 * 标尺标签与刻度身份的自检（BPM 抖动 / 缩放闪烁两项修复的回归）。
 *
 * 【要钉死的三条不变量】
 * 1. **刻度身份不随 BPM 变化**：`tick.key` 是音乐身份，改 BPM 只应改变几何
 *    （`contentPx`），不应改变 key —— 否则每次 BPM 写入都会让 React 卸载重建整棵
 *    刻度子树，滚轮调 BPM 时标尺"抽搐"。
 * 2. **标签版式不随滚动窗口变化**：`showLabel` / `labelMaxWidth` 由刻度序列本身
 *    决定。曾经由渲染层按"可见切片里的相邻标签"判定，切片边界随滚动移动，同一个
 *    标签的可见性会随滚动位置改变（右边缘的标签永远不被隐藏，一进入切片内部就可能
 *    被隐藏）—— "标尺文字时有时无"。
 * 3. **绝不整片无标签**：任何缩放 × 网格组合下，视口内至少有一个标签。
 */
import { describe, expect, it } from "vitest";

import { buildTimelineTicks } from "./buildTimelineTicks.js";
import { createTimelineAxis } from "../../renderKernel/timelineAxis.js";
import type { TempoMap } from "../../../../utils/tempoMap.ts";

function axisOf(pxPerSec: number, scrollLeftPx: number, viewportWidthPx = 1200) {
    return createTimelineAxis({ pxPerSec, scrollLeftPx, viewportWidthPx });
}

function ticksAt(args: {
    pxPerSec: number;
    scrollLeftPx?: number;
    bpm?: number;
    grid?: string;
    tempoMap?: TempoMap | null;
    minLabelSpacingPx?: number;
    viewportWidthPx?: number;
}) {
    return buildTimelineTicks({
        axis: axisOf(args.pxPerSec, args.scrollLeftPx ?? 0, args.viewportWidthPx ?? 1200),
        bpm: args.bpm ?? 120,
        beatsPerBar: 4,
        grid: args.grid ?? "1/4",
        primaryUnit: "barBeats",
        secondaryUnit: "none",
        minLabelSpacingPx: args.minLabelSpacingPx ?? 110,
        tempoMap: args.tempoMap ?? null,
    });
}

const tempoMapTwoSegments: TempoMap = {
    points: [
        { id: "a", positionSec: 0, bpm: 120, timeSignature: { numerator: 4, denominator: 4 } },
        { id: "b", positionSec: 7.4, bpm: 128, timeSignature: { numerator: 4, denominator: 4 } },
    ],
} as unknown as TempoMap;

describe("刻度身份（tick.key）", () => {
    it("★ BPM 变化只改几何，不改身份（滚轮调 BPM 不重建刻度子树）", () => {
        const at120 = ticksAt({ pxPerSec: 100, bpm: 120 });
        const at121 = ticksAt({ pxPerSec: 100, bpm: 121 });
        expect(at120.length).toBeGreaterThan(0);
        expect(at121.length).toBeGreaterThan(0);

        const keys120 = at120.map((tick) => tick.key);
        const keys121 = at121.map((tick) => tick.key);
        expect(keys121).toEqual(keys120);

        // 几何确实变了（否则说明 BPM 根本没影响刻度，用例失去意义）。
        const movedCount = at120.filter(
            (tick, index) => Math.abs(tick.contentPx - at121[index].contentPx) > 0.5,
        ).length;
        expect(movedCount).toBeGreaterThan(0);
    });

    it("身份在整条序列内唯一", () => {
        for (const pxPerSec of [8, 100, 1600]) {
            const keys = ticksAt({ pxPerSec }).map((tick) => tick.key);
            expect(new Set(keys).size).toBe(keys.length);
        }
    });

    it("Tempo Map 下身份同样只随网格变化（不随该段 BPM 变化）", () => {
        const map = tempoMapTwoSegments;
        const at120 = ticksAt({ pxPerSec: 60, tempoMap: map, bpm: 120 });
        const keys = at120.map((tick) => tick.key);
        // 两个段各自带段序号前缀，段的归属不会因为时间平移而改变。
        expect(keys.some((key) => key.startsWith("s0:"))).toBe(true);
        expect(keys.some((key) => key.startsWith("s1:"))).toBe(true);
    });
});

describe("标签版式", () => {
    it("★ 标签可见性不随滚动窗口变化（同一刻度在不同窗口下判定一致）", () => {
        const pxPerSec = 90;
        const viewportWidthPx = 1200;
        const base = ticksAt({ pxPerSec, scrollLeftPx: 0, viewportWidthPx });
        const shifted = ticksAt({ pxPerSec, scrollLeftPx: 900, viewportWidthPx });
        const byKey = new Map(shifted.map((tick) => [tick.key, tick]));

        let compared = 0;
        for (const tick of base) {
            const other = byKey.get(tick.key);
            if (other === undefined) continue;
            compared += 1;
            expect({ key: tick.key, showLabel: other.showLabel }).toEqual({
                key: tick.key,
                showLabel: tick.showLabel,
            });
        }
        expect(compared).toBeGreaterThan(5);
    });

    it("带标签的刻度间距不小于隐藏阈值", () => {
        for (const pxPerSec of [8, 40, 100, 400, 1600]) {
            const labeled = ticksAt({ pxPerSec }).filter((tick) => tick.showLabel);
            for (let i = 1; i < labeled.length; i += 1) {
                expect(labeled[i].contentPx - labeled[i - 1].contentPx).toBeGreaterThanOrEqual(26);
            }
        }
    });

    it("labelMaxWidth 非负，且不超过到下一个标签的间距", () => {
        const ticks = ticksAt({ pxPerSec: 100 });
        const labeled = ticks.filter((tick) => tick.showLabel);
        for (let i = 0; i + 1 < labeled.length; i += 1) {
            const width = labeled[i].labelMaxWidth;
            if (width === null) continue;
            expect(width).toBeGreaterThanOrEqual(0);
            expect(width).toBeLessThanOrEqual(labeled[i + 1].contentPx - labeled[i].contentPx);
        }
        // 最后一条不限宽。
        expect(labeled[labeled.length - 1].labelMaxWidth).toBeNull();
    });

    it("★ 任何缩放 × 网格组合下都不整片无标签（不出现文字消失）", () => {
        const grids = ["1/4", "1/8", "1/16", "1/3", "1/32"];
        const zooms = [0.5, 1, 4, 8, 16, 32, 64, 100, 200, 400, 800, 1600, 4000];
        for (const grid of grids) {
            for (const pxPerSec of zooms) {
                const ticks = ticksAt({ pxPerSec, grid });
                const labeled = ticks.filter((tick) => tick.showLabel);
                expect({ grid, pxPerSec, labeled: labeled.length > 0 }).toEqual({
                    grid,
                    pxPerSec,
                    labeled: true,
                });
            }
        }
    });

    it("★ Tempo Map 下同样不整片无标签", () => {
        for (const pxPerSec of [0.5, 2, 8, 30, 120, 600, 2400]) {
            for (const map of [tempoMapTwoSegments]) {
                const labeled = ticksAt({ pxPerSec, tempoMap: map }).filter(
                    (tick) => tick.showLabel,
                );
                expect({ pxPerSec, labeled: labeled.length > 0 }).toEqual({
                    pxPerSec,
                    labeled: true,
                });
            }
        }
    });
});
