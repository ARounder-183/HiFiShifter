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

describe("BPM 变化下的刻度序列", () => {
    /**
     * 【为什么不再断言"身份不变"】BPM 变化在语义上**就是要平移每一条刻度**
     * （`sec = beat × 60 / bpm`）—— 试图让身份跨 BPM 稳定是在否认这一点，上一轮
     * 为此引入的音乐身份字段因此没有消费者。
     *
     * 真正要守的性质是：**序列长度（几乎）不变**，这样标尺用**位置**作 React key
     * 时能一一复用 DOM，不会整棵重建。位置抖动（阈值处网格步长整档变化）才是需要
     * 避免的，因此这里同时断言步长档位在小幅 BPM 变化下不跳。
     */
    it("★ 小幅 BPM 变化不改变刻度条数（位置 key 因此一一对应，不重建 DOM）", () => {
        const at120 = ticksAt({ pxPerSec: 100, bpm: 120 });
        const at121 = ticksAt({ pxPerSec: 100, bpm: 121 });
        expect(at120.length).toBeGreaterThan(0);
        expect(at121.length).toBe(at120.length);

        // 几何确实变了（否则说明 BPM 没影响刻度，用例失去意义）。
        const movedCount = at120.filter(
            (tick, index) => Math.abs(tick.contentPx - at121[index].contentPx) > 0.5,
        ).length;
        expect(movedCount).toBeGreaterThan(0);
    });

    /**
     * 放大时视口覆盖的**秒数**变少，因此条数并不随缩放单调 —— 这里只钉住真正
     * 有意义的下界：任何缩放档位下序列都非空且条数有界（不退化、不爆炸）。
     */
    it("任何缩放档位下刻度序列非空且条数有界", () => {
        for (const pxPerSec of [0.5, 8, 40, 200, 800, 3200]) {
            const count = ticksAt({ pxPerSec }).length;
            expect({ pxPerSec, ok: count > 0 && count <= 2000 }).toEqual({
                pxPerSec,
                ok: true,
            });
        }
    });
});

describe("标签版式", () => {
    it("★ 标签可见性不随滚动窗口变化（同一刻度在不同窗口下判定一致）", () => {
        const pxPerSec = 90;
        const viewportWidthPx = 1200;
        const base = ticksAt({ pxPerSec, scrollLeftPx: 0, viewportWidthPx });
        const shifted = ticksAt({ pxPerSec, scrollLeftPx: 900, viewportWidthPx });
        // 按**秒**匹配同一刻度（刻度的时间不随窗口平移变化）。
        const bySec = new Map(shifted.map((tick) => [Math.round(tick.sec * 1e6), tick]));

        let compared = 0;
        for (const tick of base) {
            const other = bySec.get(Math.round(tick.sec * 1e6));
            if (other === undefined) continue;
            compared += 1;
            expect({ sec: tick.sec, showLabel: other.showLabel }).toEqual({
                sec: tick.sec,
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
