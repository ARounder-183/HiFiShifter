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
 *
 * ────────────────────────────────────────────────────────────────────────────
 * 下半部分（`标签栅格` 各 suite）是"缩放时某段之内刻度与文本消失"的回归。
 *
 * 【症状】滚轮水平缩放时，标尺自身的刻度线与文本会在**某个缩放值下、某一段之内**
 * 消失，再放大/缩小或水平滚动一定距离后又回来。不是渲染丢失 —— `TimeRulerMarks`
 * 只渲染 `showLabel` 的刻度，所以"刻度与文本一起消失"等价于"生成器在该处没给出
 * 带标签的刻度"。
 *
 * 【三个可复现的离散化缺陷（均与 DPR 无关）】
 * a. **密度兜底按生成范围计数**：生成范围 = 视口 + 两侧缓冲，比视口宽约 3.5 倍，
 *    于是同一缩放下 `stepBeats` 会随滚动位置整档翻倍（实测 pxPerSec=40 时网格
 *    间距在 scrollLeft 704→768 之间从 20px 跳到 40px），标签栅格随之跳档 —— 这是
 *    "滚动一下又回来"的直接机制；
 * b. **Tempo Map 段首死区**：段内 stride 用 `round(segStep / stepBeats)` 量化成
 *    非 2 的幂，且跨段最小间距约束"拒绝候选却不推进"，段边界出现 290px / 785px
 *    的空洞（标称间距的 2.40 倍）；
 * c. **附点/三连音网格的双栅格**：标签按"拍整除"判定，标签步长与网格步长不整除
 *    时标签整批落不到网格线上，实际间距被放大到 3.00 倍。
 *
 * 修复后：标签栅格取"网格步长的整数倍"（优先与音乐候选阶梯对齐，附点/三连音
 * 网格回退到 2 的幂倍），判定按**索引**取模（精确、与 swing 平移无关），密度兜底
 * 改为按**视口跨度**解析估算（与滚动位置无关）。
 */
import { describe, expect, it } from "vitest";

import { buildTimelineTicks, TICK_WINDOW_STEP_PX } from "./buildTimelineTicks.js";
import { createTimelineAxis } from "../../renderKernel/timelineAxis.js";
import { selectRulerStep } from "../timeFormat.js";
import { barBeatAtSec, tempoMapSegments } from "../../../../utils/tempoMap.js";
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

    it("★ 带标签的刻度间距在 [26px, 2 × 请求值] 之间（下界防重叠、上界防稀疏）", () => {
        // 【为什么需要上界】只断言"间距 >= 26"会放过"标签整体很稀"：间距彼此接近
        // 但都是请求值的 3~4 倍。而 `TimeRulerMarks` 只渲染 `showLabel` 的刻度，
        // 间距过大的那一段**既没有刻度线也没有文本** —— 正是用户报告的"某段之内的
        // 标尺刻度与文本消失"。上界取 2 × 请求值（标签栅格自身的固有粒度）。
        const requested = 110;
        const viewportWidth = 1200;
        for (const pxPerSec of [8, 40, 100, 400, 1600]) {
            const inView = ticksAt({ pxPerSec, viewportWidthPx: viewportWidth }).filter(
                (tick) => tick.contentPx >= 0 && tick.contentPx <= viewportWidth,
            );
            const labeled = inView.filter((tick) => tick.showLabel);
            for (let i = 1; i < labeled.length; i += 1) {
                const gap = labeled[i].contentPx - labeled[i - 1].contentPx;
                expect(gap, `pxPerSec=${pxPerSec} 下界`).toBeGreaterThanOrEqual(26);
                // 网格足够密时才要求上界：网格本身只有三五条时没有刻度可补，属物理稀疏。
                if (inView.length >= 8) {
                    expect(gap, `pxPerSec=${pxPerSec} 上界`).toBeLessThanOrEqual(
                        requested * 2 + 1e-6,
                    );
                }
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

// ════════════════════════════════════════════════════════════════════════════
// 标签栅格稳定性（"缩放时某段之内刻度与文本消失"的回归）
// ════════════════════════════════════════════════════════════════════════════

/** 缩放范围与 `constants.ts` 的 MIN_PX_PER_SEC / MAX_PX_PER_SEC 一致。 */
const MIN_PPS = 4;
const MAX_PPS = 8000;

/** 标签让位阈值（与 `RULER_LABEL_HIDDEN_GAP_PX` 同值）：空洞判据的余量。 */
const YIELD_PX = 26;

interface LabelGridCfg {
    bpm: number;
    beatsPerBar: number;
    grid: string;
    minLabelSpacingPx: number;
    minGridSpacingPx: number;
    swingPercent: number;
    viewportWidth: number;
    tempoMap: TempoMap | null;
}

/** 三段式 Tempo Map：段边界与 BPM 变化都落在视口内。 */
function threeSegmentTempoMap(): TempoMap {
    return {
        points: [
            {
                id: "p0",
                positionSec: 0,
                bpm: 120,
                timeSignature: { numerator: 4, denominator: 4 },
                scale: null,
            },
            {
                id: "p1",
                positionSec: 40,
                bpm: 96,
                timeSignature: { numerator: 4, denominator: 4 },
                scale: null,
            },
            {
                id: "p2",
                positionSec: 90,
                bpm: 150,
                timeSignature: { numerator: 4, denominator: 4 },
                scale: null,
            },
        ],
    };
}

function labelGridCfg(patch: Partial<LabelGridCfg>): LabelGridCfg {
    return {
        bpm: 120,
        beatsPerBar: 4,
        grid: "1/4",
        minLabelSpacingPx: 110,
        minGridSpacingPx: 8,
        swingPercent: 0,
        viewportWidth: 1500,
        tempoMap: null,
        ...patch,
    };
}

/** 复刻生产调用方式：量化锚点 + 宽度补一个量化步长（见 `createTickAxis`）。 */
function labelGridTicks(cfg: LabelGridCfg, pxPerSec: number, scrollLeft: number) {
    const anchor = Math.floor(scrollLeft / TICK_WINDOW_STEP_PX) * TICK_WINDOW_STEP_PX;
    return buildTimelineTicks({
        axis: createTimelineAxis({
            pxPerSec,
            scrollLeftPx: anchor,
            viewportWidthPx: cfg.viewportWidth + TICK_WINDOW_STEP_PX,
        }),
        bpm: cfg.bpm,
        beatsPerBar: cfg.beatsPerBar,
        grid: cfg.grid,
        primaryUnit: "barBeats",
        secondaryUnit: "clock",
        minLabelSpacingPx: cfg.minLabelSpacingPx,
        minGridSpacingPx: cfg.minGridSpacingPx,
        swingPercent: cfg.swingPercent,
        tempoMap: cfg.tempoMap,
    });
}

/** 视口内带标签刻度的内容坐标（升序）。 */
function visibleLabelXs(cfg: LabelGridCfg, pxPerSec: number, scrollLeft: number): number[] {
    const left = scrollLeft;
    const right = scrollLeft + cfg.viewportWidth;
    return labelGridTicks(cfg, pxPerSec, scrollLeft)
        .filter((tick) => tick.showLabel && tick.contentPx >= left && tick.contentPx <= right)
        .map((tick) => tick.contentPx)
        .sort((a, b) => a - b);
}

function gapsOf(xs: number[]): number[] {
    const gaps: number[] = [];
    for (let i = 1; i < xs.length; i += 1) gaps.push(xs[i] - xs[i - 1]);
    return gaps;
}

/**
 * 该内容坐标所在段的"标称"标签间距 —— 由 `selectRulerStep` 独立算出的参考值，
 * 与实现无关，用来判断实际间距是否被放大成空洞。
 */
function nominalSpacingPx(cfg: LabelGridCfg, pxPerSec: number, contentPx: number): number {
    const pxPerBeat = (60 / cfg.bpm) * pxPerSec;
    if (!cfg.tempoMap) {
        return (
            selectRulerStep({
                pxPerBeat,
                grid: cfg.grid,
                beatsPerBar: cfg.beatsPerBar,
                minLabelSpacingPx: cfg.minLabelSpacingPx,
            }) * pxPerBeat
        );
    }
    const sec = contentPx / pxPerSec;
    const segments = tempoMapSegments(cfg.tempoMap, sec + 1);
    let segment = segments[0];
    for (const candidate of segments) {
        if (sec >= candidate.startSec - 1e-9) segment = candidate;
    }
    const segPxPerBeat = (60 / Math.max(1, segment.point.bpm)) * pxPerSec;
    return (
        selectRulerStep({
            pxPerBeat: segPxPerBeat,
            grid: cfg.grid,
            beatsPerBar: Math.max(1, segment.beatsPerBar),
            minLabelSpacingPx: cfg.minLabelSpacingPx,
        }) * segPxPerBeat
    );
}

describe("标签栅格：视口内不得出现空洞", () => {
    it("★ 缩放 × 滚动 × 网格 × 拍号 × swing × 间距设定 × Tempo Map", () => {
        const configs: LabelGridCfg[] = [];
        for (const grid of ["1/4", "1/8", "1/8d", "1/8t"]) {
            for (const beatsPerBar of [3, 4]) {
                for (const swingPercent of [0, 50]) {
                    for (const minLabelSpacingPx of [40, 110, 320]) {
                        for (const tempoMap of [null, threeSegmentTempoMap()]) {
                            configs.push(
                                labelGridCfg({
                                    grid,
                                    beatsPerBar,
                                    swingPercent,
                                    minLabelSpacingPx,
                                    tempoMap,
                                }),
                            );
                        }
                    }
                }
            }
        }

        let worst = { ratio: 0, detail: "" };
        let checked = 0;
        for (const cfg of configs) {
            for (let i = 0; i < 32; i += 1) {
                const pxPerSec = MIN_PPS * Math.pow(MAX_PPS / MIN_PPS, i / 31);
                for (const scrollLeft of [0, 1234]) {
                    const xs = visibleLabelXs(cfg, pxPerSec, scrollLeft);
                    if (xs.length < 4) continue;
                    const gaps = gapsOf(xs);
                    const minGap = Math.min(...gaps);
                    const maxGap = Math.max(...gaps);
                    // 上限 = max(相邻最小间距, 标称间距) 的 2 倍 + 让位阈值。
                    // 2 倍来自候选阶梯的固有粒度（相邻档位是 2 的幂）；让位阈值来自
                    // "间距不足时左侧让位"这一条。旧实现在此判据下最坏 2.40 倍。
                    const bound =
                        Math.max(2 * minGap, 2 * nominalSpacingPx(cfg, pxPerSec, xs[0])) + YIELD_PX;
                    const ratio = maxGap / bound;
                    checked += 1;
                    if (ratio > worst.ratio) {
                        worst = {
                            ratio,
                            detail: `${JSON.stringify({
                                grid: cfg.grid,
                                beatsPerBar: cfg.beatsPerBar,
                                swingPercent: cfg.swingPercent,
                                minLabelSpacingPx: cfg.minLabelSpacingPx,
                                tempoMap: cfg.tempoMap ? "map" : null,
                            })} pxPerSec=${pxPerSec.toFixed(2)} scrollLeft=${scrollLeft} maxGap=${maxGap.toFixed(0)} minGap=${minGap.toFixed(0)} bound=${bound.toFixed(0)}`,
                        };
                    }
                }
            }
        }
        expect(checked).toBeGreaterThan(500);
        expect(worst.ratio, `最坏空洞：${worst.detail}`).toBeLessThanOrEqual(1);
    });
});

describe("标签栅格：位置不得随滚动变化", () => {
    it("★ 同一缩放值下，固定内容窗口内的刻度与标签集合恒定", () => {
        const cases: Array<[string, LabelGridCfg]> = [
            ["均匀网格", labelGridCfg({})],
            ["均匀网格 1/8d", labelGridCfg({ grid: "1/8d" })],
            ["Tempo Map", labelGridCfg({ tempoMap: threeSegmentTempoMap() })],
            ["Tempo Map 1/8", labelGridCfg({ grid: "1/8", tempoMap: threeSegmentTempoMap() })],
        ];
        for (const [name, cfg] of cases) {
            let unstableTicks = 0;
            let unstableLabels = 0;
            for (let i = 0; i < 120; i += 1) {
                const pxPerSec = MIN_PPS * Math.pow(MAX_PPS / MIN_PPS, i / 119);
                const tickSignatures = new Set<string>();
                const labelSignatures = new Set<string>();
                // 四个锚点的生成范围都覆盖查询窗口 [0, 1200]，因此窗口内的刻度
                // 必须逐位相同 —— 否则就是"滚动让刻度/标签忽有忽无"。
                for (const anchor of [0, 256, 512, 768]) {
                    const inWindow = labelGridTicks(cfg, pxPerSec, anchor).filter(
                        (tick) => tick.contentPx >= 0 && tick.contentPx <= 1200,
                    );
                    tickSignatures.add(
                        JSON.stringify(
                            inWindow.map((tick) => Math.round(tick.contentPx * 100) / 100),
                        ),
                    );
                    labelSignatures.add(
                        JSON.stringify(
                            inWindow
                                .filter((tick) => tick.showLabel)
                                .map((tick) => Math.round(tick.contentPx * 100) / 100),
                        ),
                    );
                }
                if (tickSignatures.size > 1) unstableTicks += 1;
                if (labelSignatures.size > 1) unstableLabels += 1;
            }
            expect(unstableTicks, `${name}: ${unstableTicks}/120 个缩放值下刻度随滚动变化`).toBe(0);
            expect(unstableLabels, `${name}: ${unstableLabels}/120 个缩放值下标签随滚动变化`).toBe(
                0,
            );
        }
    });
});

describe("标签栅格：附点 / 三连音网格必须均匀", () => {
    it("★ 无 Tempo Map 时标签间距 max/min 恒为 1", () => {
        for (const grid of ["1/8d", "1/8t", "1/4t", "1/16d"]) {
            let worst = 0;
            let detail = "";
            for (let i = 0; i < 120; i += 1) {
                const pxPerSec = MIN_PPS * Math.pow(MAX_PPS / MIN_PPS, i / 119);
                const cfg = labelGridCfg({ grid });
                const xs = visibleLabelXs(cfg, pxPerSec, 0);
                if (xs.length < 4) continue;
                const gaps = gapsOf(xs);
                const ratio = Math.max(...gaps) / Math.min(...gaps);
                if (ratio > worst) {
                    worst = ratio;
                    detail = `grid=${grid} pxPerSec=${pxPerSec.toFixed(2)} gaps=${[
                        ...new Set(gaps.map((g) => g.toFixed(0))),
                    ].join("/")}`;
                }
            }
            // 旧实现：标签步长与网格步长不整除，标签只能落在两者公倍数的位置，
            // 间距在 1× 与 2× 之间交替（实测最坏 3.00 倍）。
            expect(worst, detail).toBeLessThanOrEqual(1.001);
        }
    });
});

describe("网格步长不得随滚动位置变化", () => {
    it("★ 同一缩放值下，视口内的网格线间距恒定", () => {
        for (const pxPerSec of [12, 25, 40, 120]) {
            const spacings = new Set<string>();
            for (let scrollLeft = 0; scrollLeft <= 20000; scrollLeft += 64) {
                const cfg = labelGridCfg({});
                const left = scrollLeft;
                const right = scrollLeft + cfg.viewportWidth;
                const xs = labelGridTicks(cfg, pxPerSec, scrollLeft)
                    .filter((tick) => tick.contentPx >= left && tick.contentPx <= right)
                    .map((tick) => tick.contentPx);
                const gaps = gapsOf(xs);
                if (gaps.length === 0) continue;
                gaps.sort((a, b) => a - b);
                spacings.add(gaps[Math.floor(gaps.length / 2)].toFixed(1));
            }
            // 旧实现：密度兜底按**生成范围**计数，滚动让缓冲变长时步长整档翻倍
            // （pxPerSec=40 实测 20px ↔ 40px 两档）。
            expect([...spacings], `pxPerSec=${pxPerSec} 出现多档网格间距`).toHaveLength(1);
        }
    });
});

describe("swing 打开时标签落在未被平移的线上", () => {
    it("★ 带标签刻度的拍值必须是整数（否则文字会渲染成 1.2.300）", () => {
        for (const grid of ["1/4", "1/8"]) {
            for (const swingPercent of [25, 50, 100]) {
                for (const pxPerSec of [20, 60, 200, 800]) {
                    const cfg = labelGridCfg({ grid, swingPercent, minLabelSpacingPx: 110 });
                    const labeled = labelGridTicks(cfg, pxPerSec, 0).filter(
                        (tick) => tick.showLabel && tick.contentPx >= 0 && tick.contentPx <= 1500,
                    );
                    expect(labeled.length).toBeGreaterThan(0);
                    for (const tick of labeled) {
                        expect(
                            Math.abs(tick.beat - Math.round(tick.beat)),
                            `grid=${grid} swing=${swingPercent} pxPerSec=${pxPerSec} beat=${tick.beat}`,
                        ).toBeLessThan(1e-6);
                    }
                }
            }
        }
    });
});

describe("Tempo Map 变化点必须带标签", () => {
    it("★ 每个位于视口内的变化点都有 showLabel 刻度", () => {
        const tempoMap = threeSegmentTempoMap();
        for (const pxPerSec of [8, 12.11, 20, 40, 90]) {
            for (const scrollLeft of [0, 300, 800]) {
                const cfg = labelGridCfg({ tempoMap });
                const ticks = labelGridTicks(cfg, pxPerSec, scrollLeft);
                for (const point of tempoMap.points) {
                    const px = point.positionSec * pxPerSec;
                    if (px < scrollLeft || px > scrollLeft + cfg.viewportWidth) continue;
                    const hit = ticks.find((tick) => Math.abs(tick.contentPx - px) < 0.5);
                    expect(hit, `变化点 ${point.positionSec}s 处没有刻度`).toBeDefined();
                    expect(hit?.showLabel, `变化点 ${point.positionSec}s 处刻度没有标签`).toBe(
                        true,
                    );
                }
            }
        }
    });
});

/**
 * 密集 Tempo Map：8 个变化点、间隔 7.3s，BPM 在 80..159 间反复变化。
 *
 * 【为什么必须用密集图】稀疏的三段图（0/40/90s）在多数缩放下，变化点间距本身就
 * 大于 `minLabelSpacingPx`，"段首自动命中标签"与真实栅格恰好重合，看不出缺陷。
 * 只有当变化点间距**明显小于**请求的标签间距时，两套栅格的拼接才刺眼 —— 这正是
 * 用户报的"某段之内刻度与文本消失"。
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

describe("Tempo Map 标签栅格必须跨段连续", () => {
    /**
     * 【要钉死的缺陷】`buildTempoGridLines` 曾把弱线索引按**段内**编号（每段从 0
     * 开始），而标签判定是 `index % stride === 0`。于是**每个段起点的 index 恒为 0**，
     * `0 % 任何 stride` 都成立 ⇒ 每个变化点无条件获得标签，完全忽略
     * `minLabelSpacingPx`：变化点区（58px 间距）与尾段真实栅格（193px）首尾拼接，
     * 表现为前半密集、后半突然稀疏、中间一段"没有刻度与文本"。
     *
     * 修复后索引全局单调（跨段累加），段起点不再自动命中，变化点标签由 §4 显式补充。
     * 判据用"相邻标签间距 max / median"：两套栅格拼接时会远超 2；均匀栅格下接近 1。
     */
    it("★ 密集变化点下视口内标签间距 max/median ≤ 2.0", () => {
        const tempoMap = denseTempoMap();
        let worst = { ratio: 0, detail: "" };
        let checked = 0;
        for (let i = 0; i < 48; i += 1) {
            const pxPerSec = MIN_PPS * Math.pow(MAX_PPS / MIN_PPS, i / 47);
            for (const scrollLeft of [0, 700, 2600]) {
                const cfg = labelGridCfg({ tempoMap });
                const xs = visibleLabelXs(cfg, pxPerSec, scrollLeft);
                if (xs.length < 4) continue;
                const gaps = gapsOf(xs);
                const sorted = [...gaps].sort((a, b) => a - b);
                const median = sorted[Math.floor(sorted.length / 2)];
                const ratio = Math.max(...gaps) / median;
                checked += 1;
                if (ratio > worst.ratio) {
                    worst = {
                        ratio,
                        detail: `pxPerSec=${pxPerSec.toFixed(2)} scrollLeft=${scrollLeft} gaps=${[
                            ...new Set(gaps.map((g) => g.toFixed(0))),
                        ].join("/")}`,
                    };
                }
            }
        }
        expect(checked).toBeGreaterThan(50);
        // 修复前实测 4.1×（pps=6.5，两套栅格拼接）；修复后最坏 1.99×，恰为该图
        // 相邻段 BPM 之比（159/80）—— 这是"标签锚定在拍栅格上"的固有上界：跨越
        // 速度边界时，同一拍间距在两段里的像素宽度之比不可能小于 BPM 之比。
        expect(worst.ratio, `Tempo Map 标签空洞：${worst.detail}`).toBeLessThanOrEqual(2.0);
    });

    /**
     * 窄视口下视口内只有 3~5 个标签，`max/median` 的中位数不稳定（会误报），
     * 因此上面那条只在宽视口下用。这里改用"最大间距 / 标称间距"（`selectRulerStep`
     * 独立算出的参考值）作为"有没有空洞"的判据，对任意视口宽都成立。
     */
    it("★ 任意视口宽下，最大标签间距不超过标称间距的 2 倍", () => {
        const tempoMap = denseTempoMap();
        let worst = { ratio: 0, detail: "" };
        for (const viewportWidth of [320, 700, 1500, 2560]) {
            for (let i = 0; i < 48; i += 1) {
                const pxPerSec = MIN_PPS * Math.pow(MAX_PPS / MIN_PPS, i / 47);
                for (const scrollLeft of [0, 700, 2600]) {
                    const cfg = labelGridCfg({ tempoMap, viewportWidth });
                    const xs = visibleLabelXs(cfg, pxPerSec, scrollLeft);
                    if (xs.length < 4) continue;
                    const maxGap = Math.max(...gapsOf(xs));
                    const nominal = nominalSpacingPx(cfg, pxPerSec, xs[0]);
                    const ratio = maxGap / nominal;
                    if (ratio > worst.ratio) {
                        worst = {
                            ratio,
                            detail: `vw=${viewportWidth} pxPerSec=${pxPerSec.toFixed(2)} scrollLeft=${scrollLeft} maxGap=${maxGap.toFixed(0)} nominal=${nominal.toFixed(0)}`,
                        };
                    }
                }
            }
        }
        expect(worst.ratio, `空洞超过 2 倍标称间距：${worst.detail}`).toBeLessThanOrEqual(2.0);
    });

    /**
     * 索引改为全局单调后，标签判定仍是 `index % stride === 0`。swing 会把段内
     * **奇数**索引的线整体平移半步，其秒位不再对应整拍。若栅格相位让标签落回
     * 这些线上，文字会被渲染成 "1.2.125"（拍内余量非 0）。这里把"Tempo Map + swing
     * 下标签仍落在整拍上"钉死 —— 上面的 swing 用例只覆盖了无 Tempo Map 的路径。
     *
     * 判据用段内分解的拍内余量 `sub`，而不是全局拍：变化点本身可能落在非整拍
     * （例如 7.3s 处的前段 BPM 为 80 ⇒ 全局拍 9.73），但它在其所在段内是整拍起点。
     */
    it("★ Tempo Map + swing 下标签仍落在整拍上（拍内余量为 0）", () => {
        const tempoMap = denseTempoMap();
        for (const swingPercent of [25, 50, 100]) {
            for (const pxPerSec of [20, 60, 200, 800]) {
                const cfg = labelGridCfg({ tempoMap, swingPercent, minLabelSpacingPx: 110 });
                const labeled = labelGridTicks(cfg, pxPerSec, 0).filter(
                    (tick) => tick.showLabel && tick.contentPx >= 0 && tick.contentPx <= 1500,
                );
                expect(labeled.length).toBeGreaterThan(0);
                for (const tick of labeled) {
                    const { sub } = barBeatAtSec(tempoMap, tick.sec, cfg.bpm, cfg.beatsPerBar);
                    expect(
                        Math.abs(sub),
                        `swing=${swingPercent} pxPerSec=${pxPerSec} sec=${tick.sec} sub=${sub}`,
                    ).toBeLessThan(1e-6);
                }
            }
        }
    });
});
