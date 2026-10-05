/**
 * ★ 回归：**粗缩放下拖拽 clip 时波形抖动**。
 *
 * ## 缺陷形态
 *
 * 列的桶窗口原先用 `floor/ceil` 吸附到**绝对源时间**桶栅格：`[floor(loF),
 * ceil(hiF)−1]`。窗口本身随 clip 的亚像素位置连续滑动，于是"窗口覆盖几个桶"
 * 会随**亚桶相位**在 `⌊N⌋` 与 `⌊N⌋+1` 之间翻转。
 *
 * - 细档 `N ≈ 19`：多算一个桶只带来约 5% 扰动，且相邻桶相隔 0.36ms（几乎同值）
 *   ⇒ 不可见；
 * - 粗档 `N ≈ 1~2`：相邻桶相隔 11.6ms / 92.9ms，是彼此**独立**的样本 ⇒ 多算一个
 *   桶就是满量程跳变，且每移动约 `1/N` 像素翻一次 ⇒ 以指针频率抖动。
 *
 * 这正是用户报告的「水平缩放到足够小时，拖拽 clip 波形抖动、松手才恢复」。
 *
 * ## 本用例钉住什么
 *
 * 钉住**机制本身**：每列聚合的桶数（= 切片数 = `min(桶数, 16)`）必须**只由缩放
 * 决定**，与 clip 的亚桶相位无关。观测量取 `amplitudeMap` 的**总调用次数**
 * （列数固定 4、每列每次切片 min/max 各一次 ⇒ 总数 = 4 × 切片数 × 2）。
 * 旧实现下它会随相位在 16 / 20 / 24 之间变化；修复后恒为 16。
 *
 * 【为什么不直接断言包络高度】包络高度**本来就该**随内容滑动而变化（拖拽时波形
 * 会移动）；要区分"内容移动"与"窗口尺寸翻转"必须看窗口本身，故此处观察切片数。
 */
import { describe, expect, it } from "vitest";

import { buildWaveformGeometry } from "./geometry";
import type { WaveformScene } from "./sceneBuilder.ts";

/** 桶数：8 个桶、总时长 1s ⇒ 桶宽 0.125s。 */
const BUCKET_COUNT = 8;
const DATA_DURATION_SEC = 1;

/**
 * 场景：4 CSS px 宽（dpr=1 ⇒ 4 列）、源时长 0.75s ⇒ 每列覆盖 0.1875s = **1.5 个桶**
 *（粗档：`coarseSpan = 2`；旧实现下窗口桶数会在 2 / 3 之间翻转）。
 */
function makeScene(sourceStartSec: number): WaveformScene {
    return {
        segments: [
            {
                clipId: "clip",
                sourcePath: "/tone.wav",
                sourceSampleRate: 8,
                sourceStartSec,
                sourceEndSec: sourceStartSec + 0.75,
                clipStartSec: 0,
                clipLocalStartSec: 0,
                clipLocalEndSec: 0.75,
                clipTotalDurationSec: 0.75,
                screenRect: { x: 0, y: 0, width: 4, height: 100 },
                reversed: false,
                gain: 1,
                fadeInSec: 0,
                fadeOutSec: 0,
                fadeInShape: 0,
                fadeInDir: 0,
                fadeOutShape: 0,
                fadeOutDir: 0,
                alpha: 1,
                channelMode: 0,
                sourceChannels: 0,
            },
        ],
        markers: [],
    } as WaveformScene;
}

/** 逐桶**互不相同**的峰值：窗口尺寸一旦翻转，最大值立刻随之改变。 */
function varyingPeaks() {
    const max = new Float32Array(BUCKET_COUNT);
    const min = new Float32Array(BUCKET_COUNT);
    for (let i = 0; i < BUCKET_COUNT; i += 1) {
        max[i] = 0.1 + (0.9 * ((i * 7) % BUCKET_COUNT)) / BUCKET_COUNT;
        min[i] = -max[i];
    }
    return { min, max, dataStartSec: 0, dataDurationSec: DATA_DURATION_SEC };
}

/** 构建一次并返回 (amplitudeMap 调用次数, 顶点)。 */
function buildAt(sourceStartSec: number) {
    let calls = 0;
    const geometry = buildWaveformGeometry({
        scene: makeScene(sourceStartSec),
        color: "#ffffff",
        getPeaks: () => varyingPeaks() as never,
        amplitudeMap: (value) => {
            calls += 1;
            return value;
        },
    });
    return { calls, geometry };
}

describe("粗缩放：列窗口的桶数不得随亚桶相位变化", () => {
    it("★ 扫过多个桶相位，amplitudeMap 调用总数恒定（旧实现会随相位变化）", () => {
        const totals = new Set<number>();
        for (let step = 0; step <= 20; step += 1) {
            // 扫过 0.05..0.15s（≈ 0.8 个桶宽），覆盖各种亚桶相位。
            const sourceStartSec = 0.05 + (step / 20) * 0.1;
            totals.add(buildAt(sourceStartSec).calls);
        }
        // 4 列 × 2 切片 × (min,max) = 16；唯一值 ⇒ 每列桶数没有翻转。
        expect([...totals]).toEqual([16]);
    });

    it("★ 包络高度不超过真实桶峰（修复不得引入幻峰）", () => {
        // 线性映射（amplitudeMap 恒等）下，包络顶 = 桶峰本身。
        const { geometry } = buildAt(0.1);
        let tallest = Number.POSITIVE_INFINITY;
        for (let v = 0; v < geometry.vertices.length; v += 12) {
            const y = geometry.vertices[v + 1] ?? 0;
            if (y < tallest) tallest = y;
        }
        // 中心 50、半高 50：最高桶峰 1.0 ⇒ 包络顶 ≥ 0（不得低于 0 = 不得越界）。
        expect(tallest).toBeGreaterThanOrEqual(0);
        // 且必须画出真实存在的峰值（不能全被压平）。
        expect(tallest).toBeLessThan(50);
    });

    it("对照：细档（每列聚合约 19 个桶）本来就稳定", () => {
        // 4 CSS px 宽但源时长放大到 12s（桶宽不变）⇒ 每列 3s = 24 个桶 ⇒ 细档路径。
        const totals = new Set<number>();
        for (let step = 0; step <= 10; step += 1) {
            let calls = 0;
            buildWaveformGeometry({
                scene: {
                    segments: [
                        {
                            ...makeScene(step * 0.004).segments[0],
                            sourceEndSec: step * 0.004 + 12,
                            clipLocalEndSec: 12,
                            clipTotalDurationSec: 12,
                        },
                    ],
                    markers: [],
                } as never,
                color: "#ffffff",
                getPeaks: () =>
                    ({
                        min: new Float32Array(96).fill(-0.5),
                        max: new Float32Array(96).fill(0.5),
                        dataStartSec: 0,
                        dataDurationSec: 12,
                    }) as never,
                amplitudeMap: (value) => {
                    calls += 1;
                    return value;
                },
            });
            totals.add(calls);
        }
        // 细档每列 16 个切片（上限）⇒ 4 × 16 × 2 = 128，恒定。
        expect(totals.size).toBe(1);
    });
});

/* ────────────────────────────────────────────────────────────────────────────
 * ★ 回归：**Loop 瓦片边界列**不得随亚桶相位收窄窗口。
 *
 * `sceneBuilder` 把 Loop clip 切成 1 个头瓦片 + N 个整周期瓦片，**每个瓦片是独立
 * 段**、切片为 `[0, mediaDuration]`。于是每个瓦片的**首列与末列**都紧贴数据边界
 *（末列的时间窗有一半越过媒体末尾）。旧实现把 `winLo / winHi` 各自钳进
 * `[0, sampleCount−1]`，末列因此**桶数变少**，且随亚桶相位在 ⌊N⌋ / ⌈N⌉ 之间翻转
 * ⇒ 满量程台阶、以指针频率闪烁。
 *
 * 非 Loop clip 只有 2 个这样的列（clip 左右缘，通常在视口外）；Loop clip 有
 * 2 × 可见周期数个、成对嵌在波形内部 —— 这就是"抖动与循环节位置有关"。
 *
 * 本组用例钉住：窗口 origin 被钳进 `[0, sampleCount−span]`（宽度恒定），
 * 而不是把两端各自钳到数据边界（宽度缩水）。
 * ──────────────────────────────────────────────────────────────────────────── */

const LOOP_BUCKET_COUNT = 8;
const LOOP_MEDIA_DURATION_SEC = 1;

/**
 * 全媒体切片（Loop 整周期瓦片）的 4 列场景。
 *
 * `x` 是瓦片相对列栅格的**亚列相位**（拖拽时连续变化；`x ∈ (0,1)` 时列集恒为
 * `[1,4]` 共 4 列）。源时长 1s、8 桶 ⇒ 桶宽 0.125s ⇒ 每列覆盖 0.25s = 2 个桶。
 */
function makeLoopTilesScene(xs: readonly number[]): WaveformScene {
    return {
        segments: xs.map((x, index) => ({
            clipId: `loop-${index}`,
            sourcePath: "/tone.wav",
            sourceSampleRate: LOOP_BUCKET_COUNT,
            sourceStartSec: 0,
            sourceEndSec: LOOP_MEDIA_DURATION_SEC,
            clipStartSec: 0,
            clipLocalStartSec: 0,
            clipLocalEndSec: LOOP_MEDIA_DURATION_SEC,
            clipTotalDurationSec: LOOP_MEDIA_DURATION_SEC,
            screenRect: { x, y: 0, width: 4, height: 100 },
            reversed: false,
            gain: 1,
            fadeInSec: 0,
            fadeOutSec: 0,
            fadeInShape: 0,
            fadeInDir: 0,
            fadeOutShape: 0,
            fadeOutDir: 0,
            alpha: 1,
            channelMode: 0,
            sourceChannels: 0,
        })),
        markers: [],
    } as WaveformScene;
}

/** 单瓦片场景（Loop 的一个整周期瓦片）。 */
function makeLoopTileScene(x: number): WaveformScene {
    return makeLoopTilesScene([x]);
}

function loopPeaks(maxValues: readonly number[]) {
    return {
        min: new Float32Array(maxValues.map((v) => -v)),
        max: new Float32Array(maxValues),
        dataStartSec: 0,
        dataDurationSec: LOOP_MEDIA_DURATION_SEC,
    };
}

function buildLoopScene(scene: WaveformScene, maxValues: readonly number[]) {
    let calls = 0;
    const geometry = buildWaveformGeometry({
        scene,
        color: "#ffffff",
        getPeaks: () => loopPeaks(maxValues) as never,
        amplitudeMap: (value) => {
            calls += 1;
            return value;
        },
    });
    return { calls, geometry };
}

function buildLoopTileAt(x: number, maxValues: readonly number[]) {
    return buildLoopScene(makeLoopTileScene(x), maxValues);
}

/** 覆盖到的**设备列索引**（顶点 x 是 CSS 列中心，dpr=1 ⇒ 索引 = x − 0.5）。 */
function coveredColumnIndexes(vertices: ArrayLike<number>): number[] {
    const indexes = new Set<number>();
    for (let v = 0; v < vertices.length; v += 6) {
        indexes.add(Math.round((vertices[v] ?? 0) - 0.5));
    }
    return [...indexes].sort((a, b) => a - b);
}

/** 收集所有包络顶点的 y（顶点 6 个 float：x,y,r,g,b,a；每列两个顶点）。 */
function collectVertexYs(vertices: ArrayLike<number>): number[] {
    const ys: number[] = [];
    for (let v = 0; v < vertices.length; v += 6) {
        ys.push(vertices[v + 1] ?? 0);
    }
    return ys;
}

/** 只收集**某一列**（按 x 精确匹配）的顶点 y。 */
function collectColumnYs(vertices: ArrayLike<number>, x: number): number[] {
    const ys: number[] = [];
    for (let v = 0; v < vertices.length; v += 6) {
        if (Math.abs((vertices[v] ?? Number.NaN) - x) < 1e-6) {
            ys.push(vertices[v + 1] ?? 0);
        }
    }
    return ys;
}

describe("粗缩放：Loop 瓦片边界列不得随亚桶相位收窄窗口", () => {
    const flat = new Array<number>(LOOP_BUCKET_COUNT).fill(0.5);

    it("★ 扫过亚列相位，每列桶数恒定（旧实现末列在 2 / 1 之间翻转）", () => {
        const totals = new Set<number>();
        for (let step = 1; step <= 20; step += 1) {
            totals.add(buildLoopTileAt(step / 21, flat).calls);
        }
        // 4 列 × 2 桶 × (min,max) = 16；唯一值 ⇒ 末列没有缩水。
        expect([...totals]).toEqual([16]);
    });

    it("★ 边界列必须显示真实桶内容，不得引入伪静音", () => {
        // 桶 6 是大值、桶 7 是小值：末列窗口应恰好覆盖 [6,7]（画到 0.9），
        // 而不是被钳成只剩桶 7（只画到 0.2）或被补 0（画到中心线）。
        // 只看**末列**（xCss = 4.5）—— 其它列本来就能看到桶 6，混在一起会掩盖问题。
        const maxValues = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.2];
        const { geometry } = buildLoopTileAt(0.5, maxValues);
        const lastColumnYs = collectColumnYs(geometry.vertices, 4.5);
        expect(lastColumnYs).toHaveLength(2);
        // 中心 50、半高 50：0.9 ⇒ y = 50 − 45 = 5。
        // Float32 顶点缓冲：容差按 1e-5。
        expect(Math.min(...lastColumnYs)).toBeCloseTo(5, 5);
        // 下沿来自 −0.9 ⇒ y = 50 + 45 = 95（同一窗口、配对正确）。
        expect(Math.max(...lastColumnYs)).toBeCloseTo(95, 5);
    });

    it("对照：全桶同值时所有列等高（宽度恒定、无缩水、无补零）", () => {
        const { geometry } = buildLoopTileAt(0.5, flat);
        const ys = new Set(collectVertexYs(geometry.vertices));
        // 0.5 ⇒ 上沿 25、下沿 75；只有这两个值 ⇒ 没有一列缩水或补零。
        expect([...ys].sort((a, b) => a - b)).toEqual([25, 75]);
    });

    it("★ 相邻两个整周期瓦片（真实 Loop 拓扑）：扫过循环节，列覆盖与桶数恒定", () => {
        // 瓦片按 `periodSec × pxPerSec` 精确相邻（sceneBuilder 的构造方式）：
        // B 的起点 = A 的起点 + 宽度。循环节因此随拖拽连续扫过列边界。
        const widths = new Set<number>();
        const columnCounts = new Set<number>();
        for (let step = 0; step <= 20; step += 1) {
            const x = step / 20; // 0..1，循环节跨过一次列边界
            const { calls, geometry } = buildLoopScene(
                makeLoopTilesScene([x, x + 4]),
                flat,
            );
            widths.add(calls);
            columnCounts.add(coveredColumnIndexes(geometry.vertices).length);
        }
        // 8 列 × 2 桶 × (min,max) = 32，恒定。
        expect([...widths]).toEqual([32]);
        // 两瓦片首尾相接：恰好 8 列，无缝、无重叠（循环节移动不改变列数）。
        expect([...columnCounts]).toEqual([8]);
    });
});
