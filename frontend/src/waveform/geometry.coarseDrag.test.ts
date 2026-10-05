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
