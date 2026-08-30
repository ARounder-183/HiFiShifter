/**
 * 时间线统一坐标投影的自检。
 *
 * 【主要内容】
 * 1. 投影自身的数学性质：往返一致、线性性、内容/视口坐标偏移关系；
 * 2. 宽度语义：`durationToWidthPx` 带下限、`secToSpanPx` 可为 0；
 * 3. 栅格对齐：`snapPx` / `strokePx` 落在整数设备像素上；
 * 4. **跨层一致**：同一 clip 经 clip 体模型（内容坐标）与波形场景
 *    （视口坐标）投影后，像素位置严格相等。
 *
 * 【作用】第 4 组断言是本次重构的核心守护：历史上 clip 走「先乘后减」、
 * 波形走「先除后乘」，两者浮点不等价。任何图层重新引入独立换算都会被
 * 本测试拦截。
 *
 * 【与其他模块的关系】
 * 覆盖 `timelineAxis.ts`，并与 `timelineCanvasModel.ts`、`waveform/sceneBuilder.ts`
 * 联合验证投影一致性。
 */

import { test } from "vitest";

import { buildSparseClipRenderModel } from "./timelineCanvasModel.js";
import {
    contentPxToSec,
    createTimelineAxis,
    durationToWidthPx,
    MIN_FEATURE_WIDTH_PX,
    secToContentPx,
    secToSpanPx,
    secToViewportPx,
    snapPx,
    strokePx,
    viewportEndSec,
    viewportPxToSec,
    viewportStartSec,
    withAxis,
} from "./timelineAxis.js";
import { buildWaveformScene } from "../../../../waveform/sceneBuilder.ts";

function assertEqual(actual: unknown, expected: unknown, label: string): void {
    if (actual !== expected) {
        throw new Error(`${label}: expected ${String(expected)}, received ${String(actual)}`);
    }
}

function assertTrue(condition: boolean, label: string): void {
    if (!condition) throw new Error(`${label}: expected true`);
}

/**
 * 浮点近似断言。
 *
 * 特殊说明：扫描范围刻意覆盖到 1e7 量级的像素坐标，此时两次大数相减的
 * 舍入误差天然在 1e-9 px 级别（远低于可见阈值，也远低于历史上「先除后乘」
 * 造成的误差量级）。因此这里用带绝对下限的相对容差，而不是严格相等。
 */
function assertClose(actual: number, expected: number, tolerance: number, label: string): void {
    if (!Number.isFinite(actual) || !Number.isFinite(expected)) {
        throw new Error(`${label}: non-finite value (${String(actual)} vs ${String(expected)})`);
    }
    if (Math.abs(actual - expected) > tolerance) {
        throw new Error(`${label}: expected ~${expected}, received ${actual}`);
    }
}

/** 确定性伪随机（LCG），保证测试可重现，不依赖 Math.random 的全局状态。 */
function makeRandom(seed: number): () => number {
    let state = seed >>> 0;
    return () => {
        state = (state * 1664525 + 1013904223) >>> 0;
        return state / 0x100000000;
    };
}

test("components/layout/timeline/runtime/timelineAxis.test.ts scripted checks", async () => {
    // ── 1. 基础投影 ────────────────────────────────────────────────
    {
        const axis = createTimelineAxis({ pxPerSec: 100, scrollLeftPx: 250, viewportWidthPx: 800 });
        assertEqual(secToContentPx(axis, 2), 200, "secToContentPx");
        assertEqual(secToViewportPx(axis, 2), -50, "secToViewportPx");
        assertEqual(contentPxToSec(axis, 200), 2, "contentPxToSec");
        assertEqual(viewportPxToSec(axis, -50), 2, "viewportPxToSec");
        assertEqual(viewportStartSec(axis), 2.5, "viewportStartSec");
        assertEqual(viewportEndSec(axis), 10.5, "viewportEndSec");
    }

    // ── 2. pxPerSec 下限保护：0 / 负值 / NaN 不得产生除零或反向坐标 ──
    {
        for (const bad of [0, -100, Number.NaN, Number.POSITIVE_INFINITY]) {
            const axis = createTimelineAxis({ pxPerSec: bad });
            assertTrue(axis.pxPerSec > 0, `pxPerSec guard for ${String(bad)}`);
            assertTrue(Number.isFinite(secToContentPx(axis, 1)), `finite px for ${String(bad)}`);
        }
    }

    // ── 3. 往返一致与线性性（随机扫描） ────────────────────────────
    {
        const random = makeRandom(20260829);
        for (let i = 0; i < 500; i += 1) {
            const axis = createTimelineAxis({
                pxPerSec: 0.5 + random() * 2000,
                scrollLeftPx: random() * 20000 - 5000,
                viewportWidthPx: 1 + random() * 4000,
            });
            const sec = random() * 5000;

            // 往返：视口坐标 → 秒 → 视口坐标
            const viaViewport = secToViewportPx(axis, viewportPxToSec(axis, sec * axis.pxPerSec));
            assertTrue(
                Math.abs(viaViewport - sec * axis.pxPerSec) < 1e-6,
                `viewport roundtrip at i=${i}`,
            );
            // 往返：内容坐标 → 秒 → 内容坐标
            const viaContent = secToContentPx(axis, contentPxToSec(axis, sec));
            assertTrue(Math.abs(viaContent - sec) < 1e-6, `content roundtrip at i=${i}`);

            // 线性性：跨度只取决于时长，与绝对位置无关。
            // 容差按「跨度本身」与「绝对坐标量级」取相对值，避免把大坐标下的
            // 固有舍入误差误判为缺陷。
            const delta = 0.25 + random() * 3;
            const expectedSpan = delta * axis.pxPerSec;
            const scale = Math.max(1, Math.abs(expectedSpan), Math.abs(sec * axis.pxPerSec));
            const spanA = secToViewportPx(axis, sec + delta) - secToViewportPx(axis, sec);
            const spanB =
                secToViewportPx(axis, sec + 2 * delta) - secToViewportPx(axis, sec + delta);
            assertClose(spanA, expectedSpan, 1e-9 * scale, `linearity spanA at i=${i}`);
            assertClose(spanB, expectedSpan, 1e-9 * scale, `linearity spanB at i=${i}`);
        }
    }

    // ── 4. 宽度语义：下限仅在 durationToWidthPx，secToSpanPx 可为 0 ──
    {
        const axis = createTimelineAxis({ pxPerSec: 100 });
        assertEqual(durationToWidthPx(axis, 3), 300, "durationToWidthPx normal");
        assertEqual(durationToWidthPx(axis, 0.0001), MIN_FEATURE_WIDTH_PX, "durationToWidthPx min");
        assertEqual(
            durationToWidthPx(axis, -5),
            MIN_FEATURE_WIDTH_PX,
            "durationToWidthPx negative",
        );
        assertEqual(secToSpanPx(axis, 0), 0, "secToSpanPx zero stays zero");
        assertEqual(secToSpanPx(axis, -5), 0, "secToSpanPx negative clamps to zero");
        assertEqual(secToSpanPx(axis, 0.25), 25, "secToSpanPx normal");
    }

    // ── 5. 栅格对齐：snapPx 落在整数设备像素，strokePx 奇宽补半像素 ──
    {
        for (const dpr of [1, 1.25, 1.5, 2, 3]) {
            const axis = createTimelineAxis({ pxPerSec: 100, dpr });
            for (const raw of [0, 10.4, 33.333, 100.7, 999.49]) {
                const snapped = snapPx(axis, raw);
                assertTrue(
                    Math.abs(snapped * dpr - Math.round(snapped * dpr)) < 1e-9,
                    `snapPx lands on device pixel (dpr=${dpr}, raw=${raw})`,
                );
            }
            // 奇数物理宽度的线要偏移半物理像素，使线体正好覆盖整数个设备
            // 像素；偶数宽度则不做偏移。断言偏移量本身，避免 snapPx 的除法
            // 与乘法往返误差干扰判定。
            const snapped = snapPx(axis, 10.4);
            assertClose(
                strokePx(axis, 10.4, 1 / dpr) - snapped,
                0.5 / dpr,
                1e-9,
                `strokePx odd width offset (dpr=${dpr})`,
            );
            assertEqual(
                strokePx(axis, 10.4, 2 / dpr),
                snapped,
                `strokePx even width no offset (dpr=${dpr})`,
            );
        }
    }

    // ── 6. withAxis：等价 patch 必须返回原引用（供重绘去重） ────────
    {
        const base = createTimelineAxis({ pxPerSec: 100, scrollLeftPx: 10, dpr: 2 });
        assertEqual(withAxis(base, { pxPerSec: 100 }), base, "withAxis identity");
        assertEqual(withAxis(base, {}), base, "withAxis empty patch identity");
        assertTrue(withAxis(base, { scrollLeftPx: 11 }) !== base, "withAxis creates new axis");
    }

    // ── 7. 跨层一致：clip 体模型（内容坐标）vs 波形场景（视口坐标） ──
    // 这是本次重构的核心不变量：两条独立的换算路径必须给出同一个像素。
    {
        const random = makeRandom(11235);
        for (let i = 0; i < 200; i += 1) {
            const pxPerSec = 0.5 + random() * 800;
            const scrollLeftPx = random() * 4000;
            const axis = createTimelineAxis({
                pxPerSec,
                scrollLeftPx,
                viewportWidthPx: 1200,
            });
            // 让 clip 完全落在视口中部：否则会被裁剪，几何不可比。
            const viewportSpanSec = 1200 / pxPerSec;
            const startSec = scrollLeftPx / pxPerSec + viewportSpanSec * 0.25;
            const lengthSec = Math.max(1e-3, Math.min(viewportSpanSec * 0.5, 0.5 + random() * 20));

            const clipModel = buildSparseClipRenderModel({
                visibleTracks: [{ id: "t1" }],
                startTrackIndex: 0,
                visibleTrackClipsById: {
                    t1: [
                        {
                            id: "c1",
                            trackId: "t1",
                            name: "Clip",
                            startSec,
                            lengthSec,
                            gain: 1,
                            playbackRate: 1,
                            muted: false,
                            fadeInSec: 0,
                            fadeOutSec: 0,
                            fadeInShape: 0,
                            fadeInDir: 0,
                            fadeOutShape: 0,
                            fadeOutDir: 0,
                        },
                    ],
                },
                axis,
                rowHeight: 80,
                selectedClipId: null,
                multiSelectedClipIds: [],
                renamingClipId: null,
            });

            const scene = buildWaveformScene({
                axis,
                widthPx: 1200,
                rows: [
                    {
                        topPx: 0,
                        waveformTopPx: 0,
                        waveformHeightPx: 40,
                        clips: [
                            {
                                id: "c1",
                                sourcePath: "/tmp/a.wav",
                                startSec,
                                lengthSec,
                                sourceStartSec: 0,
                                sourceEndSec: lengthSec,
                                durationSec: lengthSec,
                                sourceSampleRate: 48000,
                                playbackRate: 1,
                                reversed: false,
                                loopEnabled: false,
                                gain: 1,
                                muted: false,
                                fadeInSec: 0,
                                fadeOutSec: 0,
                                fadeInShape: 0,
                                fadeInDir: 0,
                                fadeOutShape: 0,
                                fadeOutDir: 0,
                            },
                        ],
                    },
                ],
            });

            const drawn = clipModel.drawClips[0];
            const segment = scene.segments[0];
            if (!drawn || !segment) {
                throw new Error(`cross-layer: missing geometry at i=${i}`);
            }

            // 内容坐标与视口坐标必须只差 scrollLeftPx。未被左缘裁剪时，
            // 波形段还原回内容坐标应与 clip 体的 leftPx 重合（浮点级容差）。
            const notLeftClipped = segment.screenRect.x > 0;
            if (notLeftClipped) {
                assertClose(
                    segment.screenRect.x + axis.scrollLeftPx,
                    drawn.leftPx,
                    1e-6,
                    `cross-layer x at i=${i}`,
                );
            }

            // 未被画布边缘裁剪时，clip 体宽度与波形段宽度必须同源
            const fullyVisible =
                notLeftClipped && segment.screenRect.x + segment.screenRect.width < 1200;
            if (fullyVisible) {
                assertClose(
                    segment.screenRect.width,
                    drawn.widthPx,
                    1e-6,
                    `cross-layer width at i=${i}`,
                );
            }
        }
    }
});
