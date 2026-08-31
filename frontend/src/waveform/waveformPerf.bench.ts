/**
 * 波形渲染性能基准。
 *
 * 【主要内容】对「scene 构建 → geometry 构建 → 线段展开」三个阶段分别计时，
 * 并给出一帧完整绘制的合计耗时。
 *
 * 【作用】定位「10 轨 / 400 clip / 小缩放」下卡顿的真正瓶颈。这三个阶段都是
 * 纯函数（不依赖 DOM / React / WebGL），因此可以脱离浏览器在 Node 下计时，
 * 既能指导优化优先级，又能进 CI 防回归。
 *
 * 【与其他模块的关系】
 * - 输入：`runtime/timelinePerfScenario.ts` 提供 clip 分布与坐标投影，
 *   `perfFixtures.ts` 提供合成峰值与 `WaveformSceneRow[]`。
 * - 被测：`sceneBuilder.buildWaveformScene` / `geometry.buildWaveformGeometry` /
 *   `surfaceRenderer.expandLineSegmentsToQuads`。
 *
 * 【运行】`npm run bench`（等价于 `vitest bench`），或
 * `npx vitest bench waveformPerf` 只跑本文件。
 *
 * 【为什么分多个缩放档】UI 的缩放下限由 `resolveTimelineMinPxPerSec` 决定：
 * 内容装不下视口时固定在 4 px/s。因此"缩到能看全 400 个 clip"在 UI 中未必
 * 可达。这里同时测量 `"fitContent"`（理论最坏）与 `"zoomFloor"`（UI 可达
 * 最坏），避免基于错误前提做优化决策。
 */

import { bench, describe } from "vitest";

import {
    viewportEndSec,
    viewportStartSec,
} from "../components/layout/timeline/runtime/timelineAxis.js";
import { buildTimelinePerfScenario } from "../components/layout/timeline/runtime/timelinePerfScenario.js";
import { buildTimelineRenderModel } from "../components/layout/timeline/runtime/timelineRenderModel.js";
import { buildWaveformGeometry } from "./geometry.js";
import { buildSceneRows, createSyntheticPeakSource } from "./perfFixtures.js";
import { buildWaveformScene } from "./sceneBuilder.js";
import { expandLineSegmentsToQuads } from "./surfaceRenderer.js";

/** 波形描边颜色；只影响顶点颜色分量，不参与任何分支，对耗时无影响。 */
const WAVEFORM_COLOR = "#8fa3bf";

/** 轨道行高（CSS 像素），与 `DEFAULT_ROW_HEIGHT` 一致。 */
const ROW_HEIGHT = 96;

/** 单个 clip 的时长（秒）：本基准针对"1 分钟音频"的场景。 */
const CLIP_LENGTH_SEC = 60;

interface ScenarioSpec {
    label: string;
    trackCount: number;
    clipsPerTrack: number;
    pxPerSec: number | "fitContent" | "zoomFloor";
}

/**
 * 注册一个场景下的全部阶段基准，并打印该场景的规模诊断。
 *
 * 流程：
 * 1. 生成 clip 分布与投影，组装 `WaveformSceneRow[]`；
 * 2. 先跑一次完整链路做预热，同时采集分段数 / 顶点数 / 估算的峰值扫描量；
 * 3. 为四个粒度分别注册基准（scene / geometry / quads / full frame）。
 *
 * 诊断数据的意义：确认「像素列数 ≈ 视口宽 × 轨道数」这一守恒关系是否成立，
 * 以及每列实际扫描多少峰值——这两项决定了瓶颈在 per-clip 固定开销还是
 * 在峰值聚合。
 */
function registerScenario(spec: ScenarioSpec): void {
    const scenario = buildTimelinePerfScenario({
        trackCount: spec.trackCount,
        clipsPerTrack: spec.clipsPerTrack,
        clipLengthSec: CLIP_LENGTH_SEC,
        gapSec: 0,
        pxPerSec: spec.pxPerSec,
    });
    const axis = scenario.axis;
    const widthPx = axis.viewportWidthPx;
    const rows = buildSceneRows({
        tracks: scenario.tracks,
        clips: scenario.clips,
        rowHeight: ROW_HEIGHT,
    });
    const peaks = createSyntheticPeakSource({ mediaDurationSec: CLIP_LENGTH_SEC });

    const buildScene = () => buildWaveformScene({ axis, widthPx, viewportTopPx: 0, rows });
    const buildGeometry = (scene: ReturnType<typeof buildScene>) =>
        buildWaveformGeometry({ scene, color: WAVEFORM_COLOR, getPeaks: peaks.getPeaks });

    // 预热 + 采集规模
    const warmScene = buildScene();
    const warmGeometry = buildGeometry(warmScene);

    const peakDensity = 44100 / 4096; // L2 的峰值密度（peaks/s）
    let estimatedPeakScans = 0;
    for (const segment of warmScene.segments) {
        const sourceDurationSec = segment.sourceEndSec - segment.sourceStartSec;
        if (!(sourceDurationSec > 1e-9) || segment.screenRect.width <= 0) continue;
        const secPerPixel = sourceDurationSec / segment.screenRect.width;
        const peaksPerColumn = Math.max(1, secPerPixel * peakDensity);
        estimatedPeakScans += segment.screenRect.width * peaksPerColumn;
    }

    console.log(
        [
            `[${spec.label}]`,
            `pxPerSec=${axis.pxPerSec.toFixed(3)}`,
            `content=${scenario.contentEndSec.toFixed(0)}s`,
            `clips=${scenario.clips.length}`,
            `segments=${warmScene.segments.length}`,
            `markers=${warmScene.markers.length}`,
            `pixelColumns=${warmGeometry.lineCount}`,
            `vertices=${warmGeometry.vertices.length / 6}`,
            `vertexBytes=${(warmGeometry.vertices.length * 4 / 1024).toFixed(0)}KB`,
            `estPeakScans=${Math.round(estimatedPeakScans).toLocaleString("en-US")}`,
            `getPeaksCalls=${peaks.callCount()}`,
        ].join("  "),
    );

    describe(`waveform · ${spec.label}`, () => {
        /**
         * React 侧的每帧固定开销。
         *
         * `TimelinePanel` 把 `viewportStartSec` / `viewportEndSec` 作为
         * `buildTimelineRenderModel` 的依赖，而二者由 React state 的
         * `scrollLeft` / `pxPerSec` 派生——因此**每次滚动/缩放的 React 重渲染
         * 都会重新执行本函数**。它每次都要遍历全部 clip 重建 trackId → clip
         * 的 Map，再对每条可见轨道做一次 filter + map。
         *
         * 注意：这只是 React 侧可观测到的 O(N) 部分，不含 React 自身的
         * 协调（reconciliation）、TimelinePanel 其余 hook 与浏览器
         * layout/paint。真实成本只会更高。
         */
        bench("0. renderModel (React 侧)", () => {
            buildTimelineRenderModel({
                tracks: scenario.tracks,
                clips: scenario.clips,
                viewportStartSec: viewportStartSec(axis),
                viewportEndSec: viewportEndSec(axis),
                rowHeight: ROW_HEIGHT,
                scrollTopPx: 0,
                viewportHeightPx: 960,
            });
        });

        bench("1. scene", () => {
            buildScene();
        });

        bench("2. geometry", () => {
            buildGeometry(warmScene);
        });

        bench("3. quads", () => {
            expandLineSegmentsToQuads(warmGeometry.vertices);
        });

        bench("4. full frame", () => {
            const scene = buildScene();
            const geometry = buildGeometry(scene);
            expandLineSegmentsToQuads(geometry.vertices);
        });
    });
}

// ── 场景矩阵 ──────────────────────────────────────────────────────────────
// 1) 目标场景（理论最坏）：内容刚好压进视口，400 个 clip 全部参与绘制。
registerScenario({
    label: "400 clip / fitContent",
    trackCount: 10,
    clipsPerTrack: 40,
    pxPerSec: "fitContent",
});

// 2) 目标场景（UI 可达最坏）：UI 缩放下限，视口被 clip 填满但只看到一部分。
registerScenario({
    label: "400 clip / zoomFloor",
    trackCount: 10,
    clipsPerTrack: 40,
    pxPerSec: "zoomFloor",
});

// 3) 中等缩放：clip 远宽于视口，用于对比"clip 数量少但每列扫描量大"的情形。
registerScenario({
    label: "400 clip / pxPerSec=40",
    trackCount: 10,
    clipsPerTrack: 40,
    pxPerSec: 40,
});

// 4) 小规模对照：确认耗时是否随 clip 数线性增长。
registerScenario({
    label: "40 clip / fitContent",
    trackCount: 10,
    clipsPerTrack: 4,
    pxPerSec: "fitContent",
});
