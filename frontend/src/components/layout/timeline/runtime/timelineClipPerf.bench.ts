/**
 * clip 体渲染链路的离线性能基准。
 *
 * 【主要内容】对「可见集合分组 → 稀疏渲染模型 → 逐 clip 视觉样式」三阶段
 * 分别计时，并给出一帧的完整合计。
 *
 * 【作用】波形那侧的链路已有 `waveform/waveformPerf.bench.ts` 覆盖，但
 * **clip 体画布一直没有可复现的量化手段**，而它每帧要对每个可见 clip 依次
 * 执行 roundRect + clip + 多次 fillRect/stroke + 文本测量。本基准测的是其中
 * 能在 Node 下跑的**纯计算部分**，用来判断优化该往哪儿使劲。
 *
 * 【覆盖范围与局限】
 * - 已覆盖：`buildTimelineRenderModel`（全 clip 分组 + 窗口裁剪）、
 *   `buildSparseClipRenderModel`（几何投影 + 重叠检测）、
 *   `buildTimelineClipVisualStyle`（样式对象与标签字符串）。
 * - **未覆盖**：`drawTimelineCanvas` 里真正的 Canvas2D 绘制调用（需要 DOM）。
 *   因此本基准是下界——真实帧耗时只会更高。
 * - `measureText` 在 Node 下退化为固定字宽估算（见 timelineCanvasStyle 的
 *   getMeasureCtx），所以文字测量的真实成本也未被计入，同样是下界。
 *
 * 【与其他模块的关系】
 * - 被测：`timelineRenderModel.ts` / `timelineCanvasModel.ts` /
 *   `timelineCanvasStyle.ts`。
 * - 参照：`waveform/waveformPerf.bench.ts`（同样的场景与写法）。
 *
 * 【运行】`npm run bench`，或 `npx vitest bench timelineClipPerf`。
 */

import { bench, describe } from "vitest";

import { createTimelineAxis } from "./timelineAxis.js";
import { buildSparseClipRenderModel } from "./timelineCanvasModel.js";
import { buildTimelineRenderModel } from "./timelineRenderModel.js";
import { buildTimelineClipVisualStyle } from "./timelineCanvasStyle.js";

/** 轨道行高（CSS 像素），与 `DEFAULT_ROW_HEIGHT` 一致。 */
const ROW_HEIGHT = 96;

/** 单个 clip 时长（秒）：与波形基准同为「1 分钟音频」场景。 */
const CLIP_LENGTH_SEC = 60;

/** 字体族，与 `timelineCanvasStyle.resolveFontFamily()` 的缺省值一致。 */
const FONT_FAMILY = "sans-serif";

interface ClipFixture {
    id: string;
    trackId: string;
    name: string;
    startSec: number;
    lengthSec: number;
    gain: number;
    playbackRate: number;
    muted: boolean;
    fadeInSec: number;
    fadeOutSec: number;
    fadeInShape: number;
    fadeOutShape: number;
    fadeInDir: number;
    fadeOutDir: number;
}

/**
 * 构造一个规模场景。
 *
 * 与波形基准保持同一布局：每轨 `clipsPerTrack` 个 clip 首尾相接，
 * 因此单轨内重叠对数为 0（DAW 最常见的排布）。
 */
function buildFixture(
    trackCount: number,
    clipsPerTrack: number,
): {
    tracks: Array<{ id: string; color: string }>;
    clips: ClipFixture[];
} {
    const tracks = Array.from({ length: trackCount }, (_unused, index) => ({
        id: `track-${index}`,
        color: "#4f8ef7",
    }));
    const clips: ClipFixture[] = [];
    for (let trackIndex = 0; trackIndex < trackCount; trackIndex += 1) {
        for (let clipIndex = 0; clipIndex < clipsPerTrack; clipIndex += 1) {
            clips.push({
                id: `clip-${trackIndex}-${clipIndex}`,
                trackId: `track-${trackIndex}`,
                name: `Perf clip ${trackIndex}-${clipIndex}`,
                startSec: clipIndex * CLIP_LENGTH_SEC,
                lengthSec: CLIP_LENGTH_SEC,
                gain: 1,
                playbackRate: 1,
                muted: false,
                fadeInSec: 0,
                fadeOutSec: 0,
                fadeInShape: 0,
                fadeOutShape: 0,
                fadeInDir: 0,
                fadeOutDir: 0,
            });
        }
    }
    return { tracks, clips };
}

/** 每轨可见 clip 数按「全部可见」取值，用于喂 `buildSparseClipRenderModel`。 */
function groupByTrack(
    tracks: Array<{ id: string }>,
    clips: ClipFixture[],
): Record<string, ClipFixture[]> {
    const byTrack: Record<string, ClipFixture[]> = {};
    for (const track of tracks) byTrack[track.id] = [];
    for (const clip of clips) byTrack[clip.trackId]?.push(clip);
    return byTrack;
}

function registerScenario(spec: {
    label: string;
    trackCount: number;
    clipsPerTrack: number;
    pxPerSec: number;
}): void {
    const { tracks, clips } = buildFixture(spec.trackCount, spec.clipsPerTrack);
    const axis = createTimelineAxis({
        pxPerSec: spec.pxPerSec,
        viewportWidthPx: 1500,
        scrollLeftPx: 0,
        scrollTopPx: 0,
    });
    const contentEndSec = spec.clipsPerTrack * CLIP_LENGTH_SEC;
    const byTrack = groupByTrack(tracks, clips);

    const buildModel = () =>
        buildSparseClipRenderModel({
            visibleTracks: tracks,
            startTrackIndex: 0,
            visibleTrackClipsById: byTrack,
            axis,
            rowHeight: ROW_HEIGHT,
            selectedClipId: null,
            multiSelectedClipIds: [],
            renamingClipId: null,
        });

    const warmModel = buildModel();

    // 逐 clip 视觉样式：取三个代表性宽度档（窄 / 中 / 宽），覆盖
    // `timelineClipHeaderVisibility` 的不同分支成本。
    const styleWidths = [16, 80, 400];
    const buildStyles = () => {
        for (const widthPx of styleWidths) {
            for (const clip of warmModel.drawClips) {
                buildTimelineClipVisualStyle({
                    widthPx,
                    trackColor: clip.trackColor,
                    selected: clip.selected,
                    muted: clip.muted,
                    gain: clip.gain,
                    playbackRate: clip.playbackRate,
                    name: clip.name,
                    fontFamily: FONT_FAMILY,
                    isPitchAdjustment: clip.isMidiClip,
                    groupId: clip.groupId,
                    isGroupActive: false,
                    isGroupDisabled: false,
                    darkMode: true,
                });
            }
        }
    };

    console.log(
        [
            `[${spec.label}]`,
            `pxPerSec=${axis.pxPerSec}`,
            `content=${contentEndSec}s`,
            `clips=${clips.length}`,
            `drawClips=${warmModel.drawClips.length}`,
            `overlayIds=${Object.values(warmModel.overlayClipIdsByTrackId).reduce(
                (sum, ids) => sum + ids.length,
                0,
            )}`,
            `clipWidthPx=${(CLIP_LENGTH_SEC * axis.pxPerSec).toFixed(1)}`,
        ].join("  "),
    );

    describe(`clip body · ${spec.label}`, () => {
        bench("0. renderModel (全 clip 分组 + 窗口裁剪)", () => {
            buildTimelineRenderModel({
                tracks,
                clips,
                viewportStartSec: 0,
                viewportEndSec: contentEndSec,
                rowHeight: ROW_HEIGHT,
                scrollTopPx: 0,
                viewportHeightPx: 960,
            });
        });

        bench("1. sparseClipRenderModel (几何投影 + 重叠检测)", () => {
            buildModel();
        });

        bench("2. clipVisualStyle ×3 宽度档 (全部 clip)", () => {
            buildStyles();
        });

        bench("3. full frame (纯计算部分)", () => {
            buildTimelineRenderModel({
                tracks,
                clips,
                viewportStartSec: 0,
                viewportEndSec: contentEndSec,
                rowHeight: ROW_HEIGHT,
                scrollTopPx: 0,
                viewportHeightPx: 960,
            });
            const model = buildModel();
            for (const widthPx of styleWidths) {
                for (const clip of model.drawClips) {
                    buildTimelineClipVisualStyle({
                        widthPx,
                        trackColor: clip.trackColor,
                        selected: clip.selected,
                        muted: clip.muted,
                        gain: clip.gain,
                        playbackRate: clip.playbackRate,
                        name: clip.name,
                        fontFamily: FONT_FAMILY,
                        isPitchAdjustment: clip.isMidiClip,
                        groupId: clip.groupId,
                        isGroupActive: false,
                        isGroupDisabled: false,
                        darkMode: true,
                    });
                }
            }
        });
    });
}

// ── 场景矩阵 ──────────────────────────────────────────────────────────────
// 与波形基准对齐：fitContent（全部内容压进 1500px 视口）、中等缩放、小规模对照。
registerScenario({
    label: "400 clip / fitContent",
    trackCount: 10,
    clipsPerTrack: 40,
    pxPerSec: 1500 / (40 * CLIP_LENGTH_SEC),
});

registerScenario({
    label: "400 clip / pxPerSec=4",
    trackCount: 10,
    clipsPerTrack: 40,
    pxPerSec: 4,
});

registerScenario({
    label: "400 clip / pxPerSec=40",
    trackCount: 10,
    clipsPerTrack: 40,
    pxPerSec: 40,
});

registerScenario({
    label: "40 clip / fitContent",
    trackCount: 10,
    clipsPerTrack: 4,
    pxPerSec: 1500 / (4 * CLIP_LENGTH_SEC),
});
