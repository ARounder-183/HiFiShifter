/**
 * 时间线性能场景生成器。
 *
 * 【主要内容】按给定规模合成「轨道 + clip」测试数据，并派生一个坐标投影
 * （`TimelineAxis`）供渲染链路直接消费。
 *
 * 【作用】为离线基准与性能回归提供**可复现、可参数化**的输入。此前 clip 的
 * 时长与间距被写死（`lengthSec: 1.2`、`startSec = clipIndex * 1.5`），无法表达
 * 「10 轨 / 400 clip / 1 分钟音频」这类真实规模，因此这里把时长、间距、视口
 * 宽度与缩放模式全部开放为参数。
 *
 * 【投影模式的必要性】UI 的缩放下限由 `resolveTimelineMinPxPerSec` 决定：
 * 内容装不下视口时取 `base`（4 px/s），装得下才降到 0.5 px/s。这意味着
 * 「长时间轴缩到能看全」在 UI 中往往**不可达**。为了同时覆盖「UI 可达的
 * 最坏情况」与「理论最坏情况」，本文件提供三种模式：
 * - `"fitContent"`：把所有内容压进视口（可能低于 UI 下限，用于探底）；
 * - `"zoomFloor"`：UI 真实可缩到的最小值（经 `resolveTimelineMinPxPerSec`）；
 * - 数值：直接指定 pxPerSec。
 *
 * 【与其他模块的关系】
 * - 产出被 `frontend/src/waveform/waveformPerf.bench.ts` 消费，并在那里转换为
 *   `WaveformSceneRow[]`（反向依赖 `waveform/` 会形成循环，故此处只产出
 *   纯数据的 tracks / clips）。
 * - 坐标投影一律由 `timelineAxis.ts` 构造，本文件不做任何乘法换算。
 */

import { createTimelineAxis, type TimelineAxis } from "./timelineAxis.js";
import { resolveTimelineMinPxPerSec } from "./timelineZoomBounds.js";

/**
 * 缩放下限的基准值。
 *
 * 与 `constants.ts` 的 `MIN_PX_PER_SEC` 同值：本文件位于 `runtime/`，
 * 若从 `../constants` 导入会把整个时间线常量表（含大量与布局无关的定义）
 * 拖进基准链路，这里按值复制并注明来源。
 */
const BASE_MIN_PX_PER_SEC = 4;

/** 历史默认行为：clip 时长 1.2s、步距 1.5s（即间隙 0.3s）。 */
const DEFAULT_CLIP_LENGTH_SEC = 1.2;
const DEFAULT_GAP_SEC = 0.3;
const DEFAULT_VIEWPORT_WIDTH_PX = 1500;

export interface TimelinePerfScenarioArgs {
    trackCount: number;
    clipsPerTrack: number;
    /** 单个 clip 的时长（秒）。默认 1.2（保留历史行为）。 */
    clipLengthSec?: number;
    /** 相邻 clip 之间的间隙（秒）。默认 0.3（与默认时长合成 1.5 的步距）。 */
    gapSec?: number;
    /** 视口宽度（CSS 像素）。默认 1500。 */
    viewportWidthPx?: number;
    /**
     * 缩放模式：
     * - `"fitContent"`：内容刚好铺满视口（可能低于 UI 缩放下限）；
     * - `"zoomFloor"`：UI 实际可缩到的最小值；
     * - 数值：直接指定 pxPerSec。
     * 默认 `"fitContent"`。
     */
    pxPerSec?: number | "fitContent" | "zoomFloor";
}

export interface TimelinePerfScenario {
    tracks: Array<{
        id: string;
        name: string;
    }>;
    clips: Array<{
        id: string;
        trackId: string;
        startSec: number;
        lengthSec: number;
    }>;
    /** 按 `pxPerSec` 模式派生出的投影，`scrollLeftPx` 恒为 0（从工程开头看起）。 */
    axis: TimelineAxis;
    /** 内容的总时长（秒），即最后一个 clip 的结束时间。 */
    contentEndSec: number;
}

/**
 * 合成时间线性能场景。
 *
 * 流程：
 * 1. 按 `trackCount` / `clipsPerTrack` 生成轨道与 clip，步距为
 *    `clipLengthSec + gapSec`，每条轨道再叠加 `(trackIndex % 4) * 0.1` 的错位
 *    （保留历史行为，使各轨道不是完全对齐）；
 * 2. 由 clip 分布求出内容总时长；
 * 3. 按 `pxPerSec` 模式反算缩放，构造 axis。
 *
 * @returns 轨道、clip、坐标投影与内容总时长。
 */
export function buildTimelinePerfScenario(
    args: TimelinePerfScenarioArgs,
): TimelinePerfScenario {
    const clipLengthSec = args.clipLengthSec ?? DEFAULT_CLIP_LENGTH_SEC;
    const gapSec = args.gapSec ?? DEFAULT_GAP_SEC;
    const viewportWidthPx = args.viewportWidthPx ?? DEFAULT_VIEWPORT_WIDTH_PX;
    const strideSec = clipLengthSec + gapSec;

    const tracks = Array.from({ length: args.trackCount }, (_, index) => ({
        id: `track-${index}`,
        name: `Track ${index + 1}`,
    }));

    const clips = tracks.flatMap((track, trackIndex) =>
        Array.from({ length: args.clipsPerTrack }, (_, clipIndex) => ({
            id: `${track.id}-clip-${clipIndex}`,
            trackId: track.id,
            startSec: clipIndex * strideSec + (trackIndex % 4) * 0.1,
            lengthSec: clipLengthSec,
        })),
    );

    let contentEndSec = 0;
    for (const clip of clips) {
        contentEndSec = Math.max(contentEndSec, clip.startSec + clip.lengthSec);
    }

    return {
        tracks,
        clips,
        axis: createTimelineAxis({
            pxPerSec: resolvePxPerSec(args.pxPerSec, contentEndSec, viewportWidthPx),
            scrollLeftPx: 0,
            viewportWidthPx,
            scrollTopPx: 0,
            dpr: 1,
        }),
        contentEndSec,
    };
}

/**
 * 按模式求出 pxPerSec。
 *
 * 特殊说明：`"fitContent"` 的结果可能远低于 UI 允许的缩放下限
 * （例如 2400 秒内容压进 1500 px 得到 0.625 px/s，而 UI 下限是 4）。
 * 这是**故意**的——它给出理论上界，用于判断最坏情况是否值得优化；
 * 真实可达的最坏情况请用 `"zoomFloor"`。
 */
function resolvePxPerSec(
    mode: number | "fitContent" | "zoomFloor" | undefined,
    contentEndSec: number,
    viewportWidthPx: number,
): number {
    if (typeof mode === "number") return mode;
    if (mode === "zoomFloor") {
        return resolveTimelineMinPxPerSec({
            baseMinPxPerSec: BASE_MIN_PX_PER_SEC,
            projectSec: contentEndSec,
            viewportWidthPx,
        });
    }
    // 默认 "fitContent"
    return viewportWidthPx / Math.max(1e-6, contentEndSec);
}
