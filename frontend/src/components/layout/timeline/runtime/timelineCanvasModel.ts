/**
 * Clip 体画布的稀疏渲染模型。
 *
 * 【主要内容】把窗口内可见的 clip 元数据转换为画布绘制所需的像素几何
 * （`TimelineCanvasClipModel[]`），并判定哪些 clip 需要额外的 DOM 覆盖层。
 *
 * 【作用】clip **本体**由 canvas 绘制（数量多、重绘频繁），只有选中/悬停/
 * 重命名/重叠的 clip 才额外渲染 DOM 覆盖层以获得交互手柄。本文件负责这条
 * 分工的判定与几何产出。
 *
 * 【与其他模块的关系】
 * - 上游：`TimelinePanel` 在渲染期调用 `buildSparseClipRenderModel()`，结果
 *   交给 `TimelineCanvasViewport` 绘制、交给 `TrackLane` 决定 DOM 覆盖层。
 * - 横向：所有时间↔像素换算一律走 `timelineAxis.ts`，本文件不得自行
 *   执行 `sec * pxPerSec`，否则会与波形、网格产生错位。
 * - 下游：`runtime/timelineCanvasRenderer.ts` 消费 `drawClips`。
 */

import { CLIP_BODY_PADDING_Y, CLIP_HEADER_HEIGHT } from "../constants.js";
import { clipDisplayName } from "../../../../features/session/sessionTypes";
import {
    durationToWidthPx,
    secToContentPx,
    secToSpanPx,
    type TimelineAxis,
} from "./timelineAxis.js";

type SparseRenderClip = {
    id: string;
    trackId: string;
    name: string;
    startSec: number;
    lengthSec: number;
    gain: number;
    playbackRate: number;
    muted: boolean;
    takes?: Array<{ id: string; name: string }>;
    activeTakeId?: string;
    midiNoteCount?: number;
    groupId?: string;
    fadeInSec: number;
    fadeOutSec: number;
    fadeInShape: number;
    fadeOutShape: number;
    fadeInDir: number;
    fadeOutDir: number;
    /** 自动交叉淡化长度（可选；缺省 0），用于“有效 fade”显示。 */
    autoFadeInSec?: number;
    autoFadeOutSec?: number;
    /** 吸附偏移（秒，相对 Clip 起点；缺省 0）—— 左下角 ◣ 标记。 */
    snapOffsetSec?: number;
};
export type TimelineCanvasClipModel = {
    id: string;
    trackId: string;
    name: string;
    leftPx: number;
    topPx: number;
    widthPx: number;
    heightPx: number;
    headerHeightPx: number;
    fadeInPx: number;
    fadeOutPx: number;
    fadeInShape: number;
    fadeOutShape: number;
    fadeInDir: number;
    fadeOutDir: number;
    selected: boolean;
    muted: boolean;
    gain: number;
    playbackRate: number;
    groupId?: string;
    isMidiClip: boolean;
    trackColor?: string;
    isRenaming: boolean;
    /** 吸附偏移（已换算为像素，相对 Clip 左缘）—— 左下角 ◣ 标记。 */
    snapOffsetPx: number;
};

/**
 * 构建 clip 体画布的稀疏渲染模型。
 *
 * 流程：
 * 1. 判定需要 DOM 覆盖层的 clip（重命名中 / 悬停 / 选中 / 同组 / 同轨重叠）；
 * 2. 把每个可见 clip 的秒级字段经 `TimelineAxis` 投影为像素几何。
 *
 * 特殊说明：
 * - 输出的是**内容绝对坐标**（原点为工程 0 秒），竖直滚动由画布统一
 *   `translate` 平移，因此这里不减 scrollTopPx。
 * - 宽度与淡入淡出必须分别用 `durationToWidthPx` / `secToSpanPx`：
 *   前者带最小宽度下限（保证可命中），后者可为 0（0 淡入不应出现角标）。
 *
 * @param args.axis 统一坐标投影（唯一的时间↔像素来源）。
 * @param args.visibleTracks 窗口内可见轨道，决定绘制顺序与轨道配色。
 * @param args.startTrackIndex 窗口首行的绝对轨道索引，用于算竖直内容坐标。
 * @param args.visibleTrackClipsById 各轨道的可见 clip。
 * @param args.rowHeight 轨道行高（CSS 像素）。
 * @param args.selectedClipId / multiSelectedClipIds / renamingClipId /
 *        hoveredClipId / disabledGroupIds 决定 DOM 覆盖层的归属。
 * @returns `drawClips` 供 canvas 绘制；`overlayClipIdsByTrackId` 供
 *          `TrackLane` 决定哪些 clip 额外挂 DOM 交互层。
 */
export function buildSparseClipRenderModel(args: {
    visibleTracks: Array<{ id: string; color?: string }>;
    /** 窗口首行的绝对轨道索引：clip body 画布使用内容绝对坐标绘制，
     * 竖直滚动时由 scrollTopPx 统一平移（与 DOM 内容层同帧提交）。 */
    startTrackIndex: number;
    visibleTrackClipsById: Record<string, SparseRenderClip[]>;
    axis: TimelineAxis;
    rowHeight: number;
    selectedClipId: string | null;
    multiSelectedClipIds: string[];
    renamingClipId: string | null;
    hoveredClipId?: string | null;
    disabledGroupIds?: string[];
    /**
     * 每个 clip 在"前导重叠区"（被同轨前一个 clip 压住的部分）的秒数。
     * 渲染端据此在上 clip 重叠区画半透色块，让下 clip 的色块与波形都能看见——
     * 否则两层不透明色块会"叠加"成脏色。
     */
    leadingOverlapSecByClipId?: Record<string, number>;
}): {
    drawClips: TimelineCanvasClipModel[];
    overlayClipIdsByTrackId: Record<string, string[]>;
} {
    const overlayClipIds = new Set<string>();
    if (args.renamingClipId) {
        overlayClipIds.add(args.renamingClipId);
    }
    if (args.hoveredClipId) {
        overlayClipIds.add(args.hoveredClipId);
    }
    if (args.multiSelectedClipIds.length > 0) {
        for (const clipId of args.multiSelectedClipIds) {
            overlayClipIds.add(clipId);
        }
    } else if (args.selectedClipId) {
        overlayClipIds.add(args.selectedClipId);
    }

    // Expand overlay to include all clips that share a group with any overlay clip,
    // unless the group is disabled.
    {
        const activeGroupIds = new Set<string>();
        for (const trackClips of Object.values(args.visibleTrackClipsById)) {
            for (const clip of trackClips) {
                if (
                    clip.groupId != null &&
                    overlayClipIds.has(clip.id) &&
                    !args.disabledGroupIds?.includes(clip.groupId)
                ) {
                    activeGroupIds.add(clip.groupId);
                }
            }
        }
        if (activeGroupIds.size > 0) {
            for (const trackClips of Object.values(args.visibleTrackClipsById)) {
                for (const clip of trackClips) {
                    if (clip.groupId != null && activeGroupIds.has(clip.groupId)) {
                        overlayClipIds.add(clip.id);
                    }
                }
            }
        }
    }

    const multiSelectedSet =
        args.multiSelectedClipIds.length > 0 ? new Set(args.multiSelectedClipIds) : null;

    // 重叠区可编辑性：把“同轨道存在重叠”的两个 clip 都加入 DOM overlay。
    // 否则重叠时只有“后绘制/选中的那个”有 DOM 边缘/淡入淡出手柄，较早 clip 的
    // 边缘（延长截短/拉伸/淡入淡出）完全不可达。配合 ClipItem 去掉会建立独立层叠
    // 上下文的 transform，交叉处两个 clip 的手柄都位于其它 clip body 之上、可被编辑。
    for (const trackClips of Object.values(args.visibleTrackClipsById)) {
        if (trackClips.length < 2) continue;
        // 排序后线性扫描（原实现是 O(n²) 全对比较）。
        //
        // 判据不变：仍把每一对**存在重叠**的两个 clip 都加入 overlay。
        // 按 startSec 升序后，对固定的 a 而言，一旦某个 b 的起点越过了 a 的
        // 末端，它后面所有 clip 的起点只会更靠右，必然也不与 a 重叠 ——
        // 因此可以直接 `break`，把内层从「与 a 之后的全部 clip 比较」收敛为
        // 「与 a 真正相交的那一段」。
        //
        // 复杂度：排序 O(n log n) + 扫描 O(n + 重叠对数)。单轨 400 clip 时
        // 原实现固定 8 万次比较；首尾相接的常见排布下重叠对数为 0，扫描退化为
        // O(n)。必须复制后再排序——入参数组是上游 memo 的缓存值，就地排序会
        // 破坏缓存引用并让下游 memo 每帧失效。
        const sorted = trackClips.slice().sort((a, b) => a.startSec - b.startSec);
        for (let i = 0; i < sorted.length; i += 1) {
            const a = sorted[i];
            const aStart = a.startSec;
            const aEnd = aStart + a.lengthSec;
            for (let j = i + 1; j < sorted.length; j += 1) {
                const b = sorted[j];
                const bStart = b.startSec;
                // 升序 ⇒ 越过后不可能再重叠，提前跳出内层。
                if (bStart >= aEnd - 1e-9) break;
                const bEnd = bStart + b.lengthSec;
                if (Math.min(aEnd, bEnd) > bStart + 1e-9) {
                    overlayClipIds.add(a.id);
                    overlayClipIds.add(b.id);
                }
            }
        }
    }

    const drawClips = args.visibleTracks.flatMap((track, visibleIndex) =>
        (args.visibleTrackClipsById[track.id] ?? []).map((clip) => ({
            id: clip.id,
            trackId: clip.trackId,
            name: clipDisplayName(clip),
            leftPx: secToContentPx(args.axis, clip.startSec),
            topPx: (args.startTrackIndex + visibleIndex) * args.rowHeight,
            widthPx: durationToWidthPx(args.axis, clip.lengthSec),
            heightPx: Math.max(1, args.rowHeight - CLIP_BODY_PADDING_Y),
            headerHeightPx: CLIP_HEADER_HEIGHT,
            fadeInPx: secToSpanPx(
                args.axis,
                (clip.autoFadeInSec ?? 0) > 0 ? clip.autoFadeInSec! : clip.fadeInSec,
            ),
            fadeOutPx: secToSpanPx(
                args.axis,
                (clip.autoFadeOutSec ?? 0) > 0 ? clip.autoFadeOutSec! : clip.fadeOutSec,
            ),
            fadeInShape: Number.isFinite(clip.fadeInShape) ? clip.fadeInShape : 0,
            fadeOutShape: Number.isFinite(clip.fadeOutShape) ? clip.fadeOutShape : 0,
            fadeInDir: clip.fadeInDir ?? 0,
            fadeOutDir: clip.fadeOutDir ?? 0,
            selected:
                multiSelectedSet != null
                    ? multiSelectedSet.has(clip.id)
                    : args.selectedClipId === clip.id,
            muted: clip.muted,
            gain: clip.gain,
            playbackRate: clip.playbackRate,
            groupId: clip.groupId,
            isMidiClip: clip.midiNoteCount != null,
            trackColor: track.color,
            isRenaming: clip.id === args.renamingClipId,
            snapOffsetPx: secToSpanPx(args.axis, Number(clip.snapOffsetSec) || 0),
            leadingOverlapPx: secToSpanPx(
                args.axis,
                args.leadingOverlapSecByClipId?.[clip.id] ?? 0,
            ),
        })),
    );

    const overlayClipIdsByTrackId = Object.fromEntries(
        args.visibleTracks.map((track) => [
            track.id,
            (args.visibleTrackClipsById[track.id] ?? [])
                .filter((clip) => overlayClipIds.has(clip.id))
                .map((clip) => clip.id),
        ]),
    ) as Record<string, string[]>;

    return {
        drawClips,
        overlayClipIdsByTrackId,
    };
}
