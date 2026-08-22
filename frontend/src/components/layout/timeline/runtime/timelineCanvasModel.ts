import { CLIP_BODY_PADDING_Y, CLIP_HEADER_HEIGHT } from "../constants.js";

type SparseRenderClip = {
    id: string;
    trackId: string;
    name: string;
    startSec: number;
    lengthSec: number;
    gain: number;
    playbackRate: number;
    muted: boolean;
    midiNoteCount?: number;
    groupId?: string;
    fadeInSec: number;
    fadeOutSec: number;
    fadeInCurve: "linear" | "sine" | "exponential" | "logarithmic" | "scurve";
    fadeOutCurve: "linear" | "sine" | "exponential" | "logarithmic" | "scurve";
    /** 自动交叉淡化长度（可选；缺省 0），用于“有效 fade”显示。 */
    autoFadeInSec?: number;
    autoFadeOutSec?: number;
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
    fadeInCurve: "linear" | "sine" | "exponential" | "logarithmic" | "scurve";
    fadeOutCurve: "linear" | "sine" | "exponential" | "logarithmic" | "scurve";
    selected: boolean;
    muted: boolean;
    gain: number;
    playbackRate: number;
    groupId?: string;
    isMidiClip: boolean;
    trackColor?: string;
    isRenaming: boolean;
};

export function buildSparseClipRenderModel(args: {
    visibleTracks: Array<{ id: string; color?: string }>;
    visibleTrackClipsById: Record<string, SparseRenderClip[]>;
    pxPerSec: number;
    rowHeight: number;
    scrollLeft: number;
    selectedClipId: string | null;
    multiSelectedClipIds: string[];
    renamingClipId: string | null;
    hoveredClipId?: string | null;
    disabledGroupIds?: string[];
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
        for (let i = 0; i < trackClips.length; i += 1) {
            const a = trackClips[i];
            for (let j = i + 1; j < trackClips.length; j += 1) {
                const b = trackClips[j];
                const aStart = a.startSec;
                const aEnd = aStart + a.lengthSec;
                const bStart = b.startSec;
                const bEnd = bStart + b.lengthSec;
                if (Math.min(aEnd, bEnd) > Math.max(aStart, bStart) + 1e-9) {
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
            name: clip.name,
            leftPx: clip.startSec * args.pxPerSec - args.scrollLeft,
            topPx: visibleIndex * args.rowHeight,
            widthPx: Math.max(1, clip.lengthSec * args.pxPerSec),
            heightPx: Math.max(1, args.rowHeight - CLIP_BODY_PADDING_Y),
            headerHeightPx: CLIP_HEADER_HEIGHT,
            fadeInPx: Math.max(
                0,
                ((clip.autoFadeInSec ?? 0) > 0 ? clip.autoFadeInSec! : clip.fadeInSec) *
                    args.pxPerSec,
            ),
            fadeOutPx: Math.max(
                0,
                ((clip.autoFadeOutSec ?? 0) > 0 ? clip.autoFadeOutSec! : clip.fadeOutSec) *
                    args.pxPerSec,
            ),
            fadeInCurve: clip.fadeInCurve,
            fadeOutCurve: clip.fadeOutCurve,
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
