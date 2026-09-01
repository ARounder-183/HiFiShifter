import { computeVisibleTrackWindow, sliceVisibleClipIds } from "./timelineWindowing.js";

/**
 * React `scrollLeft` 的量化步长（CSS 像素）。
 *
 * React 侧真正消费 scrollLeft 的只有**窗口化**（visibleTracks / drawClips）
 * 与刻度窗口——两者都带 ≥256px 的内容缓冲。因此把提交按此步长量化后：
 * - 滚动帧的 React 提交次数减少一个数量级（探针实测 react p50 16ms/帧，而
 *   commit 只有 2ms——重渲染才是滚动帧率的瓶颈）；
 * - 快速移动的视觉层（clip 体 / 波形 / 网格 / 标尺 / 播放头）全部走
 *   视口总线命令式更新，不经过 React，**不受这个滞后影响**。
 */
export const REACT_SCROLL_STEP_PX = 256;

export function buildTimelineRenderModel(args: {
    tracks: Array<{ id: string }>;
    clips: Array<{
        id: string;
        trackId: string;
        startSec: number;
        lengthSec: number;
    }>;
    viewportStartSec: number;
    viewportEndSec: number;
    pxPerSec: number;
    rowHeight: number;
    scrollTopPx: number;
    viewportHeightPx: number;
}): {
    startIndex: number;
    endIndex: number;
    visibleTrackIds: string[];
    visibleClipIdsByTrackId: Record<string, string[]>;
} {
    const visibleTrackWindow = computeVisibleTrackWindow({
        totalTracks: args.tracks.length,
        rowHeight: args.rowHeight,
        scrollTopPx: args.scrollTopPx,
        viewportHeightPx: args.viewportHeightPx,
        // overscan 提升到 4 行：竖直滚动时 React state（窗口化）可能比
        // 原生滚动晚一帧提交，足够的内容缓冲保证 sticky 画布（内容绝对
        // 坐标 + scrollTopPx 同帧平移）在窗口更新前不缺行。
        overscanRows: 4,
    });

    const visibleTrackIds = args.tracks
        .slice(visibleTrackWindow.startIndex, visibleTrackWindow.endIndex + 1)
        .map((track) => track.id);
    const clipsByTrackId = new Map<string, typeof args.clips>();

    for (const clip of args.clips) {
        const next = clipsByTrackId.get(clip.trackId);
        if (next) {
            next.push(clip);
        } else {
            clipsByTrackId.set(clip.trackId, [clip]);
        }
    }

    const visibleClipIdsByTrackId = Object.fromEntries(
        visibleTrackIds.map((trackId) => [
            trackId,
            sliceVisibleClipIds({
                viewportStartSec: args.viewportStartSec,
                viewportEndSec: args.viewportEndSec,
                // 横向缓冲必须覆盖 scrollLeft 的量化滞后：React 里的
                // viewportStart/EndSec 最多落后真实值一个步长，缓冲不足会让
                // 刚滚入视野的 clip 晚一拍才出现在 drawClips 里。
                bufferSec: Math.max(1.5, REACT_SCROLL_STEP_PX / Math.max(1e-9, args.pxPerSec)),
                clips: clipsByTrackId.get(trackId) ?? [],
            }),
        ]),
    ) as Record<string, string[]>;

    return {
        ...visibleTrackWindow,
        visibleTrackIds,
        visibleClipIdsByTrackId,
    };
}
