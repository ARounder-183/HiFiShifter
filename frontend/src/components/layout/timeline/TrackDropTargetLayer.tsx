import React from "react";
import { useSyncExternalStore } from "react";

import {
    getTrackDropTargetSnapshot,
    subscribeTrackDropTarget,
    type TrackDropTargetSnapshot,
} from "./trackDropTargetHighlight";

/**
 * 时间轴 · 目标轨道描边层
 *
 * 【主要内容】把 {@link TrackDropTargetSnapshot} 里的轨道 id 投影为整行圆角
 * 描边。**单一颜色**（与吸附竖线同色）：描边只表示「吸附锁定」，不区分
 * move / copy —— 那种区分让「没吸附也画着描边」看起来像正常状态。
 *
 * 【坐标】容器由 TimelineKernelView 的内容坐标层承载（与吸附高亮同一层），
 * 因此轨道环会随水平 / 纵向滚动一起平移，不需要自己读 scrollTop。
 *
 * 【边界】只做视觉反馈：不参与行命中、不改落点、不渲染文字或按钮。拖到
 * “新建轨道”哨兵时不发布任何 trackId，所以这里自然没有 ring；未吸附时
 * store 为空，同样没有 ring。
 */

export interface TrackDropTargetRingsProps {
    readonly snapshot: TrackDropTargetSnapshot;
    readonly tracks: ReadonlyArray<{ readonly id: string }>;
    readonly rowHeight: number;
    readonly contentWidth: number;
    readonly contentHeight: number;
}

/**
 * 描边颜色：与吸附竖线同一个 token。
 *
 * 两处画的是**同一件事**（这一刻被吸附钉住了），用两种颜色会让用户以为它们是
 * 两条独立的提示。固定 opacity 0.95：不再有「非吸附态淡一点」的分支。
 */
const TRACK_DROP_TARGET_COLOR = "var(--qt-snap-origin)";
const TRACK_DROP_TARGET_OPACITY = 0.95;

/** Pure ring renderer; exported so tests can render it without a live store. */
export const TrackDropTargetRings: React.FC<TrackDropTargetRingsProps> = ({
    snapshot,
    tracks,
    rowHeight,
    contentWidth,
    contentHeight,
}) => {
    if (snapshot.trackIds.length === 0 || rowHeight <= 0) return null;
    if (contentWidth <= 0 || contentHeight <= 0) return null;

    const rowIndexById = new Map<string, number>();
    tracks.forEach((track, index) => rowIndexById.set(track.id, index));
    const rows = snapshot.trackIds
        .map((trackId) => ({ trackId, index: rowIndexById.get(trackId) }))
        .filter((row): row is { trackId: string; index: number } => row.index !== undefined);
    if (rows.length === 0) return null;

    const color = TRACK_DROP_TARGET_COLOR;
    return (
        <div
            className="absolute top-0 left-0 pointer-events-none overflow-hidden"
            style={{ width: contentWidth, height: contentHeight }}
            aria-hidden
        >
            {rows.map((row) => {
                const top = row.index * rowHeight + 2;
                const height = Math.max(6, rowHeight - 4);
                return (
                    <div
                        key={row.trackId}
                        data-track-drop-target={row.trackId}
                        className="absolute rounded-[6px]"
                        style={{
                            left: 4,
                            right: 4,
                            top,
                            height,
                            border: "2px solid " + color,
                            background: "color-mix(in oklab, " + color + " 9%, transparent)",
                            boxShadow:
                                "0 0 0 1px color-mix(in oklab, " +
                                color +
                                " 35%, transparent), 0 0 12px color-mix(in oklab, " +
                                color +
                                " 30%, transparent)",
                            opacity: TRACK_DROP_TARGET_OPACITY,
                            transition: "opacity 80ms ease-out",
                        }}
                    />
                );
            })}
        </div>
    );
};

/**
 * Live adapter. The third argument is intentionally the same snapshot getter:
 * the layer is also rendered by node-based tests / SSR where React requires a
 * server snapshot, and the store is a plain module singleton in both worlds.
 */
export const TrackDropTargetLayer: React.FC<{
    readonly rowHeight: number;
    readonly tracks: ReadonlyArray<{ readonly id: string }>;
    readonly contentWidth: number;
    readonly contentHeight: number;
}> = (props) => {
    const snapshot = useSyncExternalStore(
        subscribeTrackDropTarget,
        getTrackDropTargetSnapshot,
        getTrackDropTargetSnapshot,
    );
    return (
        <TrackDropTargetRings
            snapshot={snapshot}
            tracks={props.tracks}
            rowHeight={props.rowHeight}
            contentWidth={props.contentWidth}
            contentHeight={props.contentHeight}
        />
    );
};
