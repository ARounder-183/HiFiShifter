import React from "react";
import { useSyncExternalStore } from "react";

import {
    getTrackDropTargetSnapshot,
    subscribeTrackDropTarget,
    type TrackDropTargetSnapshot,
    type TrackDropTargetTone,
} from "./trackDropTargetHighlight";

/**
 * 时间轴 · 目标轨道描边层
 *
 * 【主要内容】把 {@link TrackDropTargetSnapshot} 里的轨道 id 投影为整行圆角
 * 描边；move / copy / snap 三种 tone 使用不同颜色。
 *
 * 【坐标】容器由 TimelineKernelView 的内容坐标层承载（与吸附高亮同一层），
 * 因此轨道环会随水平 / 纵向滚动一起平移，不需要自己读 scrollTop。
 *
 * 【边界】只做视觉反馈：不参与行命中、不改落点、不渲染文字或按钮。拖到
 * “新建轨道”哨兵时不发布任何 trackId，所以这里自然没有 ring。
 */

export interface TrackDropTargetRingsProps {
    readonly snapshot: TrackDropTargetSnapshot;
    readonly tracks: ReadonlyArray<{ readonly id: string }>;
    readonly rowHeight: number;
    readonly contentWidth: number;
    readonly contentHeight: number;
}

/** Resolve the semantic color token for a drop tone. */
function trackDropTargetColor(tone: TrackDropTargetTone): string {
    if (tone === "snap") return "var(--qt-snap-origin)";
    if (tone === "copy") return "var(--qt-track-drop-copy)";
    return "var(--qt-track-drop-target)";
}

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

    const color = trackDropTargetColor(snapshot.tone);
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
                            opacity: snapshot.tone === "snap" ? 0.98 : 0.72,
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
