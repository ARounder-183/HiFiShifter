import React from "react";

import type { ClipInfo, TrackInfo } from "../../../features/session/sessionTypes";
import { timelineViewportBus } from "../../../utils/timelineViewportBus";
import { BackgroundGrid } from "./BackgroundGrid";
import { invokeGridRedrawHandler } from "./gridRedrawBridge";
import { TimelineCanvasViewport } from "./TimelineCanvasViewport";
import { TimelineWaveformSurface } from "./TimelineWaveformSurface";

export const TimelineSurface = React.memo(function TimelineSurface(props: {
    tracks: readonly TrackInfo[];
    /** 窗口首行的绝对轨道索引（供波形面内容绝对坐标使用）。 */
    startTrackIndex: number;
    clipsByTrackId: Readonly<Record<string, readonly ClipInfo[]>>;
    rowHeight: number;
    widthPx: number;
    heightPx: number;
    topPx: number;
    viewportStartSec: number;
    viewportEndSec: number;
    pxPerSec: number;
    scrollLeft: number;
    playheadSec: number;
    clipModel: {
        drawClips: import("./runtime/timelineCanvasModel").TimelineCanvasClipModel[];
        activeGroupIds?: Set<string>;
        disabledGroupIds?: string[];
    };
    /** Sticky 背景网格参数（与内容层同坐标系，随滚动同步重绘）。 */
    contentWidth: number;
    pxPerBeat: number;
    grid: string;
    beatsPerBar: number;
    gridVisible: boolean;
    gridMinSpacingPx?: number;
    gridSwingPercent?: number;
    gridWeakLineXs?: number[] | null;
    gridStrongLineXs?: number[] | null;
    /** 网格内容底部边界（内容绝对坐标 y，通常为最后一条轨道底边）。 */
    gridBottomPx: number;
    gridLayerRef: React.RefObject<HTMLDivElement | null>;
    gridOverlayLayerRef: React.RefObject<HTMLDivElement | null>;
    /** 播放光标竖直参考线：随滚动/缩放与其它 sticky 层在同一帧移动。 */
    playheadLineRef: React.Ref<HTMLDivElement>;
}) {
    const gridBaseProps = {
        contentWidth: props.contentWidth,
        contentHeight: props.heightPx,
        pxPerBeat: props.pxPerBeat,
        grid: props.grid,
        beatsPerBar: props.beatsPerBar,
        viewportWidth: props.widthPx,
        scrollLeft: props.scrollLeft,
        visible: props.gridVisible,
        minSpacingPx: props.gridMinSpacingPx,
        swingPercent: props.gridSwingPercent,
        weakLineXs: props.gridWeakLineXs ?? null,
        strongLineXs: props.gridStrongLineXs ?? null,
        sticky: true,
        viewportTopPx: 0,
        contentBottomPx: props.gridBottomPx,
    } as const;

    // 挂载时按总线快照立即同步一次网格：恢复滚动位置时 React 的 scrollLeft
    // state 可能仍是 rAF 前值，不能让网格与 Clip 体画布在首帧分叉。
    React.useLayoutEffect(() => {
        const viewport = timelineViewportBus.getSnapshot();
        invokeGridRedrawHandler(
            props.gridLayerRef.current,
            viewport.scrollLeft,
            viewport.scrollTopPx,
        );
        invokeGridRedrawHandler(
            props.gridOverlayLayerRef.current,
            viewport.scrollLeft,
            viewport.scrollTopPx,
        );
    }, [props.gridLayerRef, props.gridOverlayLayerRef, props.gridVisible]);

    return (
        <div
            className="sticky left-0 top-0 pointer-events-none"
            style={{ width: props.widthPx, zIndex: 1 }}
        >
            <BackgroundGrid {...gridBaseProps} layerRef={props.gridLayerRef} />
            <div
                className="absolute pointer-events-none"
                style={{
                    top: props.topPx,
                    width: props.widthPx,
                    height: props.heightPx,
                }}
            >
                <TimelineCanvasViewport
                    width={props.widthPx}
                    height={props.heightPx}
                    model={props.clipModel}
                    rowGuides={{
                        startTrackIndex: props.startTrackIndex,
                        rowCount: props.tracks.length,
                        rowHeight: props.rowHeight,
                    }}
                />
            </div>
            <div
                className="absolute pointer-events-none"
                style={{
                    top: props.topPx,
                    width: props.widthPx,
                    height: props.heightPx,
                }}
            >
                <TimelineWaveformSurface
                    tracks={props.tracks}
                    startTrackIndex={props.startTrackIndex}
                    clipsByTrackId={props.clipsByTrackId}
                    rowHeight={props.rowHeight}
                    widthPx={props.widthPx}
                    heightPx={props.heightPx}
                    viewportStartSec={props.viewportStartSec}
                    viewportEndSec={props.viewportEndSec}
                    pxPerSec={props.pxPerSec}
                />
            </div>
            <BackgroundGrid
                {...gridBaseProps}
                layerRef={props.gridOverlayLayerRef}
                lineOpacity={0.38}
            />
            <div
                ref={props.playheadLineRef}
                className="absolute w-px bg-qt-playhead z-20 pointer-events-none"
                style={{
                    top: props.topPx,
                    height: props.heightPx,
                    left: (Number(props.playheadSec) || 0) * props.pxPerSec - props.scrollLeft,
                }}
            />
        </div>
    );
});
