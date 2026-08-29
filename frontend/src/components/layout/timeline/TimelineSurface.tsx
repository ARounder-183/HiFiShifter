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
    gridLayerRef: React.RefObject<HTMLDivElement | null>;
    gridBoundaryRef: React.RefObject<HTMLDivElement | null>;
    gridOverlayLayerRef: React.RefObject<HTMLDivElement | null>;
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
    } as const;

    // 挂载时按总线快照立即同步一次网格：恢复滚动位置时 React 的 scrollLeft
    // state 可能仍是 rAF 前值，不能让网格与 Clip 体画布在首帧分叉。
    React.useLayoutEffect(() => {
        const viewport = timelineViewportBus.getSnapshot();
        invokeGridRedrawHandler(props.gridLayerRef.current, viewport.scrollLeft);
        invokeGridRedrawHandler(props.gridOverlayLayerRef.current, viewport.scrollLeft);
    }, [props.gridLayerRef, props.gridOverlayLayerRef, props.gridVisible]);

    return (
        <div
            className="sticky left-0 top-0 pointer-events-none"
            style={{ width: props.widthPx, zIndex: 1 }}
        >
            <BackgroundGrid
                {...gridBaseProps}
                layerRef={props.gridLayerRef}
                boundaryRef={props.gridBoundaryRef}
            />
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
                showBoundary={false}
            />
        </div>
    );
});
