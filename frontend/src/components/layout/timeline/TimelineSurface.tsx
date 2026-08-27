import React from "react";

import type { ClipInfo, TrackInfo } from "../../../features/session/sessionTypes";
import { TimelineCanvasViewport } from "./TimelineCanvasViewport";
import { TimelineWaveformSurface } from "./TimelineWaveformSurface";

export const TimelineSurface = React.memo(function TimelineSurface(props: {
    tracks: readonly TrackInfo[];
    clipsByTrackId: Readonly<Record<string, readonly ClipInfo[]>>;
    rowHeight: number;
    widthPx: number;
    heightPx: number;
    topPx: number;
    viewportStartSec: number;
    viewportEndSec: number;
    pxPerSec: number;
    clipModel: {
        drawClips: import("./runtime/timelineCanvasModel").TimelineCanvasClipModel[];
        activeGroupIds?: Set<string>;
        disabledGroupIds?: string[];
    };
}) {
    return (
        <div
            className="sticky left-0 top-0 pointer-events-none"
            style={{ width: props.widthPx, zIndex: 1 }}
        >
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
                    clipsByTrackId={props.clipsByTrackId}
                    rowHeight={props.rowHeight}
                    widthPx={props.widthPx}
                    heightPx={props.heightPx}
                    viewportStartSec={props.viewportStartSec}
                    viewportEndSec={props.viewportEndSec}
                    pxPerSec={props.pxPerSec}
                />
            </div>
        </div>
    );
});
