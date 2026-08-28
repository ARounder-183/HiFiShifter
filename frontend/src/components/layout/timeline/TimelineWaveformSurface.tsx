import React from "react";

import type { ClipInfo, TrackInfo } from "../../../features/session/sessionTypes";
import { useAppTheme } from "../../../theme/AppThemeProvider";
import { getWaveformColors } from "../../../theme/waveformColors";
import { timelineViewportBus } from "../../../utils/timelineViewportBus";
import { WaveformSurface } from "../../../waveform/WaveformSurface";
import type { WaveformSceneClip, WaveformSceneRow } from "../../../waveform/sceneBuilder";
import { CLIP_BODY_PADDING_Y, CLIP_HEADER_HEIGHT } from "./constants";
import { computeLeadingOverlapSecByClipId } from "./TrackLane";

function toSceneClip(clip: ClipInfo): WaveformSceneClip | null {
    if (!clip.sourcePath) return null;
    return {
        id: clip.id,
        sourcePath: clip.sourcePath,
        startSec: clip.startSec,
        lengthSec: clip.lengthSec,
        sourceStartSec: clip.sourceStartSec,
        sourceEndSec: clip.sourceEndSec,
        durationSec: clip.durationSec,
        durationFrames: clip.durationFrames,
        sourceSampleRate: clip.sourceSampleRate,
        playbackRate: clip.playbackRate,
        reversed: clip.reversed,
        loopEnabled: clip.loopEnabled,
        gain: clip.gain,
        muted: clip.muted,
        fadeInSec: clip.fadeInSec,
        fadeOutSec: clip.fadeOutSec,
        autoFadeInSec: clip.autoFadeInSec,
        autoFadeOutSec: clip.autoFadeOutSec,
        fadeInShape: Number.isFinite(clip.fadeInShape) ? clip.fadeInShape : 0,
        fadeInDir: clip.fadeInDir ?? 0,
        fadeOutShape: Number.isFinite(clip.fadeOutShape) ? clip.fadeOutShape : 0,
        fadeOutDir: clip.fadeOutDir ?? 0,
    };
}

export const TimelineWaveformSurface = React.memo(function TimelineWaveformSurface(props: {
    tracks: readonly TrackInfo[];
    /** 窗口首行的绝对轨道索引：行 topPx 使用内容绝对坐标，
     * 竖直滚动时由总线 scrollTopPx 统一平移（与 DOM 内容层同帧提交）。 */
    startTrackIndex: number;
    clipsByTrackId: Readonly<Record<string, readonly ClipInfo[]>>;
    rowHeight: number;
    widthPx: number;
    heightPx: number;
    viewportStartSec: number;
    viewportEndSec: number;
    pxPerSec: number;
}) {
    const { mode } = useAppTheme();
    const color = React.useMemo(() => getWaveformColors(mode, "timeline").stroke, [mode]);
    const rows = React.useMemo<WaveformSceneRow[]>(
        () =>
            props.tracks.map((track, index) => {
                const clips = props.clipsByTrackId[track.id] ?? [];
                return {
                    topPx: (props.startTrackIndex + index) * props.rowHeight,
                    waveformTopPx: CLIP_HEADER_HEIGHT,
                    waveformHeightPx: Math.max(
                        1,
                        props.rowHeight - CLIP_BODY_PADDING_Y - CLIP_HEADER_HEIGHT,
                    ),
                    clips: clips
                        .map(toSceneClip)
                        .filter((clip): clip is WaveformSceneClip => clip != null),
                    leadingOverlapSecByClipId: computeLeadingOverlapSecByClipId([...clips]),
                };
            }),
        [props.clipsByTrackId, props.rowHeight, props.startTrackIndex, props.tracks],
    );

    return (
        <WaveformSurface
            rows={rows}
            widthPx={props.widthPx}
            heightPx={props.heightPx}
            viewportStartSec={props.viewportStartSec}
            viewportEndSec={props.viewportEndSec}
            pxPerSec={props.pxPerSec}
            viewportTopPx={props.startTrackIndex * props.rowHeight}
            color={color}
            viewportSource={timelineViewportBus}
        />
    );
});
