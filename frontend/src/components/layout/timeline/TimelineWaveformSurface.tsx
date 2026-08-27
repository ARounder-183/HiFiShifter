import React from "react";

import type { ClipInfo, TrackInfo } from "../../../features/session/sessionTypes";
import { useAppTheme } from "../../../theme/AppThemeProvider";
import { getWaveformColors } from "../../../theme/waveformColors";
import { timelineViewportBus } from "../../../utils/timelineViewportBus";
import { WaveformSurface } from "../../../waveform/WaveformSurface";
import type { WaveformSceneClip, WaveformSceneRow } from "../../../waveform/sceneBuilder";
import { CLIP_BODY_PADDING_Y, CLIP_HEADER_HEIGHT } from "./constants";

function leadingOverlaps(clips: readonly ClipInfo[]): Record<string, number> {
    const sorted = [...clips].sort((a, b) => a.startSec - b.startSec || a.id.localeCompare(b.id));
    const result: Record<string, number> = {};
    for (let index = 0; index < sorted.length; index += 1) {
        const clip = sorted[index];
        let overlapEnd = clip.startSec;
        for (let priorIndex = 0; priorIndex < index; priorIndex += 1) {
            const prior = sorted[priorIndex];
            overlapEnd = Math.max(
                overlapEnd,
                Math.min(clip.startSec + clip.lengthSec, prior.startSec + prior.lengthSec),
            );
        }
        result[clip.id] = Math.max(0, overlapEnd - clip.startSec);
    }
    return result;
}

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
        fadeInCurve: clip.fadeInCurve,
        fadeOutCurve: clip.fadeOutCurve,
    };
}

export const TimelineWaveformSurface = React.memo(function TimelineWaveformSurface(props: {
    tracks: readonly TrackInfo[];
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
                    topPx: index * props.rowHeight,
                    waveformTopPx: CLIP_HEADER_HEIGHT,
                    waveformHeightPx: Math.max(
                        1,
                        props.rowHeight - CLIP_BODY_PADDING_Y - CLIP_HEADER_HEIGHT,
                    ),
                    clips: clips
                        .map(toSceneClip)
                        .filter((clip): clip is WaveformSceneClip => clip != null),
                    leadingOverlapSecByClipId: leadingOverlaps(clips),
                };
            }),
        [props.clipsByTrackId, props.rowHeight, props.tracks],
    );

    return (
        <WaveformSurface
            rows={rows}
            widthPx={props.widthPx}
            heightPx={props.heightPx}
            viewportStartSec={props.viewportStartSec}
            viewportEndSec={props.viewportEndSec}
            pxPerSec={props.pxPerSec}
            color={color}
            style={{ zIndex: 2 }}
            viewportSource={timelineViewportBus}
            compensateNativeScroll
        />
    );
});
