import React from "react";

import { WaveformSurface } from "../../../waveform/WaveformSurface";
import type { WaveformSceneClip, WaveformSceneRow } from "../../../waveform/sceneBuilder";
import type { WaveformColors } from "../../../theme/waveformColors";
import type { ClipPeaksEntry } from "./useClipsPeaksForPianoRoll";

function toSceneClip(entry: ClipPeaksEntry): WaveformSceneClip | null {
    if (!entry.sourcePath || entry.muted) return null;
    return {
        id: entry.clipId,
        sourcePath: entry.sourcePath,
        startSec: entry.startSec,
        lengthSec: entry.lengthSec,
        sourceStartSec: entry.sourceStartSec,
        sourceEndSec: entry.sourceEndSec,
        durationSec: entry.sourceDurationSec,
        sourceSampleRate: entry.sourceSampleRate,
        playbackRate: entry.playbackRate,
        reversed: entry.reversed,
        loopEnabled: entry.loopEnabled,
        gain: entry.gain,
        muted: false,
        fadeInSec: entry.fadeInSec,
        fadeOutSec: entry.fadeOutSec,
        autoFadeInSec: entry.autoFadeInSec,
        autoFadeOutSec: entry.autoFadeOutSec,
        fadeInCurve: entry.fadeInCurve,
        fadeOutCurve: entry.fadeOutCurve,
    };
}

export const PianoRollWaveformSurface = React.memo(function PianoRollWaveformSurface(props: {
    clips: readonly ClipPeaksEntry[];
    widthPx: number;
    heightPx: number;
    scrollLeftPx: number;
    pxPerSec: number;
    colors: WaveformColors;
}) {
    const rows = React.useMemo<WaveformSceneRow[]>(
        () => [
            {
                topPx: 0,
                waveformTopPx: 0,
                waveformHeightPx: props.heightPx,
                clips: props.clips
                    .map(toSceneClip)
                    .filter((clip): clip is WaveformSceneClip => clip != null),
            },
        ],
        [props.clips, props.heightPx],
    );
    const pxPerSec = Math.max(1e-9, props.pxPerSec);
    const viewportStartSec = props.scrollLeftPx / pxPerSec;

    return (
        <WaveformSurface
            rows={rows}
            widthPx={props.widthPx}
            heightPx={props.heightPx}
            viewportStartSec={viewportStartSec}
            viewportEndSec={viewportStartSec + props.widthPx / pxPerSec}
            pxPerSec={pxPerSec}
            color={props.colors.stroke}
            style={{ opacity: 0.86 }}
        />
    );
});
