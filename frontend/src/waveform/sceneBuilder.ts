import type { FadeCurveType } from "../features/session/sessionTypes.ts";
import {
    modEuclid,
    resolveLoopMediaDurationSec,
    resolvePlaybackWindowSec,
} from "../utils/loopRender.ts";

export interface WaveformSceneClip {
    id: string;
    sourcePath: string;
    startSec: number;
    lengthSec: number;
    sourceStartSec: number;
    sourceEndSec: number;
    durationSec?: number;
    durationFrames?: number;
    sourceSampleRate?: number;
    playbackRate: number;
    reversed: boolean;
    loopEnabled: boolean;
    gain: number;
    muted: boolean;
    fadeInSec: number;
    fadeOutSec: number;
    autoFadeInSec?: number;
    autoFadeOutSec?: number;
    fadeInCurve: FadeCurveType;
    fadeOutCurve: FadeCurveType;
}

export interface WaveformSceneRow {
    topPx: number;
    waveformTopPx: number;
    waveformHeightPx: number;
    clips: readonly WaveformSceneClip[];
    leadingOverlapSecByClipId?: Readonly<Record<string, number>>;
}

export interface WaveformSceneSegment {
    clipId: string;
    sourcePath: string;
    sourceSampleRate: number;
    sourceStartSec: number;
    sourceEndSec: number;
    clipLocalStartSec: number;
    clipLocalEndSec: number;
    clipTotalDurationSec: number;
    screenRect: { x: number; y: number; width: number; height: number };
    reversed: boolean;
    gain: number;
    fadeInSec: number;
    fadeOutSec: number;
    fadeInCurve: FadeCurveType;
    fadeOutCurve: FadeCurveType;
    alpha: number;
}

export interface WaveformSceneMarker {
    clipId: string;
    timelineSec: number;
    xPx: number;
    yPx: number;
    heightPx: number;
    kind: "loop" | "media-boundary";
}

export interface WaveformScene {
    segments: WaveformSceneSegment[];
    markers: WaveformSceneMarker[];
}

interface SourceTile {
    localStartSec: number;
    durationSec: number;
    sourceStartSec: number;
    sourceEndSec: number;
}

function finitePositive(value: number, fallback: number): number {
    return Number.isFinite(value) && value > 1e-6 ? value : fallback;
}

function effectiveFade(auto: number | undefined, manual: number): number {
    const automatic = Number(auto ?? 0);
    return automatic > 0 ? automatic : Math.max(0, Number(manual) || 0);
}

function validLocalInterval(
    tile: SourceTile,
    reversed: boolean,
    playbackRate: number,
    mediaDurationSec: number,
    localStartSec: number,
    localEndSec: number,
): [number, number] | null {
    if (!(mediaDurationSec > 0)) return null;

    const domainStart = reversed
        ? tile.localStartSec + (tile.sourceEndSec - mediaDurationSec) / playbackRate
        : tile.localStartSec + (0 - tile.sourceStartSec) / playbackRate;
    const domainEnd = reversed
        ? tile.localStartSec + tile.sourceEndSec / playbackRate
        : tile.localStartSec + (mediaDurationSec - tile.sourceStartSec) / playbackRate;
    const start = Math.max(localStartSec, Math.min(domainStart, domainEnd));
    const end = Math.min(localEndSec, Math.max(domainStart, domainEnd));
    return end > start + 1e-9 ? [start, end] : null;
}

function sourceRangeForLocal(
    tile: SourceTile,
    reversed: boolean,
    playbackRate: number,
    localStartSec: number,
    localEndSec: number,
): [number, number] {
    if (reversed) {
        return [
            tile.sourceEndSec - (localEndSec - tile.localStartSec) * playbackRate,
            tile.sourceEndSec - (localStartSec - tile.localStartSec) * playbackRate,
        ];
    }
    return [
        tile.sourceStartSec + (localStartSec - tile.localStartSec) * playbackRate,
        tile.sourceStartSec + (localEndSec - tile.localStartSec) * playbackRate,
    ];
}

export function buildWaveformScene(args: {
    viewportStartSec: number;
    viewportEndSec: number;
    pxPerSec: number;
    widthPx: number;
    rows: readonly WaveformSceneRow[];
}): WaveformScene {
    const segments: WaveformSceneSegment[] = [];
    const markers: WaveformSceneMarker[] = [];
    const pxPerSec = finitePositive(args.pxPerSec, 1);
    const viewportStartSec = Number.isFinite(args.viewportStartSec) ? args.viewportStartSec : 0;
    const viewportEndSec = Math.max(viewportStartSec, args.viewportEndSec);
    const widthPx = Math.max(1, args.widthPx);

    for (const row of args.rows) {
        for (const clip of row.clips) {
            if (!clip.sourcePath || !(clip.lengthSec > 1e-9)) continue;
            const clipEndSec = clip.startSec + clip.lengthSec;
            const visibleStartSec = Math.max(clip.startSec, viewportStartSec);
            const visibleEndSec = Math.min(clipEndSec, viewportEndSec);
            if (visibleEndSec <= visibleStartSec) continue;

            const mediaDurationSec = resolveLoopMediaDurationSec(clip);
            if (!(mediaDurationSec > 1e-9)) continue;
            const playbackRate = finitePositive(clip.playbackRate, 1);
            const reversed = Boolean(clip.reversed);
            const loopEnabled = Boolean(clip.loopEnabled);
            const sourceStartSec = Number(clip.sourceStartSec) || 0;
            const sourceEndSec = Number(clip.sourceEndSec) || mediaDurationSec;
            const window = resolvePlaybackWindowSec({
                loopEnabled,
                reversed,
                sourceStartSec,
                sourceEndSec,
                playbackRate,
                lengthSec: clip.lengthSec,
            });
            const visibleLocalStartSec = visibleStartSec - clip.startSec;
            const visibleLocalEndSec = visibleEndSec - clip.startSec;
            const tiles: SourceTile[] = [];

            if (!loopEnabled) {
                tiles.push({
                    localStartSec: 0,
                    durationSec: clip.lengthSec,
                    sourceStartSec: window.winStartSec,
                    sourceEndSec: window.winEndSec,
                });
            } else {
                const anchorForward = modEuclid(sourceStartSec, mediaDurationSec);
                const anchorReverse = modEuclid(sourceEndSec, mediaDurationSec);
                const headDurationSec =
                    (reversed ? anchorReverse : mediaDurationSec - anchorForward) / playbackRate;
                const periodSec = mediaDurationSec / playbackRate;

                if (headDurationSec > 1e-9 && visibleLocalStartSec < headDurationSec) {
                    tiles.push({
                        localStartSec: 0,
                        durationSec: headDurationSec,
                        sourceStartSec: reversed ? 0 : anchorForward,
                        sourceEndSec: reversed ? anchorReverse : mediaDurationSec,
                    });
                }

                const firstPeriod = Math.max(
                    0,
                    Math.floor((visibleLocalStartSec - headDurationSec - 1e-9) / periodSec),
                );
                for (
                    let localStartSec = headDurationSec + firstPeriod * periodSec, guard = 0;
                    localStartSec < visibleLocalEndSec - 1e-9 && guard < 4096;
                    localStartSec += periodSec, guard += 1
                ) {
                    tiles.push({
                        localStartSec,
                        durationSec: periodSec,
                        sourceStartSec: 0,
                        sourceEndSec: mediaDurationSec,
                    });
                }

                const firstMarker = Math.max(
                    0,
                    Math.ceil((visibleLocalStartSec - headDurationSec - 1e-9) / periodSec),
                );
                for (
                    let markerLocalSec = headDurationSec + firstMarker * periodSec, guard = 0;
                    markerLocalSec < Math.min(clip.lengthSec, visibleLocalEndSec) - 1e-9 &&
                    guard < 4096;
                    markerLocalSec += periodSec, guard += 1
                ) {
                    if (markerLocalSec <= 1e-9) continue;
                    markers.push({
                        clipId: clip.id,
                        timelineSec: clip.startSec + markerLocalSec,
                        xPx: (clip.startSec + markerLocalSec - viewportStartSec) * pxPerSec,
                        yPx: row.topPx + row.waveformTopPx,
                        heightPx: row.waveformHeightPx,
                        kind: "loop",
                    });
                }
            }

            const leadingOverlapSec = Math.max(
                0,
                Math.min(
                    clip.lengthSec,
                    Number(row.leadingOverlapSecByClipId?.[clip.id] ?? 0) || 0,
                ),
            );
            const baseAlpha = clip.muted ? 0.4 : 1;
            const fadeInSec = effectiveFade(clip.autoFadeInSec, clip.fadeInSec);
            const fadeOutSec = effectiveFade(clip.autoFadeOutSec, clip.fadeOutSec);

            for (const tile of tiles) {
                const tileEndSec = tile.localStartSec + tile.durationSec;
                const clippedLocalStart = Math.max(tile.localStartSec, visibleLocalStartSec);
                const clippedLocalEnd = Math.min(tileEndSec, visibleLocalEndSec);
                const valid = validLocalInterval(
                    tile,
                    reversed,
                    playbackRate,
                    mediaDurationSec,
                    clippedLocalStart,
                    clippedLocalEnd,
                );
                if (!valid) continue;

                const boundaries = [valid[0]];
                if (leadingOverlapSec > valid[0] + 1e-9 && leadingOverlapSec < valid[1] - 1e-9) {
                    boundaries.push(leadingOverlapSec);
                }
                boundaries.push(valid[1]);

                for (let index = 0; index + 1 < boundaries.length; index += 1) {
                    const localStartSec = boundaries[index];
                    const localEndSec = boundaries[index + 1];
                    const [pieceSourceStartSec, pieceSourceEndSec] = sourceRangeForLocal(
                        tile,
                        reversed,
                        playbackRate,
                        localStartSec,
                        localEndSec,
                    );
                    const x = (clip.startSec + localStartSec - viewportStartSec) * pxPerSec;
                    const right = (clip.startSec + localEndSec - viewportStartSec) * pxPerSec;
                    const clippedX = Math.max(0, x);
                    const clippedRight = Math.min(widthPx, right);
                    if (clippedRight <= clippedX) continue;

                    segments.push({
                        clipId: clip.id,
                        sourcePath: clip.sourcePath,
                        sourceSampleRate: finitePositive(clip.sourceSampleRate ?? 44100, 44100),
                        sourceStartSec: Math.max(0, pieceSourceStartSec),
                        sourceEndSec: Math.min(mediaDurationSec, pieceSourceEndSec),
                        clipLocalStartSec: localStartSec,
                        clipLocalEndSec: localEndSec,
                        clipTotalDurationSec: clip.lengthSec,
                        screenRect: {
                            x: clippedX,
                            y: row.topPx + row.waveformTopPx,
                            width: clippedRight - clippedX,
                            height: row.waveformHeightPx,
                        },
                        reversed,
                        gain: Number.isFinite(clip.gain) ? Math.max(0, clip.gain) : 1,
                        fadeInSec,
                        fadeOutSec,
                        fadeInCurve: clip.fadeInCurve ?? "linear",
                        fadeOutCurve: clip.fadeOutCurve ?? "linear",
                        alpha:
                            localStartSec < leadingOverlapSec - 1e-9 ? baseAlpha * 0.5 : baseAlpha,
                    });
                }
            }

            if (!loopEnabled) {
                for (const boundarySec of [0, mediaDurationSec]) {
                    const localSec = reversed
                        ? (window.winEndSec - boundarySec) / playbackRate
                        : (boundarySec - window.winStartSec) / playbackRate;
                    if (
                        localSec <= visibleLocalStartSec + 1e-9 ||
                        localSec >= visibleLocalEndSec - 1e-9 ||
                        localSec <= 1e-9 ||
                        localSec >= clip.lengthSec - 1e-9
                    ) {
                        continue;
                    }
                    markers.push({
                        clipId: clip.id,
                        timelineSec: clip.startSec + localSec,
                        xPx: (clip.startSec + localSec - viewportStartSec) * pxPerSec,
                        yPx: row.topPx + row.waveformTopPx,
                        heightPx: row.waveformHeightPx,
                        kind: "media-boundary",
                    });
                }
            }
        }
    }

    return { segments, markers };
}
