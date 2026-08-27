import type { FadeCurveType } from "../features/session/sessionTypes";

export const WAVEFORM_TILE_PEAKS = 4096;

export type WaveformLevel = 0 | 1 | 2;
export type WaveformTileKey = `${string}|${WaveformLevel}|${number}`;

export interface WaveformViewportSnapshot {
    revision: number;
    scrollLeftPx: number;
    pxPerSec: number;
    widthPx: number;
    heightPx: number;
    devicePixelRatio: number;
}

export type WaveformViewportPatch = Partial<Omit<WaveformViewportSnapshot, "revision">>;

export interface WaveformViewportSource {
    getSnapshot(): WaveformViewportSnapshot;
    subscribe(listener: () => void): () => void;
}

export interface WaveformViewportStore extends WaveformViewportSource {
    set(patch: WaveformViewportPatch): WaveformViewportSnapshot;
}

export interface WaveformManifestLevel {
    level: WaveformLevel;
    divisionFactor: number;
    peakCount: number;
    tileCount: number;
}

export interface WaveformManifest {
    sourcePath: string;
    revision: string;
    sampleRate: number;
    totalFrames: number;
    channels: number;
    durationSec: number;
    tilePeaks: number;
    levels: readonly WaveformManifestLevel[];
}

export interface WaveformTileIdentity {
    sourcePath: string;
    sourceRevision: string;
    level: WaveformLevel;
    tileIndex: number;
}

export interface WaveformTileData extends WaveformTileIdentity {
    peakStart: number;
    peakCount: number;
    divisionFactor: number;
    sampleRate: number;
    minMax: Float32Array;
}

export interface WaveformTileLease {
    key: WaveformTileKey;
    tile: WaveformTileData;
    release(): void;
}

export interface WaveformScreenRect {
    x: number;
    y: number;
    width: number;
    height: number;
}

export interface WaveformDrawSegment {
    clipId: string;
    sourcePath: string;
    sourceRevision: string;
    level: WaveformLevel;
    sourceStartSec: number;
    sourceEndSec: number;
    timelineStartSec: number;
    timelineDurationSec: number;
    clipTimeOffsetSec: number;
    clipTotalDurationSec: number;
    screenRect: WaveformScreenRect;
    playbackRate: number;
    reversed: boolean;
    gain: number;
    fadeInSec: number;
    fadeOutSec: number;
    fadeInCurve: FadeCurveType;
    fadeOutCurve: FadeCurveType;
    alpha: number;
    color: string;
}

export interface WaveformTileNeed extends WaveformTileIdentity {
    key: WaveformTileKey;
    priority: "visible" | "overscan" | "refinement";
}

export interface WaveformMarker {
    clipId: string;
    timelineSec: number;
    xPx: number;
    yPx: number;
    heightPx: number;
    color: string;
    kind: "loop" | "media-boundary";
}

export interface WaveformFrame {
    viewport: WaveformViewportSnapshot;
    segments: readonly WaveformDrawSegment[];
    requiredTiles: readonly WaveformTileNeed[];
    markers: readonly WaveformMarker[];
}
