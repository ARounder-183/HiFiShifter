export type DrawToolMode = "draw" | "line" | "vibrato";
export type ToolModeGroup = "select" | "draw";
export type ToolMode = "select" | DrawToolMode;
export type PitchSnapUnit = "semitone" | "scale";
export type ScaleHighlightMode = "always" | "off";
export type DragDirection = "free" | "x-only" | "y-only";
export type DrawDragDirection = "free" | "x-only";
export type FadeCurveType = "linear" | "sine" | "exponential" | "logarithmic" | "scurve";
// EditParam 是一个字符串，可以是 "pitch" 或声码器额外参数 ID（如 "breath_gain"、"hifigan_tension"）
// 具体可用值由后端 `get_processor_params` 动态返回
export type EditParam = string;
export type GridSize =
    | "1/1"
    | "1/2"
    | "1/4"
    | "1/8"
    | "1/16"
    | "1/32"
    | "1/64"
    | "1/1d"
    | "1/2d"
    | "1/4d"
    | "1/8d"
    | "1/16d"
    | "1/32d"
    | "1/64d"
    | "1/1t"
    | "1/2t"
    | "1/4t"
    | "1/8t"
    | "1/16t"
    | "1/32t"
    | "1/64t";

export interface TrackInfo {
    id: string;
    name: string;
    parentId?: string | null;
    depth?: number;
    childTrackIds?: string[];
    muted: boolean;
    solo: boolean;
    volume: number;

    composeEnabled: boolean;
    pitchAnalysisAlgo: string;
    /** 轨道主题色，hex 字符串，如 "#4f8ef7" */
    color?: string;
}

export interface TrackMeterInfo {
    peakLinear: number;
    maxPeakLinear: number;
    clipped: boolean;
}

export interface ClipInfo {
    id: string;
    trackId: string;
    name: string;
    startSec: number;
    lengthSec: number;
    color: "blue" | "violet" | "emerald" | "amber";
    sourcePath?: string;
    durationSec?: number;
    durationFrames?: number; // 精确frame总数
    sourceSampleRate?: number; // 源文件采样率
    gain: number;
    muted: boolean;
    sourceStartSec: number;
    sourceEndSec: number;
    playbackRate: number;
    reversed: boolean;
    fadeInSec: number;
    fadeOutSec: number;
    fadeInCurve: FadeCurveType;
    fadeOutCurve: FadeCurveType;
}

export type WaveformPreview = number[] | { l: number[]; r: number[] };

export interface LinkedParamCurves {
    framePeriodMs: number;
    pitchEdit: number[];
    tensionEdit: number[];
    extraCurves: Record<string, number[]>;
}

export type ClipTemplate = Partial<Omit<ClipInfo, "id" | "color">> & {
    trackId: string;
    name: string;
    startSec: number;
    lengthSec: number;
    waveformPreview?: WaveformPreview;
    linkedParams?: LinkedParamCurves;
};
export interface AutomationPoint {
    id: string;
    beat: number;
    value: number;
}
