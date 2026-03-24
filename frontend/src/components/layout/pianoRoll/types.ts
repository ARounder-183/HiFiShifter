import type { ParamReferenceKind } from "../../../types/api";

// ParamName 是一个字符串，可以是 "pitch"、"tension" 或声码器额外参数 ID。
// 具体可用值由后端 `get_processor_params` 动态返回。
export type ParamName = string;

export type StrokeMode = "draw" | "restore";

export type StrokePoint = { frame: number; value: number };

export type ValueViewport = { center: number; span: number };

export type WavePeaksSegment = {
    key: string;
    startSec: number;
    durSec: number;
    columns: number;
    min: number[];
    max: number[];
};

export type ParamViewSegment = {
    key: string;
    framePeriodMs: number;
    startFrame: number;
    stride: number;
    referenceKind: ParamReferenceKind;
    orig: number[];
    edit: number[];
};

export type ParamMorphPointKind = "left" | "mid1" | "mid2" | "right";

export type ParamMorphControlPoint = {
    kind: ParamMorphPointKind;
    frame: number;
    value: number;
};

export type ParamMorphOverlay = {
    selectionStartFrame: number;
    selectionEndFrame: number;
    meanValue: number;
    baselineValues: number[];
    points: ParamMorphControlPoint[];
};
