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
    /**
     * dyn 专用：后端 `edit_sentinel`「未画帧」位图，与 `edit` 逐帧对齐
     * （同一次取数、同一 stride）。
     *
     * 【为什么进 pv 数据模型】读-变换-写回类提交（拉伸 / morph / 选区拖拽）
     * 的数据源就是 pv；没有位图，"未画帧"在提交时会被物化成显式基线 ——
     * 日后 clip 移动触发基线重分析时这些帧不再跟随，响度静默漂移（见
     * `paramRanges.restoreDynSentinels`）。非 dyn 参数恒 undefined。
     */
    editSentinel?: boolean[];
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
