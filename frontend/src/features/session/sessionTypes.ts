export type DrawToolMode = "draw" | "line" | "vibrato";
export type ToolModeGroup = "select" | "draw";
export type ToolMode = "select" | DrawToolMode;
export type PitchSnapUnit = "semitone" | "scale";
export type ScaleHighlightMode = "always" | "off";
export type DragDirection = "free" | "x-only" | "y-only";
export type DrawDragDirection = "free" | "x-only";
export type FadeCurveType = "linear" | "sine" | "exponential" | "logarithmic" | "scurve";

/**
 * 分割过渡的“淡化曲线”设置（新版 REAPER 形状/曲率模型）。
 *
 * - `"keep"`：分割后**保留原 Clip 的淡化曲线类型，不再修改**（默认值）；
 * - 其余取值与 `timeline/reaperFade.ts` 的 `FADE_PRESETS` 一一对应，
 *   写入边界淡变时同时应用该预设的默认曲率（淡入/淡出各自方向）。
 */
export type SplitTransitionCurveType =
    | "keep"
    | "linear"
    | "convexSlight"
    | "lateSlight"
    | "convexSharp"
    | "lateSharp"
    | "sSlight"
    | "sSharp";

/**
 * 兼容迁移：旧命名曲线（v3/早期 v4，淡化模型重构前）→ 新版预设 id。
 * 未知值一律回退为 "keep"（保留原曲线，不再强行改写）。
 * - linear → linear；exponential → lateSlight；logarithmic → convexSlight；
 * - sine / scurve → sSlight（与后端 legacy_curve_to_fade_spec 同语义）。
 */
export function normalizeSplitTransitionCurve(value: string): SplitTransitionCurveType {
    switch (value) {
        case "linear":
        case "convexSlight":
        case "lateSlight":
        case "convexSharp":
        case "lateSharp":
        case "sSlight":
        case "sSharp":
            return value;
        case "exponential":
            return "lateSlight";
        case "logarithmic":
            return "convexSlight";
        case "sine":
        case "scurve":
            return "sSlight";
        default:
            return "keep";
    }
}
export type TimeUnit = "barBeats" | "barDivisions" | "seconds" | "clock";
export type TimeUnitChoice = TimeUnit | "none";
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

/**
 * 时间轴吸附/网格设置（对标 REAPER Snap/Grid Settings）。
 *
 * 该对象同时持久化到后端 UI 设置，并通过 Redux `session.timelineSnap` 全局共享。
 */
export interface TimelineSnapSettings {
    // ── Grid ──
    /** 显示背景网格线。 */
    gridVisible: boolean;
    /** 网格线最小像素间距。 */
    gridMinSpacingPx: number;
    /** 开启 Swing（摇摆）网格。 */
    swingEnabled: boolean;
    /** Swing 强度（0-100）。 */
    swingPercent: number;
    /** 调整 Swing 时自动按新 Swing 网格重新对齐现有 Clip。 */
    adjustClipsOnSwingChange: boolean;

    // ── Snap master ──
    /** 吸附总开关（工具栏按钮同步该值）。 */
    enabled: boolean;
    /** 鼠标距吸附目标的像素距离阈值。 */
    snapDistancePx: number;
    /** 吸附位置相对网格保留原偏移。 */
    snapRelativeToGrid: boolean;
    /**
     * 拖拽时显示吸附竖线高亮（同时标记吸附对象与被吸附对象的吸附处）。
     * 纯视觉开关，不影响吸附行为本身；默认开启。
     */
    snapHighlightEnabled: boolean;

    // ── Snap targets / objects matrix ──
    /** Clip 吸附到 选择/标记/光标。 */
    snapClipsToSelectionMarkersCursor: boolean;
    /** Clip 吸附到网格。 */
    snapClipsToGrid: boolean;
    /** 选区吸附到 选择/标记/光标。 */
    snapSelectionToSelectionMarkersCursor: boolean;
    /** 选区吸附到网格。 */
    snapSelectionToGrid: boolean;
    /** 光标吸附到 选择/标记/光标。 */
    snapCursorToSelectionMarkersCursor: boolean;
    /** 光标吸附到网格。 */
    snapCursorToGrid: boolean;

    // ── Grid snap behavior ──
    /** 吸附设置跟随网格显示状态。 */
    snapFollowsGridVisibility: boolean;
    /** 任意距离吸附网格（激进模式）。 */
    snapToGridAnyDistance: boolean;
    /** 使用独立于显示网格的吸附间距。 */
    useIndependentSnapSpacing: boolean;
    /** 独立吸附间距。 */
    snapSpacing: GridSize;
    /** 独立吸附间距的最小像素值。 */
    snapSpacingMinPx: number;

    // ── Clip & special interactions ──
    /** 其他 Clip 的起点/终点参与吸附。 */
    snapClipEdges: boolean;
    /** Clip 内容起始点（snap offset）参与吸附。 */
    snapClipSnapOffset: boolean;
    /** 允许吸附到其他轨道的媒体项。 */
    snapAcrossTracks: boolean;
    /** 允许跨多少条轨道吸附（0 = 仅本轨）。 */
    snapTrackDistance: number;
    /** 剃刀/分割边缘吸附。 */
    snapRazorEdits: boolean;

    // ── Advanced ──
    /** 吸附到工程采样率（sample accurate）。 */
    snapToProjectSampleRate: boolean;
    /** Clip 边缘吸附到源素材首尾（循环节吸附）。 */
    snapClipsToSourceMedia: boolean;
    /** 强制选区为网格倍数。 */
    forceSelectionsToMultiples: boolean;
    /** 强制选区倍数网格。 */
    selectionMultiple: GridSize;
    /** Arrange 视图与 MIDI/参数编辑器共用同一网格划分。 */
    syncArrangeAndMidiGrid: boolean;
}

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

export interface ClipTakeInfo {
    id: string;
    name: string;
    gain: number;
    sourcePath?: string;
    sourcePathRelative?: string;
    durationSec?: number;
    durationFrames?: number;
    sourceSampleRate?: number;
    sourceStartSec: number;
    sourceEndSec: number;
    playbackRate: number;
    reversed: boolean;
    loopEnabled: boolean;
    midiNoteData?: MidiNoteEvent[];
    midiFillGaps?: boolean;
}

export interface ClipInfo {
    id: string;
    trackId: string;
    name: string;
    startSec: number;
    lengthSec: number;
    color: "blue" | "violet" | "emerald" | "amber" | "cyan";
    /** 全部 take 元数据；active take 的媒体字段同时平铺在 ClipInfo 上。 */
    takes?: ClipTakeInfo[];
    activeTakeId?: string;
    sourcePath?: string;
    sourcePathRelative?: string;
    durationSec?: number;
    durationFrames?: number; // 精确frame总数
    sourceSampleRate?: number; // 源文件采样率
    gain: number;
    muted: boolean;
    /** When set, this clip belongs to a group of clips sharing the same UUID. */
    groupId?: string;
    sourceStartSec: number;
    sourceEndSec: number;
    playbackRate: number;
    /** Clip 级播放倍率；playbackRate = clipPlaybackRate × activeTake.playbackRate。 */
    clipPlaybackRate?: number;
    reversed: boolean;
    /** Loop（循环源）：延伸超出源媒体区间时按周期回绕产生循环内容。 */
    loopEnabled: boolean;
    /**
     * 吸附偏移（秒）：相对 Clip 起点的偏移，默认 0。与倒放无关 ——
     * 倒放时它依然表示"距 Clip 起点偏移 X"的位置（对标 REAPER/VEGAS
     * 的 item snap offset）。作为其他拖拽的吸附目标参与匹配；
     * Clip 被拉伸时按长度比例同步缩放。
     */
    snapOffsetSec: number;
    fadeInSec: number;
    fadeOutSec: number;
    /** 淡入形状：REAPER 浮点形状 id（整数 0..6 七预设；小数变体透传）。 */
    fadeInShape: number;
    /** 淡出形状（语义同 fadeInShape）。 */
    fadeOutShape: number;
    /** 淡入曲率（REAPER D_FADEINDIR），范围 [-1, 1]。 */
    fadeInDir: number;
    /** 淡出曲率（REAPER D_FADEOUTDIR），范围 [-1, 1]。 */
    fadeOutDir: number;
    /** 自动交叉淡化长度（秒），与手动 fade（fadeInSec/fadeOutSec）分离存储。 */
    autoFadeInSec?: number;
    autoFadeOutSec?: number;
    formantMorph?: ClipFormantMorph;
    midiNoteCount?: number;
    midiNoteData?: MidiNoteEvent[];
    midiFillGaps?: boolean;
}

export interface ClipFormantMorph {
    enabled: boolean;
    targetF1Hz: number;
    targetF2Hz: number;
    strength: number;
}

/**
 * Clip 源共振峰分析状态（供共振峰工具窗口的"源点 → 目标点"可视化）。
 * 数据来自 analyze_clip_formants 命令，与 clipFormantStatus 平行存储。
 */
export interface ClipFormantAnalysisState {
    status: "loading" | "ready" | "failed";
    /** 统计源 F1（检出帧中位数，Hz；无检出为 0） */
    sourceF1Hz: number;
    /** 统计源 F2（Hz；无检出为 0） */
    sourceF2Hz: number;
    /** 稀疏轨迹 [t_norm, f1_hz, f2_hz] */
    track: Array<[number, number, number]>;
    /** 检出候选的分析帧占比 [0,1] */
    voicedRatio: number;
    /** 诊断消息："source_too_short" / "no_voiced_frames" */
    message: string | null;
}

export type WaveformPreview = number[] | { l: number[]; r: number[] };

/**
 * Clip 头部展示名：
 * - 多 Take 时显示 active take 名称与序号（VEGAS 风格：`name (n / count)`）；
 * - 单 Take 时只显示名称，不暴露 Take 计数。
 */
export function clipDisplayName(clip: {
    name: string;
    takes?: Array<{ id: string; name: string }>;
    activeTakeId?: string;
}): string {
    const takes = clip.takes ?? [];
    if (takes.length <= 1) return clip.name;
    const index = Math.max(
        0,
        takes.findIndex((take) => take.id === clip.activeTakeId),
    );
    const take = takes[index];
    const takeName = take?.name || clip.name;
    return `${takeName} (${index + 1} / ${takes.length})`;
}

/** 当前可编辑的名称（改名输入框的预填值）。
 *
 * 路由规则与 commitTrackLaneRename 一致：多 Take（>1）时编辑 active take
 * 名；单 Take / 无 takes 时展示名与提交目标都是容器 name —— 预填必须跟随
 * 提交目标，否则"多 take 删到剩 1 个"后预填残留 take 名、提交却写容器名，
 * 用户会看到"名字没变"。 */
export function activeClipTakeName(clip: {
    name: string;
    takes?: Array<{ id: string; name: string }>;
    activeTakeId?: string;
}): string {
    const takes = clip.takes ?? [];
    if (takes.length <= 1) return clip.name;
    const activeTake = takes.find((take) => take.id === clip.activeTakeId) ?? takes[0];
    return activeTake?.name || clip.name;
}

export interface MidiNoteEvent {
    startSec: number;
    endSec: number;
    note: number;
    velocity: number;
    channel: number;
}

export interface LinkedParamCurves {
    framePeriodMs: number;
    pitchEdit: number[];
    tensionEdit: number[];
    extraCurves: Record<string, number[]>;
}

export type ClipTemplate = Partial<Omit<ClipInfo, "id" | "color" | "groupId">> & {
    trackId: string;
    name: string;
    startSec: number;
    lengthSec: number;
    sourceClipId?: string;
    waveformPreview?: WaveformPreview;
    linkedParams?: LinkedParamCurves;
};
export interface AutomationPoint {
    id: string;
    beat: number;
    value: number;
}
