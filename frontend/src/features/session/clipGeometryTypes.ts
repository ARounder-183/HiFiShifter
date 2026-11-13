/**
 * 跨窗口手势快照所用的 Clip 源几何类型。
 *
 * 【为什么独立成模块】同一份投影同时服务两件事：
 * 1. 参数编辑器的响度基线与动态增益映射（只需要「素材落在哪一刻」的标量字段）；
 * 2. 参数块冻结（还需要 takes / 音节 / MIDI 等源域结构）。
 *
 * 发布方（时间轴）与消费方（可能位于另一个 WebView 的参数编辑器）必须使用
 * **同一份字段口径**。若把字段清单写在两个面板里，新增一个 Take 字段时必然
 * 只改一边，最终表现为「内置窗口正确、独立窗口缺一块」。
 *
 * 本模块只描述可序列化的纯数据；不得引入 waveform / waveformPreview、函数或
 * 任何与具体窗口绑定的对象。手势快照会作为 Redux action 经 detachBridge 发送。
 */

/** 源域音符/音节片段（参数块投影与 GAME 基线共用）。 */
export interface ClipSourceNoteSpan {
    readonly startSec: number;
    readonly endSec: number;
    readonly note?: number | null;
    readonly eventId?: string;
    /** 参数块内部标注来源；Clip 自身的数据不带此字段。 */
    readonly source?: "syllable" | "gameBaseline" | "midiNote";
}

/** GAME 全源分段基线。 */
export interface ClipSourceGameBaseline {
    readonly revision?: number;
    readonly algorithm?: string;
    readonly algorithmVersion?: number;
    readonly sourceSpansSec?: Array<ClipSourceNoteSpan>;
    readonly analyzedWindowSec?: [number, number];
}

/** 源域 MIDI 音符。 */
export interface ClipSourceMidiNote {
    readonly startSec: number;
    readonly endSec: number;
    readonly note: number;
}

/** Take 的源域几何（参数块合成需要的那一部分）。 */
export interface ClipSourceTake {
    readonly id: string;
    readonly sourceStartSec: number;
    readonly sourceEndSec: number;
    readonly playbackRate: number;
    readonly reversed: boolean;
    readonly loopEnabled: boolean;
    readonly durationSec?: number;
    readonly durationFrames?: number;
    readonly sourceSampleRate?: number;
    readonly syllableSpansSec?: Array<ClipSourceNoteSpan>;
    readonly syllableNotes?: Array<number | null>;
    readonly syllableEventIds?: Array<string>;
    readonly syllableAnalysisWindowSec?: [number, number];
    readonly gameBlockBaseline?: ClipSourceGameBaseline;
    readonly midiNoteData?: Array<ClipSourceMidiNote>;
}

/**
 * 手势开始时记录的单个 clip 几何（决定「哪段素材在哪个时刻被播放」的全部字段）。
 *
 * 这是响度映射的输入，故意只保留标量/小数组；完整源几何在
 * ClipGeometryPreviewState.clips 中另存一份。
 */
export interface ClipGeometrySnapshot {
    readonly clipId: string;
    /** 时间轴起点（秒）。 */
    readonly startSec: number;
    /** 时间轴长度（秒）。 */
    readonly lengthSec: number;
    /** 静态 clip 增益（响度基线的物理口径包含它）。 */
    readonly gain: number;
    /** 源窗口（秒）；Slip / 延伸 / 截短都靠它识别。 */
    readonly sourceStartSec: number;
    readonly sourceEndSec: number;
    /** 播放速率（playbackRate，含 take 倍率）。 */
    readonly playbackRate: number;
    readonly reversed: boolean;
    readonly loopEnabled: boolean;
    /** 媒体元数据：Loop 回绕周期 D 的取值链。 */
    readonly durationSec: number | null;
    readonly durationFrames: number | null;
    readonly sourceSampleRate: number | null;
    readonly sourcePath: string | null;
    readonly midiNoteData: Array<{ endSec: number }> | null;
}

/**
 * Clip 的完整源几何投影。
 *
 * 字段是 ClipInfo 与 ClipTakeInfo 的**子集**，刻意不含显示/波形字段：
 * - 无 waveform / waveformPreview；
 * - 无颜色、名称、静音、淡化形状等与参数块/响度映射无关的字段。
 */
export interface ClipSourceGeometry {
    readonly id: string;
    readonly trackId: string;
    readonly startSec: number;
    readonly lengthSec: number;
    /** 静态 clip 增益；响度基线的物理口径包含它。 */
    readonly gain?: number;
    readonly sourceStartSec?: number;
    readonly sourceEndSec?: number;
    readonly playbackRate?: number;
    /** Clip 级播放倍率；与 take.playbackRate 相乘得到有效速率。 */
    readonly clipPlaybackRate?: number;
    readonly reversed?: boolean;
    readonly loopEnabled?: boolean;
    /** 宿主给出的 Loop 周期（秒）；优先于媒体时长。 */
    readonly loopPeriodSec?: number | null;
    /** 源媒体总时长（秒）；Loop 回绕周期的兜底（旧名）。 */
    readonly contentDurationSec?: number | null;
    /** 源媒体总时长（秒）；手势快照使用的名字。 */
    readonly durationSec?: number | null;
    readonly durationFrames?: number | null;
    readonly sourceSampleRate?: number | null;
    readonly sourcePath?: string | null;
    readonly activeTakeId?: string;
    readonly takes?: Array<ClipSourceTake>;
    readonly syllableSpansSec?: Array<ClipSourceNoteSpan>;
    readonly syllableNotes?: Array<number | null>;
    readonly syllableEventIds?: Array<string>;
    readonly syllableAnalysisWindowSec?: [number, number];
    readonly gameBlockBaseline?: ClipSourceGameBaseline;
    readonly midiNoteData?: Array<ClipSourceMidiNote>;
}

function cloneSpans(
    spans: Array<ClipSourceNoteSpan> | undefined,
): ClipSourceNoteSpan[] | undefined {
    return spans?.map((span) => ({ ...span }));
}

function cloneNumbers(values: Array<number | null> | undefined): Array<number | null> | undefined {
    return values ? [...values] : undefined;
}

function cloneStrings(values: Array<string> | undefined): string[] | undefined {
    return values ? [...values] : undefined;
}

function cloneWindow(window: [number, number] | undefined): [number, number] | undefined {
    return window ? [window[0], window[1]] : undefined;
}

function cloneGameBaseline(
    baseline: ClipSourceGameBaseline | undefined,
): ClipSourceGameBaseline | undefined {
    if (!baseline) return undefined;
    return {
        ...baseline,
        sourceSpansSec: cloneSpans(baseline.sourceSpansSec),
        analyzedWindowSec: cloneWindow(baseline.analyzedWindowSec),
    };
}

function cloneMidiNotes(
    notes: Array<ClipSourceMidiNote> | undefined,
): ClipSourceMidiNote[] | undefined {
    return notes?.map((note) => ({ ...note }));
}

/**
 * 把任意结构兼容的 Clip 投影成手势快照可安全携带的纯数据。
 *
 * 这里**逐字段复制**而不是只做浅拷贝：快照必须与后续 Immer 乐观更新彻底隔离，
 * 否则「按下时几何」会跟着手势一起移动，响度映射退化成恒等映射。
 */
export function toClipSourceGeometry(clip: ClipSourceGeometry): ClipSourceGeometry {
    const contentDurationSec =
        clip.contentDurationSec ?? (clip.durationSec === undefined ? null : clip.durationSec);
    return {
        id: clip.id,
        trackId: clip.trackId,
        startSec: clip.startSec,
        lengthSec: clip.lengthSec,
        gain: clip.gain,
        sourceStartSec: clip.sourceStartSec,
        sourceEndSec: clip.sourceEndSec,
        playbackRate: clip.playbackRate,
        clipPlaybackRate: clip.clipPlaybackRate,
        reversed: clip.reversed,
        loopEnabled: clip.loopEnabled,
        loopPeriodSec: clip.loopPeriodSec ?? null,
        contentDurationSec,
        durationSec: clip.durationSec ?? contentDurationSec,
        durationFrames: clip.durationFrames ?? null,
        sourceSampleRate: clip.sourceSampleRate ?? null,
        sourcePath: clip.sourcePath ?? null,
        activeTakeId: clip.activeTakeId,
        takes: clip.takes?.map((take) => ({
            ...take,
            syllableSpansSec: cloneSpans(take.syllableSpansSec),
            syllableNotes: cloneNumbers(take.syllableNotes),
            syllableEventIds: cloneStrings(take.syllableEventIds),
            syllableAnalysisWindowSec: cloneWindow(take.syllableAnalysisWindowSec),
            gameBlockBaseline: cloneGameBaseline(take.gameBlockBaseline),
            midiNoteData: cloneMidiNotes(take.midiNoteData),
        })),
        syllableSpansSec: cloneSpans(clip.syllableSpansSec),
        syllableNotes: cloneNumbers(clip.syllableNotes),
        syllableEventIds: cloneStrings(clip.syllableEventIds),
        syllableAnalysisWindowSec: cloneWindow(clip.syllableAnalysisWindowSec),
        gameBlockBaseline: cloneGameBaseline(clip.gameBlockBaseline),
        midiNoteData: cloneMidiNotes(clip.midiNoteData),
    };
}
