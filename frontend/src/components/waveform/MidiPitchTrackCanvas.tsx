/**
 * MidiPitchTrackCanvas - 轨道级 MIDI 音高预览 Canvas 组件
 *
 * 为 MIDI clip 在轨道上绘制音高线预览，类似于音频 clip 的波形图。
 * 使用与 WaveformTrackCanvas 相同的 rAF+invalidate 架构和 timelineViewportBus。
 *
 * 数据来源：
 *   - 优先从 clip.midiNoteData 即时生成音高曲线（拖拽/拉伸/Slip 时实时更新）
 *   - 回退到 Redux clipPitchCurves（后端 clip_pitch_data 事件推送）
 *
 * 渲染流程：
 *   1. 遍历可见的 MIDI clip，对每个 clip：
 *      a. 若有 midiNoteData，即时生成 midiCurve（适配当前 sourceStartSec / playbackRate / reversed）
 *      b. 否则从 Redux clipPitchCurves 读取
 *      c. 将每帧 MIDI 值映射到 canvas Y 坐标（高音在上）
 *      d. 使用 clip 自身颜色绘制连续折线
 */

import React from "react";
import type { ClipInfo } from "../../features/session/sessionTypes";
import { useAppSelector } from "../../app/hooks";
import { timelineViewportBus } from "../../utils/timelineViewportBus";
import {
    drawLoopMarkers,
    modEuclid,
    resolveClipContentDurationSec,
    resolveSourceEndSec,
} from "../../utils/loopRender";

// ========================================
// 常量
// ========================================

const FRAME_PERIOD_MS = 5;

const CLIP_COLOR_TO_STROKE: Record<string, string> = {
    blue: "rgba(96, 165, 250, 0.85)",
    violet: "rgba(167, 139, 250, 0.85)",
    emerald: "rgba(52, 211, 153, 0.85)",
    amber: "rgba(251, 191, 36, 0.85)",
    cyan: "rgba(34, 211, 238, 0.85)",
};

// ========================================
// 工具函数
// ========================================

function strokeColorForClip(clip: { color: string }): string {
    return CLIP_COLOR_TO_STROKE[clip.color] ?? "rgba(34, 211, 238, 0.78)";
}

/**
 * Loop（循环源）回绕描述 —— 与后端 clip_loop_cycle_span_sec /
 * place_note_occurrence_frames 的锚点数学逐帧一致。
 */
interface LoopCycleDescriptor {
    /** 回绕周期 D（源域秒）：音频 clip = 整个媒体文件时长；纯 MIDI clip = 窗口跨度。 */
    cycleSec: number;
    /** 正放锚点（原始 sourceStartSec，可为负，floor_mod 环绕）。 */
    fwdAnchorSec: number;
    /**
     * 倒放锚点末端：周期来自媒体时长时 clamp 到 D；周期退化为窗口跨度时
     * 保持原始 sourceEndSec（与后端 place_note_occurrence_in_loop 一致 ——
     * 否则 slip 窗口 [2,7] 的跨度 clamp 会把倒放相位错误平移 ss=2s）。
     */
    revAnchorEndSec: number;
    /**
     * 周期是否来自真实媒体时长：
     * - true：坐标是**文件域**，首个回绕点在 headDur = 进入段耗尽处；
     * - false（纯 MIDI 窗口跨度）：坐标是**窗口相对域**，入口即窗口起点，
     *   首个回绕点在一个完整窗口周期之后。
     */
    cycleFromMedia: boolean;
}

/**
 * 解析 Loop 回绕描述。
 *
 * 关键语义：所有可发声内容（带源媒体的 clip、以及纯音高参考块）的实际
 * 声音/音高按**整个内容** floor_mod 回绕（与 WaveformTrackCanvas / 后端
 * 引擎一致），音高曲线与回绕标记必须使用同一周期，否则相位错位；
 * contentDurationSec 为 null（连音符内容都无法确定）时退化为窗口跨度。
 */
function resolveLoopCycleDescriptor(args: {
    loopEnabled: boolean;
    /** 内容时长（秒）：resolveClipContentDurationSec 的结果；null = 退化。 */
    contentDurationSec: number | null;
    sourceStartSec: number;
    sourceEndSec: number;
}): LoopCycleDescriptor | null {
    if (!args.loopEnabled) return null;
    const mediaDur = args.contentDurationSec ?? 0;
    const windowSpan = Math.abs(
        Number(args.sourceEndSec ?? 0) - Number(args.sourceStartSec ?? 0),
    );
    const cycleSec = mediaDur > 1e-9 ? mediaDur : windowSpan;
    if (!Number.isFinite(cycleSec) || cycleSec <= 1e-9) return null;
    const srcStart = Number(args.sourceStartSec ?? 0);
    const srcEnd = Number(args.sourceEndSec ?? 0);
    return {
        cycleSec,
        fwdAnchorSec: srcStart,
        revAnchorEndSec:
            mediaDur > 1e-9
                ? Math.min(Math.max(srcEnd, 0), mediaDur)
                : Math.max(srcEnd, 0),
        cycleFromMedia: mediaDur > 1e-9,
    };
}

/**
 * 从 MIDI note data 即时生成音高曲线。
 * 逻辑与后端 emit_clip_pitch_data_for_clip 的 MIDI 分支一致，
 * 支持 source range trim、playbackRate 拉伸、reversed 倒放，
 * 以及 Loop（循环源）：按媒体时长的锚点回绕重复铺满
 * （`loopCycle` 为 null 时走非循环路径）。
 */
function generateMidiCurveFromNotes(
    notes: Array<{ startSec: number; endSec: number; note: number }>,
    clipLengthSec: number,
    sourceStartSec: number,
    sourceEndSec: number,
    playbackRate: number,
    reversed: boolean,
    fillGaps: boolean,
    loopCycle: LoopCycleDescriptor | null,
): number[] {
    const fp = Math.max(FRAME_PERIOD_MS, 0.1);
    const targetFrames = Math.max(1, Math.round((clipLengthSec * 1000) / fp));
    const curve = new Array<number>(targetFrames).fill(0);

    const pr = Number.isFinite(playbackRate) && playbackRate > 0 ? playbackRate : 1;
    const srcTotalLen = sourceEndSec - sourceStartSec;

    for (const note of notes) {
        // Loop（循环源）：按媒体时长 D 的锚点回绕放置（与音频渲染的
        // floor_mod 映射一致）。不能用窗口比较过滤可见性 —— split 产生的
        // "环绕窗口"（start > end）会把所有音符误判为越界而全部丢弃。
        if (loopCycle && note.endSec - note.startSec > 1e-9) {
            const { cycleSec, fwdAnchorSec, revAnchorEndSec } = loopCycle;
            const u0 = reversed
                ? modEuclid(revAnchorEndSec - note.endSec, cycleSec)
                : modEuclid(note.startSec - fwdAnchorSec, cycleSec);
            const firstStartFrame = Math.round(((u0 / pr) * 1000) / fp);
            const lenFrames = Math.max(
                1,
                Math.round((((note.endSec - note.startSec) / pr) * 1000) / fp),
            );
            const cycleFrames = Math.max(1, Math.round(((cycleSec / pr) * 1000) / fp));
            const noteValue = note.note;
            for (
                let cycleOffset = 0;
                cycleOffset < targetFrames;
                cycleOffset += cycleFrames
            ) {
                const writeStart = cycleOffset + firstStartFrame;
                const writeEnd = Math.min(
                    cycleOffset + firstStartFrame + lenFrames,
                    targetFrames,
                );
                if (writeStart >= writeEnd) break;
                for (let frame = writeStart; frame < writeEnd; frame++) {
                    if (noteValue > curve[frame] || curve[frame] <= 0) {
                        curve[frame] = noteValue;
                    }
                }
            }
            continue;
        }

        if (note.endSec <= sourceStartSec || note.startSec >= sourceEndSec) continue;
        const relStart = Math.max(0, note.startSec - sourceStartSec);
        const relEnd = Math.min(srcTotalLen, note.endSec - sourceStartSec);
        if (relEnd <= relStart) continue;

        const [effStart, effEnd] = reversed
            ? [Math.max(0, srcTotalLen - relEnd), Math.min(srcTotalLen, srcTotalLen - relStart)]
            : [relStart, relEnd];
        if (effEnd <= effStart) continue;

        const noteStartFrame = Math.round(((effStart / pr) * 1000) / fp);
        const noteEndFrame = Math.round(((effEnd / pr) * 1000) / fp);
        const noteValue = note.note;

        // 非 Loop：单次写入（Loop 已在上方 placement 分支处理）。
        const writeEnd = Math.min(noteEndFrame, targetFrames);
        for (let frame = noteStartFrame; frame < writeEnd; frame++) {
            if (noteValue > curve[frame] || curve[frame] <= 0) {
                curve[frame] = noteValue;
            }
        }
    }

    // 填补音符之间的空隙（与后端 fill_gaps_in_pitch_edit 逻辑一致）
    if (fillGaps && curve.length > 0) {
        let first = -1;
        for (let i = 0; i < curve.length; i++) {
            if (curve[i] > 0) {
                first = i;
                break;
            }
        }
        let last = -1;
        for (let i = curve.length - 1; i >= 0; i--) {
            if (curve[i] > 0) {
                last = i;
                break;
            }
        }
        if (first >= 0 && last > first) {
            let lastPitch = 0;
            for (let i = first; i <= last; i++) {
                if (curve[i] > 0) {
                    lastPitch = curve[i];
                } else if (lastPitch > 0) {
                    curve[i] = lastPitch;
                }
            }
        }
    }

    return curve;
}

/**
 * 曲线生成缓存（F6 性能修复）：
 * generateMidiCurveFromNotes 在每个绘制帧对每个可见 MIDI clip 全量重算 ——
 * 分配 lengthSec×200 的数组（10000s clip ⇒ 2M 元素），且 Loop 分支按重复
 * 周期放大写入量。绘制帧之间同一 clip 的（notes 引用, 几何参数）几乎总是
 * 不变，直接复用上次结果即可。
 *
 * 缓存键：notes 数组**引用**（WeakMap 随 clip 释放，无泄漏）+ 几何参数串。
 * 每个 notes 引用最多保留 4 份几何变体（拖拽编辑时的中间态）。
 */
const midiCurveCache = new WeakMap<
    Array<{ startSec: number; endSec: number; note: number }>,
    Map<string, number[]>
>();
const MIDI_CURVE_CACHE_MAX_PER_NOTES = 4;

function getCachedMidiCurve(
    notes: Array<{ startSec: number; endSec: number; note: number }>,
    clipLengthSec: number,
    sourceStartSec: number,
    sourceEndSec: number,
    playbackRate: number,
    reversed: boolean,
    fillGaps: boolean,
    loopCycle: LoopCycleDescriptor | null,
): number[] {
    let inner = midiCurveCache.get(notes);
    if (!inner) {
        inner = new Map();
        midiCurveCache.set(notes, inner);
    }
    const key =
        `${clipLengthSec}|${sourceStartSec}|${sourceEndSec}|${playbackRate}|` +
        `${reversed ? 1 : 0}|${fillGaps ? 1 : 0}|` +
        (loopCycle
            ? `${loopCycle.cycleSec}|${loopCycle.fwdAnchorSec}|${loopCycle.revAnchorEndSec}|${loopCycle.cycleFromMedia ? 1 : 0}`
            : "null");
    const hit = inner.get(key);
    if (hit) return hit;
    const curve = generateMidiCurveFromNotes(
        notes,
        clipLengthSec,
        sourceStartSec,
        sourceEndSec,
        playbackRate,
        reversed,
        fillGaps,
        loopCycle,
    );
    if (inner.size >= MIDI_CURVE_CACHE_MAX_PER_NOTES) {
        const oldest = inner.keys().next().value;
        if (oldest !== undefined) inner.delete(oldest);
    }
    inner.set(key, curve);
    return curve;
}

// ========================================
// 类型定义
// ========================================

export interface MidiPitchTrackCanvasProps {
    /** 当前轨道上的完整 clip 列表 */
    clips: ClipInfo[];
    /** 轨道高度（像素） */
    trackHeight: number;
    /** 音高预览区域的 top 偏移 */
    waveformTop: number;
    /** 音高预览区域高度 */
    waveformHeight: number;
    /** 每秒像素数 */
    pxPerSec: number;
    /** 视口宽度（CSS 像素） */
    viewportWidthPx: number;
    /** 视口起始时间（秒） */
    viewportStartSec: number;
    /** 视口结束时间（秒） */
    viewportEndSec: number;
    /** 描边宽度 */
    strokeWidth?: number;
}

export const MidiPitchTrackCanvas = React.memo(
    function MidiPitchTrackCanvas(props: MidiPitchTrackCanvasProps) {
        const { clips, waveformTop, waveformHeight, viewportWidthPx, strokeWidth = 1.5 } = props;

        // 从 Redux 读取 MIDI 音高数据（回退数据源）
        const clipPitchCurves = useAppSelector((s) => s.session.clipPitchCurves);
        const clipPitchRanges = useAppSelector((s) => s.session.clipPitchRanges);

        // ========================================
        // refs：高频变化的参数存 ref
        // ========================================
        const canvasRef = React.useRef<HTMLCanvasElement | null>(null);
        const rafRef = React.useRef<number | null>(null);

        const pxPerSecRef = React.useRef(props.pxPerSec);
        const viewportStartSecRef = React.useRef(props.viewportStartSec);
        const viewportEndSecRef = React.useRef(props.viewportEndSec);
        const clipsRef = React.useRef(clips);
        const waveformHeightRef = React.useRef(waveformHeight);
        const strokeWidthRef = React.useRef(strokeWidth);
        const viewportWidthPxRef = React.useRef(viewportWidthPx);
        const clipPitchCurvesRef = React.useRef(clipPitchCurves);
        const clipPitchRangesRef = React.useRef(clipPitchRanges);

        pxPerSecRef.current = props.pxPerSec;
        viewportStartSecRef.current = props.viewportStartSec;
        viewportEndSecRef.current = props.viewportEndSec;
        clipsRef.current = clips;
        waveformHeightRef.current = waveformHeight;
        strokeWidthRef.current = strokeWidth;
        viewportWidthPxRef.current = viewportWidthPx;
        clipPitchCurvesRef.current = clipPitchCurves;
        clipPitchRangesRef.current = clipPitchRanges;

        // ========================================
        // invalidate + rAF 帧合并
        // ========================================
        const drawRef = React.useRef<() => void>(() => {});

        const invalidate = React.useCallback(() => {
            if (rafRef.current != null) return;
            rafRef.current = requestAnimationFrame(() => {
                rafRef.current = null;
                drawRef.current();
            });
        }, []);

        // ========================================
        // 核心绘制函数
        // ========================================
        drawRef.current = () => {
            const canvas = canvasRef.current;
            if (!canvas) return;

            const currentPxPerSec = pxPerSecRef.current;
            const currentViewportStartSec = viewportStartSecRef.current;
            const currentViewportEndSec = viewportEndSecRef.current;
            const currentClips = clipsRef.current;
            const currentWaveformHeight = waveformHeightRef.current;
            const currentStrokeWidth = strokeWidthRef.current;
            const currentViewportWidthPx = viewportWidthPxRef.current;
            const currentClipPitchCurves = clipPitchCurvesRef.current;
            const currentClipPitchRanges = clipPitchRangesRef.current;

            const displayW = Math.max(1, Math.ceil(currentViewportWidthPx));
            const displayH = currentWaveformHeight;

            const dpr = window.devicePixelRatio || 1;
            const internalW = Math.max(1, Math.floor(displayW * dpr));
            const internalH = Math.max(1, Math.floor(displayH * dpr));

            if (canvas.width !== internalW) canvas.width = internalW;
            if (canvas.height !== internalH) canvas.height = internalH;

            const ctx = canvas.getContext("2d");
            if (!ctx) return;

            const scaleX = internalW / Math.max(1, displayW);
            const scaleY = internalH / Math.max(1, displayH);
            ctx.setTransform(scaleX, 0, 0, scaleY, 0, 0);
            ctx.clearRect(0, 0, displayW, displayH);

            canvas.style.width = `${displayW}px`;
            canvas.style.height = `${displayH}px`;

            // 只处理 MIDI clip
            for (const clip of currentClips) {
                if (clip.midiNoteCount == null) continue;
                if (!clip.lengthSec || clip.lengthSec <= 0) continue;

                const clipStartSec = clip.startSec;
                const clipEndSec = clipStartSec + clip.lengthSec;

                // clip 与视口的交集
                const visStartSec = Math.max(clipStartSec, currentViewportStartSec);
                const visEndSec = Math.min(clipEndSec, currentViewportEndSec);
                if (visEndSec <= visStartSec) continue;

                const viewportStartPx = currentViewportStartSec * currentPxPerSec;
                const clipStartPx = clipStartSec * currentPxPerSec;
                const clipEndPx = clipEndSec * currentPxPerSec;
                const visLeftPx = Math.max(0, clipStartPx - viewportStartPx);
                const visRightPx = Math.min(displayW, clipEndPx - viewportStartPx);
                if (visRightPx <= visLeftPx) continue;

                // ── 即时生成或回退读取音高曲线 ──
                let midiCurve: number[] | undefined;
                let curveStartSec: number;
                let framePeriodMs: number;

                if (clip.midiNoteData && clip.midiNoteData.length > 0) {
                    // 内容时长（循环周期 D）：有源媒体 → 媒体总时长；
                    // 纯音高参考块 → 音符内容最大结束时间 —— 与普通媒体
                    // Clip 完全一致（回绕整个内容，窗口之外为静音）。
                    const contentDurSec = resolveClipContentDurationSec({
                        sourcePath: clip.sourcePath,
                        midiNoteData: clip.midiNoteData,
                        durationFrames: clip.durationFrames,
                        sourceSampleRate: clip.sourceSampleRate,
                        durationSec: clip.durationSec,
                    });
                    // 曲线裁剪窗口：非 Loop 正放按派生窗口（起点+长度×速率），
                    // 与音频渲染一致；Loop/倒放保持原字段。
                    const srcEnd = resolveSourceEndSec({
                        loopEnabled: Boolean(clip.loopEnabled),
                        reversed: Boolean(clip.reversed),
                        sourceStartSec: Number(clip.sourceStartSec) || 0,
                        playbackRate: Math.abs(Number(clip.playbackRate) || 1),
                        lengthSec: clip.lengthSec,
                        sourceEndSec:
                            clip.sourceEndSec > 0
                                ? clip.sourceEndSec
                                : clip.midiNoteData.reduce((max, n) => Math.max(max, n.endSec), 0),
                    });
                    const loopCycle = resolveLoopCycleDescriptor({
                        loopEnabled: Boolean(clip.loopEnabled),
                        contentDurationSec: contentDurSec,
                        sourceStartSec: clip.sourceStartSec,
                        sourceEndSec: srcEnd,
                    });
                    midiCurve = getCachedMidiCurve(
                        clip.midiNoteData,
                        clip.lengthSec,
                        clip.sourceStartSec,
                        srcEnd,
                        clip.playbackRate,
                        clip.reversed,
                        clip.midiFillGaps ?? false,
                        loopCycle,
                    );
                    curveStartSec = clipStartSec;
                    framePeriodMs = FRAME_PERIOD_MS;
                } else {
                    const pitchData = currentClipPitchCurves[clip.id];
                    if (!pitchData || !pitchData.midiCurve || pitchData.midiCurve.length < 2)
                        continue;
                    midiCurve = pitchData.midiCurve;
                    curveStartSec = pitchData.curveStartSec ?? clipStartSec;
                    framePeriodMs = pitchData.framePeriodMs || FRAME_PERIOD_MS;
                }

                if (!midiCurve || midiCurve.length < 2) continue;

                // 计算音高范围
                const pitchRange = currentClipPitchRanges[clip.id];
                const minNote = pitchRange?.min ?? 0;
                const maxNote = pitchRange?.max ?? 127;
                const noteSpan = Math.max(1, maxNote - minNote);

                // 曲线的时间跨度
                const curveDurationSec = (midiCurve.length * framePeriodMs) / 1000;
                const curveEndSec = curveStartSec + curveDurationSec;

                // 曲线与可见区域的交集（在时间线上）
                const overlapStartSec = Math.max(visStartSec, curveStartSec);
                const overlapEndSec = Math.min(visEndSec, curveEndSec);
                if (overlapEndSec <= overlapStartSec) continue;

                // 映射到帧索引范围
                const frameStartFrac = ((overlapStartSec - curveStartSec) * 1000) / framePeriodMs;
                const frameEndFrac = ((overlapEndSec - curveStartSec) * 1000) / framePeriodMs;
                const frameStart = Math.max(0, Math.floor(frameStartFrac));
                const frameEnd = Math.min(midiCurve.length - 1, Math.ceil(frameEndFrac));
                if (frameEnd <= frameStart) continue;

                // 每帧对应的画布像素步长
                const frameToPx = (framePeriodMs / 1000) * currentPxPerSec;

                // 绘制连续折线
                ctx.save();
                ctx.beginPath();
                ctx.rect(visLeftPx, 0, visRightPx - visLeftPx, displayH);
                ctx.clip();

                const clipColor = strokeColorForClip(clip);
                ctx.strokeStyle = clipColor;
                ctx.lineWidth = currentStrokeWidth;
                ctx.lineJoin = "round";
                ctx.lineCap = "round";

                const alpha = clip.muted ? 0.4 : 0.85;
                ctx.globalAlpha = alpha;

                let pathStarted = false;
                const minFrameStep = Math.max(1, Math.floor(0.5 / Math.max(0.01, frameToPx)));

                for (let fi = frameStart; fi <= frameEnd; fi += minFrameStep) {
                    const midiValue = midiCurve[fi];
                    if (midiValue <= 0) {
                        pathStarted = false;
                        continue;
                    }

                    const frameTimeSec = curveStartSec + (fi * framePeriodMs) / 1000;
                    const x = frameTimeSec * currentPxPerSec - viewportStartPx;

                    const normalized = (midiValue - minNote) / noteSpan;
                    const padding = displayH * 0.1;
                    const y = displayH - padding - normalized * (displayH - 2 * padding);
                    const clampedY = Math.max(padding, Math.min(displayH - padding, y));

                    if (!pathStarted) {
                        ctx.moveTo(x, clampedY);
                        pathStarted = true;
                    } else {
                        ctx.lineTo(x, clampedY);
                    }
                }

                ctx.stroke();
                ctx.restore();

                // ── 循环节点倒三角标记 ──
                // 周期与曲线平铺一致：内容时长 D（媒体总时长 / 音符内容范围）。
                const markerContentDur = resolveClipContentDurationSec({
                    sourcePath: clip.sourcePath,
                    midiNoteData: clip.midiNoteData ?? null,
                    durationFrames: clip.durationFrames,
                    sourceSampleRate: clip.sourceSampleRate,
                    durationSec: clip.durationSec,
                });
                const markerCycle = resolveLoopCycleDescriptor({
                    loopEnabled: Boolean(clip.loopEnabled),
                    contentDurationSec: markerContentDur,
                    sourceStartSec: clip.sourceStartSec,
                    sourceEndSec: Number(clip.sourceEndSec) || 0,
                });
                const markerRate =
                    Math.abs(Number(clip.playbackRate ?? 1) || 1) < 1e-6
                        ? 1
                        : Math.abs(Number(clip.playbackRate ?? 1) || 1);
                const markerBodyDur = markerCycle
                    ? markerCycle.cycleSec / markerRate
                    : 0;
                if (markerBodyDur > 0 && clip.lengthSec > markerBodyDur + 1e-6) {
                    // 标记必须锚定在**实际回绕点**：与 WaveformTrackCanvas 的
                    // 分段边界一致 ——
                    // - 周期来自媒体：头部进入段耗尽处（headDur）及此后每个
                    //   整文件周期边界；
                    // - 纯 MIDI 窗口跨度（窗口相对域）：入口即窗口起点，
                    //   首个回绕点在一个完整周期之后。
                    // 不能一律用 k·周期：窗口不从文件原点进入时（trim/split），
                    // 标记会与音频/曲线相位错开。
                    const desc = markerCycle;
                    const headDur = !desc
                        ? 0
                        : desc.cycleFromMedia
                          ? ((clip.reversed
                                ? desc.revAnchorEndSec
                                : desc.cycleSec -
                                  modEuclid(desc.fwdAnchorSec, desc.cycleSec)) /
                              markerRate)
                          : markerBodyDur;
                    const markers: number[] = [];
                    {
                        // 直接跳到可视范围内的第一个回绕点：既避免从 clip 入口
                        // 逐周期空转数千次，也修复"深入长循环 clip 后标记消失"
        //（旧实现受 guard<8192 限制，波形分段是直接寻址的）。
                        const visLocalStart =
                            viewportStartPx / Math.max(1e-9, currentPxPerSec) - clipStartSec;
                        const k0 = Math.max(
                            0,
                            Math.ceil((visLocalStart - headDur - 1e-6) / markerBodyDur),
                        );
                        for (
                            let markerT = headDur + k0 * markerBodyDur;
                            markerT < clip.lengthSec - 1e-6 && markers.length < 4096;
                            markerT += markerBodyDur
                        ) {
                            const mx =
                                (clipStartSec + markerT) * currentPxPerSec - viewportStartPx;
                            if (mx > displayW + 8) break;
                            // 恰好在 clip 起点/终点的回绕点不绘制（loopRender 约定；
                            // 倒放整文件 Loop 的 revAnchor=D 时 headDur=0 会命中）。
                            if (markerT <= 1e-6) continue;
                            if (mx < -8) continue;
                            markers.push(Math.round(mx * 2) / 2);
                        }
                    }
                    if (markers.length > 0) {
                        drawLoopMarkers(ctx, markers, displayH, clipColor);
                    }
                } else if (markerContentDur != null && !clip.reversed) {
                    // ── 非 Loop：媒体/内容边界标记 ──
                    // 循环节 = 源媒体（或音符内容）在该 Clip 内的真实起始/
                    // 终止位置（音频与静音的分界线），落在 Clip 内部时绘制。
                    const mediaDur = markerContentDur;
                    {
                        const rate =
                            Math.abs(Number(clip.playbackRate ?? 1) || 1) < 1e-6
                                ? 1
                                : Math.abs(Number(clip.playbackRate ?? 1) || 1);
                        const markers: number[] = [];
                        for (const b of [0, mediaDur]) {
                            const tLocal = (b - (Number(clip.sourceStartSec) || 0)) / rate;
                            if (tLocal <= 1e-6 || tLocal >= clip.lengthSec - 1e-6) continue;
                            const mx =
                                (clipStartSec + tLocal) * currentPxPerSec - viewportStartPx;
                            if (mx < -8 || mx > displayW + 8) continue;
                            markers.push(Math.round(mx * 2) / 2);
                        }
                        if (markers.length > 0) {
                            drawLoopMarkers(ctx, markers, displayH, clipColor);
                        }
                    }
                }
            }

            if (ctx.globalAlpha !== 1) {
                ctx.globalAlpha = 1;
            }
        };

        // ========================================
        // 监听 Redux pitch curves 变化时触发 invalidate
        // ========================================
        React.useEffect(() => {
            invalidate();
        }, [clipPitchCurves, clipPitchRanges, invalidate]);

        // ========================================
        // 监听低频 props 变化时 invalidate
        // ========================================
        React.useEffect(() => {
            invalidate();
        }, [clips, waveformHeight, strokeWidth, viewportWidthPx, invalidate]);

        // ========================================
        // 订阅事件总线
        // ========================================
        React.useEffect(() => {
            const unsub = timelineViewportBus.subscribe((scrollLeft, pxPerSec, viewportWidth) => {
                pxPerSecRef.current = pxPerSec;
                const vpStartSec = scrollLeft / pxPerSec;
                const vpEndSec = vpStartSec + viewportWidth / pxPerSec;
                viewportStartSecRef.current = vpStartSec;
                viewportEndSecRef.current = vpEndSec;
                viewportWidthPxRef.current = viewportWidth;
                if (canvasRef.current) {
                    canvasRef.current.style.transform = `translate3d(${scrollLeft}px,0,0)`;
                }
                invalidate();
            });
            return unsub;
        }, [invalidate]);

        // 组件卸载时取消待执行的 rAF
        React.useEffect(() => {
            return () => {
                if (rafRef.current != null) {
                    cancelAnimationFrame(rafRef.current);
                    rafRef.current = null;
                }
            };
        }, []);

        return (
            <canvas
                ref={canvasRef}
                style={{
                    position: "absolute",
                    top: waveformTop,
                    height: waveformHeight,
                    pointerEvents: "none",
                    zIndex: 2,
                    left: 0,
                    willChange: "transform",
                }}
            />
        );
    },
    // 自定义比较函数：忽略高频 props
    (prev, next) => {
        return (
            prev.clips === next.clips &&
            prev.trackHeight === next.trackHeight &&
            prev.waveformTop === next.waveformTop &&
            prev.waveformHeight === next.waveformHeight &&
            prev.viewportWidthPx === next.viewportWidthPx &&
            prev.strokeWidth === next.strokeWidth
        );
    },
);
