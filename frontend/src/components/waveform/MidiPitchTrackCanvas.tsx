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
import { drawLoopMarkers, loopCycleSec } from "../../utils/loopRender";

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
 * 从 MIDI note data 即时生成音高曲线。
 * 逻辑与后端 emit_clip_pitch_data_for_clip 的 MIDI 分支一致，
 * 支持 source range trim、playbackRate 拉伸、reversed 倒放，
 * 以及 Loop（循环源）：超出单个循环周期的内容按周期回绕重复铺满。
 */
function generateMidiCurveFromNotes(
    notes: Array<{ startSec: number; endSec: number; note: number }>,
    clipLengthSec: number,
    sourceStartSec: number,
    sourceEndSec: number,
    playbackRate: number,
    reversed: boolean,
    fillGaps: boolean,
    loopEnabled: boolean,
): number[] {
    const fp = Math.max(FRAME_PERIOD_MS, 0.1);
    const targetFrames = Math.max(1, Math.round((clipLengthSec * 1000) / fp));
    const curve = new Array<number>(targetFrames).fill(0);

    const pr = Number.isFinite(playbackRate) && playbackRate > 0 ? playbackRate : 1;
    const srcTotalLen = sourceEndSec - sourceStartSec;

    // Loop（循环源）：单个循环周期占用的帧数。
    const loopCycleFrames =
        loopEnabled && srcTotalLen > 1e-9
            ? Math.max(1, Math.round((srcTotalLen / pr) * 1000 / fp))
            : 0;

    for (const note of notes) {
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

        // Loop：按周期重复写入，铺满整个可见区间；否则只写一次。
        for (
            let cycleOffset = 0;
            cycleOffset < targetFrames;
            cycleOffset += loopCycleFrames > 0 ? loopCycleFrames : targetFrames
        ) {
            const writeStart = cycleOffset + noteStartFrame;
            const writeEnd = Math.min(cycleOffset + noteEndFrame, targetFrames);
            if (writeStart >= writeEnd) break;
            for (let frame = writeStart; frame < writeEnd; frame++) {
                if (noteValue > curve[frame] || curve[frame] <= 0) {
                    curve[frame] = noteValue;
                }
            }
            if (loopCycleFrames <= 0) break;
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
                    const srcEnd =
                        clip.sourceEndSec > 0
                            ? clip.sourceEndSec
                            : clip.midiNoteData.reduce((max, n) => Math.max(max, n.endSec), 0);
                    midiCurve = generateMidiCurveFromNotes(
                        clip.midiNoteData,
                        clip.lengthSec,
                        clip.sourceStartSec,
                        srcEnd,
                        clip.playbackRate,
                        clip.reversed,
                        clip.midiFillGaps ?? false,
                        Boolean(clip.loopEnabled),
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

                // ── 循环节点倒三角标记（Loop 启用且存在内部回绕点时）──
                const cycleSec = loopCycleSec({
                    loopEnabled: Boolean(clip.loopEnabled),
                    sourceStartSec: clip.sourceStartSec,
                    sourceEndSec:
                        clip.sourceEndSec > 0
                            ? clip.sourceEndSec
                            : (clip.midiNoteData?.reduce(
                                  (max, n) => Math.max(max, n.endSec),
                                  0,
                              ) ?? 0),
                    playbackRate: clip.playbackRate,
                });
                if (cycleSec > 0 && clip.lengthSec > cycleSec + 1e-6) {
                    const markers: number[] = [];
                    for (
                        let markerT = cycleSec;
                        markerT < clip.lengthSec - 1e-6;
                        markerT += cycleSec
                    ) {
                        const mx =
                            (clipStartSec + markerT) * currentPxPerSec - viewportStartPx;
                        if (mx < -8 || mx > displayW + 8) continue;
                        markers.push(Math.round(mx * 2) / 2);
                        if (markers.length >= 4096) break;
                    }
                    if (markers.length > 0) {
                        drawLoopMarkers(ctx, markers, displayH, clipColor);
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
