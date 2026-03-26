/**
 * PianoRoll 渲染模块
 *
 * 负责钢琴卷帘界面的可视化渲染，包括：
 * - 音高网格和键盘可视化
 * - 音频波形渲染
 * - 参数曲线绘制（音高、音量等）
 * - 选区、播放头等交互元素
 *
 * @module render
 */

import type {
    ParamMorphOverlay,
    ParamName,
    ParamViewSegment,
    ValueViewport,
} from "./types";
import type { ClipPeaksEntry } from "./useClipsPeaksForPianoRoll";
import { clamp } from "../timeline";
import { AXIS_W, PITCH_MAX_MIDI, PITCH_MIN_MIDI } from "./constants";
import { framesToTime, timeToPixel } from "./utils";
import { resolveSecondaryOverlayValues } from "./secondaryOverlaySelection";
import {
    applyGainsToPeaks,
    renderWaveform,
    type WaveformRenderParams,
} from "../../../utils/waveformRenderer";
import { waveformMipmapStore } from "../../../utils/waveformMipmapStore";
import { resolveScaleNotes } from "../../../utils/musicalScales";
import type { ScaleLike } from "../../../utils/musicalScales";

/** 为数值轴选择"好看"的刻度步长 */
function niceAxisStep(range: number, targetCount: number): number {
    const roughStep = range / targetCount;
    const mag = Math.pow(10, Math.floor(Math.log10(roughStep)));
    const normalized = roughStep / mag;
    let nice: number;
    if (normalized < 1.5) nice = 1;
    else if (normalized < 3.5) nice = 2;
    else if (normalized < 7.5) nice = 5;
    else nice = 10;
    return nice * mag;
}

/** 格式化轴标记数值，避免浮点噪声 */
function formatAxisMark(v: number): string {
    // 最多保留 4 位有效数字，去掉尾随零
    const s = parseFloat(v.toPrecision(4)).toString();
    return s;
}

function isBlackKey(midi: number): boolean {
    const pc = ((midi % 12) + 12) % 12;
    return pc === 1 || pc === 3 || pc === 6 || pc === 8 || pc === 10;
}

function midiToLabel(midi: number): string {
    const NOTE_NAMES = [
        "C",
        "C#",
        "D",
        "D#",
        "E",
        "F",
        "F#",
        "G",
        "G#",
        "A",
        "A#",
        "B",
    ];
    const octave = Math.floor(midi / 12) - 1;
    const name = NOTE_NAMES[((midi % 12) + 12) % 12];
    return `${name}${octave}`;
}

function drawCurveTimed(args: {
    ctx: CanvasRenderingContext2D;
    values: number[];
    param: ParamName;
    w: number;
    h: number;
    startFrame: number;
    stride: number;
    framePeriodMs: number;
    visibleStartSec: number;
    visibleDurSec: number;
    valueToY: (param: ParamName, v: number, h: number) => number;
}) {
    const {
        ctx,
        values,
        param,
        w,
        h,
        startFrame,
        stride,
        framePeriodMs,
        visibleStartSec,
        visibleDurSec,
        valueToY,
    } = args;

    if (values.length < 2) return;
    const fp = Math.max(1e-6, framePeriodMs);
    const step = Math.max(1, Math.floor(stride));

    // Check debug flag
    const debugEnabled =
        typeof window !== "undefined" &&
        window.localStorage?.getItem("hifishifter.debugPianoRoll") === "1";

    // DEBUG: 验证曲线时间参数（使用统一转换函数�?
    const curveStartSec = framesToTime(startFrame, fp);
    const curveEndSec = framesToTime(
        startFrame + (values.length - 1) * step,
        fp,
    );
    const curveTotalDurSec = curveEndSec - curveStartSec;

    if (debugEnabled) {
        console.log("[drawCurveTimed] Params:", {
            param,
            visibleStartSec,
            visibleDurSec,
            visibleEndSec: visibleStartSec + visibleDurSec,
            startFrame,
            stride: step,
            framePeriodMs: fp,
            valuesLength: values.length,
            firstValue: values[0],
            lastValue: values[values.length - 1],
            curveStartSec,
            curveEndSec,
            curveTotalDurSec,
            canvasWidth: w,
        });
    }

    let started = false;
    let firstPoint: { frame: number; tSec: number; x: number } | null = null;
    let lastPoint: { frame: number; tSec: number; x: number } | null = null;

    ctx.beginPath();
    for (let i = 0; i < values.length; i += 1) {
        const frame = startFrame + i * step;
        const tSec = framesToTime(frame, fp);
        if (tSec < visibleStartSec || tSec > visibleStartSec + visibleDurSec) {
            started = false;
            continue;
        }
        const x = timeToPixel(tSec, visibleStartSec, visibleDurSec, w);

        // Track first and last points for debugging
        if (!firstPoint && started === false) {
            firstPoint = { frame, tSec, x };
        }
        lastPoint = { frame, tSec, x };

        // pitch 曲线：MIDI �?N 应绘制在 N 键中心（N �?N+1 区间的中点），加 0.5 偏移
        const rawValue = values[i] ?? 0;
        const mappedValue = param === "pitch" ? rawValue + 0.5 : rawValue;
        const y = valueToY(param, mappedValue, h);
        if (!started) {
            ctx.moveTo(x, y);
            started = true;
        } else {
            ctx.lineTo(x, y);
        }
    }

    // DEBUG: Log first and last rendered points
    if (debugEnabled && firstPoint && lastPoint) {
        console.log("[drawCurveTimed] Rendered points:", {
            param,
            firstPoint: {
                frame: firstPoint.frame,
                tSec: firstPoint.tSec,
                x: firstPoint.x,
                // Verify conversion
                verifyTime: framesToTime(firstPoint.frame, fp),
                verifyPixel: timeToPixel(
                    firstPoint.tSec,
                    visibleStartSec,
                    visibleDurSec,
                    w,
                ),
            },
            lastPoint: {
                frame: lastPoint.frame,
                tSec: lastPoint.tSec,
                x: lastPoint.x,
                // Verify conversion
                verifyTime: framesToTime(lastPoint.frame, fp),
                verifyPixel: timeToPixel(
                    lastPoint.tSec,
                    visibleStartSec,
                    visibleDurSec,
                    w,
                ),
            },
            pixelSpan: lastPoint.x - firstPoint.x,
            timeSpan: lastPoint.tSec - firstPoint.tSec,
            pxPerSec:
                (lastPoint.x - firstPoint.x) /
                (lastPoint.tSec - firstPoint.tSec),
        });
    }

    ctx.stroke();
}

function drawParamMorphOverlay(args: {
    ctx: CanvasRenderingContext2D;
    overlay: ParamMorphOverlay;
    editParam: ParamName;
    framePeriodMs: number;
    visibleStartSec: number;
    visibleDurSec: number;
    w: number;
    h: number;
    valueToY: (param: ParamName, v: number, h: number) => number;
    isDark: boolean;
}) {
    const {
        ctx,
        overlay,
        editParam,
        framePeriodMs,
        visibleStartSec,
        visibleDurSec,
        w,
        h,
        valueToY,
        isDark,
    } = args;
    const fp = Math.max(1e-6, framePeriodMs);
    const points = overlay.points.slice().sort((a, b) => a.frame - b.frame);
    if (points.length !== 4) return;

    const lineColor = isDark
        ? "rgba(255, 210, 95, 0.9)"
        : "rgba(160, 90, 10, 0.9)";
    const fillColor = isDark
        ? "rgba(255, 210, 95, 0.22)"
        : "rgba(160, 90, 10, 0.18)";

    const toCanvasX = (frame: number) => {
        const sec = framesToTime(frame, fp);
        return timeToPixel(sec, visibleStartSec, visibleDurSec, w);
    };

    ctx.save();
    ctx.strokeStyle = lineColor;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    for (let i = 0; i < points.length; i += 1) {
        const p = points[i];
        const mappedValue = editParam === "pitch" ? p.value + 0.5 : p.value;
        const x = toCanvasX(p.frame);
        const y = valueToY(editParam, mappedValue, h);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.setLineDash([]);

    for (const p of points) {
        const mappedValue = editParam === "pitch" ? p.value + 0.5 : p.value;
        const x = toCanvasX(p.frame);
        const y = valueToY(editParam, mappedValue, h);
        const radius = p.kind === "left" || p.kind === "right" ? 4 : 5;
        ctx.fillStyle = fillColor;
        ctx.strokeStyle = lineColor;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
    }
    ctx.restore();
}

/**
 * per-clip 检测音高曲线（来自后端 clip_pitch_data 事件），
 * 在参数面板 pitch 视图中作为参考线渲染。
 */
export interface DetectedPitchCurve {
    /** MIDI 曲线第 0 帧对应的 timeline 绝对时间（秒），直接来自后端 */
    curveStartSec: number;
    /** MIDI 音高曲线，每帧一个值，0 表示无声 */
    midiCurve: number[];
    /** WORLD 帧周期（毫秒） */
    framePeriodMs: number;
}

export function drawPianoRoll(args: {
    axisCanvas: HTMLCanvasElement | null;
    canvas: HTMLCanvasElement | null;
    viewSize: { w: number; h: number };
    editParam: ParamName;
    pitchView: ValueViewport;
    /** 每个参数 id 的视口（非音高参数用） */
    paramViews: Record<string, ValueViewport>;
    valueToY: (param: ParamName, v: number, h: number) => number;
    clipPeaks: ClipPeaksEntry[];
    paramView: ParamViewSegment | null;
    secondaryParamViews: Partial<Record<ParamName, ParamViewSegment>>;
    secondaryParamIds: ParamName[];
    showSecondaryParam: boolean;
    overlayText?: string | null;
    liveEditOverride: { key: string; edit: number[] } | null;
    selection: { aBeat: number; bBeat: number } | null;
    pxPerSec: number;
    scrollLeft: number;
    secPerBeat: number;
    playheadSec: number; // 播放头位置（秒）
    pitchAnalysisPending?: boolean;
    waveformColors?: { fill: string; stroke: string };
    /** 检测音高曲线列表，在 pitch 模式下渲染为参考线 */
    detectedPitchCurves?: DetectedPitchCurve[];
    /** 是否为深色主题（默认 true） */
    isDark?: boolean;
    /** 剪贴板预览数据（选区内渲染半透明预览曲线） */
    clipboardPreview?: {
        param: ParamName;
        framePeriodMs: number;
        values: number[];
    } | null;
    // pitch snap visual helpers
    pitchSnapUnit?: "semitone" | "scale";
    projectScale?: ScaleLike | null;
    toolMode?: string;
    snapToggleHeld?: boolean;
    scaleHighlightMode?: import("../../../features/session/sessionTypes").ScaleHighlightMode;
    paramMorphOverlay?: ParamMorphOverlay | null;
}) {
    const {
        axisCanvas,
        canvas,
        viewSize,
        editParam,
        pitchView,
        paramViews,
        valueToY,
        clipPeaks,
        paramView,
        secondaryParamViews,
        secondaryParamIds,
        showSecondaryParam,
        overlayText,
        liveEditOverride,
        selection,
        pxPerSec,
        scrollLeft,
        secPerBeat,
        playheadSec,
        pitchAnalysisPending,
        waveformColors = {
            fill: "rgba(255,255,255,0.2)",
            stroke: "rgba(255,255,255,0.5)",
        },
        detectedPitchCurves,
        isDark = true,
        clipboardPreview,
        paramMorphOverlay,
    } = args;

    // 主题颜色查找表
    const colors = isDark
        ? {
              // 琴键区
              axisBorder: "rgba(255,255,255,0.08)",
              whiteKey: "#e8e8e8",
              blackKey: "#1a1a1a",
              blackKeyGradient: "rgba(0,0,0,0.35)",
              cLabel: "#3b82f6",
              whiteKeyLabel: "rgba(80,80,80,0.70)",
              blackKeyLabel: "rgba(220,220,220,0.80)",
              cSeparator: "rgba(100,100,100,0.45)",
              keySeparator: "rgba(160,160,160,0.20)",
              tensionLabel: "rgba(255,255,255,0.55)",
              tensionLine: "rgba(255,255,255,0.10)",
              // 网格线
              pitchGridC: "rgba(255,255,255,0.10)",
              pitchGridOther: "rgba(255,255,255,0.05)",
              // 曲线
              origCurve: "rgba(200,200,200,0.55)",
              editCurve: "rgba(255,255,255,0.90)",
              selectionCurve: "rgba(100,200,255,0.95)",
              // 叠加文字 & 播放头
              overlayTextColor: "rgba(255,255,255,0.35)",
              playheadLine: "rgba(255,255,255,0.25)",
          }
        : {
              // 浅色主题
              axisBorder: "rgba(0,0,0,0.10)",
              whiteKey: "#ffffff",
              blackKey: "#3a3a3a",
              blackKeyGradient: "rgba(0,0,0,0.25)",
              cLabel: "#2563eb",
              whiteKeyLabel: "rgba(80,80,80,0.65)",
              blackKeyLabel: "rgba(255,255,255,0.85)",
              cSeparator: "rgba(0,0,0,0.25)",
              keySeparator: "rgba(0,0,0,0.12)",
              tensionLabel: "rgba(0,0,0,0.55)",
              tensionLine: "rgba(0,0,0,0.10)",
              // 网格线
              pitchGridC: "rgba(0,0,0,0.12)",
              pitchGridOther: "rgba(0,0,0,0.06)",
              // 曲线
              origCurve: "rgba(100,120,150,0.50)",
              editCurve: "rgba(50,80,140,0.80)",
              selectionCurve: "rgba(30,120,200,0.95)",
              // 叠加文字 & 播放头
              overlayTextColor: "rgba(0,0,0,0.35)",
              playheadLine: "rgba(0,0,0,0.20)",
          };

    // Draw axis (left labels)
    if (axisCanvas) {
        const ctx = axisCanvas.getContext("2d");
        if (ctx) {
            const h = viewSize.h;
            const w = AXIS_W;
            const dpr = Math.max(1, window.devicePixelRatio || 1);
            const cw = Math.max(1, Math.floor(w * dpr));
            const ch = Math.max(1, Math.floor(h * dpr));
            if (axisCanvas.width !== cw || axisCanvas.height !== ch) {
                axisCanvas.width = cw;
                axisCanvas.height = ch;
                axisCanvas.style.width = `${w}px`;
                axisCanvas.style.height = `${h}px`;
            }
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
            ctx.clearRect(0, 0, w, h);

            ctx.strokeStyle = colors.axisBorder;
            ctx.beginPath();
            ctx.moveTo(w - 0.5, 0);
            ctx.lineTo(w - 0.5, h);
            ctx.stroke();

            if (editParam === "pitch") {
                const absMin = PITCH_MIN_MIDI;
                const absMax = PITCH_MAX_MIDI;
                const view = pitchView;
                const span = clamp(view.span, 1e-6, absMax - absMin);
                const min = clamp(
                    view.center - span / 2,
                    absMin,
                    absMax - span,
                );
                const max = min + span;
                const startMidi = clamp(Math.floor(min), absMin, absMax);
                const endMidi = clamp(Math.ceil(max), absMin, absMax);
                for (let midi = startMidi; midi < endMidi; midi += 1) {
                    const y0 = valueToY("pitch", midi, h);
                    const y1 = valueToY("pitch", midi + 1, h);
                    const top = Math.min(y0, y1);
                    const bottom = Math.max(y0, y1);
                    const keyH = Math.max(1, bottom - top);

                    const black = isBlackKey(midi);
                    const pc = ((midi % 12) + 12) % 12;

                    // 白键
                    if (!black) {
                        ctx.fillStyle = colors.whiteKey;
                        ctx.fillRect(0, top, w, keyH);
                    }

                    // 黑键：深色覆盖，宽度 72%
                    if (black) {
                        ctx.fillStyle = colors.blackKey;
                        ctx.fillRect(0, top, w * 0.72, keyH);
                        // 黑键右侧渐变边缘
                        const grad = ctx.createLinearGradient(
                            w * 0.62,
                            0,
                            w * 0.72,
                            0,
                        );
                        grad.addColorStop(0, "rgba(0,0,0,0)");
                        grad.addColorStop(1, colors.blackKeyGradient);
                        ctx.fillStyle = grad;
                        ctx.fillRect(w * 0.62, top, w * 0.1, keyH);
                    }

                    // 所有琴键音名标注（高度足够时）
                    if (keyH >= 6) {
                        ctx.textBaseline = "middle";
                        const midY = top + keyH / 2;
                        if (!black) {
                            // 白键：C 音用蓝色加粗，其他用灰色
                            ctx.fillStyle =
                                pc === 0 ? colors.cLabel : colors.whiteKeyLabel;
                            ctx.font =
                                pc === 0
                                    ? "bold 9px sans-serif"
                                    : "9px sans-serif";
                            ctx.fillText(midiToLabel(midi), 4, midY);
                        } else {
                            // 黑键：在黑键宽度内裁剪绘制
                            ctx.save();
                            ctx.beginPath();
                            ctx.rect(0, top, w * 0.7, keyH);
                            ctx.clip();
                            ctx.fillStyle = colors.blackKeyLabel;
                            ctx.font = "8px sans-serif";
                            ctx.fillText(midiToLabel(midi), 3, midY);
                            ctx.restore();
                        }
                    }

                    // 分隔线：C 音用较深的线，其他用浅线
                    ctx.strokeStyle =
                        pc === 0 ? colors.cSeparator : colors.keySeparator;
                    ctx.lineWidth = pc === 0 ? 1 : 0.5;
                    ctx.beginPath();
                    ctx.moveTo(0, top + 0.5);
                    ctx.lineTo(w, top + 0.5);
                    ctx.stroke();
                    ctx.lineWidth = 1;
                }
            } else {
                // 非音高参数轴标签：根据当前视口动态计算刷分标记
                const view = paramViews[editParam] ?? { center: 0.5, span: 1 };
                const span = Math.max(1e-6, view.span);
                const vMin = view.center - span / 2;
                const vMax = view.center + span / 2;
                // 若干标尺刻度：选择让屏内展示 3–5 个标记
                const niceStep = niceAxisStep(span, 4);
                const firstMark = Math.ceil(vMin / niceStep) * niceStep;
                ctx.fillStyle = colors.tensionLabel;
                ctx.font = "10px sans-serif";
                ctx.textBaseline = "middle";
                for (
                    let m = firstMark;
                    m <= vMax + niceStep * 0.01;
                    m += niceStep
                ) {
                    const y = valueToY(editParam, m, h);
                    ctx.fillText(formatAxisMark(m), 6, y);
                    ctx.strokeStyle = colors.tensionLine;
                    ctx.lineWidth = 1;
                    ctx.beginPath();
                    ctx.moveTo(0, y + 0.5);
                    ctx.lineTo(w, y + 0.5);
                    ctx.stroke();
                }
            }
        }
    }

    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const { w, h } = viewSize;
    const dpr = Math.max(1, window.devicePixelRatio || 1);
    const cw = Math.max(1, Math.floor(w * dpr));
    const ch = Math.max(1, Math.floor(h * dpr));
    if (canvas.width !== cw || canvas.height !== ch) {
        canvas.width = cw;
        canvas.height = ch;
        canvas.style.width = `${w}px`;
        canvas.style.height = `${h}px`;
    }

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    // 统一用 sec 坐标系：所有 x 坐标 = timeSec * pxPerSec - scrollLeft
    const visibleStartSec = scrollLeft / Math.max(1e-9, pxPerSec);
    const visibleDurSec = w / Math.max(1e-9, pxPerSec);
    // beat 坐标系辅助（仅用于 selection 等仍以 beat 为单位的数据）
    const pxPerBeat = pxPerSec * secPerBeat;

    // Horizontal pitch grid
    if (editParam === "pitch") {
        const absMin = PITCH_MIN_MIDI;
        const absMax = PITCH_MAX_MIDI;
        const view = pitchView;
        const span = clamp(view.span, 1e-6, absMax - absMin);
        const min = clamp(view.center - span / 2, absMin, absMax - span);
        const max = min + span;
        const startMidi = clamp(Math.floor(min), absMin, absMax);
        const endMidi = clamp(Math.ceil(max), absMin, absMax);
        const highlightActive = (() => {
            if (!args.projectScale) return false;
            const mode = args.scaleHighlightMode ?? "off";
            if (mode === "off") return false;
            return mode === "always";
        })();
        const projectScaleNotes = args.projectScale
            ? resolveScaleNotes(args.projectScale)
            : [];

        for (let midi = startMidi; midi <= endMidi; midi += 1) {
            const y = valueToY("pitch", midi + 0.5, h);
            const pc = ((midi % 12) + 12) % 12;
            const isScaleNote = highlightActive
                ? projectScaleNotes.includes(pc)
                : false;

            if (isScaleNote) {
                ctx.strokeStyle = isDark
                    ? "rgba(255,200,80,0.22)"
                    : "rgba(200,120,20,0.22)";
                ctx.lineWidth = 2;
            } else {
                ctx.strokeStyle =
                    pc === 0 ? colors.pitchGridC : colors.pitchGridOther;
                ctx.lineWidth = 1;
            }
            ctx.beginPath();
            ctx.moveTo(0, y + 0.5);
            ctx.lineTo(w, y + 0.5);
            ctx.stroke();
        }
    }

    // 创建离屏 canvas 缓冲（复用，避免每帧创建）
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    const drawPianoRollRef = drawPianoRoll as unknown as {
        _offCanvas?: HTMLCanvasElement;
        _lastLevelByPath?: Record<string, 0 | 1 | 2>;
    };
    // 初始化 previousLevel 跟踪映射（与 WaveformTrackCanvas 的 lastLevelByPathRef 一致）
    if (!drawPianoRollRef._lastLevelByPath) {
        drawPianoRollRef._lastLevelByPath = {};
    }
    const lastLevelByPath = drawPianoRollRef._lastLevelByPath;
    const offCanvas =
        drawPianoRollRef._offCanvas || document.createElement("canvas");
    drawPianoRollRef._offCanvas = offCanvas;
    // 限制 dpr 为 1，与主 canvas 一致
    const offDpr = 1;
    const displayedOffW = Math.max(1, w);
    const displayedOffH = Math.max(1, h);
    const internalOffW = Math.max(1, Math.floor(displayedOffW * offDpr));
    const internalOffH = Math.max(1, Math.floor(displayedOffH * offDpr));
    if (offCanvas.width !== internalOffW) {
        offCanvas.width = internalOffW;
    }
    if (offCanvas.height !== internalOffH) {
        offCanvas.height = internalOffH;
    }
    offCanvas.style.width = `${displayedOffW}px`;
    offCanvas.style.height = `${displayedOffH}px`;
    const offCtx = offCanvas.getContext("2d");
    if (!offCtx) return;

    // Background waveform: per-clip 叠加绘制
    // 与 WaveformTrackCanvas 保持一致的数据路径：
    // waveformMipmapStore.getInterleavedSlice() → applyGainsToPeaks → renderWaveform
    console.info(
        `[PianoRoll] 🎨 波形渲染开始 | clips=${clipPeaks.length} | pxPerSec=${pxPerSec.toFixed(1)} | visible=[${visibleStartSec.toFixed(2)}s, ${(visibleStartSec + visibleDurSec).toFixed(2)}s]`,
    );
    let prRenderedCount = 0;
    let prSkippedCount = 0;

    for (const entry of clipPeaks) {
        if (!entry.sourcePath) continue;

        const pr = entry.playbackRate > 0 ? entry.playbackRate : 1;
        const sourceStartSec = entry.sourceStartSec ?? 0;
        const sourceDurSec = entry.sourceDurationSec;
        if (sourceDurSec <= 0) continue;

        const clipStartSec = entry.startSec;
        const clipEndSec = clipStartSec + entry.lengthSec;
        const clipWidthPx = entry.lengthSec * pxPerSec;
        if (clipWidthPx <= 0) continue;

        // 只渲染当前视口内的片段
        const visStartSec = Math.max(clipStartSec, visibleStartSec);
        const visEndSec = Math.min(clipEndSec, visibleStartSec + visibleDurSec);
        if (visEndSec <= visStartSec) continue;

        const visibleClipStartX = visStartSec * pxPerSec - scrollLeft;
        const visibleClipWidthPx = Math.max(
            1,
            Math.ceil((visEndSec - visStartSec) * pxPerSec),
        );

        // 可见部分在 clip 内的比例 → 映射到源文件时间
        const clipLen = entry.lengthSec;
        const ratioStart = (visStartSec - clipStartSec) / Math.max(1e-6, clipLen);
        const ratioEnd = (visEndSec - clipStartSec) / Math.max(1e-6, clipLen);

        const clipSourceEndSec =
            Number(entry.sourceEndSec ?? sourceDurSec) || sourceDurSec;
        const clipSourceSpanSec = Math.max(
            0,
            Math.min(clipLen * pr, clipSourceEndSec - sourceStartSec),
        );
        const sourceTimeStart = Math.max(0, sourceStartSec + ratioStart * clipSourceSpanSec);
        const sourceTimeEnd = Math.min(sourceDurSec, sourceStartSec + ratioEnd * clipSourceSpanSec);
        const sourceDuration = Math.max(0.001, sourceTimeEnd - sourceTimeStart);

        // 选择 mipmap 级别（与 WaveformTrackCanvas 一致，使用 previousLevel 实现滞后防抖）
        const sampleRate = entry.sourceSampleRate || 44100;
        const spp = Math.max(1, Math.round(sampleRate / pxPerSec));
        const previousLevel = lastLevelByPath[entry.sourcePath];
        const stableLevel = waveformMipmapStore.selectLevelStable(spp, previousLevel);
        lastLevelByPath[entry.sourcePath] = stableLevel;

        const levelLabels = ["L0(div=64)", "L1(div=512)", "L2(div=4096)"];
        const fileName = entry.sourcePath.split(/[/\\]/).pop() ?? entry.sourcePath;

        // 从 mipmap 缓存获取 interleaved 数据
        const result = waveformMipmapStore.getInterleavedSlice(
            entry.sourcePath,
            stableLevel,
            sourceTimeStart,
            sourceDuration,
        );
        if (!result || result.interleaved.length < 4) {
            console.warn(
                `[PianoRoll] ⏭️ clip 跳过: "${fileName}" | level=${levelLabels[stableLevel]} | spp=${spp} | 原因=${!result ? "数据未加载" : "interleaved 太短(" + result.interleaved.length + ")"}`,
            );
            prSkippedCount++;
            continue;
        }

        console.debug(
            `[PianoRoll] ✅ clip 渲染: "${fileName}" | level=${levelLabels[stableLevel]} | spp=${spp} | srcTime=[${sourceTimeStart.toFixed(3)}s, ${sourceTimeEnd.toFixed(3)}s] | interleaved=${result.interleaved.length}`,
        );
        prRenderedCount++;

        // clip 内的像素偏移
        const clipPixelOffset = (visStartSec - clipStartSec) * pxPerSec;

        // 构建渲染参数（与 WaveformTrackCanvas 一致的参数结构）
        const params: WaveformRenderParams = {
            canvasWidth: visibleClipWidthPx,
            canvasHeight: displayedOffH,
            centerY: displayedOffH / 2,
            sourceStartSec,
            clipDuration: entry.lengthSec,
            playbackRate: pr,
            sourceDurationSec: sourceDurSec,
            volumeGain: Number(entry.gain ?? 1) || 1,
            fadeInSec: Number(entry.fadeInSec ?? 0) || 0,
            fadeOutSec: Number(entry.fadeOutSec ?? 0) || 0,
            fadeInCurve: entry.fadeInCurve ?? "linear",
            fadeOutCurve: entry.fadeOutCurve ?? "linear",
            dataStartSec: result.dataStartSec,
            dataDurationSec: result.dataDurationSec,
            clipPixelOffset,
            clipTotalWidthPx: Math.max(1, clipWidthPx),
        };

        // 应用增益（音量 + 淡入淡出）
        const withGains = applyGainsToPeaks(result.interleaved, params);

        // 使用固定尺寸的离屏 canvas，避免缩放过程中频繁重设宽度导致闪烁
        offCtx.setTransform(offDpr, 0, 0, offDpr, 0, 0);
        offCtx.clearRect(0, 0, displayedOffW, displayedOffH);

        // 渲染波形（line 竖线模式）
        renderWaveform(offCtx, withGains, params, waveformColors.stroke, 0.5, "line");

        // 裁剪到 clip 的可视 x 范围，避免溢出到相邻 clip
        ctx.save();
        ctx.beginPath();
        ctx.rect(visibleClipStartX, 0, visibleClipWidthPx, h);
        ctx.clip();

        // 静音 clip 半透明
        ctx.globalAlpha = entry.muted ? 0.2 : 0.5;

        // 使用 drawImage 精确绘制
        ctx.drawImage(
            offCanvas,
            0,
            0,
            Math.max(1, Math.floor(visibleClipWidthPx * offDpr)),
            internalOffH,
            visibleClipStartX,
            0,
            visibleClipWidthPx,
            h,
        );

        ctx.restore();
    }

    console.info(
        `[PianoRoll] 🎨 波形渲染完成 | 渲染=${prRenderedCount} | 跳过=${prSkippedCount} | 总clips=${clipPeaks.length}`,
    );

    // Selection (time band)
    if (selection) {
        const a = Math.min(selection.aBeat, selection.bBeat);
        const b = Math.max(selection.aBeat, selection.bBeat);
        const x0 = a * pxPerBeat - scrollLeft;
        const x1 = b * pxPerBeat - scrollLeft;
        ctx.fillStyle = "rgba(100, 200, 255, 0.08)";
        ctx.fillRect(x0, 0, x1 - x0, h);
        ctx.strokeStyle = "rgba(100, 200, 255, 0.30)";
        ctx.strokeRect(x0 + 0.5, 0.5, Math.max(0, x1 - x0 - 1), h - 1);
    }

    // 若音高分析进行中，跳过曲线绘制（进度条已显示状态）
    if (pitchAnalysisPending) {
        return;
    }

    // 检测音高参考线：在 pitch 模式下，将后端推送的 per-clip 检测曲线渲染为半透明彩色参考线�?
    // 渲染在用户编辑曲线下方，不干扰主曲线的视觉层次�?
    if (
        editParam === "pitch" &&
        detectedPitchCurves &&
        detectedPitchCurves.length > 0
    ) {
        // �?clip 时循环颜色，增强区分�?
        const DETECTED_COLORS = [
            "rgba(80, 220, 180, 0.55)", // 青绿
            "rgba(255, 180, 60, 0.55)", // 橙黄
            "rgba(180, 120, 255, 0.55)", // �?
            "rgba(60, 180, 255, 0.55)", // �?
        ];

        for (let ci = 0; ci < detectedPitchCurves.length; ci++) {
            const curve = detectedPitchCurves[ci];
            if (!curve.midiCurve || curve.midiCurve.length < 2) continue;

            const fp = Math.max(1e-6, curve.framePeriodMs);
            // 曲线起始时间（秒）：直接来自后端，无需帧→秒转换
            const curveStartSec = curve.curveStartSec;

            ctx.save();

            ctx.strokeStyle;
            ctx.strokeStyle = DETECTED_COLORS[ci % DETECTED_COLORS.length];
            ctx.lineWidth = 1.5;
            ctx.setLineDash([]);
            ctx.globalAlpha = 1;

            ctx.beginPath();
            let hasStarted = false;

            for (let i = 0; i < curve.midiCurve.length; i++) {
                const midi = curve.midiCurve[i];
                if (midi == null || !isFinite(midi)) continue;

                // 计算当前帧的时间（秒），统一用 sec 坐标系
                const frameSec = curveStartSec + (i * fp) / 1000;
                const x = frameSec * pxPerSec - scrollLeft;
                // 裁剪到可见区域
                if (x < -10 || x > w + 10) continue;

                // 无声帧（midi <= 0）：跳过，但保持连续性
                if (midi <= 0) {
                    continue;
                }

                // pitch 曲线加 0.5 偏移，使点落在键中心
                const y = valueToY("pitch", midi + 0.5, h);

                if (!hasStarted) {
                    ctx.moveTo(x, y);
                    hasStarted = true;
                } else {
                    ctx.lineTo(x, y);
                }
            }
            ctx.stroke();
            ctx.restore();
        }
    }

    // Curves
    // 副参数曲线（半透明、细线，绘制在主参数曲线下方�?
    if (
        showSecondaryParam &&
        secondaryParamIds.length > 0
    ) {
        const secondaryPalette = [
            "rgba(100, 200, 255, 0.45)",
            "rgba(255, 180, 60, 0.45)",
            "rgba(160, 120, 255, 0.45)",
            "rgba(90, 220, 160, 0.45)",
        ];
        secondaryParamIds.forEach((paramId, index) => {
            const secondaryParamView = secondaryParamViews[paramId];
            if (
                !secondaryParamView ||
                Math.max(
                    secondaryParamView.orig.length,
                    secondaryParamView.edit.length,
                ) < 2
            ) {
                return;
            }
            const secondaryValues = resolveSecondaryOverlayValues({
                orig: secondaryParamView.orig,
                edit: secondaryParamView.edit,
            });
            const secondaryColor =
                paramId === "pitch"
                    ? "rgba(100, 200, 255, 0.45)"
                    : secondaryPalette[index % secondaryPalette.length];
            ctx.save();
            ctx.strokeStyle = secondaryColor;
            ctx.lineWidth = 1.5;
            ctx.setLineDash([]);
            drawCurveTimed({
                ctx,
                values: secondaryValues,
                param: paramId,
                w,
                h,
                startFrame: secondaryParamView.startFrame,
                stride: secondaryParamView.stride,
                framePeriodMs: secondaryParamView.framePeriodMs,
                visibleStartSec,
                visibleDurSec,
                valueToY,
            });
            ctx.restore();
        });
    }

    if (paramView && paramView.orig.length >= 2 && paramView.edit.length >= 2) {
        const editValues =
            liveEditOverride && liveEditOverride.key === paramView.key
                ? liveEditOverride.edit
                : paramView.edit;

        // original (dashed)
        ctx.save();
        ctx.strokeStyle = colors.origCurve;
        ctx.lineWidth = 1.5;
        ctx.setLineDash([6, 6]);
        drawCurveTimed({
            ctx,
            values: paramView.orig,
            param: editParam,
            w,
            h,
            startFrame: paramView.startFrame,
            stride: paramView.stride,
            framePeriodMs: paramView.framePeriodMs,
            visibleStartSec,
            visibleDurSec,
            valueToY,
        });
        ctx.restore();

        // edited (solid)
        ctx.save();
        ctx.strokeStyle = colors.editCurve;
        ctx.lineWidth = 2;
        ctx.setLineDash([]);
        drawCurveTimed({
            ctx,
            values: editValues,
            param: editParam,
            w,
            h,
            startFrame: paramView.startFrame,
            stride: paramView.stride,
            framePeriodMs: paramView.framePeriodMs,
            visibleStartSec,
            visibleDurSec,
            valueToY,
        });
        ctx.restore();

        // 选区内曲线高亮：在选区范围内用亮蓝色加粗重绘编辑曲线
        if (selection) {
            const selMinBeat = Math.min(selection.aBeat, selection.bBeat);
            const selMaxBeat = Math.max(selection.aBeat, selection.bBeat);
            const selX0 = selMinBeat * pxPerBeat - scrollLeft;
            const selX1 = selMaxBeat * pxPerBeat - scrollLeft;

            ctx.save();
            // 裁剪到选区范围
            ctx.beginPath();
            ctx.rect(selX0, 0, selX1 - selX0, h);
            ctx.clip();

            ctx.strokeStyle = colors.selectionCurve;
            ctx.lineWidth = 3;
            ctx.setLineDash([]);
            drawCurveTimed({
                ctx,
                values: editValues,
                param: editParam,
                w,
                h,
                startFrame: paramView.startFrame,
                stride: paramView.stride,
                framePeriodMs: paramView.framePeriodMs,
                visibleStartSec,
                visibleDurSec,
                valueToY,
            });
            ctx.restore();
        }

        // 剪贴板预览曲线：在选区范围内渲染半透明虚线预览
        // 起始点与选区起始点对齐，超出选区的部分直接裁掉（不压缩）
        if (
            clipboardPreview &&
            selection &&
            clipboardPreview.param === editParam &&
            clipboardPreview.values.length > 0
        ) {
            const selMinBeat = Math.min(selection.aBeat, selection.bBeat);
            const selMaxBeat = Math.max(selection.aBeat, selection.bBeat);
            const selStartSec = selMinBeat * secPerBeat;
            const selEndSec = selMaxBeat * secPerBeat;

            const cbFp = Math.max(1e-6, clipboardPreview.framePeriodMs);

            const selX0 = selMinBeat * pxPerBeat - scrollLeft;
            const selX1 = selMaxBeat * pxPerBeat - scrollLeft;

            ctx.save();
            // 裁剪到选区范围
            ctx.beginPath();
            ctx.rect(selX0, 0, selX1 - selX0, h);
            ctx.clip();

            ctx.strokeStyle = isDark
                ? "rgba(255, 180, 60, 0.65)"
                : "rgba(220, 140, 20, 0.65)";
            ctx.lineWidth = 2;
            ctx.setLineDash([4, 4]);
            ctx.beginPath();

            let started = false;
            for (let i = 0; i < clipboardPreview.values.length; i++) {
                // 不缩放，直接按原始帧间距排列
                const tSec = selStartSec + (i * cbFp) / 1000;
                // 超出选区结束点则停止
                if (tSec > selEndSec) break;
                const x = timeToPixel(tSec, visibleStartSec, visibleDurSec, w);
                const rawValue = clipboardPreview.values[i] ?? 0;
                const mappedValue =
                    editParam === "pitch" ? rawValue + 0.5 : rawValue;
                const y = valueToY(editParam, mappedValue, h);
                if (!started) {
                    ctx.moveTo(x, y);
                    started = true;
                } else {
                    ctx.lineTo(x, y);
                }
            }
            ctx.stroke();
            ctx.restore();
        }

        if (paramMorphOverlay) {
            drawParamMorphOverlay({
                ctx,
                overlay: paramMorphOverlay,
                editParam,
                framePeriodMs: paramView.framePeriodMs,
                visibleStartSec,
                visibleDurSec,
                w,
                h,
                valueToY,
                isDark,
            });
        }
    }

    if (overlayText) {
        ctx.save();
        ctx.fillStyle = colors.overlayTextColor;
        ctx.font = "12px sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(overlayText, w / 2, h * 0.88);
        ctx.restore();
    }

    // Playhead（统一用 sec 坐标系）
    const phx = playheadSec * pxPerSec - scrollLeft;
    ctx.strokeStyle = colors.playheadLine;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(phx + 0.5, 0);
    ctx.lineTo(phx + 0.5, h);
    ctx.stroke();
}
