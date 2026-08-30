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

import type { ParamMorphOverlay, ParamName, ParamViewSegment, ValueViewport } from "./types";
import type { ClipPeaksEntry } from "./useClipsPeaksForPianoRoll";
import { clamp } from "../timeline";
import { rasterize } from "../timeline/runtime/canvasRaster";
import {
    durationToWidthPx,
    secToContentPx,
    secToSpanPx,
    secToViewportPx,
    viewportEndSec,
    viewportStartSec,
    type TimelineAxis,
} from "../timeline/runtime/timelineAxis";
import { AXIS_W, PITCH_MAX_MIDI, PITCH_MIN_MIDI } from "./constants";
import { framesToTime } from "./utils";
import { resolveSecondaryOverlayValues } from "./secondaryOverlaySelection";
import {
    applyGainsToPeaks,
    releaseGainBuffer,
    renderWaveform,
    type WaveformRenderParams,
} from "../../../utils/waveformRenderer";
import { waveformMipmapStore } from "../../../utils/waveformMipmapStore";
import { modEuclid, resolvePlaybackWindowSec } from "../../../utils/loopRender";
import { resolveScaleNotes } from "../../../utils/musicalScales";
import type { ScaleLike } from "../../../utils/musicalScales";
import {
    childPitchOffsetValueToDisplay,
    isChildPitchOffsetCentsParam,
    isChildPitchOffsetDegreesParam,
    isChildFormantOffsetCentsParam,
} from "./childPitchOffsetParams";

function isLegacyWaveformRendererEnabled(): boolean {
    return false;
}

/**
 * 返回视觉上固定像素长度的虚线参数，避免随 dpr/缩放产生样式漂移。
 */
function getFixedDashPattern(baseDashPx: number, baseGapPx: number): number[] {
    const dpr = Math.max(1, window.devicePixelRatio || 1);
    const toAlignedCssPx = (v: number) => Math.max(1, Math.round(v * dpr) / dpr);
    return [toAlignedCssPx(baseDashPx), toAlignedCssPx(baseGapPx)];
}

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
function formatAxisMark(v: number, param?: ParamName): string {
    const displayValue = param != null ? childPitchOffsetValueToDisplay(param, v) : v;
    // 最多保留 4 位有效数字，去掉尾随零
    const s = parseFloat(displayValue.toPrecision(4)).toString();
    return s;
}

function isBlackKey(midi: number): boolean {
    const pc = ((midi % 12) + 12) % 12;
    return pc === 1 || pc === 3 || pc === 6 || pc === 8 || pc === 10;
}

function midiToLabel(midi: number): string {
    const NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
    const octave = Math.floor(midi / 12) - 1;
    const name = NOTE_NAMES[((midi % 12) + 12) % 12];
    return `${name}${octave}`;
}

/**
 * 绘制一条参数曲线。
 *
 * 流程：按帧周期把帧号还原成工程时间 → 用统一投影换成视口 x → 逐点连线。
 *
 * 特殊规则：x 坐标**只允许**经 `secToViewportPx(axis, tSec)` 得到。此前这里走
 * 「先除后乘」的 `timeToPixel(t, scrollLeft/p, w/p, w)`，与其余图层的「先乘后减」
 * 在 IEEE754 下不等价，是曲线与网格/播放头错位的来源。二者的等价性由
 * `renderProjection.test.ts` 的 2 万组随机比对守护（相对误差 < 1e-9）。
 */
function drawCurveTimed(args: {
    ctx: CanvasRenderingContext2D;
    values: number[];
    param: ParamName;
    w: number;
    h: number;
    startFrame: number;
    stride: number;
    framePeriodMs: number;
    /** 统一投影：曲线与其它图层的唯一坐标来源。 */
    axis: TimelineAxis;
    valueToY: (param: ParamName, v: number, h: number) => number;
}) {
    const { ctx, values, param, w, h, startFrame, stride, framePeriodMs, axis, valueToY } = args;

    if (values.length < 2) return;
    const fp = Math.max(1e-6, framePeriodMs);
    const step = Math.max(1, Math.floor(stride));
    // 可见区间只用于裁剪；必须由 axis 提供，禁止用 scrollLeft / pxPerSec 还原。
    const visibleStartSec = viewportStartSec(axis);
    const visibleDurSec = viewportEndSec(axis) - visibleStartSec;

    // Check debug flag
    const debugEnabled =
        typeof window !== "undefined" &&
        window.localStorage?.getItem("hifishifter.debugPianoRoll") === "1";

    // DEBUG: 验证曲线时间参数（使用统一转换函数�?
    const curveStartSec = framesToTime(startFrame, fp);
    const curveEndSec = framesToTime(startFrame + (values.length - 1) * step, fp);
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
        if (tSec > visibleStartSec + visibleDurSec) {
            break;
        }
        if (tSec < visibleStartSec) {
            started = false;
            continue;
        }
        const x = secToViewportPx(axis, tSec);

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
                verifyPixel: secToViewportPx(axis, firstPoint.tSec),
            },
            lastPoint: {
                frame: lastPoint.frame,
                tSec: lastPoint.tSec,
                x: lastPoint.x,
                // Verify conversion
                verifyTime: framesToTime(lastPoint.frame, fp),
                verifyPixel: secToViewportPx(axis, lastPoint.tSec),
            },
            pixelSpan: lastPoint.x - firstPoint.x,
            timeSpan: lastPoint.tSec - firstPoint.tSec,
            pxPerSec: (lastPoint.x - firstPoint.x) / (lastPoint.tSec - firstPoint.tSec),
        });
    }

    ctx.stroke();
}

function drawParamMorphOverlay(args: {
    ctx: CanvasRenderingContext2D;
    overlay: ParamMorphOverlay;
    editParam: ParamName;
    framePeriodMs: number;
    /** 统一投影：与曲线、网格、播放头同源。 */
    axis: TimelineAxis;
    h: number;
    valueToY: (param: ParamName, v: number, h: number) => number;
    isDark: boolean;
}) {
    const { ctx, overlay, editParam, framePeriodMs, axis, h, valueToY, isDark } = args;
    const fp = Math.max(1e-6, framePeriodMs);
    const points = overlay.points.slice().sort((a, b) => a.frame - b.frame);
    if (points.length !== 4) return;

    // 变形预览与参数线同向：深=浅色、浅=深色（虚线+半透明填充区分本体）。
    const lineColor = isDark ? "rgba(255, 255, 255, 0.9)" : "rgba(28, 32, 40, 0.9)";
    const fillColor = isDark ? "rgba(255, 255, 255, 0.20)" : "rgba(28, 32, 40, 0.16)";

    const toCanvasX = (frame: number) => {
        const sec = framesToTime(frame, fp);
        return secToViewportPx(axis, sec);
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

export interface ReferencePitchOverlay {
    rootTrackId: string;
    strokeColor: string;
    highlighted: boolean;
    paramView: ParamViewSegment;
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
    /**
     * 统一投影：本函数内**所有**时间↔像素换算的唯一来源。
     * 不再单独接收 pxPerSec / scrollLeft，避免图层各自执行 `t*p - s`。
     */
    axis: TimelineAxis;
    /** 每拍秒数。仅用于 beat↔sec 换算（选区数据以 beat 为单位），不参与投影。 */
    secPerBeat: number;
    playheadSec: number; // 播放头位置（秒）
    pitchAnalysisPending?: boolean;
    waveformColors?: { fill: string; stroke: string };
    referencePitchOverlays?: ReferencePitchOverlay[];
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
    /** Tempo Map 音阶高亮分段（null = 无 Tempo Map 音阶数据，使用单音阶路径）。 */
    scaleSegments?: Array<{
        startSec: number;
        endSec: number;
        scale: ScaleLike | null;
    }> | null;
    toolMode?: string;
    snapToggleHeld?: boolean;
    scaleHighlightMode?: import("../../../features/session/sessionTypes").ScaleHighlightMode;
    paramMorphOverlay?: ParamMorphOverlay | null;
    /** 自定义字体族，用于 canvas 文本渲染 */
    fontFamily?: string;
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
        axis,
        secPerBeat,
        playheadSec,
        pitchAnalysisPending,
        waveformColors = {
            fill: "rgba(255,255,255,0.2)",
            stroke: "rgba(255,255,255,0.5)",
        },
        referencePitchOverlays,
        detectedPitchCurves,
        isDark = true,
        clipboardPreview,
        paramMorphOverlay,
        fontFamily,
    } = args;

    const resolvedFontFamily = fontFamily || "sans-serif";

    // 主题颜色查找表
    const colors = isDark
        ? {
              // 琴键区（白键降亮一档、黑键提亮一档：在深色画布上既不刺眼也不淹没）
              axisBorder: "rgba(255,255,255,0.08)",
              whiteKey: "#d7dade",
              blackKey: "#2e3136",
              blackKeyGradient: "rgba(0,0,0,0.35)",
              cLabel: "#3b82f6",
              whiteKeyLabel: "rgba(60,63,70,0.75)",
              blackKeyLabel: "rgba(220,220,220,0.80)",
              cSeparator: "rgba(100,100,100,0.45)",
              keySeparator: "rgba(160,160,160,0.20)",
              tensionLabel: "rgba(255,255,255,0.55)",
              tensionLine: "rgba(255,255,255,0.10)",
              // 网格线
              pitchGridC: "rgba(255,255,255,0.10)",
              pitchGridOther: "rgba(255,255,255,0.05)",
              // 曲线：参数线深浅随主题（深=白、浅=黑），在色块化的界面里
              // 永远是最清晰的一条；原始音高 = 橄榄黄虚线；选区高亮 = 青蓝。
              origCurve: "rgba(200,164,60,0.70)",
              editCurve: "rgba(255,255,255,0.92)",
              selectionCurve: "rgba(100,200,255,0.95)",
              // 叠加文字 & 播放头（画布中央的操作提示文字，需保持可读：
              // 旧值 35% 不透明度在两套主题下都只剩 1.5-1.8:1）
              overlayTextColor: "rgba(235,240,248,0.45)",
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
              // 曲线：参数线深浅随主题（深=白、浅=黑）。
              origCurve: "rgba(132,104,26,0.80)",
              editCurve: "rgba(28,32,40,0.95)",
              selectionCurve: "rgba(0,116,200,1)",
              // 叠加文字 & 播放头（画布中央的操作提示文字，需保持可读）
              overlayTextColor: "rgba(30,36,48,0.60)",
              playheadLine: "rgba(0,0,0,0.20)",
          };

    // 网格线设备像素对齐：分数 DPR（125%/150%）下 1px CSS 线覆盖 1~2 物理像素，
    // 随落点相位粗细不一。hairline = 1 物理像素、strong = 2 物理像素。
    const dpr = window.devicePixelRatio || 1;
    const hairlineY = (cssY: number): number => (Math.round(cssY * dpr) + 0.5) / dpr;
    const hairlineW = 1 / dpr;
    const strongW = 2 / dpr;

    // Draw axis (left labels)
    if (axisCanvas) {
        const ctx = axisCanvas.getContext("2d");
        if (ctx) {
            const h = viewSize.h;
            const w = AXIS_W;
            // 统一光栅化契约：与时间线画布、波形面共用同一套取整规则。
            const target = rasterize(axisCanvas, w, h, window.devicePixelRatio || 1);
            ctx.setTransform(target.dpr, 0, 0, target.dpr, 0, 0);
            ctx.clearRect(0, 0, target.cssWidthPx, target.cssHeightPx);

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
                const min = clamp(view.center - span / 2, absMin, absMax - span);
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
                        const grad = ctx.createLinearGradient(w * 0.62, 0, w * 0.72, 0);
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
                            ctx.fillStyle = pc === 0 ? colors.cLabel : colors.whiteKeyLabel;
                            ctx.font =
                                pc === 0
                                    ? `bold 9px ${resolvedFontFamily}`
                                    : `9px ${resolvedFontFamily}`;
                            ctx.fillText(midiToLabel(midi), 4, midY);
                        } else {
                            // 黑键：在黑键宽度内裁剪绘制
                            ctx.save();
                            ctx.beginPath();
                            ctx.rect(0, top, w * 0.7, keyH);
                            ctx.clip();
                            ctx.fillStyle = colors.blackKeyLabel;
                            ctx.font = `8px ${resolvedFontFamily}`;
                            ctx.fillText(midiToLabel(midi), 3, midY);
                            ctx.restore();
                        }
                    }

                    // 分隔线：C 音用较深的线，其他用浅线
                    ctx.strokeStyle = pc === 0 ? colors.cSeparator : colors.keySeparator;
                    ctx.lineWidth = pc === 0 ? 1 : 0.5;
                    ctx.beginPath();
                    ctx.moveTo(0, top + 0.5);
                    ctx.lineTo(w, top + 0.5);
                    ctx.stroke();
                    ctx.lineWidth = 1;
                }
            } else {
                // 非音高参数轴标签：对 child-pitch-offset 做特殊处理以配合横线（音分/度数）
                const view = paramViews[editParam] ?? { center: 0.5, span: 1 };
                const span = Math.max(1e-6, view.span);
                const vMin = view.center - span / 2;
                const vMax = view.center + span / 2;
                ctx.fillStyle = colors.tensionLabel;
                ctx.font = `10px ${resolvedFontFamily}`;
                ctx.textBaseline = "middle";

                if (isChildPitchOffsetCentsParam(editParam)) {
                    // 候选步长（以音分为单位），从大到小
                    const range = vMax - vMin;
                    const candidates = [1200, 600, 300, 200, 100, 50, 25, 10, 5, 1];
                    let chosen = candidates[candidates.length - 1];
                    for (const c of candidates) {
                        const count = Math.ceil(range / c) + 1;
                        if (count >= 5 && count <= 12) {
                            chosen = c;
                            break;
                        }
                    }
                    // 退化：若跨度相对较大，回退到更粗的步长以避免过多刻度
                    const approxCount = range / chosen;
                    if (approxCount > 12) {
                        // 使用针对约 8 个刻度的 "好看" 步长作为回退，
                        // 并确保它比当前 chosen 更大；否则尝试下一个更大的候选值。
                        const niceStep = niceAxisStep(range, 8);
                        if (niceStep > chosen) {
                            chosen = niceStep;
                        } else {
                            const largerCandidate = candidates.find((c) => c > chosen);
                            if (largerCandidate !== undefined) {
                                chosen = largerCandidate;
                            }
                        }
                    }

                    const firstMark = Math.ceil(vMin / chosen) * chosen;
                    for (let m = firstMark; m <= vMax + chosen * 0.01; m += chosen) {
                        const y = valueToY(editParam, m, h);
                        const isStrong = Math.round(m) % 1200 === 0;
                        ctx.fillText(formatAxisMark(m, editParam), 6, y);
                        ctx.strokeStyle = isStrong ? colors.tensionLine : colors.tensionLine;
                        ctx.lineWidth = isStrong ? 1.25 : 1;
                        ctx.beginPath();
                        ctx.moveTo(0, y + 0.5);
                        ctx.lineTo(w, y + 0.5);
                        ctx.stroke();
                    }
                } else if (isChildFormantOffsetCentsParam(editParam)) {
                    // 共振峰差使用 cents 单位，强线每 600 cents。
                    const range = vMax - vMin;
                    const candidates = [1200, 600, 300, 200, 100, 50, 25, 10, 5, 1];
                    let chosen = candidates[candidates.length - 1];
                    for (const c of candidates) {
                        const count = Math.ceil(range / c) + 1;
                        if (count >= 5 && count <= 12) {
                            chosen = c;
                            break;
                        }
                    }
                    const firstMark = Math.ceil(vMin / chosen) * chosen;
                    for (let m = firstMark; m <= vMax + chosen * 0.01; m += chosen) {
                        const y = valueToY(editParam, m, h);
                        const isStrong = Math.round(m) % 600 === 0;
                        ctx.fillText(formatAxisMark(m, editParam), 6, y);
                        ctx.strokeStyle = colors.tensionLine;
                        ctx.lineWidth = isStrong ? 1.25 : 1;
                        ctx.beginPath();
                        ctx.moveTo(0, y + 0.5);
                        ctx.lineTo(w, y + 0.5);
                        ctx.stroke();
                    }
                } else if (isChildPitchOffsetDegreesParam(editParam)) {
                    // 度数使用内部 degree-step 单位，强线每 7 个单位
                    const candidates = [14, 7, 3, 1];
                    let chosen = candidates[candidates.length - 1];
                    for (const c of candidates) {
                        const count = Math.ceil((vMax - vMin) / c) + 1;
                        if (count >= 5 && count <= 12) {
                            chosen = c;
                            break;
                        }
                    }
                    const firstMark = Math.ceil(vMin / chosen) * chosen;
                    for (let m = firstMark; m <= vMax + chosen * 0.01; m += chosen) {
                        const y = valueToY(editParam, m, h);
                        const rounded = Math.round(m);
                        const isStrong = rounded % 7 === 0;
                        ctx.fillText(formatAxisMark(m, editParam), 6, y);
                        ctx.strokeStyle = isStrong ? colors.tensionLine : colors.tensionLine;
                        ctx.lineWidth = isStrong ? 1.25 : 1;
                        ctx.beginPath();
                        ctx.moveTo(0, y + 0.5);
                        ctx.lineTo(w, y + 0.5);
                        ctx.stroke();
                    }
                    // 确保 0 的刻度一定显示
                    const y0 = valueToY(editParam, 0, h);
                    ctx.fillText(formatAxisMark(0, editParam), 6, y0);
                } else {
                    // 回退：使用常规的“nice”步长
                    const niceStep = niceAxisStep(span, 4);
                    const firstMark = Math.ceil(vMin / niceStep) * niceStep;
                    for (let m = firstMark; m <= vMax + niceStep * 0.01; m += niceStep) {
                        const y = valueToY(editParam, m, h);
                        ctx.fillText(formatAxisMark(m, editParam), 6, y);
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
    }

    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const { w, h } = viewSize;
    // 统一光栅化契约：此前这里用 Math.floor，而波形面用 Math.round，两者在
    // 半像素 DPR 下会差一整个物理像素；现在全部收敛到 rasterize()。
    const target = rasterize(canvas, w, h, window.devicePixelRatio || 1);
    ctx.setTransform(target.dpr, 0, 0, target.dpr, 0, 0);
    ctx.clearRect(0, 0, target.cssWidthPx, target.cssHeightPx);

    // 所有 x 坐标 = axis.secToViewportPx(sec)，与时间线侧同一实现。
    // 可见区间仅用于裁剪，且只能由 axis 提供：此前这里写作
    // `scrollLeft / pxPerSec`（先除后乘），与其余图层不等价，是错位根源之一。
    const visibleStartSec = viewportStartSec(axis);
    const visibleDurSec = viewportEndSec(axis) - visibleStartSec;
    // beat → sec 的换算系数（选区/剪贴板预览数据仍以 beat 为单位）。
    // 注意：不构造 pxPerBeat —— 像素投影一律走 axis，beat 先转 sec 再投影。
    const beatToSec = Math.max(1e-9, secPerBeat);

    // Horizontal grid lines
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
        const projectScaleNotes = args.projectScale ? resolveScaleNotes(args.projectScale) : [];
        const scaleSegments = args.scaleSegments ?? null;

        for (let midi = startMidi; midi <= endMidi; midi += 1) {
            const y = hairlineY(valueToY("pitch", midi + 0.5, h));
            const pc = ((midi % 12) + 12) % 12;
            const isScaleNote = highlightActive ? projectScaleNotes.includes(pc) : false;

            const normalColor = pc === 0 ? colors.pitchGridC : colors.pitchGridOther;
            ctx.strokeStyle = normalColor;
            ctx.lineWidth = hairlineW;
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(w, y);
            ctx.stroke();

            if (!highlightActive) continue;

            if (scaleSegments && scaleSegments.length > 0) {
                // Tempo Map 路径：按时间段绘制高亮段。
                ctx.strokeStyle = isDark ? "rgba(255,200,80,0.22)" : "rgba(200,120,20,0.22)";
                ctx.lineWidth = 2;
                for (const segment of scaleSegments) {
                    if (!segment.scale) continue;
                    const segmentNotes = resolveScaleNotes(segment.scale);
                    if (!segmentNotes.includes(pc)) continue;
                    const x0 = secToViewportPx(axis, segment.startSec);
                    const x1 = secToViewportPx(axis, segment.endSec);
                    if (x1 < 0 || x0 > w) continue;
                    ctx.beginPath();
                    ctx.moveTo(Math.max(0, x0), y);
                    ctx.lineTo(Math.min(w, x1), y);
                    ctx.stroke();
                }
                continue;
            }

            if (isScaleNote) {
                ctx.strokeStyle = isDark ? "rgba(255,200,80,0.22)" : "rgba(200,120,20,0.22)";
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(w, y);
                ctx.stroke();
            }
        }
    } else if (isChildPitchOffsetCentsParam(editParam)) {
        const view = paramViews[editParam] ?? { center: 0, span: 1 };
        const span = Math.max(1e-6, view.span);
        const vMin = view.center - span / 2;
        const vMax = view.center + span / 2;
        const step = 100;
        const start = Math.ceil(vMin / step) * step;

        for (let v = start; v <= vMax + step * 0.01; v += step) {
            const isStrong = Math.round(v) % 1200 === 0;
            const y = isStrong
                ? Math.round(valueToY(editParam, v, h) * dpr) / dpr
                : hairlineY(valueToY(editParam, v, h));
            ctx.strokeStyle = isStrong
                ? isDark
                    ? "rgba(255,255,255,0.14)"
                    : "rgba(0,0,0,0.16)"
                : isDark
                  ? "rgba(255,255,255,0.07)"
                  : "rgba(0,0,0,0.08)";
            ctx.lineWidth = isStrong ? strongW : hairlineW;
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(w, y);
            ctx.stroke();
        }
    } else if (isChildPitchOffsetDegreesParam(editParam)) {
        const view = paramViews[editParam] ?? { center: 0, span: 1 };
        const span = Math.max(1e-6, view.span);
        const vMin = view.center - span / 2;
        const vMax = view.center + span / 2;
        const step = 1;
        const start = Math.ceil(vMin / step) * step;

        for (let v = start; v <= vMax + step * 0.01; v += step) {
            const rounded = Math.round(v);
            const isStrong = rounded % 7 === 0;
            const y = isStrong
                ? Math.round(valueToY(editParam, v, h) * dpr) / dpr
                : hairlineY(valueToY(editParam, v, h));
            ctx.strokeStyle = isStrong
                ? isDark
                    ? "rgba(255,255,255,0.14)"
                    : "rgba(0,0,0,0.16)"
                : isDark
                  ? "rgba(255,255,255,0.07)"
                  : "rgba(0,0,0,0.08)";
            ctx.lineWidth = isStrong ? strongW : hairlineW;
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(w, y);
            ctx.stroke();
        }
    } else if (isChildFormantOffsetCentsParam(editParam)) {
        const view = paramViews[editParam] ?? { center: 0, span: 1 };
        const span = Math.max(1e-6, view.span);
        const vMin = view.center - span / 2;
        const vMax = view.center + span / 2;
        const step = 50;
        const start = Math.ceil(vMin / step) * step;

        for (let v = start; v <= vMax + step * 0.01; v += step) {
            const rounded = Math.round(v);
            const isStrong = rounded % 600 === 0;
            const y = isStrong
                ? Math.round(valueToY(editParam, v, h) * dpr) / dpr
                : hairlineY(valueToY(editParam, v, h));
            ctx.strokeStyle = isStrong
                ? isDark
                    ? "rgba(255,255,255,0.14)"
                    : "rgba(0,0,0,0.16)"
                : isDark
                  ? "rgba(255,255,255,0.07)"
                  : "rgba(0,0,0,0.08)";
            ctx.lineWidth = isStrong ? strongW : hairlineW;
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(w, y);
            ctx.stroke();
        }
    }

    // Shared WaveformSurface now owns background waveform rendering. Keep this
    // legacy block unreachable until its remaining helper imports are removed.
    if (isLegacyWaveformRendererEnabled()) {
        // ========================================
        // 废弃离屏 Canvas，保留 mipmap 级数状态即可
        // ========================================
        const drawPianoRollRef = drawPianoRoll as unknown as {
            _lastLevelByClip?: Record<string, 0 | 1 | 2>;
        };
        if (!drawPianoRollRef._lastLevelByClip) {
            drawPianoRollRef._lastLevelByClip = {};
        }
        const lastLevelByClip = drawPianoRollRef._lastLevelByClip;

        // 级别提示键清理：该 map 挂在模块级函数属性上（跨卸载存活），
        // 已删除/不可见 clip 的 `${path}::${clipId}` 键若不清理会无限累积。
        {
            const liveKeys = new Set<string>();
            for (const entry of clipPeaks) {
                if (entry.sourcePath) liveKeys.add(`${entry.sourcePath}::${entry.clipId}`);
            }
            for (const key of Object.keys(lastLevelByClip)) {
                if (!liveKeys.has(key)) delete lastLevelByClip[key];
            }
        }

        // Background waveform: per-clip 叠加绘制
        // 与 WaveformTrackCanvas 保持一致的数据路径：
        // waveformMipmapStore.getInterleavedSlice() → applyGainsToPeaks → renderWaveform
        for (const entry of clipPeaks) {
            if (!entry.sourcePath) continue;
            if (entry.muted) continue;

            const pr = entry.playbackRate > 0 ? entry.playbackRate : 1;
            const sourceStartSec = entry.sourceStartSec ?? 0;
            const sourceDurSec = entry.sourceDurationSec;
            if (sourceDurSec <= 0) continue;

            const clipStartSec = entry.startSec;
            const clipEndSec = clipStartSec + entry.lengthSec;
            const clipWidthPx = secToSpanPx(axis, entry.lengthSec);
            if (clipWidthPx <= 0) continue;

            // 只渲染当前视口内的片段
            const visStartSec = Math.max(clipStartSec, visibleStartSec);
            const visEndSec = Math.min(clipEndSec, visibleStartSec + visibleDurSec);
            if (visEndSec <= visStartSec) continue;

            const viewportStartPx = Math.round(axis.scrollLeftPx);
            const clipStartPx = Math.round(secToContentPx(axis, clipStartSec));

            const isLoop = Boolean(entry.loopEnabled);
            const mediaDurPiano = Math.max(0, Number(entry.sourceDurationSec) || 0);
            const storedSourceEndSec = Number(entry.sourceEndSec ?? sourceDurSec) || sourceDurSec;
            // 消费窗口模型（与后端 clip_playback_window_sec / WaveformTrackCanvas
            // 一致）：正放 win=[ss, ss+len·r)、倒放 win=[se−len·r, se)。倒放的
            // sourceStartSec 不参与取窗 —— 否则延伸/trim 写入的域外锚点会让波形
            // 与音频错位。
            const { winStartSec, winEndSec } = resolvePlaybackWindowSec({
                loopEnabled: isLoop,
                reversed: Boolean(entry.reversed),
                sourceStartSec,
                playbackRate: pr,
                lengthSec: entry.lengthSec,
                sourceEndSec: storedSourceEndSec,
            });
            const effSrcEndPiano = Math.min(winEndSec, mediaDurPiano || winEndSec);
            let clipSourceSpanSec: number;
            if (!isLoop) {
                // 非 Loop：窗口宽度恒为 len·r（域外为静音，无数据 → 空白）。
                clipSourceSpanSec = Math.max(0, winEndSec - winStartSec);
            } else if (mediaDurPiano > 1e-6) {
                // Loop（循环源）：回绕发生在整个媒体文件上，音频只由锚点与媒体
                // 时长决定 —— split 等编辑会产生 sourceStart > sourceEnd 的"环绕
                // 窗口"，可用性只取决于媒体时长本身（否则波形会整体消失）。
                clipSourceSpanSec = mediaDurPiano;
            } else {
                // 无有效媒体时长：退化为窗口跨度
                clipSourceSpanSec = Math.max(
                    0,
                    Math.min(winEndSec, sourceDurSec || winEndSec) - winStartSec,
                );
            }
            if (clipSourceSpanSec <= 0) continue;

            // ── 循环分段（Loop = 循环原始音频文件，与 WaveformTrackCanvas 一致）──
            // 语义：正放 src(t)=mod(sourceStart+t·pr, D)、倒放 src(t)=mod(sourceEnd−t·pr, D)，
            // D = 完整媒体时长。分段 = 头部进入段 + 整文件重复段；
            // 每段携带自己的源窗口，clipDuration === 源跨度/pr（倒放镜像成立），
            // 超出 clip 长度的部分仅通过绘制矩形裁掉。
            // 淡入淡出按 clip 局部时间求值（每段携带完整淡化参数）。
            interface PianoRollTile {
                localStartSec: number;
                durationSec: number;
                srcWinStart: number;
                srcWinEnd: number;
            }
            const tiles: PianoRollTile[] = [];
            if (!isLoop) {
                tiles.push({
                    localStartSec: 0,
                    durationSec: entry.lengthSec,
                    srcWinStart: winStartSec,
                    srcWinEnd: winEndSec,
                });
            } else if (!(mediaDurPiano > 1e-6)) {
                // 无有效媒体时长：退化为单片近似
                tiles.push({
                    localStartSec: 0,
                    durationSec: entry.lengthSec,
                    srcWinStart: winStartSec,
                    srcWinEnd: winEndSec,
                });
            } else {
                // 锚点用 floor_mod 归一化（与引擎 mod(anchor ± t·pr, D) 一致，
                // 与 WaveformTrackCanvas 相同）—— 负 / 超界存储锚点正确环绕。
                const anchorFwd = modEuclid(sourceStartSec, mediaDurPiano);
                const anchorRev = modEuclid(effSrcEndPiano, mediaDurPiano);
                const headDur = (entry.reversed ? anchorRev : mediaDurPiano - anchorFwd) / pr;
                const bodyDur = mediaDurPiano / pr;
                const visLocalStart = Math.max(0, visStartSec - clipStartSec);
                const visLocalEnd = Math.min(entry.lengthSec, visEndSec - clipStartSec);
                // 分段数按【可见区间】估算（与 WaveformTrackCanvas 一致）——
                // 长循环 clip 不会落入"单片拉伸近似"。
                // 首个重复段取**包含视口左缘**的那一段（floor），避免左缘空隙。
                const firstBodyIndex = Math.max(
                    0,
                    Math.floor((visLocalStart - headDur - 1e-9) / bodyDur),
                );
                const approxCount =
                    2 + Math.ceil((visLocalEnd - Math.max(headDur, visLocalStart)) / bodyDur);
                if (visLocalEnd > visLocalStart && approxCount <= 4096) {
                    if (headDur > 1e-9 && visLocalStart < headDur) {
                        tiles.push({
                            localStartSec: 0,
                            durationSec: headDur,
                            srcWinStart: entry.reversed ? 0 : anchorFwd,
                            srcWinEnd: entry.reversed ? anchorRev : mediaDurPiano,
                        });
                    }
                    let segOffset = headDur + firstBodyIndex * bodyDur;
                    for (
                        let guard = 0;
                        segOffset < visLocalEnd - 1e-9 && guard < 4096;
                        guard += 1
                    ) {
                        tiles.push({
                            localStartSec: segOffset,
                            durationSec: bodyDur,
                            srcWinStart: 0,
                            srcWinEnd: mediaDurPiano,
                        });
                        segOffset += bodyDur;
                    }
                } else {
                    // 退化保护：单片近似。
                    // 用"进入段"窗口（锚点 → 媒体末端/起点）近似，
                    // 避免按 [start, start+span] 取到越界源区间。
                    tiles.push({
                        localStartSec: 0,
                        durationSec: entry.lengthSec,
                        srcWinStart: entry.reversed ? 0 : anchorFwd,
                        srcWinEnd: entry.reversed ? anchorRev : mediaDurPiano,
                    });
                }
                if (tiles.length === 0) continue;
            }

            // ── 同窗口切片缓存 ─────────────────────────────────────────────
            // 单条目切片缓存：窗口参数相同的相邻瓦片（或重绘间未变化的窗口）
            // 直接复用上一次切片，缓存持有 store buffer 直到换窗或本 clip
            // 结束（与 WaveformTrackCanvas 保持一致）。
            let fetchCacheKey: string | null = null;
            let fetchCacheResult: {
                interleaved: Float32Array;
                dataStartSec: number;
                dataDurationSec: number;
            } | null = null;
            const releaseFetchCache = () => {
                if (fetchCacheResult) {
                    waveformMipmapStore.releaseInterleaved(fetchCacheResult.interleaved);
                    fetchCacheResult = null;
                }
                fetchCacheKey = null;
            };

            // 选择 mipmap 级别（与 WaveformTrackCanvas 一致，使用 previousLevel
            // 实现滞后防抖）。级别只依赖 pxPerSec 与采样率，对本 clip 的所有
            // 瓦片都相同 —— 移到循环外，避免每瓦片重复计算与写回。
            const sampleRate = entry.sourceSampleRate || 44100;
            // 采样密度（每像素采样数）依赖缩放，但**不是**坐标投影：
            // 它不产生任何 x 坐标，只是挑选 mipmap 等级，因此直接读 axis 的
            // 缩放标量是安全的（坐标一律走 secToViewportPx / secToContentPx）。
            const spp = Math.max(1, Math.round(sampleRate / axis.pxPerSec));
            const levelKey = `${entry.sourcePath}::${entry.clipId}`;
            const previousLevel = lastLevelByClip[levelKey];
            const stableLevel = waveformMipmapStore.selectLevelStable(spp, previousLevel);
            lastLevelByClip[levelKey] = stableLevel;

            // 边缘外扩（与 WaveformTrackCanvas 相同的公式）：保证像素列插值
            // 在瓦片可见边界处不缺数据。
            const sourcePadSecPiano = Math.max(0.005, (2 / Math.max(1, axis.pxPerSec)) * pr);

            for (const tile of tiles) {
                const tileLocalEndSec = tile.localStartSec + tile.durationSec;
                const visLocalStart = Math.max(tile.localStartSec, visStartSec - clipStartSec);
                const visLocalEnd = Math.min(tileLocalEndSec, visEndSec - clipStartSec);
                if (visLocalEnd <= visLocalStart) continue;

                // 该分段自己的源窗口（头部进入段 / 整文件重复段）。
                // 只请求当前可见部分对应的源数据 —— 此前整窗取数（正放/倒放重复
                // 段即整个媒体 [0, D]），长媒体的每个循环瓦片都要对全量 peaks 跑
                // applyGains/renderWaveform 的索引换算，开销随媒体时长线性放大；
                // 与 WaveformTrackCanvas 一致地取"瓦片 ∩ 视口"后，成本只与
                // 可见像素相关。renderWaveform 依据 dataStartSec/dataDurationSec
                // 把部分数据映射回正确的屏幕位置，绘制结果不变。
                const tileSpanStartSec = tile.srcWinStart;
                const tileSpanEndSec = tile.srcWinEnd;
                const sourceVisStartSec = entry.reversed
                    ? tileSpanEndSec - (visLocalEnd - tile.localStartSec) * pr
                    : tileSpanStartSec + (visLocalStart - tile.localStartSec) * pr;
                const sourceVisEndSec = entry.reversed
                    ? tileSpanEndSec - (visLocalStart - tile.localStartSec) * pr
                    : tileSpanStartSec + (visLocalEnd - tile.localStartSec) * pr;
                // 取数范围 clamp 到媒体 [0, mediaDurPiano]：消费窗口（尤其倒放
                // 延伸后的 [se−len·r, se]）可越出媒体域，缺失区间无数据、自然
                // 渲染为空白 —— 与音频的静音表达一致。
                const sourceTimeStart = Math.max(
                    0,
                    tileSpanStartSec,
                    Math.min(sourceVisStartSec, sourceVisEndSec) - sourcePadSecPiano,
                );
                const sourceTimeEnd = Math.min(
                    mediaDurPiano,
                    tileSpanEndSec,
                    Math.max(sourceVisStartSec, sourceVisEndSec) + sourcePadSecPiano,
                );
                // 可见区与媒体域无交集（纯静音段）：跳过取数与绘制，防止退化
                // 请求把媒体开头的 1ms 数据错误映射进静音区。
                if (!(sourceTimeEnd > sourceTimeStart + 1e-9)) {
                    releaseFetchCache();
                    continue;
                }
                const sourceDuration = Math.max(0.001, sourceTimeEnd - sourceTimeStart);

                // 从 mipmap 缓存获取 interleaved 数据（相同源窗口的瓦片复用同一切片）
                const fetchKey = `${sourceTimeStart}|${sourceDuration}`;
                if (!fetchCacheResult || fetchKey !== fetchCacheKey) {
                    releaseFetchCache();
                    fetchCacheKey = fetchKey;
                    fetchCacheResult = waveformMipmapStore.getInterleavedSlice(
                        entry.sourcePath,
                        stableLevel,
                        sourceTimeStart,
                        sourceDuration,
                    );
                }
                const result = fetchCacheResult;
                if (!result || result.interleaved.length < 4) {
                    releaseFetchCache();
                    continue;
                }

                // 瓦片在画布上的可见像素范围
                const tileVisLeft =
                    Math.round(secToContentPx(axis, clipStartSec + visLocalStart)) -
                    viewportStartPx;
                const tileVisRight =
                    Math.round(secToContentPx(axis, clipStartSec + visLocalEnd)) -
                    viewportStartPx;
                if (tileVisRight <= tileVisLeft) {
                    continue;
                }

                // clipPixelOffset = canvas 左边缘对应的瓦片局部像素：
                // renderWaveform 内部 screenX = globalTilePx − clipPixelOffset。
                // 与 WaveformTrackCanvas 一致量化到半像素，消除大浮点数相减的
                // 子像素漂移。
                const tileStartTimelinePx =
                    clipStartPx + secToSpanPx(axis, tile.localStartSec);
                const clipPixelOffset = Math.round((viewportStartPx - tileStartTimelinePx) * 2) / 2;

                const effectiveFadeInPiano =
                    Number(entry.autoFadeInSec ?? 0) > 0
                        ? Number(entry.autoFadeInSec) || 0
                        : Number(entry.fadeInSec ?? 0) || 0;
                const effectiveFadeOutPiano =
                    Number(entry.autoFadeOutSec ?? 0) > 0
                        ? Number(entry.autoFadeOutSec) || 0
                        : Number(entry.fadeOutSec ?? 0) || 0;

                // 构建渲染参数（以单个循环分段为坐标系；
                // sourceStart + clipDuration·rate === 该段源窗口终点，倒放镜像成立）
                const params: WaveformRenderParams = {
                    canvasWidth: w,
                    canvasHeight: h,
                    centerY: h / 2,
                    zeroDbHalfHeight: h / 2,
                    sourceStartSec: tile.srcWinStart,
                    clipDuration: tile.durationSec,
                    playbackRate: pr,
                    reversed: entry.reversed,
                    sourceDurationSec: sourceDurSec,
                    volumeGain: Number(entry.gain ?? 1) || 1,
                    // 每个分段都携带完整淡化参数（增益按 clip 局部时间求值），
                    // 长于一个周期的淡化横跨多段时包络保持连续。
                    fadeInSec: effectiveFadeInPiano,
                    fadeOutSec: effectiveFadeOutPiano,
                    fadeInShape: Number.isFinite(entry.fadeInShape) ? entry.fadeInShape : 0,
                    fadeInDir: entry.fadeInDir ?? 0,
                    fadeOutShape: Number.isFinite(entry.fadeOutShape) ? entry.fadeOutShape : 0,
                    fadeOutDir: entry.fadeOutDir ?? 0,
                    dataStartSec: result.dataStartSec,
                    dataDurationSec: result.dataDurationSec,
                    clipTimeOffsetSec: isLoop ? tile.localStartSec : 0,
                    clipTotalDurationSec: entry.lengthSec,
                    clipPixelOffset,
                    // 与 clip 体画布共用同一宽度下限（durationToWidthPx），
                    // 否则极小瓦片处波形与 clip 体宽度会分叉。
                    clipTotalWidthPx: durationToWidthPx(axis, tile.durationSec),
                };

                // 应用增益（音量 + 淡入淡出）
                const withGains = applyGainsToPeaks(result.interleaved, params);

                ctx.save();
                // beginPath 必须先于 rect：canvas 路径不受 save/restore 管理，
                // 缺失会让 rect 永久累积（clip 区域 = 历史所有矩形并集，
                // 跨瓦片/跨 clip 渗透，且每帧路径增长造成渐进卡顿）。
                ctx.beginPath();
                ctx.rect(tileVisLeft, 0, tileVisRight - tileVisLeft, h);
                ctx.clip();

                // 静音 clip 半透明
                ctx.globalAlpha = entry.muted ? 0.3 : 0.86;
                renderWaveform(ctx, withGains, params, waveformColors.stroke, 0.5, "line");

                ctx.restore();
                if (withGains !== result.interleaved) {
                    releaseGainBuffer(withGains);
                }
                // store 复用池 buffer 由 fetchCache 统一持有/归还。
            }
            releaseFetchCache();
        }
    }

    // Selection (time band)
    if (selection) {
        const a = Math.min(selection.aBeat, selection.bBeat);
        const b = Math.max(selection.aBeat, selection.bBeat);
        // 选区数据是 beat 单位：先转 sec 再统一投影，不构造 pxPerBeat。
        const x0 = secToViewportPx(axis, a * beatToSec);
        const x1 = secToViewportPx(axis, b * beatToSec);
        ctx.fillStyle = "rgba(100, 200, 255, 0.08)";
        ctx.fillRect(x0, 0, x1 - x0, h);
        ctx.strokeStyle = "rgba(100, 200, 255, 0.30)";
        ctx.strokeRect(x0 + 0.5, 0.5, Math.max(0, x1 - x0 - 1), h - 1);
    }

    // 若音高分析进行中，跳过曲线绘制（进度条已显示状态）
    if (pitchAnalysisPending) {
        return;
    }

    if (editParam === "pitch" && referencePitchOverlays && referencePitchOverlays.length > 0) {
        referencePitchOverlays.forEach((overlay) => {
            const values = resolveSecondaryOverlayValues({
                orig: overlay.paramView.orig,
                edit: overlay.paramView.edit,
            });
            if (values.length < 2) return;
            ctx.save();
            ctx.strokeStyle = overlay.strokeColor;
            ctx.lineWidth = overlay.highlighted ? 3.2 : 2.6;
            ctx.setLineDash([]);
            drawCurveTimed({
                ctx,
                values,
                param: "pitch",
                w,
                h,
                startFrame: overlay.paramView.startFrame,
                stride: overlay.paramView.stride,
                framePeriodMs: overlay.paramView.framePeriodMs,
                axis,
                valueToY,
            });
            ctx.restore();
        });
    }

    // 检测音高参考线：在 pitch 模式下，将后端推送的 per-clip 检测曲线渲染为半透明彩色参考线�?
    // 渲染在用户编辑曲线下方，不干扰主曲线的视觉层次�?
    if (editParam === "pitch" && detectedPitchCurves && detectedPitchCurves.length > 0) {
        // �?clip 时循环颜色，增强区分�?
        // 候选曲线色板：按主题给两套 —— 浅色主题提高不透明度并加深，
        // 否则在白底上几乎隐形（旧版青绿在白底仅 ~1.4:1）。
        // 橙黄一员改为玫红：琥珀色现在是编辑包络线的专属色相，避免混淆。
        const DETECTED_COLORS = isDark
            ? [
                  "rgba(80, 220, 180, 0.56)", // 青绿
                  "rgba(255, 110, 197, 0.60)", // 玫红
                  "rgba(180, 120, 255, 0.56)", // 紫色
                  "rgba(60, 180, 255, 0.56)", // 天蓝
              ]
            : [
                  "rgba(0, 150, 118, 0.80)", // 青绿
                  "rgba(214, 44, 140, 0.75)", // 玫红
                  "rgba(124, 58, 237, 0.70)", // 紫色
                  "rgba(2, 132, 199, 0.80)", // 天蓝
              ];

        for (let ci = 0; ci < detectedPitchCurves.length; ci++) {
            const curve = detectedPitchCurves[ci];
            if (!curve.midiCurve || curve.midiCurve.length < 2) continue;

            const fp = Math.max(1e-6, curve.framePeriodMs);
            // 曲线起始时间（秒）：直接来自后端，无需帧→秒转换
            const curveStartSec = curve.curveStartSec;

            ctx.save();
            ctx.strokeStyle = DETECTED_COLORS[ci % DETECTED_COLORS.length];
            ctx.lineWidth = 2;
            ctx.setLineDash([]);
            ctx.globalAlpha = 1;

            ctx.beginPath();
            let hasStarted = false;

            for (let i = 0; i < curve.midiCurve.length; i++) {
                const midi = curve.midiCurve[i];
                if (midi == null || !isFinite(midi)) continue;

                // 计算当前帧的时间（秒），统一用 sec 坐标系
                const frameSec = curveStartSec + (i * fp) / 1000;
                const x = secToViewportPx(axis, frameSec);

                if (x > w + 10) break;

                // 裁剪左侧不可见区域
                if (x < -10) continue;

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
    if (showSecondaryParam && secondaryParamIds.length > 0) {
        // 副参数曲线调色板：按主题给两套（浅色主题加深加浓，否则在白底上发飘）；
        // 琥珀成员换成玫红 —— 琥珀是编辑包络线的专属色相，避免撞色。
        const secondaryPalette = isDark
            ? [
                  "rgba(100, 200, 255, 0.62)",
                  "rgba(255, 110, 197, 0.62)",
                  "rgba(160, 120, 255, 0.62)",
                  "rgba(90, 220, 160, 0.62)",
              ]
            : [
                  "rgba(2, 132, 199, 0.75)",
                  "rgba(214, 44, 140, 0.72)",
                  "rgba(124, 58, 237, 0.72)",
                  "rgba(22, 163, 116, 0.75)",
              ];
        secondaryParamIds.forEach((paramId, index) => {
            const secondaryParamView = secondaryParamViews[paramId];
            if (
                !secondaryParamView ||
                Math.max(secondaryParamView.orig.length, secondaryParamView.edit.length) < 2
            ) {
                return;
            }
            const secondaryValues = resolveSecondaryOverlayValues({
                orig: secondaryParamView.orig,
                edit: secondaryParamView.edit,
            });
            const secondaryColor =
                paramId === "pitch"
                    ? "rgba(100, 200, 255, 0.65)"
                    : secondaryPalette[index % secondaryPalette.length];
            ctx.save();
            ctx.strokeStyle = secondaryColor;
            ctx.lineWidth = 2;
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
                axis,
                valueToY,
            });
            ctx.restore();
        });
    }

    if (paramView) {
        const editValues =
            liveEditOverride && liveEditOverride.key === paramView.key
                ? liveEditOverride.edit
                : paramView.edit;

        if (paramView.orig.length >= 2) {
            // original (dashed)
            ctx.save();
            ctx.strokeStyle = colors.origCurve;
            ctx.lineWidth = 1.8;
            ctx.setLineDash(getFixedDashPattern(6, 6));
            drawCurveTimed({
                ctx,
                values: paramView.orig,
                param: editParam,
                w,
                h,
                startFrame: paramView.startFrame,
                stride: paramView.stride,
                framePeriodMs: paramView.framePeriodMs,
                axis,
                valueToY,
            });
            ctx.restore();
        }

        if (editValues.length >= 2) {
            // edited (solid)
            ctx.save();
            ctx.strokeStyle = colors.editCurve;
            ctx.lineWidth = 2.6;
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
                axis,
                valueToY,
            });
            ctx.restore();
        }

        // 选区内曲线高亮：在选区范围内用亮蓝色加粗重绘编辑曲线
        if (selection && editValues.length >= 2) {
            const selMinBeat = Math.min(selection.aBeat, selection.bBeat);
            const selMaxBeat = Math.max(selection.aBeat, selection.bBeat);
            const selX0 = secToViewportPx(axis, selMinBeat * beatToSec);
            const selX1 = secToViewportPx(axis, selMaxBeat * beatToSec);

            ctx.save();
            // 裁剪到选区范围
            ctx.beginPath();
            ctx.rect(selX0, 0, selX1 - selX0, h);
            ctx.clip();

            ctx.strokeStyle = colors.selectionCurve;
            ctx.lineWidth = 3.6;
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
                axis,
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
            const selStartSec = selMinBeat * beatToSec;
            const selEndSec = selMaxBeat * beatToSec;

            const cbFp = Math.max(1e-6, clipboardPreview.framePeriodMs);

            const selX0 = secToViewportPx(axis, selStartSec);
            const selX1 = secToViewportPx(axis, selEndSec);

            ctx.save();
            // 裁剪到选区范围
            ctx.beginPath();
            ctx.rect(selX0, 0, selX1 - selX0, h);
            ctx.clip();

            // 剪贴板预览与选区高亮同用青蓝色相（虚线+降不透明度区分），
            // 不再占用琥珀色相 —— 琥珀属于编辑包络线本体。
            ctx.strokeStyle = isDark ? "rgba(100, 200, 255, 0.55)" : "rgba(0, 116, 200, 0.60)";
            ctx.lineWidth = 2;
            ctx.setLineDash(getFixedDashPattern(4, 4));
            ctx.beginPath();

            let started = false;
            for (let i = 0; i < clipboardPreview.values.length; i++) {
                // 不缩放，直接按原始帧间距排列
                const tSec = selStartSec + (i * cbFp) / 1000;
                // 超出选区结束点则停止
                if (tSec > selEndSec) break;
                const x = secToViewportPx(axis, tSec);
                const rawValue = clipboardPreview.values[i] ?? 0;
                const mappedValue = editParam === "pitch" ? rawValue + 0.5 : rawValue;
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
                axis,
                h,
                valueToY,
                isDark,
            });
        }
    }

    if (overlayText) {
        ctx.save();
        ctx.fillStyle = colors.overlayTextColor;
        ctx.font = `12px ${resolvedFontFamily}`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(overlayText, w / 2, h * 0.88);
        ctx.restore();
    }

    // Playhead（统一用 sec 坐标系）
    const phx = secToViewportPx(axis, playheadSec);
    ctx.strokeStyle = colors.playheadLine;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(phx + 0.5, 0);
    ctx.lineTo(phx + 0.5, h);
    ctx.stroke();
}
