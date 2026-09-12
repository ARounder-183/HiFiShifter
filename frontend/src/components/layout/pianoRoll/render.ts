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
import { resolvePianoRollColors } from "./colors";
import { clamp } from "../timeline";
import { clearCanvasPhysical, rasterize } from "../timeline/runtime/canvasRaster";
import {
    secToViewportPx,
    strokePx,
    viewportEndSec,
    viewportStartSec,
    type TimelineAxis,
} from "../timeline/runtime/timelineAxis";
import { wholeDevicePxLength } from "../../../utils/devicePixelLine";
import { AXIS_W, PITCH_MAX_MIDI, PITCH_MIN_MIDI } from "./constants";
import { framesToTime } from "./utils";
import { resolveSecondaryOverlayValues } from "./secondaryOverlaySelection";
import { resolveScaleNotes } from "../../../utils/musicalScales";
import type { ScaleLike } from "../../../utils/musicalScales";
import {
    childPitchOffsetValueToDisplay,
    isChildPitchOffsetCentsParam,
    isChildPitchOffsetDegreesParam,
    isChildFormantOffsetCentsParam,
} from "./childPitchOffsetParams";

// 调试开关缓存：drawCurveTimed 每帧会被调用多次（主曲线 / 编辑线 / 选区叠加 /
// 每条副参数 / 每条参考线），逐次同步读 localStorage 会拖慢绘制热路径；
// 1 秒 TTL 足以跟踪开关变化。
let debugFlagCache = { value: false, checkedAt: 0 };
function isPianoRollDebugEnabled(): boolean {
    const now = Date.now();
    if (now - debugFlagCache.checkedAt > 1000) {
        debugFlagCache = {
            value:
                typeof window !== "undefined" &&
                window.localStorage?.getItem("hifishifter.debugPianoRoll") === "1",
            checkedAt: now,
        };
    }
    return debugFlagCache.value;
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

    // DEBUG: 验证曲线时间参数（使用统一转换函数）
    const debugEnabled = isPianoRollDebugEnabled();
    const curveStartSec = debugEnabled ? framesToTime(startFrame, fp) : 0;
    const curveEndSec = debugEnabled
        ? framesToTime(startFrame + (values.length - 1) * step, fp)
        : 0;
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
    // 调试点跟踪用标量而非对象：每帧数千次迭代的循环里逐点分配对象
    // 是纯浪费，仅在 debugEnabled 时才进入日志输出。
    let firstFrame = -1;
    let firstTSec = 0;
    let firstX = 0;
    let lastFrame = 0;
    let lastTSec = 0;
    let lastX = 0;

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

        if (firstFrame === -1) {
            firstFrame = frame;
            firstTSec = tSec;
            firstX = x;
        }
        lastFrame = frame;
        lastTSec = tSec;
        lastX = x;

        // pitch 曲线：MIDI 值 N 应绘制在 N 键中心（N 到 N+1 区间的中点），加 0.5 偏移
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
    if (debugEnabled && firstFrame !== -1) {
        console.log("[drawCurveTimed] Rendered points:", {
            param,
            firstPoint: {
                frame: firstFrame,
                tSec: firstTSec,
                x: firstX,
                // Verify conversion
                verifyTime: framesToTime(firstFrame, fp),
                verifyPixel: secToViewportPx(axis, firstTSec),
            },
            lastPoint: {
                frame: lastFrame,
                tSec: lastTSec,
                x: lastX,
                // Verify conversion
                verifyTime: framesToTime(lastFrame, fp),
                verifyPixel: secToViewportPx(axis, lastTSec),
            },
            pixelSpan: lastX - firstX,
            timeSpan: lastTSec - firstTSec,
            pxPerSec: (lastX - firstX) / (lastTSec - firstTSec),
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
        referencePitchOverlays,
        detectedPitchCurves,
        isDark = true,
        clipboardPreview,
        paramMorphOverlay,
        fontFamily,
    } = args;

    const resolvedFontFamily = fontFamily || "sans-serif";

    // 主题颜色查找表（提取到 colors.ts：阶段 2 起 GL 层与 Canvas2D 层共用同一份配色）
    const colors = resolvePianoRollColors(isDark);

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
            // 全物理清屏：round 向上取整时 CSS 尺寸清屏会在底部遗留残影。
            clearCanvasPhysical(ctx, target);

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
    // 全物理清屏：round 向上取整时 CSS 尺寸清屏会在底部遗留残影
    // （贴底曲线/网格/选区带/播放头的颜色永久残留在画布最底边）。
    clearCanvasPhysical(ctx, target);

    // 所有 x 坐标 = axis.secToViewportPx(sec)，与时间线侧同一实现。
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
        // 段级音阶音级只依赖段本身，与行无关：提升到行循环外只求值一次，
        // 否则 60fps 下是 行数 × 段数 次重复的音阶解析 + 数组分配。
        const segmentNotesList =
            highlightActive && scaleSegments && scaleSegments.length > 0
                ? scaleSegments.map((segment) =>
                      segment.scale ? resolveScaleNotes(segment.scale) : null,
                  )
                : null;

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

            if (segmentNotesList) {
                // Tempo Map 路径：按时间段绘制高亮段。
                ctx.strokeStyle = isDark ? "rgba(255,200,80,0.22)" : "rgba(200,120,20,0.22)";
                ctx.lineWidth = 2;
                for (let si = 0; si < scaleSegments!.length; si += 1) {
                    const segmentNotes = segmentNotesList[si];
                    if (!segmentNotes || !segmentNotes.includes(pc)) continue;
                    const segment = scaleSegments![si];
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
    //
    // 设备像素对齐（与 56238d45 网格修复同法，遵循 timelineAxis 强制约束 3
    // 「描边必须经 snapPx/strokePx 对齐」）：旧写法 `lineWidth=1 + x+0.5` 是
    // dpr=1 时代的整像素技巧 —— 分数 DPR（125%/150%）下 1 CSS px = 1.25/1.5
    // 物理像素，且 +0.5 不再对齐设备像素边界，播放头每帧重绘时覆盖的物理
    // 像素数随落点相位变化，就是画布版"播放时粗细不一"。
    // 修正：线宽取整物理像素；奇数物理像素宽的居中描边由 strokePx 补半个
    // 设备像素，使线体恰好覆盖整数个设备列。
    const phWidthPx = wholeDevicePxLength(1, axis.dpr);
    const phx = strokePx(axis, secToViewportPx(axis, playheadSec), phWidthPx);
    ctx.strokeStyle = colors.playheadLine;
    ctx.lineWidth = phWidthPx;
    ctx.beginPath();
    ctx.moveTo(phx, 0);
    ctx.lineTo(phx, h);
    ctx.stroke();
}
