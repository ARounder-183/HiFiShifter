/**
 * timelineCanvasStyle.ts - Timeline Clip 视觉样式与文字标签计算。
 *
 * 主要内容：
 * - 计算 Clip 头部各种 badge / label 的可见性、宽度、坐标。
 * - 根据 gain / playbackRate / 名称等数据生成显示文本与字号宽度。
 *
 * 与其他模块的关系：
 * - 被 ClipHeader.tsx、timelineCanvasRenderer.ts 调用消费。
 * - 依赖 timelineClipHeaderVisibility.ts 决定哪些元素在当前宽度下可见。
 *
 * 维护说明：
 * - playbackRate 显示走 `formatPlaybackRateLabel`，统一规则：保留至多 2 位小数，
 *   再去除末尾多余的 0；这样 1 → "x1"、1.5 → "x1.5"、1.23 → "x1.23"，
 *   避免出现 "x1.50" 这种带冗余尾零或 "x1.0" 让用户误以为没拉伸的情况
 *   （Bug 修复，2026-06-30）。
 */
import { gainToDb } from "../math.js";
import { resolveTimelineClipHeaderVisibility } from "./timelineClipHeaderVisibility.js";

// ── Font helpers ─────────────────────────────────────────────────────────
//
// 测量 helper 在浏览器和 Node 测试环境下都需要可用：
// - 浏览器：通过共享一个 measure-canvas 用 ctx.measureText() 拿到精确像素宽度。
// - Node（vitest / tsx 运行单元测试时）：不存在 `document` / `<canvas>`，
//   `getMeasureCtx()` 会抛 ReferenceError。此时退化到一个粗略的固定字符宽度
//   估算 —— 仅用于让纯逻辑断言可跑通，不参与运行时视觉效果。

let _measureCtx: CanvasRenderingContext2D | null = null;
let _measureCtxResolved = false;

/**
 * 取（并缓存）一个用于文字宽度测量的离屏 2D context。
 *
 * 在没有 DOM 的运行时（Node / SSR）下返回 null；调用方必须做好 null 兜底。
 * 第一次失败后会缓存"无 ctx"状态，避免每次测量都尝试访问 document。
 */
function getMeasureCtx(): CanvasRenderingContext2D | null {
    if (_measureCtxResolved) return _measureCtx;
    _measureCtxResolved = true;
    if (typeof document === "undefined") {
        _measureCtx = null;
        return null;
    }
    try {
        const canvas = document.createElement("canvas");
        _measureCtx = canvas.getContext("2d");
    } catch {
        _measureCtx = null;
    }
    return _measureCtx;
}

/**
 * 估算字符串宽度（像素）。
 *
 * 流程：
 * 1. 优先使用浏览器 canvas 的 `ctx.measureText()` 拿到与字体严格一致的宽度。
 * 2. 在 Node 等无 DOM 环境下退化到固定单字符宽估算：从 fontStyle 中解析像素
 *    字号（默认 12），按 0.55 的字符宽高比作为单字符估算宽度。这只在测试
 *    跑断言时被用到，不影响线上视觉。
 *
 * 参数：
 * - `text`：待测字符串。
 * - `fontStyle`：CSS font-style 段（例如 "12px" 或 "bold 10px"）。
 * - `fontFamily`：CSS font-family。
 */
export function measureTextWidth(text: string, fontStyle: string, fontFamily: string): number {
    const ctx = getMeasureCtx();
    if (ctx) {
        ctx.font = `${fontStyle} ${fontFamily}`;
        return ctx.measureText(text).width;
    }
    // Fallback：按字号 × 0.55 估算单字符宽
    const sizeMatch = fontStyle.match(/(\d+(?:\.\d+)?)px/);
    const sizePx = sizeMatch ? Number.parseFloat(sizeMatch[1]) : 12;
    return text.length * sizePx * 0.55;
}

/** Read the current font-family from the --qt-font-family CSS custom property. */
export function resolveFontFamily(): string {
    if (typeof document === "undefined") return "sans-serif";
    const font = getComputedStyle(document.documentElement)
        .getPropertyValue("--qt-font-family")
        .trim();
    return font || "sans-serif";
}

const NAME_FONT_STYLE = "12px";
const LABEL_FONT_STYLE = "10px";

/** A representative character set for estimating average char width. */
const CHAR_SAMPLE = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

function clamp(value: number, min: number, max: number): number {
    return Math.min(max, Math.max(min, value));
}

/**
 * 将 playbackRate 数值格式化为 Clip 头部用的简洁标签。
 *
 * 规则：
 * - 非有限 / 非正值 → "x1"（兜底）。
 * - 保留至多 2 位小数，再去除末尾的 0 与可能多余的小数点：
 *   1.0 → "x1"、1.5 → "x1.5"、1.23 → "x1.23"、0.85 → "x0.85"。
 * - 这样可避免出现 "x1.50" 这种带冗余尾零，也避免历史实现里 1.0001 显示
 *   为 "x1.0" 时让用户难以判断是否真正发生了拉伸（Bug 修复，2026-06-30）。
 *
 * @param rate 播放速率（>0）
 * @returns 形如 "x1" / "x1.5" / "x1.23" 的标签。
 */
export function formatPlaybackRateLabel(rate: number): string {
    if (!Number.isFinite(rate) || rate <= 0) return "x1";
    // toFixed(2) → "1.00" / "1.50" / "1.23"，再去除末尾的 0 和孤立小数点
    const trimmed = rate.toFixed(2).replace(/\.?0+$/, "");
    return `x${trimmed.length > 0 ? trimmed : "1"}`;
}

function parseHexColor(color: string): { r: number; g: number; b: number } | null {
    if (!color.startsWith("#")) return null;
    const hex = color.slice(1);
    const normalized =
        hex.length === 3
            ? hex
                  .split("")
                  .map((part) => part + part)
                  .join("")
            : hex;
    if (normalized.length !== 6) return null;
    return {
        r: Number.parseInt(normalized.slice(0, 2), 16),
        g: Number.parseInt(normalized.slice(2, 4), 16),
        b: Number.parseInt(normalized.slice(4, 6), 16),
    };
}

function mixHexColor(
    color: string,
    target: { r: number; g: number; b: number },
    ratio: number,
): { r: number; g: number; b: number } {
    const base = parseHexColor(color) ?? { r: 104, g: 131, b: 157 };
    const t = clamp(ratio, 0, 1);
    return {
        r: Math.round(base.r * (1 - t) + target.r * t),
        g: Math.round(base.g * (1 - t) + target.g * t),
        b: Math.round(base.b * (1 - t) + target.b * t),
    };
}

function rgba(rgb: { r: number; g: number; b: number }, alpha: number): string {
    return `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${alpha})`;
}

function ellipsizeText(text: string, maxChars: number): string {
    if (maxChars <= 0) return "";
    if (text.length <= maxChars) return text;
    if (maxChars <= 3) return ".".repeat(maxChars);
    return `${text.slice(0, maxChars - 3)}...`;
}

export function computeTimelineFadeShadeRange(args: {
    widthPx: number;
    fadeInPx: number;
    fadeOutPx: number;
}): {
    startPx: number;
    endPx: number;
} | null {
    const widthPx = Math.max(1, args.widthPx);
    const startPx = clamp(args.fadeInPx, 0, widthPx);
    const endPx = clamp(widthPx - args.fadeOutPx, 0, widthPx);
    if (endPx <= startPx) return null;
    return { startPx, endPx };
}

export function buildTimelineClipVisualStyle(args: {
    widthPx: number;
    trackColor?: string;
    selected: boolean;
    muted: boolean;
    gain: number;
    playbackRate: number;
    name: string;
    fontFamily?: string;
    isPitchAdjustment?: boolean;
    groupId?: string;
    isGroupActive?: boolean;
    isGroupDisabled?: boolean;
}): {
    headerFill: string;
    bodyFill: string;
    borderStroke: string;
    textFill: string;
    muteBadgeFill: string;
    muteBadgeStroke: string;
    muteBadgeTextFill: string;
    muteBadgeLabel: string;
    muteBadgeWidth: number;
    muteBadgeHeight: number;
    muteBadgeRadius: number;
    muteBadgeOffsetX: number;
    muteBadgeOffsetY: number;
    chainBadgeFill: string;
    chainBadgeStroke: string;
    chainBadgeTextFill: string;
    chainBadgeWidth: number;
    chainBadgeHeight: number;
    chainBadgeRadius: number;
    chainBadgeOffsetX: number;
    chainBadgeOffsetY: number;
    formantBadgeFill: string;
    formantBadgeStroke: string;
    formantBadgeTextFill: string;
    formantBadgeLabel: string;
    formantBadgeWidth: number;
    formantBadgeHeight: number;
    formantBadgeRadius: number;
    formantBadgeOffsetX: number;
    formantBadgeOffsetY: number;
    gainKnobFill: string;
    gainKnobStroke: string;
    gainKnobIndicator: string;
    gainKnobCoreFill: string;
    gainKnobAngleDeg: number;
    gainKnobRadius: number;
    gainKnobCenterOffsetX: number;
    gainKnobCenterOffsetY: number;
    showPlaybackRate: boolean;
    playbackRateLabel: string;
    gainLabel: string;
    displayName: string;
    mutedAlpha: number;
    leadingControlsWidth: number;
    trailingReservePx: number;
    showMuteBadge: boolean;
    showChainBadge: boolean;
    showFormantBadge: boolean;
    showGainKnob: boolean;
    showGainLabel: boolean;
    showName: boolean;
    borderLineWidth: number;
} {
    const fontFamily = args.fontFamily || resolveFontFamily();
    const trackColor = args.trackColor ?? "#68839d";
    const headerRgb = mixHexColor(trackColor, { r: 160, g: 171, b: 183 }, 0.16);
    const bodyRgb = mixHexColor(trackColor, { r: 58, g: 63, b: 71 }, 0.68);
    const borderRgb = mixHexColor(trackColor, { r: 133, g: 144, b: 156 }, 0.3);
    const knobRgb = mixHexColor(trackColor, { r: 205, g: 212, b: 220 }, 0.24);
    const controlRgb = mixHexColor(trackColor, { r: 40, g: 46, b: 55 }, 0.52);
    const controlActiveRgb = mixHexColor(trackColor, { r: 120, g: 64, b: 69 }, 0.4);
    const isPitchAdj = args.isPitchAdjustment === true;
    const {
        showChain,
        showMute,
        showFormant,
        showGainKnob,
        showPlaybackRate,
        showGainLabel,
        showName,
    } = resolveTimelineClipHeaderVisibility(args.widthPx, isPitchAdj);
    const showChainBadge = showChain && args.groupId != null;

    // Compute labels early so we can measure their widths with the correct font
    const gainDb = gainToDb(args.gain);
    const clampedGainDb = clamp(gainDb, -12, 12);
    const playbackRate =
        Number.isFinite(args.playbackRate) && args.playbackRate > 0 ? args.playbackRate : 1;
    const playbackRateLabel = formatPlaybackRateLabel(playbackRate);
    const gainLabel = `${gainDb >= 0 ? "+" : ""}${gainDb.toFixed(1)}dB`;

    // Font-aware trailing reserve: measure actual label widths
    const gainLabelWidth = showGainLabel
        ? measureTextWidth(gainLabel, LABEL_FONT_STYLE, fontFamily)
        : 0;
    const rateLabelWidth =
        showGainLabel && showPlaybackRate
            ? measureTextWidth(playbackRateLabel, LABEL_FONT_STYLE, fontFamily)
            : 0;
    const trailingReservePx = showGainLabel
        ? showPlaybackRate
            ? rateLabelWidth + gainLabelWidth + 16
            : gainLabelWidth + 12
        : showGainKnob
          ? 26
          : 10;

    const muteBadgeWidth = 20;
    const muteBadgeHeight = 14;
    const muteBadgeRadius = 4;
    const chainBadgeWidth = 20;
    const chainBadgeHeight = 14;
    const chainBadgeRadius = 4;
    const formantBadgeWidth = 20;
    const formantBadgeHeight = 14;
    const formantBadgeRadius = 4;
    const gainKnobRadius = 7;
    const gainKnobCenterOffsetX = 15;
    const gainKnobCenterOffsetY = 10;
    const chainBadgeOffsetX = showGainKnob ? 28 : 8;
    const chainBadgeOffsetY = 3;
    const muteBadgeOffsetX = showChainBadge
        ? chainBadgeOffsetX + chainBadgeWidth + 2
        : chainBadgeOffsetX;
    const muteBadgeOffsetY = 3;
    const formantBadgeOffsetX = muteBadgeOffsetX + muteBadgeWidth + 2;
    const formantBadgeOffsetY = 3;

    // Compute right edge of left-side controls dynamically (chain-aware)
    const controlsRightEdge = showFormant
        ? formantBadgeOffsetX + formantBadgeWidth
        : showMute
          ? muteBadgeOffsetX + muteBadgeWidth
          : showChainBadge
            ? chainBadgeOffsetX + chainBadgeWidth
            : showGainKnob
              ? gainKnobCenterOffsetX + gainKnobRadius + 2
              : 8;
    const leadingControlsWidth = controlsRightEdge + 10;

    // Chain badge: red when group is disabled, golden when active, neutral otherwise
    const chainBadgeFill = args.isGroupDisabled
        ? "rgba(220, 70, 70, 0.45)"
        : args.isGroupActive
          ? "rgba(255, 200, 50, 0.55)"
          : `rgba(${controlRgb.r}, ${controlRgb.g}, ${controlRgb.b}, 0.55)`;
    const chainBadgeStroke = args.isGroupDisabled
        ? "rgba(200, 50, 50, 0.80)"
        : args.isGroupActive
          ? "rgba(255, 200, 50, 0.90)"
          : `rgba(${borderRgb.r}, ${borderRgb.g}, ${borderRgb.b}, 0.50)`;
    const chainBadgeTextFill = args.isGroupDisabled
        ? "rgba(180, 40, 40, 1)"
        : args.isGroupActive
          ? "rgba(180, 120, 10, 1)"
          : "rgba(210, 215, 225, 0.85)";

    const textStartPx = controlsRightEdge + 6;

    // Font-aware average char width for name truncation
    const avgCharWidth = Math.max(
        1,
        measureTextWidth(CHAR_SAMPLE, NAME_FONT_STYLE, fontFamily) / CHAR_SAMPLE.length,
    );
    const maxChars = Math.max(
        1,
        Math.floor((args.widthPx - textStartPx - trailingReservePx) / avgCharWidth),
    );

    return {
        headerFill: rgba(headerRgb, 0.95),
        bodyFill: rgba(bodyRgb, 0.74),
        borderStroke: args.selected
            ? getComputedStyle(document.documentElement)
                .getPropertyValue("--qt-clip-selected-border")
                .trim() || rgba(borderRgb, 1)
            : rgba(borderRgb, 0.74),
        borderLineWidth: args.selected ? 2 : 1,
        textFill: "rgba(241, 245, 249, 0.94)",
        muteBadgeFill: rgba(args.muted ? controlActiveRgb : controlRgb, args.muted ? 0.96 : 0.9),
        muteBadgeStroke: rgba(
            args.muted
                ? mixHexColor(trackColor, { r: 216, g: 187, b: 191 }, 0.26)
                : mixHexColor(trackColor, { r: 182, g: 193, b: 206 }, 0.18),
            args.muted ? 0.95 : 0.66,
        ),
        muteBadgeTextFill: args.muted ? "#fbebeb" : "rgba(244, 247, 250, 0.94)",
        muteBadgeLabel: "M",
        muteBadgeWidth,
        muteBadgeHeight,
        muteBadgeRadius,
        muteBadgeOffsetX,
        muteBadgeOffsetY,
        chainBadgeFill,
        chainBadgeStroke,
        chainBadgeTextFill,
        chainBadgeWidth,
        chainBadgeHeight,
        chainBadgeRadius,
        chainBadgeOffsetX,
        chainBadgeOffsetY,
        formantBadgeFill: rgba(controlRgb, 0.9),
        formantBadgeStroke: rgba(mixHexColor(trackColor, { r: 182, g: 193, b: 206 }, 0.18), 0.66),
        formantBadgeTextFill: "rgba(244, 247, 250, 0.94)",
        formantBadgeLabel: "F",
        formantBadgeWidth,
        formantBadgeHeight,
        formantBadgeRadius,
        formantBadgeOffsetX,
        formantBadgeOffsetY,
        gainKnobFill: rgba(knobRgb, 0.94),
        gainKnobStroke: rgba(mixHexColor(trackColor, { r: 34, g: 40, b: 48 }, 0.46), 0.92),
        gainKnobIndicator: "rgba(248, 251, 255, 0.94)",
        gainKnobCoreFill: rgba(mixHexColor(trackColor, { r: 246, g: 250, b: 255 }, 0.38), 0.9),
        gainKnobAngleDeg: (clampedGainDb / 12) * 135,
        gainKnobRadius,
        gainKnobCenterOffsetX,
        gainKnobCenterOffsetY,
        showPlaybackRate,
        playbackRateLabel,
        gainLabel,
        displayName: ellipsizeText(args.name, maxChars),
        mutedAlpha: args.muted ? 0.29 : 1,
        leadingControlsWidth,
        trailingReservePx,
        showMuteBadge: showMute,
        showChainBadge,
        showFormantBadge: showFormant,
        showGainKnob,
        showGainLabel,
        showName,
    };
}
