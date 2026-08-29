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

function rgba(rgb: { r: number; g: number; b: number }, alpha: number): string {
    return `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${alpha})`;
}

// ── HSL 色彩归一化（Ableton 式"亮色块 + 深色波形"）──────────────────────
//
// 设计思路：Clip 主体是一块**明亮的轨道色**（粉彩明度带），波形与文字用
// 深色画在色块上。此前走的是"中等亮度色块 + 白波形"——为了保白波形对比
// 不得不把色块压暗，压暗后的颜色发闷发脏；翻转对比方向后色块可以放开
// 亮度，任何轨道色都出来"干净的糖果色"。
//
// 轨道色是用户随意挑的，归一化把任意输入收敛到可控区间：饱和度夹在
// [0.45, 0.68]、亮度夹在 [0.55, 0.72]，再按感知亮度二次校正。

type Hsl = { h: number; s: number; l: number };

function rgbToHsl(rgb: { r: number; g: number; b: number }): Hsl {
    const r = rgb.r / 255;
    const g = rgb.g / 255;
    const b = rgb.b / 255;
    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    const l = (max + min) / 2;
    if (max === min) return { h: 0, s: 0, l };
    const d = max - min;
    const s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
    let h: number;
    if (max === r) h = ((g - b) / d + (g < b ? 6 : 0)) * 60;
    else if (max === g) h = ((b - r) / d + 2) * 60;
    else h = ((r - g) / d + 4) * 60;
    return { h, s, l };
}

function hueToRgb(p: number, q: number, tRaw: number): number {
    let t = tRaw;
    if (t < 0) t += 1;
    if (t > 1) t -= 1;
    if (t < 1 / 6) return p + (q - p) * 6 * t;
    if (t < 1 / 2) return q;
    if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
    return p;
}

function hslToRgb(hsl: Hsl): { r: number; g: number; b: number } {
    const { h, s, l } = hsl;
    if (s === 0) {
        const v = Math.round(l * 255);
        return { r: v, g: v, b: v };
    }
    const hue = ((h % 360) + 360) % 360 / 360;
    const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
    const p = 2 * l - q;
    return {
        r: Math.round(hueToRgb(p, q, hue + 1 / 3) * 255),
        g: Math.round(hueToRgb(p, q, hue) * 255),
        b: Math.round(hueToRgb(p, q, hue - 1 / 3) * 255),
    };
}

/** 感知亮度（0–1，Rec.601 加权）：人眼对绿最亮、对蓝最暗。 */
function perceivedLuminance(rgb: { r: number; g: number; b: number }): number {
    return (rgb.r * 0.299 + rgb.g * 0.587 + rgb.b * 0.114) / 255;
}

/** 默认轨道色：中性灰。未设色/异常色的轨道呈现安静的灰块。 */
export const DEFAULT_TRACK_COLOR = "#8a9099";

/** 色块的感知亮度目标带：统一的中等偏亮明度，所有轨道色视觉重量一致，
 * 时间线整体安静、有序（Cubase 默认主题式"彩色灰"）。 */
const CLIP_LUMINANCE_BAND = { min: 0.50, max: 0.6 } as const;

/**
 * 轨道色 → 归一化 HSL。
 *
 * 两级收敛：
 * 1. HSL 夹取（饱和度、亮度各自限幅）；
 * 2. **感知亮度**校正 —— HSL 的 l 不等于人眼亮度：蓝紫色在 l=0.48 时感知
 *    亮度只有 ~0.21（发黑，暗背景上隐形），黄绿色则偏亮。按感知亮度迭代
 *    微调 l，把任意色相的色块都收敛到同一视觉明度带，时间线才整齐。
 */
function normalizeTrackHsl(color: string): Hsl {
    const base = rgbToHsl(parseHexColor(color) ?? { r: 138, g: 144, b: 153 });
    // 中性灰直通：默认轨道色是灰色，不能被强行拉成彩色 —— 低饱和输入
    // 只做明度归一，保持无彩。
    if (base.s < 0.1) {
        return { h: base.h, s: 0, l: clamp(base.l, 0.48, 0.6) };
    }
    const hsl: Hsl = {
        h: base.h,
        // 中饱和带：8 个轨道色相需保持可辨识，同时不过度主导画面；
        // 亮度带统一所有轨道的视觉重量（0.48-0.58 偏暗，已提亮一档）。
        s: clamp(base.s, 0.3, 0.46),
        l: clamp(base.l, 0.44, 0.58),
    };
    for (let i = 0; i < 16; i += 1) {
        const lum = perceivedLuminance(hslToRgb(hsl));
        if (lum < CLIP_LUMINANCE_BAND.min) {
            hsl.l = Math.min(0.85, hsl.l + 0.02);
        } else if (lum > CLIP_LUMINANCE_BAND.max) {
            hsl.l = Math.max(0.05, hsl.l - 0.02);
        } else {
            break;
        }
    }
    return hsl;
}

/**
 * 轨道色 → 归一化后的 CSS 颜色（轨道头取色显示用）。
 *
 * Clip 色块画的是归一化色，轨道头的色条/取色块**必须显示同一个颜色**——
 * 否则"挑的颜色"和"时间线上实际出现的颜色"对不上，用户会以为取色器坏了。
 */
export function normalizedTrackColorCss(color: string | undefined | null): string {
    const rgb = hslToRgb(normalizeTrackHsl(color ?? DEFAULT_TRACK_COLOR));
    return `rgb(${rgb.r}, ${rgb.g}, ${rgb.b})`;
}

/**
 * 从基准 HSL 派生明度变体（header 调暗、badge 更暗、旋钮提亮……）。
 * 整个 Clip 的所有颜色都从同一个归一化基准派生 —— 它们天然同调，
 * 不会出现"header 和 body 两张皮"。
 */
function shadeHsl(base: Hsl, lightnessDelta: number, saturationDelta = 0): Hsl {
    return {
        h: base.h,
        s: clamp(base.s + saturationDelta, 0, 1),
        l: clamp(base.l + lightnessDelta, 0, 1),
    };
}

/**
 * Clip 圆角半径（px）。
 *
 * 参考 Ableton Live / REAPER：Clip 近乎直角 —— 边界即命中边界，用户对
 * "边缘在哪、能不能抓"的判断不被圆角干扰。保留 1.5px 只为消除直角的
 * 像素锯齿，视觉上不可辨。绘制端按 Clip 实际尺寸再收敛一次。
 */
export const CLIP_CORNER_RADIUS_PX = 1.5;

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
    const trackColor = args.trackColor ?? DEFAULT_TRACK_COLOR;
    // ── 色块配色（Ableton 式"亮色块 + 深色前景"）──────────────────
    // Clip 主体 = 归一化后的明亮轨道色；header 是同色轻微压深的一条带。
    // 文字 / 波形 / 徽章 / 旋钮全部用**深色**画在亮块上 —— 对比方向与旧方案
    // 相反，这正是"干净不发闷"的关键：色块可以亮，前景永远深。
    //
    // muted：饱和度压到近灰、明度不动 —— 亮灰块一眼可辨，不靠降 alpha
    // （降 alpha 会透出背景、显脏）。
    const baseHsl = normalizeTrackHsl(trackColor);
    // 选中 = 色块提亮（Ableton 式）+ 白描边；muted = 同色相灰且压暗一档 ——
    // 深底上等亮度灰块像"脏水泥"，沉下去后活跃 clip 的层级自然浮现。
    const clipHsl: Hsl = args.muted
        ? { h: baseHsl.h, s: 0.06, l: Math.max(0.08, baseHsl.l - 0.12) }
        : args.selected
          ? shadeHsl(baseHsl, 0.08)
          : baseHsl;
    const bodyRgb = hslToRgb(clipHsl);
    const headerRgb = hslToRgb(shadeHsl(clipHsl, -0.05));

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

    // Chain badge：禁用=红（白字）、激活=金（深字）、中性=半透明深底 + 深字。
    const chainBadgeFill = args.isGroupDisabled
        ? "rgba(189, 54, 54, 0.95)"
        : args.isGroupActive
          ? "rgba(233, 185, 47, 0.95)"
          : "rgba(0, 0, 0, 0.16)";
    const chainBadgeStroke = args.isGroupDisabled
        ? "rgba(120, 22, 22, 0.8)"
        : args.isGroupActive
          ? "rgba(122, 88, 6, 0.75)"
          : "rgba(0, 0, 0, 0.28)";
    const chainBadgeTextFill = args.isGroupDisabled
        ? "rgba(255, 244, 244, 0.96)"
        : args.isGroupActive
          ? "rgba(56, 42, 4, 0.96)"
          : "rgba(28, 32, 40, 0.92)";

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
        headerFill: rgba(headerRgb, 1),
        bodyFill: rgba(bodyRgb, 1),
        // 描边：选中 = 白色 2px（在提亮的色块上清晰醒目，REAPER 惯例）；
        // 未选中 = 同色调深描边 —— 纯黑低透明描边在深底上不可见，同色相
        // 加深的描边让色块边缘"闭合"且与色块同调。
        borderStroke: args.selected
            ? "rgba(255, 255, 255, 0.92)"
            : rgba(headerRgb, 0.55),
        borderLineWidth: args.selected ? 2 : 1,
        textFill: "rgba(28, 32, 40, 0.92)",
        muteBadgeFill: args.muted ? "rgba(189, 54, 54, 0.95)" : "rgba(0, 0, 0, 0.16)",
        muteBadgeStroke: args.muted ? "rgba(120, 22, 22, 0.8)" : "rgba(0, 0, 0, 0.28)",
        muteBadgeTextFill: args.muted
            ? "rgba(255, 244, 244, 0.96)"
            : "rgba(28, 32, 40, 0.92)",
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
        formantBadgeFill: "rgba(0, 0, 0, 0.16)",
        formantBadgeStroke: "rgba(0, 0, 0, 0.28)",
        formantBadgeTextFill: "rgba(28, 32, 40, 0.92)",
        formantBadgeLabel: "F",
        formantBadgeWidth,
        formantBadgeHeight,
        formantBadgeRadius,
        formantBadgeOffsetX,
        formantBadgeOffsetY,
        gainKnobFill: "rgba(0, 0, 0, 0.2)",
        gainKnobStroke: "rgba(0, 0, 0, 0.38)",
        gainKnobIndicator: "rgba(28, 32, 40, 0.95)",
        gainKnobCoreFill: "rgba(255, 255, 255, 0.4)",
        gainKnobAngleDeg: (clampedGainDb / 12) * 135,
        gainKnobRadius,
        gainKnobCenterOffsetX,
        gainKnobCenterOffsetY,
        showPlaybackRate,
        playbackRateLabel,
        gainLabel,
        displayName: ellipsizeText(args.name, maxChars),
        // muted 已由去饱和的亮灰块表达，保持不透明（透背景会显脏）。
        mutedAlpha: 1,
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
