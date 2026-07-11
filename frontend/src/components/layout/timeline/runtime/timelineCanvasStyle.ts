/**
 * timelineCanvasStyle.ts - Timeline Clip 视觉样式与文字标签计算。
 *
 * 主要内容：
 * - 计算 Clip 头部各种 badge / label 的可见性、宽度、坐标。
 * - 根据 gain / playbackRate / 名称等数据生成显示文本与字号宽度。
 * - 输出 Logic Pro 风格的扁平视觉参数（去饱和 body + 全饱和 accent bar）。
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
 * - Clip 视觉风格在 2026-06-30 重做为 Logic Pro 扁平风格：取消所有外圆角、
 *   body 用 trackColor 派生的低饱和深底（不再透明叠在 lane 上）、左侧 3px
 *   accent bar 承担色相识别、选中态用 1px 内描边 + body 提亮。这套方案
 *   的设计目标是降低视觉噪声、提升专业感，并让多 clip 排列时颜色不互相干扰。
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

/** 左侧 accent bar 的固定宽度（px），在 clip 最左侧承担色相识别。 */
const ACCENT_BAR_WIDTH_PX = 3;

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

type Rgb = { r: number; g: number; b: number };
type Hsl = { h: number; s: number; l: number };

/**
 * RGB(0..255) → HSL(h:0..360, s:0..1, l:0..1)。
 *
 * 用于把 trackColor 拆解为色相 / 饱和度 / 明度三轴，方便单独调整某一项
 * （比如保留色相但大幅降低饱和度，得到"识别得出但视觉低噪"的 body 颜色）。
 */
function rgbToHsl(rgb: Rgb): Hsl {
    const r = rgb.r / 255;
    const g = rgb.g / 255;
    const b = rgb.b / 255;
    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    const l = (max + min) / 2;
    if (max === min) {
        return { h: 0, s: 0, l };
    }
    const d = max - min;
    const s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
    let h: number;
    switch (max) {
        case r:
            h = ((g - b) / d + (g < b ? 6 : 0)) * 60;
            break;
        case g:
            h = ((b - r) / d + 2) * 60;
            break;
        default:
            h = ((r - g) / d + 4) * 60;
            break;
    }
    return { h, s, l };
}

function hue2rgbChannel(p: number, q: number, t: number): number {
    let tt = t;
    if (tt < 0) tt += 1;
    if (tt > 1) tt -= 1;
    if (tt < 1 / 6) return p + (q - p) * 6 * tt;
    if (tt < 1 / 2) return q;
    if (tt < 2 / 3) return p + (q - p) * (2 / 3 - tt) * 6;
    return p;
}

/** HSL → RGB(0..255)，与 rgbToHsl 互逆，用于颜色调整后回写到 ctx.fillStyle。 */
function hslToRgb(hsl: Hsl): Rgb {
    const h = ((hsl.h % 360) + 360) % 360 / 360;
    const s = clamp(hsl.s, 0, 1);
    const l = clamp(hsl.l, 0, 1);
    if (s === 0) {
        const v = Math.round(l * 255);
        return { r: v, g: v, b: v };
    }
    const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
    const p = 2 * l - q;
    return {
        r: Math.round(hue2rgbChannel(p, q, h + 1 / 3) * 255),
        g: Math.round(hue2rgbChannel(p, q, h) * 255),
        b: Math.round(hue2rgbChannel(p, q, h - 1 / 3) * 255),
    };
}

/**
 * 基于 trackColor 派生 Clip 视觉所需的全套颜色（Logic Pro 扁平风格）。
 *
 * 流程：
 * 1. trackColor → HSL 色相 H 提取（保留色相是身份识别的关键）。
 * 2. accent：保持原 H、把 S 拉满（≥0.85）、L 适中（≈0.55），用作左侧细条与
 *    选中态描边，是唯一携带高饱和度的视觉元素。
 * 3. body：保持 H、把 S 大幅压低（≈0.22）、L 设到深灰区（≈0.20），得到一个
 *    "你能感觉到色相，但绝不刺眼"的暗底。这个底是 fillRect 直接写入，不再
 *    透明叠层，从而避免颜色和 lane 背景互相污染产生的"泥糊"质感。
 * 4. header：在 body 基础上 L -0.04 得到稍深一档的顶条。
 * 5. selectedBody：在 body 基础上 L +0.06，作为选中态的 body 提亮。
 *
 * 异常 trackColor（无效 hex / 未提供）回退到一个低饱和蓝灰色基准，保持视觉
 * 一致性。
 */
function buildClipPalette(trackColor: string | undefined): {
    accent: Rgb;
    body: Rgb;
    header: Rgb;
    selectedBody: Rgb;
    selectedHeader: Rgb;
} {
    const fallback: Rgb = { r: 104, g: 131, b: 157 };
    const base = parseHexColor(trackColor ?? "") ?? fallback;
    const hsl = rgbToHsl(base);
    const accent = hslToRgb({ h: hsl.h, s: Math.max(hsl.s, 0.85), l: 0.55 });
    const body = hslToRgb({ h: hsl.h, s: 0.22, l: 0.2 });
    const header = hslToRgb({ h: hsl.h, s: 0.22, l: 0.16 });
    const selectedBody = hslToRgb({ h: hsl.h, s: 0.26, l: 0.26 });
    const selectedHeader = hslToRgb({ h: hsl.h, s: 0.26, l: 0.22 });
    return { accent, body, header, selectedBody, selectedHeader };
}

function rgba(rgb: Rgb, alpha: number): string {
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

/**
 * 构建单个 Clip 在 canvas 上的全部视觉样式参数。
 *
 * 流程：
 * 1. 调色板构建（buildClipPalette）：从 trackColor 派生 accent/body/header/selectedBody。
 * 2. 控件可见性：基于宽度决定 gain knob / mute / formant / chain / name 是否显示。
 * 3. 控件几何：计算各 badge 的偏移、文字预留宽度、name 的可用绘制宽度。
 * 4. 选中态分支：选中时 body 用 selectedBody（更亮）、border 用 accent；
 *    非选中态 border 几乎不可见（极淡黑色）以保持扁平。
 * 5. 静音态：通过 mutedAlpha 整体降透明度，保留交互可见性的同时弱化存在感。
 *
 * 返回的字段供 timelineCanvasRenderer.ts 直接消费，渲染端不应再做颜色再加工。
 */
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
    accentBarFill: string;
    accentBarWidthPx: number;
    headerSeparatorFill: string;
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
} {
    const fontFamily = args.fontFamily || resolveFontFamily();
    const trackColor = args.trackColor ?? "#68839d";
    const palette = buildClipPalette(trackColor);
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
    // gain knob 整体右移 ACCENT_BAR_WIDTH_PX，给左侧 accent bar 留位置
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
    // 中性态走偏暗的灰底，保持与新扁平风格一致
    const chainBadgeFill = args.isGroupDisabled
        ? "rgba(220, 70, 70, 0.42)"
        : args.isGroupActive
          ? "rgba(255, 200, 50, 0.5)"
          : "rgba(255, 255, 255, 0.06)";
    const chainBadgeStroke = args.isGroupDisabled
        ? "rgba(220, 70, 70, 0.85)"
        : args.isGroupActive
          ? "rgba(255, 200, 50, 0.9)"
          : "rgba(255, 255, 255, 0.14)";
    const chainBadgeTextFill = args.isGroupDisabled
        ? "rgba(255, 200, 200, 0.95)"
        : args.isGroupActive
          ? "rgba(255, 235, 170, 0.95)"
          : "rgba(255, 255, 255, 0.78)";

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

    // 选中态：body 提亮、border 用饱和 accent；非选中态 border 极淡（仅作分隔提示）
    const bodyRgb = args.selected ? palette.selectedBody : palette.body;
    const headerRgb = args.selected ? palette.selectedHeader : palette.header;
    const borderColor = args.selected
        ? rgba(palette.accent, 0.95)
        : "rgba(0, 0, 0, 0.32)";
    // header 与 body 的分隔线：扁平风格下用极淡的白色提分层，避免黑色硬线
    const headerSeparatorFill = "rgba(255, 255, 255, 0.06)";

    // mute / formant badge：统一使用半透明白系，避免与 trackColor 强耦合
    const muteIdleFill = "rgba(255, 255, 255, 0.06)";
    const muteActiveFill = "rgba(220, 80, 80, 0.55)";
    const muteIdleStroke = "rgba(255, 255, 255, 0.14)";
    const muteActiveStroke = "rgba(220, 80, 80, 0.9)";

    return {
        headerFill: rgba(headerRgb, 1),
        bodyFill: rgba(bodyRgb, 1),
        borderStroke: borderColor,
        accentBarFill: rgba(palette.accent, 1),
        accentBarWidthPx: ACCENT_BAR_WIDTH_PX,
        headerSeparatorFill,
        textFill: "rgba(255, 255, 255, 0.92)",
        muteBadgeFill: args.muted ? muteActiveFill : muteIdleFill,
        muteBadgeStroke: args.muted ? muteActiveStroke : muteIdleStroke,
        muteBadgeTextFill: args.muted ? "rgba(255, 230, 230, 0.98)" : "rgba(255, 255, 255, 0.85)",
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
        formantBadgeFill: muteIdleFill,
        formantBadgeStroke: muteIdleStroke,
        formantBadgeTextFill: "rgba(255, 255, 255, 0.85)",
        formantBadgeLabel: "F",
        formantBadgeWidth,
        formantBadgeHeight,
        formantBadgeRadius,
        formantBadgeOffsetX,
        formantBadgeOffsetY,
        gainKnobFill: "rgba(255, 255, 255, 0.10)",
        gainKnobStroke: "rgba(255, 255, 255, 0.35)",
        gainKnobIndicator: "rgba(255, 255, 255, 0.92)",
        gainKnobCoreFill: "rgba(255, 255, 255, 0.55)",
        gainKnobAngleDeg: (clampedGainDb / 12) * 135,
        gainKnobRadius,
        gainKnobCenterOffsetX,
        gainKnobCenterOffsetY,
        showPlaybackRate,
        playbackRateLabel,
        gainLabel,
        displayName: ellipsizeText(args.name, maxChars),
        mutedAlpha: args.muted ? 0.4 : 1,
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

