/**
 * 声像参数的显示层：左/右/中、百分比、轴方向。
 *
 * 两条通道刻意分开：纵轴刻度用语言无关的紧凑字母（`L`/`C`/`R`），ToolTip / 拖拽
 * 读数 / Set Value 用完整词（`左 35%`）。前者要在一列里并排扫读，后者只需精确一次。
 *
 * 本模块不改变内部值：输入/输出仍是 -1..1 的 HS 声像。它只负责
 * “这个值对用户怎么说”以及“左/右在上还是在下”。曲线、选区、参数块、
 * 命中与拖拽的投影都读取同一个方向设置。
 */

export const PAN_PARAM_ID = "pan";

export type PanAxisDirection = "left-bottom" | "left-top";
export const DEFAULT_PAN_AXIS_DIRECTION: PanAxisDirection = "left-bottom";

export type PanAxisDirections = Record<string, PanAxisDirection>;

export interface PanTextLabels {
    readonly left: string;
    readonly center: string;
    readonly right: string;
}

export function isPanParam(param: string | null | undefined): boolean {
    return param === PAN_PARAM_ID;
}

export function resolvePanAxisDirection(
    directions: PanAxisDirections | null | undefined,
    param: string,
): PanAxisDirection {
    return directions?.[param] === "left-top" ? "left-top" : DEFAULT_PAN_AXIS_DIRECTION;
}

export function nextPanAxisDirection(direction: PanAxisDirection): PanAxisDirection {
    return direction === "left-top" ? "left-bottom" : "left-top";
}

export function normalizePanAxisDirections(raw: unknown): PanAxisDirections {
    const out: PanAxisDirections = {};
    if (raw == null || typeof raw !== "object") return out;
    for (const [key, value] of Object.entries(raw as Record<string, unknown>)) {
        if (!isPanParam(key)) continue;
        if (value === "left-bottom" || value === "left-top") out[key] = value;
    }
    return out;
}

/** 内部值 → 左/中/右；0 附近按 0 处理，避免 -0.0001 显示成“右 0%”。 */
export function panSideOf(value: number): "left" | "center" | "right" {
    if (!Number.isFinite(value) || Math.abs(value) < 1e-6) return "center";
    return value < 0 ? "left" : "right";
}

export function panPercentOf(value: number): number {
    if (!Number.isFinite(value)) return 0;
    return Math.round(Math.min(1, Math.abs(value)) * 100);
}

/**
 * 纵轴刻度的单字母表：**语言无关**，因此不进 i18n、不随语言变长。
 *
 * 轴列只有几十像素宽、刻度还会在小值域下变密，任何完整单词（`Left`/`Center`）都会
 * 互相挤占；单字母可以在同样宽度内容纳更多刻度。字母描述的是**声像方向**
 * （左/右声道），与纵轴方向翻转无关——`left-top` 只改投影，不改字母含义。
 */
export const PAN_AXIS_LEFT = "L";
export const PAN_AXIS_CENTER = "C";
export const PAN_AXIS_RIGHT = "R";

/**
 * 纵轴刻度：紧凑、语言无关。端点与中心只给字母，中间值给「字母 + 百分比」。
 *
 * 例：`L` / `L50` / `C` / `R50` / `R`（最长 4 字符）。
 * 不带 `%`、不带空格：轴列空间有限，且同一列里的每个刻度都已经是百分比口径。
 */
export function formatPanAxisCompactLabel(value: number): string {
    const side = panSideOf(value);
    if (side === "center") return PAN_AXIS_CENTER;
    const pct = panPercentOf(value);
    const letter = side === "left" ? PAN_AXIS_LEFT : PAN_AXIS_RIGHT;
    return pct >= 100 ? letter : `${letter}${pct}`;
}

/** ToolTip / 读数：明确写百分比。例：左 35% / 右 35% / 中。 */
export function formatPanValue(value: number, labels: PanTextLabels): string {
    const side = panSideOf(value);
    if (side === "center") return labels.center;
    const pct = panPercentOf(value);
    const sideLabel = side === "left" ? labels.left : labels.right;
    return `${sideLabel} ${pct}%`;
}
