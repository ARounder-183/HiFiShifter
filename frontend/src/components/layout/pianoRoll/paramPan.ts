/**
 * 声像参数的显示层：左/右/中、百分比、轴方向。
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

/** 纵轴刻度：短、可并排。例：左100 / 左50 / 中 / 右50 / 右100。 */
export function formatPanAxisLabel(value: number, labels: PanTextLabels): string {
    const side = panSideOf(value);
    if (side === "center") return labels.center;
    const pct = panPercentOf(value);
    const sideLabel = side === "left" ? labels.left : labels.right;
    return `${sideLabel}${pct}`;
}

/** ToolTip / 读数：明确写百分比。例：左 35% / 右 35% / 中。 */
export function formatPanValue(value: number, labels: PanTextLabels): string {
    const side = panSideOf(value);
    if (side === "center") return labels.center;
    const pct = panPercentOf(value);
    const sideLabel = side === "left" ? labels.left : labels.right;
    return `${sideLabel} ${pct}%`;
}
