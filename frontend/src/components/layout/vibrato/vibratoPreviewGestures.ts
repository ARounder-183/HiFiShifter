/**
 * 颤音预览画布的手势命中与映射（纯函数，可单测）。
 *
 * 【设计原则】预览画布在编辑时就是**另一组滑杆**：拖动的是与参数表单同一个草稿，
 * 不是独立的第二事实源。因此这里的映射规则必须与渲染端一致 ——
 * 渐入 / 渐出与渲染端同样封顶在"整段时长的一半"（见 `vibratoCurve.ts` 的
 * `maxHalf`），深度 / 相位的换算是绘制所用几何的逆运算。
 *
 * 【为什么单独成文件】命中测试与换算全是可以脱离 React 验证的算术；混在画布里
 * 就只能靠手感回归。"拖不动 / 拖反了"这类缺陷不会抛错，只能靠单测钉住。
 */

import { VIBRATO_LIMITS } from "../../../features/vibrato/vibratoPresets";
import type { VibratoPreset } from "../../../features/vibrato/vibratoTypes";

/** 手势区域：左右缘手柄，或波形主体。 */
export type PreviewZone = { kind: "attack" } | { kind: "release" } | { kind: "body" };

/** 手柄的归一化横向位置（0..1）。 */
export interface PreviewHandleLayout {
    attackFrac: number;
    releaseFrac: number;
}

/** 手柄的命中半径（CSS 像素）。 */
export const HANDLE_HIT_PX = 9;
/** 外缘兜底命中带占画布宽度的比例（手柄被拖到别处时的后备抓手）。 */
export const EDGE_ZONE_FRAC = 0.12;

/**
 * 命中测试：**先手柄、后外缘、最后主体**。
 *
 * 【为什么手柄优先】手柄画在包络斜坡的实际位置（渐入很长时可能到画布中部），
 * 若只按"左右 12%"判定，用户抓住看得见的手柄却落在主体区 —— 拖出的是相位而不是
 * 渐入。手柄附近优先保证"抓哪个就是哪个"；外缘 12% 是手柄被拖远后的后备。
 */
export function hitTestPreviewZone(
    x: number,
    width: number,
    layout: PreviewHandleLayout,
): PreviewZone {
    // 画布尚未布局（宽度为 0）时没有可抓手柄，一律当主体 —— 也避免用 1px 的
    // 假宽度去比 9px 的命中半径（那会把整条轴都判成手柄）。
    if (!(width > 0)) return { kind: "body" };
    const attackX = layout.attackFrac * width;
    const releaseX = layout.releaseFrac * width;
    if (Math.abs(x - attackX) <= HANDLE_HIT_PX) return { kind: "attack" };
    if (Math.abs(x - releaseX) <= HANDLE_HIT_PX) return { kind: "release" };
    if (x < width * EDGE_ZONE_FRAC) return { kind: "attack" };
    if (x > width * (1 - EDGE_ZONE_FRAC)) return { kind: "release" };
    return { kind: "body" };
}

/** 区域对应的鼠标指针形状。 */
export function cursorForZone(zone: PreviewZone): string {
    return zone.kind === "body" ? "move" : "ew-resize";
}

/**
 * 一次手势的起点快照。
 *
 * 【为什么必须快照】主体拖动的纵向换算依赖画布**当前的**纵轴定标，而深度一改，
 * 定标（按峰值自适应）也跟着变 —— 若每帧都重新取标尺，拖动会变成非线性甚至
 * 反向。起点取一次，整段手势按同一套几何走。
 */
export interface PreviewGestureSnapshot {
    attackMs: number;
    releaseMs: number;
    depthCents: number;
    startPhaseDeg: number;
    /** 可见窗口时长（ms）。 */
    windowMs: number;
    /** 画布宽度（CSS 像素）。 */
    widthPx: number;
    /** 一个可见周期的像素宽度（相位换算用）。 */
    cycleWidthPx: number;
    /** 纵向每像素对应的 cents（与绘制同一套换算）。 */
    centsPerPx: number;
}

function clamp(value: number, min: number, max: number): number {
    return Math.min(max, Math.max(min, value));
}

/** 相位取模到 [0,360)。 */
export function wrapPhaseDeg(value: number): number {
    const wrapped = value % 360;
    const positive = wrapped < 0 ? wrapped + 360 : wrapped;
    // `-0` 与 `0` 数值相等但 `Object.is` 不等；归一成 `0` 以免下游断言 / 显示出现 "-0°"。
    return Object.is(positive, -0) ? 0 : positive;
}

/**
 * 由手势位移求出要写进草稿的字段。
 *
 * | 区域 | 映射 |
 * | - | - |
 * | 渐入 | 水平位移 → `attackMs`（`dx/width × windowMs`），钳 `0..windowMs/2` |
 * | 渐出 | 同上（向右加长）→ `releaseMs` |
 * | 主体 | `dx → startPhaseDeg`（一个可见周期 = 360°）；`dy → depthCents`（向上加深） |
 */
export function applyPreviewGesture(
    zone: PreviewZone,
    snapshot: PreviewGestureSnapshot,
    deltaX: number,
    deltaY: number,
): Partial<VibratoPreset> {
    const width = snapshot.widthPx > 0 ? snapshot.widthPx : 1;
    const maxHalf = Math.max(1, snapshot.windowMs) / 2;
    const msPerPx = snapshot.windowMs / width;

    if (zone.kind === "attack") {
        return { attackMs: clamp(snapshot.attackMs + deltaX * msPerPx, 0, maxHalf) };
    }
    if (zone.kind === "release") {
        return { releaseMs: clamp(snapshot.releaseMs + deltaX * msPerPx, 0, maxHalf) };
    }

    const cycleWidthPx = snapshot.cycleWidthPx > 0 ? snapshot.cycleWidthPx : width;
    const phase = wrapPhaseDeg(snapshot.startPhaseDeg + (deltaX / cycleWidthPx) * 360);
    const centsPerPx = snapshot.centsPerPx > 0 ? snapshot.centsPerPx : 1;
    const depthCents = clamp(
        snapshot.depthCents - deltaY / centsPerPx,
        VIBRATO_LIMITS.depthCents.min,
        VIBRATO_LIMITS.depthCents.max,
    );
    return { startPhaseDeg: phase, depthCents };
}

/**
 * 求一个可见周期的像素宽度。
 *
 * 与渲染端同一套有效速率：`hz` 模式直接用 `rateHz`，`cycles` 模式把整段周期数
 * 摊到窗口时长上（见 `vibratoCurve.ts` 的 `effectiveRateHz`）。
 */
export function cycleWidthPxFor(preset: VibratoPreset, widthPx: number, windowMs: number): number {
    const windowSec = Math.max(1e-6, windowMs / 1000);
    const rateHz =
        preset.rateMode === "cycles"
            ? Math.max(0, preset.cycles) / windowSec
            : Math.max(0, preset.rateHz);
    if (!(rateHz > 0)) return widthPx;
    return widthPx / (rateHz * windowSec);
}

/** 手柄的归一化横向位置：渐入在 `attackMs` 处，渐出在 `windowMs - releaseMs` 处。 */
export function handleLayoutFor(
    preset: Pick<VibratoPreset, "attackMs" | "releaseMs">,
    windowMs: number,
): PreviewHandleLayout {
    const window = Math.max(1, windowMs);
    return {
        attackFrac: clamp(preset.attackMs / window, 0, 1),
        releaseFrac: clamp(1 - preset.releaseMs / window, 0, 1),
    };
}
