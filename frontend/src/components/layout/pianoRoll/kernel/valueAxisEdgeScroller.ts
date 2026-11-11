/**
 * 参数编辑器内核 · 值轴边缘滚动积分器。
 *
 * 【主要内容】
 * 1. 值轴专用手感常量（带宽 / 满带速度 / 软启动 / **单帧上限**）；
 * 2. `resolveValueAxisEdgeStep`：比例 + 帧时长 + span → Δ值（纯函数）；
 * 3. `createValueAxisEdgeScrollAxis`：把值域中心包装成 `EdgeScrollAxis`；
 * 4. `valueAxisScrollable`：值域是否还有可移动的居中范围。
 *
 * 【为什么单位必须是「值」】
 * 值轴的物理刻度是参数值。旧实现把它翻译成 1600px 的 `scrollTop` 再翻译回来，
 * 于是「单帧最多走多远」这件事在值域上没有任何约束：一次卡顿帧可以跨过 7px，
 * 放大后甚至跨过 0.4 个半音。用户报告的「步进太大、抓不住位置」就是它。
 *
 * 【手感标定（视口 600px、pitch span=24、25px/半音）】
 * | 距边缘 | ratio | 屏幕速度 | 说明 |
 * | 20px（刚进带） | 1.00 | 60 px/s | 满带：10 秒走完一屏 |
 * | 10px | 0.50 | 15 px/s | 中段 |
 * | 4px | 0.20 | 2.4 px/s | 微调区入口 |
 * | 2px | 0.10 | 0.6 px/s | 几乎静止，可精确停位 |
 * 旧实现（18px / 1.6 / 0.12）在 2px 处是 2.2 px/s、在 10px 处是 28.6 px/s。
 *
 * 【为什么单帧上限用「视口比例」而不是像素】
 * 视口高会变、值域 span 会变，只有「占视口的比例」能保证任何缩放下的最大跳跃
 * 都是同一段视觉距离。
 *
 * 【它不做什么】
 * - 不排帧、不读时钟（`edgeScrollDriver` 的事）；
 * - 不把中心换算成 `scrollTop`（滚动条自己按 `verticalScrollMapping` 同步）；
 * - 不决定可滚范围（调用方给 `canScroll`）。
 *
 * 【违反边界的症状】
 * 若这里重新引入 `scrollTop`，症状是「缩放后上下滚动的速度跟着变」以及
 * 「拖动时视图突然跳一段」—— 因为 `scrollTop` 代理域与值域不是线性同构。
 */
import {
    EDGE_SCROLL_MAX_AXIS_FRAME_MS,
    type EdgeScrollAxis,
    type EdgeScrollAxisPolicy,
} from "./edgeScrollAxis";

/** 值轴边缘带宽：只占 20px，避免块靠近上下边时误触发。 */
export const VALUE_EDGE_SCROLL_BAND_PX = 20;

/** 满带速度：每秒移动 0.10 个可见值域高度（10 秒走完一屏）。 */
export const VALUE_EDGE_SCROLL_VIEWPORTS_PER_SEC = 0.1;

/** 拖出视口后不再加速；值轴不需要时间轴那种 1.5 倍"甩出去"的手感。 */
export const VALUE_EDGE_SCROLL_MAX_RATIO = 1;

/** 软启动：靠近带内缘时速度按 ratio^2 增长（二次曲线 ⇒ 真正的微调区）。 */
export const VALUE_EDGE_SCROLL_RAMP_EXPONENT = 2;

/**
 * 单帧最多走 0.0025 个视口（600px 视口约 1.5px）。
 *
 * 【为什么是这个数】满带标称速度 0.10 视口/秒在 60Hz 下单帧是 0.00167 视口，
 * 所以这个上限**不会**在正常帧率下生效 —— 否则 60Hz 会被削到 0.09 视口/秒、
 * 而 120Hz 不受影响，同一段拖拽在两种屏幕上快慢不同。它只在 ≥25ms 的卡顿帧上
 * 生效（0.0025 / 0.10 = 25ms），把「补一帧」限制在 1.5px 以内。
 */
export const VALUE_EDGE_SCROLL_MAX_STEP_VIEWPORTS = 0.0025;

/** 值轴不接受长帧补偿：超过 32ms 的帧按 32ms 算。 */
export const VALUE_EDGE_SCROLL_MAX_FRAME_MS = EDGE_SCROLL_MAX_AXIS_FRAME_MS;

/** 值轴手感策略（驱动与可滚性判断共用的唯一来源）。 */
export const VALUE_EDGE_SCROLL_POLICY: EdgeScrollAxisPolicy = {
    bandPx: VALUE_EDGE_SCROLL_BAND_PX,
    maxRatio: VALUE_EDGE_SCROLL_MAX_RATIO,
    rampExponent: VALUE_EDGE_SCROLL_RAMP_EXPONENT,
};

/** 值域是否还有可移动的居中范围；span >= range 时滚动条不应再产生位移。 */
export function valueAxisScrollable(input: {
    readonly min: number;
    readonly max: number;
    readonly span: number;
}): boolean {
    if (
        !Number.isFinite(input.min) ||
        !Number.isFinite(input.max) ||
        !Number.isFinite(input.span)
    ) {
        return false;
    }
    const range = Math.max(0, input.max - input.min);
    return range - Math.max(0, input.span) > 1e-9;
}

/**
 * 比例 + 帧时长 + span → 本帧 Δ值。
 *
 * 推导：满带速度 = `viewportsPerSec × span` 值单位/秒；乘以比例与帧时长即本帧
 * 位移；最后夹到 `maxStepViewports × span`（单帧上限）。
 *
 * 特殊说明 1：非法输入一律返回 0（宁可这一帧不滚，也不要把 NaN 写进值域中心）。
 * 特殊说明 2：**返回值的符号跟随 ratio**（正 = 往「后」= 更低的值）。取负号
 * 是调用方（`apply`）的事，因为只有它知道「中心增大 = 看到更高的值」。
 */
export function resolveValueAxisEdgeStep(args: {
    readonly ratio: number;
    readonly frameMs: number;
    readonly span: number;
}): number {
    const span = Number.isFinite(args.span) ? Math.max(0, args.span) : 0;
    const frameMs = Number.isFinite(args.frameMs) ? Math.max(0, args.frameMs) : 0;
    const ratio = Number.isFinite(args.ratio) ? args.ratio : 0;
    if (span <= 0 || frameMs <= 0 || ratio === 0) return 0;
    const perSecond = ratio * VALUE_EDGE_SCROLL_VIEWPORTS_PER_SEC * span;
    const raw = (perSecond * frameMs) / 1000;
    const cap = VALUE_EDGE_SCROLL_MAX_STEP_VIEWPORTS * span;
    return Math.max(-cap, Math.min(cap, raw));
}

export interface ValueAxisEdgeScrollAxisOptions {
    /** 当前可见值域高度（值单位）。 */
    readonly getSpan: () => number;
    /** 当前值域中心（值单位）。 */
    readonly getCenter: () => number;
    /** 写入新的值域中心；实现方负责钳制（`clampViewport`）。 */
    readonly setCenter: (center: number) => void;
    /** span >= range 时为 false：驱动据此不排帧。 */
    readonly canScroll: () => boolean;
}

/**
 * 把「值域中心」包装成一个轴。
 *
 * 【符号】驱动传进来的 `ratio` 正 = 指针偏「后」（下）。值域中心增大 = 看到更高
 * 的值，所以这里对步长取**负号**。这是本模块唯一容易搞反的地方。
 *
 * 【为什么写完要读回】中心会被 `clampViewport` 夹在可移动范围里。若只在写之前
 * 比较，滚到边界后 `apply` 会一直返回 true，rAF 永不停止（CPU 空转 + 每帧标脏）。
 */
export function createValueAxisEdgeScrollAxis(
    options: ValueAxisEdgeScrollAxisOptions,
): EdgeScrollAxis {
    return {
        id: "y",
        policy: VALUE_EDGE_SCROLL_POLICY,
        maxFrameMs: VALUE_EDGE_SCROLL_MAX_FRAME_MS,
        canScroll: () => options.canScroll(),
        apply: (ratio, frameMs) => {
            if (!options.canScroll()) return false;
            // 指针偏向下方（ratio > 0）⇒ 想看更低的参数值 ⇒ 视口中心下移。
            const delta = -resolveValueAxisEdgeStep({
                ratio,
                frameMs,
                span: options.getSpan(),
            });
            if (Math.abs(delta) < 1e-9) return false;
            const before = options.getCenter();
            const next = before + delta;
            if (!Number.isFinite(next) || next === before) return false;
            options.setCenter(next);
            // 写完读回：被钳住时返回 false，让驱动停止排帧。
            return options.getCenter() !== before;
        },
    };
}
