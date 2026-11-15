/**
 * 参数编辑器内核 · 边缘自动滚屏的轴抽象。
 *
 * 【主要内容】
 * 1. `EdgeScrollAxis`：一个轴只需要回答「带内比例 + 本帧时长 → 本帧位移」；
 * 2. `EdgeScrollAxisPolicy`：带宽 / 比例上限 / 软启动指数；
 * 3. `resolveEdgeScrollRatio`：指针位置 → 带内比例的**唯一**几何（转发
 *    `shared/edgeAutoScroll` 的实现，几何本身只有一份）；
 * 4. `createHorizontalEdgeScrollAxis`：水平轴（像素域）的标准实现。
 *
 * 【为什么把单位交给轴】
 * 水平轴的单位是 CSS px（`scrollLeft`），竖直轴的单位是**参数值**（值域中心）。
 * 旧实现要求两个轴都先换算成 px，于是值轴被迫借道 1600px 的 `scrollTop` 代理域，
 * 单帧步长在「值」这个单位上完全没有上限 —— 一次卡顿帧能跨过好几个像素对应的
 * 参数值，放大后甚至跨过 0.4 个半音。轴自己持有单位之后，驱动不需要知道任何换算，
 * 值轴也能直接表达「这一帧最多走 0.0015 个视口」。
 *
 * 【它不做什么】
 * - 不排帧、不读时钟（那是 `edgeScrollDriver` 的事）；
 * - 不碰 React state、不认识工具与参数块；
 * - 不决定「滚了之后要重算什么」（调用方通过 `onScrolled` 决定）。
 *
 * 【违反边界的症状】
 * 若某个轴又开始返回「像素」而不是自己的单位，症状是：缩放后滚动手感随缩放变化
 * （值域越窄滚得越快），或者「指针停在边缘不动，视图却在几帧后突然跳一段」。
 */
import {
    EDGE_SCROLL_BAND_PX,
    EDGE_SCROLL_MAX_RATIO,
    resolveEdgeScrollRatio as resolveSharedEdgeScrollRatio,
} from "../../shared/edgeAutoScroll";

/** 单帧时长的上限（毫秒）。超过 32ms 的帧按 32ms 算，见 `edgeScrollDriver`。 */
export const EDGE_SCROLL_MAX_AXIS_FRAME_MS = 32;

/** 位移小于这个量（轴自己的单位）时视为「本帧没动」。 */
export const EDGE_SCROLL_MIN_STEP = 0.01;

/**
 * 单轴手感策略。
 *
 * 三个字段都是**必填**：轴的手感是它身份的一部分，缺省值散落在调用点会让
 * 「同一个面板里两个轴用不同带宽」这种缺陷无法被发现。默认值集中在
 * `DEFAULT_EDGE_SCROLL_AXIS_POLICY`。
 */
export interface EdgeScrollAxisPolicy {
    /** 触发带宽（CSS px）。 */
    readonly bandPx: number;
    /** 指针拖出视口后的比例上限。 */
    readonly maxRatio: number;
    /** 比例软启动指数；1 = 线性，>1 = 越靠近带内缘越慢。 */
    readonly rampExponent: number;
}

/** 水平轴（时间轴 / 参数编辑器横向）的既有手感：32px 带、1.5 上限、线性。 */
export const DEFAULT_EDGE_SCROLL_AXIS_POLICY: EdgeScrollAxisPolicy = {
    bandPx: EDGE_SCROLL_BAND_PX,
    maxRatio: EDGE_SCROLL_MAX_RATIO,
    rampExponent: 1,
};

export interface EdgeScrollAxis {
    readonly id: "x" | "y";
    readonly policy: EdgeScrollAxisPolicy;
    /** 当前是否可滚；不可滚时驱动既不排帧也不把本帧算作 moved。 */
    canScroll(): boolean;
    /**
     * 应用一帧位移。
     *
     * @param ratio 带内比例；**正 = 指针偏向「后」侧**（右 / 下）。
     * @param frameMs 本帧时长（毫秒，已由驱动钳制到轴自报的上限）。
     * @returns 是否真的产生了位移（写完读回比较，被钳住时返回 false）。
     */
    apply(ratio: number, frameMs: number): boolean;
    /** 本帧时长的上限（毫秒）；缺省 32。 */
    readonly maxFrameMs?: number;
}

/**
 * 指针位置 → 带内比例（`-maxRatio .. maxRatio`）。
 *
 * 【符号约定】正 = 指针靠近「后」侧边界（右 / 下）。
 * 【退化输入】视口宽/高非正、指针非有限 → 0（本帧不滚）。
 *
 * 实现转发 `shared/edgeAutoScroll.resolveEdgeScrollRatio`：几何只有一份，
 * 时间轴与参数编辑器不可能再各写一套（那正是"两个面板手感不一样"的来源）。
 */
export function resolveEdgeScrollRatio(args: {
    readonly pointerPx: number;
    readonly minPx: number;
    readonly maxPx: number;
    readonly policy: EdgeScrollAxisPolicy;
}): number {
    return resolveSharedEdgeScrollRatio(args);
}

export interface HorizontalEdgeScrollAxisOptions {
    /** 读**内核真值**（绘制坐标），不要读原生 `scrollLeft` 镜像。 */
    readonly getScrollLeft: () => number;
    readonly setScrollLeft: (next: number) => void;
    readonly getMaxScrollLeft: () => number;
    readonly getMinScrollLeft: () => number;
    /** 满带速度（CSS px/秒）。参数编辑器 420，时间轴 720。 */
    readonly maxSpeedPxPerSec: number;
    readonly policy?: EdgeScrollAxisPolicy;
}

/**
 * 水平轴：像素域，速度 × 帧时长，再夹到 `[min, max]`。
 *
 * 【为什么单独成工厂而不是写进调用点】这段算术有三条被单测钉住的约定：
 * 1. 步长与**真实帧时长**成正比（与事件频率无关）；
 * 2. 下界可以是负数（同步时间轴时左侧预留留白是合法可滚区间），写死 0 会让
 *    向左滚每帧被夹回 0、与内核下界打架 —— 用户看到的是"闪现"；
 * 3. 位置被钳住时返回 false，驱动据此停止排帧（不空转）。
 *
 * 它不读时钟：帧时长由驱动算好后传进来（一次手势一个时钟）。
 */
export function createHorizontalEdgeScrollAxis(
    options: HorizontalEdgeScrollAxisOptions,
): EdgeScrollAxis {
    const policy = options.policy ?? DEFAULT_EDGE_SCROLL_AXIS_POLICY;
    return {
        id: "x",
        policy,
        maxFrameMs: EDGE_SCROLL_MAX_AXIS_FRAME_MS,
        canScroll: () => options.getMaxScrollLeft() > options.getMinScrollLeft(),
        apply: (ratio, frameMs) => {
            const max = options.getMaxScrollLeft();
            const min = options.getMinScrollLeft();
            if (!(max > min)) return false;
            const before = options.getScrollLeft();
            const delta = (ratio * options.maxSpeedPxPerSec * Math.max(0, frameMs)) / 1000;
            if (!Number.isFinite(delta) || Math.abs(delta) < EDGE_SCROLL_MIN_STEP) return false;
            const next = Math.min(max, Math.max(min, before + delta));
            if (Math.abs(next - before) < EDGE_SCROLL_MIN_STEP) return false;
            options.setScrollLeft(next);
            return true;
        },
    };
}
