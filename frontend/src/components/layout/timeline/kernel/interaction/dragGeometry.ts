/**
 * 时间轴渲染内核 · 拖拽几何换算
 *
 * 【主要内容】
 * 把「指针在内容坐标下的水平位移」换算为 clip 的新起始时间，并钳制到工程范围。
 *
 * 【作用】
 * 内核手势只负责**几何**：命中、位移换算、目标轨道判定。编辑语义（吸附、事务、
 * 乐观更新、后端提交）一律留在 React 侧复用既有实现——两处都做吸附会让规则分叉
 * （旧实现的吸附点是 `snapTimelineDetailed`，带轨道间避让与网格步长等上下文）。
 *
 * 【与其他模块的关系】
 * - 上游：`host/timelineKernelHost` 的 `clip-drag` 手势在每帧调用。
 * - 下游：回调把结果交给 React 侧（`TimelinePanel` 的拖拽提交链路）。
 * - 独立性：纯函数，无 DOM / React 依赖，可直接单测。
 *
 * 【设计约束】
 * 1. 钳制在**写入时一次算清**（与 `ScrollKernel` 同一约定）：调用方拿到的
 *    `startSec` 恒为合法值，不存在「先写越界值、读回被修正」的中间态。
 * 2. 右边界按「工程长度 − clip 长度」钳制：clip 不允许越出工程末端。
 * 3. `pxPerSec` 非法（0 / NaN / 负数）时退化为「不产生位移」而不是产生 `NaN`
 *    ——拖拽热路径上一旦出现 `NaN`，几何与命中会同时失效且难以定位。
 */

/** 拖拽位移换算参数。 */
export interface DragDeltaArgs {
    /** 指针相对按下位置的水平位移（内容坐标 CSS px，右为正）。 */
    readonly deltaContentXPx: number;
    /** 当前水平缩放（px/秒）。 */
    readonly pxPerSec: number;
    /** clip 按下时的起始时间（秒）。 */
    readonly startSec: number;
    /** clip 长度（秒），用于右边界钳制。 */
    readonly lengthSec: number;
    /** 工程总时长（秒）。 */
    readonly projectSec: number;
}

/** 拖拽位移换算结果。 */
export interface DragDeltaResult {
    /** 钳制后的新起始时间（秒，恒 >= 0）。 */
    readonly startSec: number;
    /** 实际生效的时间位移（秒，已含钳制）。 */
    readonly deltaSec: number;
}

/**
 * 把内容坐标位移换算为新的起始时间。
 *
 * 流程：位移 → 时间位移（除以 `pxPerSec`）→ 加上原起始时间 → 钳制到
 * `[0, max(0, projectSec - lengthSec)]` → 回算实际生效的位移。
 *
 * @param args 换算参数。
 * @returns 新起始时间与实际生效位移。
 */
export function resolveDragDelta(args: DragDeltaArgs): DragDeltaResult {
    const pxPerSec = Number.isFinite(args.pxPerSec) && args.pxPerSec > 0 ? args.pxPerSec : 0;
    const rawDelta = pxPerSec > 0 ? args.deltaContentXPx / pxPerSec : 0;
    const lengthSec = Number.isFinite(args.lengthSec) ? Math.max(0, args.lengthSec) : 0;
    const projectSec = Number.isFinite(args.projectSec) ? Math.max(0, args.projectSec) : 0;
    const maxStart = Math.max(0, projectSec - lengthSec);
    const baseStart = Number.isFinite(args.startSec) ? args.startSec : 0;
    const startSec = Math.min(maxStart, Math.max(0, baseStart + rawDelta));
    return { startSec, deltaSec: startSec - baseStart };
}

/** trim 的边。 */
export type TrimEdge = "left" | "right";

/** trim 换算参数。 */
export interface TrimEdgeArgs {
    /** 拖动的边。 */
    readonly edge: TrimEdge;
    /** 指针相对按下位置的水平位移（内容坐标 CSS px，右为正）。 */
    readonly deltaContentXPx: number;
    readonly pxPerSec: number;
    /** clip 按下时的起始时间（秒）。 */
    readonly startSec: number;
    /** clip 按下时的长度（秒）。 */
    readonly lengthSec: number;
    /** 工程总时长（秒）。 */
    readonly projectSec: number;
    /** 允许的最小长度（秒）：防止裁到 0 长度而无法再选中。 */
    readonly minLengthSec: number;
}

/** trim 换算结果。 */
export interface TrimEdgeResult {
    /** 新的起始时间（秒）。 */
    readonly startSec: number;
    /** 新的长度（秒）。 */
    readonly lengthSec: number;
    /**
     * 实际生效的变化量（秒）。
     *
     * 特殊说明：左边缘拖动改的是 `startSec`、右边缘改的是 `lengthSec`，
     * 调用方据此决定把哪个值写进乐观态——用同一个字段表达可避免调用方
     * 自己判断边缘类型而写错分支。
     */
    readonly deltaSec: number;
}

/**
 * 把边缘拖拽位移换算为 clip 的新起始时间与长度。
 *
 * 规则：
 * - **左边缘**：右端固定（`startSec + lengthSec` 不变），拖右 = 裁短、拖左 = 延长；
 * - **右边缘**：左端固定，拖右 = 延长、拖左 = 裁短；
 * - 两个方向都保证长度 >= `minLengthSec`，且不越出 `[0, projectSec]`。
 *
 * @param args 换算参数。
 * @returns 新的起始时间、长度与实际变化量。
 */
export function resolveTrimEdge(args: TrimEdgeArgs): TrimEdgeResult {
    const pxPerSec = Number.isFinite(args.pxPerSec) && args.pxPerSec > 0 ? args.pxPerSec : 0;
    const rawDelta = pxPerSec > 0 ? args.deltaContentXPx / pxPerSec : 0;
    const minLengthSec =
        Number.isFinite(args.minLengthSec) && args.minLengthSec > 0 ? args.minLengthSec : 1e-6;
    const projectSec = Number.isFinite(args.projectSec) ? Math.max(0, args.projectSec) : 0;
    const startSec = Number.isFinite(args.startSec) ? Math.max(0, args.startSec) : 0;
    const lengthSec = Number.isFinite(args.lengthSec)
        ? Math.max(minLengthSec, args.lengthSec)
        : minLengthSec;

    if (args.edge === "left") {
        // 右端固定：newStart + newLength = 原右端。
        const rightEdge = startSec + lengthSec;
        const maxStart = Math.max(0, rightEdge - minLengthSec);
        const nextStart = Math.min(maxStart, Math.max(0, startSec + rawDelta));
        return { startSec: nextStart, lengthSec: rightEdge - nextStart, deltaSec: nextStart - startSec };
    }

    // 右边缘：左端固定，长度受「工程末端 − 起点」与最小长度双向约束。
    const maxLength = Math.max(minLengthSec, projectSec - startSec);
    const nextLength = Math.min(maxLength, Math.max(minLengthSec, lengthSec + rawDelta));
    return { startSec, lengthSec: nextLength, deltaSec: nextLength - lengthSec };
}

/**
 * 把内容坐标的纵向位置换算为目标轨道下标。
 *
 * 规则：按行高取整，并钳制到 `[0, trackCount - 1]`；轨道数为 0 时返回 -1
 * （调用方据此放弃本次拖拽的跨轨部分）。
 *
 * @param contentY 指针的内容坐标 y（CSS px）。
 * @param rowHeight 单条轨道高度（CSS px）。
 * @param trackCount 轨道总数。
 * @returns 目标轨道下标；无轨道时为 -1。
 */
export function resolveTargetTrackIndex(
    contentY: number,
    rowHeight: number,
    trackCount: number,
): number {
    if (!Number.isFinite(trackCount) || trackCount <= 0) return -1;
    const safeRowHeight = Number.isFinite(rowHeight) && rowHeight > 0 ? rowHeight : 1;
    const y = Number.isFinite(contentY) ? Math.max(0, contentY) : 0;
    const index = Math.floor(y / safeRowHeight);
    return Math.min(trackCount - 1, Math.max(0, index));
}
