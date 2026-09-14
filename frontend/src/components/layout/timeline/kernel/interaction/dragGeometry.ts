/**
 * 时间轴渲染内核 · 拖拽几何换算
 *
 * 【主要内容】
 * 把「指针在内容坐标下的水平位移」换算为 clip 的新起始时间，并钳制到工程范围；
 * 把纵向位置换算为目标轨道下标，并判定「是否落在全部轨道之下」（新建轨道哨兵）。
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
 * 2. 移动（`resolveDragDelta`）**只受下界 0 约束，没有上界**：右移应能越过当前
 *    工程末端，由 `moveClipStart` 的自动扩展与后端 `ensure_project_end_sec` 增长
 *    工程时长。旧的「工程长度 − clip 长度」上界是自指边界（被拖 clip 自己定义上界），
 *    会让自动扩展永不触发，表现为"向右拖有隐形边界"——见该函数注释。
 *
 *    裁切 / 拉伸（`resolveTrimEdge`）同样**不按工程末端钳制**：它改的是**长度**，
 *    没有"自动扩展工程"的语义承接，因此与旧实现一致，上界只保留
 *    `TRIM_MAX_LENGTH_SEC` 这个防呆值（旧实现 `useEditDrag` 的
 *    `clamp(…, minLen, 10_000)` 同源）；越出工程末端的部分由渲染管线按尾静音处理，
 *    工程时长按 `resolveScrollableProjectSec`（后端时长与最右 clip 末端的较大者）
 *    自然增长。把这里钳到 `projectSec` 会让「把 clip 裁/拉到工程外」整体失效，
 *    且用户看不到任何提示（与"向右拖有隐形边界"同类）。
 * 3. `pxPerSec` 非法（0 / NaN / 负数）时退化为「不产生位移」而不是产生 `NaN`
 *    ——拖拽热路径上一旦出现 `NaN`，几何与命中会同时失效且难以定位。
 */

/**
 * 裁切 / 拉伸允许的最大长度（秒）。
 *
 * 与旧实现 `useEditDrag` 的 `clamp(…, minLen, 10_000)` 同源：只是一个防呆上界
 * （防止指针坐标异常时算出天文数字），**不是**工程末端钳制——越出工程末端是
 * 允许的，工程时长会随之自动增长（见文件头设计约束 2）。
 */
export const TRIM_MAX_LENGTH_SEC = 10_000;

/** 拖拽位移换算参数。 */
export interface DragDeltaArgs {
    /** 指针相对按下位置的水平位移（内容坐标 CSS px，右为正）。 */
    readonly deltaContentXPx: number;
    /** 当前水平缩放（px/秒）。 */
    readonly pxPerSec: number;
    /** clip 按下时的起始时间（秒）。 */
    readonly startSec: number;
}

/** 拖拽位移换算结果。 */
export interface DragDeltaResult {
    /** 新的起始时间（秒，恒 >= 0；上界不设限，见函数注释）。 */
    readonly startSec: number;
    /** 实际生效的时间位移（秒）。 */
    readonly deltaSec: number;
}

/**
 * 把内容坐标位移换算为新的起始时间。
 *
 * 流程：位移 → 时间位移（除以 `pxPerSec`）→ 加上原起始时间 → **只受下界 0 约束**。
 *
 * 【为什么没有上界（去掉 `projectSec − lengthSec` 的原因）】
 * 旧实现把结果钳到 `[0, projectSec − lengthSec]`，理由是「clip 不允许越出工程末端」。
 * 但喂进来的 `projectSec` 并不是后端时长真值，而是
 * `resolveScrollableProjectSec()` = `max(后端时长, 最右 clip 末端)`。当被拖的 clip
 * **本身就是最右那个**时，上界由它自己的末端推出 —— **自指边界**：它永远无法越过，
 * 且因为本函数保证 `startSec + lengthSec <= projectSec`，`moveClipStart` 里那段
 * 「拖动超出边界时自动扩展工程时长」与后端的 `ensure_project_end_sec` **都变成死代码**，
 * 边界永不增长。现场表现即「使劲向右拖会在某个位置被卡住，像有隐形边界」。
 *
 * 旧实现（重构前的 `useClipDrag`）**完全没有上界**（只做 `Math.max(0, …)`），
 * 因此这是内核迁移引入的回归。去掉上界后，工程时长由既有的自动扩展路径负责增长
 * —— 那正是它们被写出来的目的。
 *
 * 特殊说明：入参**刻意不含** `projectSec` / `lengthSec`。它们曾是右边界钳制的来源，
 * 现在既无用途，留在签名里只会诱导后来者把上界加回来（缺陷复发）。去掉上界后
 * 工程时长的增长由 `moveClipStart` 的自动扩展与 `ensure_project_end_sec` 负责。
 *
 * @param args 换算参数。
 * @returns 新起始时间与实际生效位移。
 */
export function resolveDragDelta(args: DragDeltaArgs): DragDeltaResult {
    const pxPerSec = Number.isFinite(args.pxPerSec) && args.pxPerSec > 0 ? args.pxPerSec : 0;
    const rawDelta = pxPerSec > 0 ? args.deltaContentXPx / pxPerSec : 0;
    const baseStart = Number.isFinite(args.startSec) ? args.startSec : 0;
    const startSec = Math.max(0, baseStart + rawDelta);
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
    /** 允许的最小长度（秒）；<= 0 时退化为一个极小正数（旧实现 `minLen = 0`）。 */
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
 * - 两个方向都保证长度 >= `minLengthSec`；右边缘上界为 `TRIM_MAX_LENGTH_SEC`
 *   （**不**钳到工程末端，见文件头设计约束 2）。
 *
 * @param args 换算参数。
 * @returns 新的起始时间、长度与实际变化量。
 */
export function resolveTrimEdge(args: TrimEdgeArgs): TrimEdgeResult {
    const pxPerSec = Number.isFinite(args.pxPerSec) && args.pxPerSec > 0 ? args.pxPerSec : 0;
    const rawDelta = pxPerSec > 0 ? args.deltaContentXPx / pxPerSec : 0;
    const minLengthSec =
        Number.isFinite(args.minLengthSec) && args.minLengthSec > 0 ? args.minLengthSec : 1e-6;
    const startSec = Number.isFinite(args.startSec) ? Math.max(0, args.startSec) : 0;
    const lengthSec = Number.isFinite(args.lengthSec)
        ? Math.max(minLengthSec, args.lengthSec)
        : minLengthSec;

    if (args.edge === "left") {
        // 右端固定：newStart + newLength = 原右端。
        const rightEdge = startSec + lengthSec;
        const maxStart = Math.max(0, rightEdge - minLengthSec);
        const nextStart = Math.min(maxStart, Math.max(0, startSec + rawDelta));
        return {
            startSec: nextStart,
            lengthSec: rightEdge - nextStart,
            deltaSec: nextStart - startSec,
        };
    }

    // 右边缘：左端固定，长度受最小长度与防呆上界双向约束。
    const nextLength = Math.min(TRIM_MAX_LENGTH_SEC, Math.max(minLengthSec, lengthSec + rawDelta));
    return { startSec, lengthSec: nextLength, deltaSec: nextLength - lengthSec };
}

/** 淡变的侧。 */
export type FadeSide = "in" | "out";

/** 淡变角拖拽参数。 */
export interface FadeDragArgs {
    readonly side: FadeSide;
    /** 指针相对按下位置的水平位移（内容坐标 CSS px，右为正）。 */
    readonly deltaContentXPx: number;
    readonly pxPerSec: number;
    /** 按下时的淡变长度（秒）。 */
    readonly currentSec: number;
    /** clip 长度（秒）：淡变不能长于 clip 本身。 */
    readonly lengthSec: number;
}

/** 淡变角拖拽结果。 */
export interface FadeDragResult {
    /** 新的淡变长度（秒）。 */
    readonly fadeSec: number;
    /** 实际生效的变化量（秒）。 */
    readonly deltaSec: number;
}

/**
 * 把淡变角拖拽位移换算为新的淡变长度。
 *
 * 方向约定（与既有实现一致）：
 * - **淡入角**（clip 左边缘）：向右拖 = 变长、向左拖 = 变短；
 * - **淡出角**（clip 右边缘）：向左拖 = 变长、向右拖 = 变短。
 *
 * 两者都钳制到 `[0, lengthSec]`——淡变长于 clip 没有意义，且会让绘制端的
 * 曲线超出矩形。
 *
 * @param args 换算参数。
 * @returns 新的淡变长度与实际变化量。
 */
export function resolveFadeDrag(args: FadeDragArgs): FadeDragResult {
    const pxPerSec = Number.isFinite(args.pxPerSec) && args.pxPerSec > 0 ? args.pxPerSec : 0;
    const rawDelta = pxPerSec > 0 ? args.deltaContentXPx / pxPerSec : 0;
    const lengthSec = Number.isFinite(args.lengthSec) ? Math.max(0, args.lengthSec) : 0;
    const baseSec = Number.isFinite(args.currentSec) ? Math.max(0, args.currentSec) : 0;
    // 淡出角的"向外"方向与淡入相反（在右边缘向左拖才是变长）。
    const signedDelta = args.side === "in" ? rawDelta : -rawDelta;
    const fadeSec = Math.min(lengthSec, Math.max(0, baseSec + signedDelta));
    return { fadeSec, deltaSec: fadeSec - baseSec };
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

/**
 * 判定纵向位置是否落在**最后一条轨道之下**（拖到空白处新建轨道的哨兵）。
 *
 * 【为什么需要它】`resolveTargetTrackIndex` 把越界位置钳制回最后一行——这对
 * 「拖拽移动 / trim」是正确的（落点必须落在已有轨道上），但会**吞掉**「用户想
 * 拖到轨道列表下方新建一条轨道」的意图。旧实现靠 `trackIdFromClientY` 返回
 * `null` 表达这件事（`drag.lastTrackId == null` → `dropToNewTrack`）。
 *
 * 判定按**行下标的整数比较**而不是"y < 内容总高"：行高可能不是整数，用浮点高度
 * 比较会在最后一行底部产生一个"看起来仍在行内、实际被判为越界"的窄带。
 *
 * @param contentY 指针的内容坐标 y（CSS px）。
 * @param rowHeight 单条轨道高度（CSS px）。
 * @param trackCount 轨道总数。
 * @returns 是否位于全部轨道之下；无轨道时恒为 false（没有"下方"可言）。
 */
export function isContentYBelowTracks(
    contentY: number,
    rowHeight: number,
    trackCount: number,
): boolean {
    if (!Number.isFinite(trackCount) || trackCount <= 0) return false;
    const safeRowHeight = Number.isFinite(rowHeight) && rowHeight > 0 ? rowHeight : 1;
    const y = Math.max(0, Number.isFinite(contentY) ? contentY : 0);
    return Math.floor(y / safeRowHeight) > trackCount - 1;
}
