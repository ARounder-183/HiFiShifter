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
    /**
     * **未钳制**的指针位移（秒，右为正）。
     *
     * 与 `deltaSec` 的差别只在 `startSec + rawDeltaSec < 0` 时：`deltaSec` 被下界
     * 钳住而本字段不会。Slip 这类"位移即语义"的手势必须用它（见
     * {@link resolveDragDeltaSec}）。
     */
    readonly rawDeltaSec: number;
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
    const rawDelta = resolveDragDeltaSec(args);
    const baseStart = Number.isFinite(args.startSec) ? args.startSec : 0;
    const startSec = Math.max(0, baseStart + rawDelta);
    return { startSec, deltaSec: startSec - baseStart, rawDeltaSec: rawDelta };
}

/**
 * 把内容坐标位移换算为**纯位移**（秒，右为正）—— **不施加任何钳制**。
 *
 * 【为什么必须与 `resolveDragDelta` 分开】后者的下界钳制是**位置**语义（clip 不能
 * 从负时间开始），它同时把返回值里的 `deltaSec` 也钳住了（`deltaSec = startSec −
 * baseStart`）。但 Slip 手势消费的不是位置而是**位移**：`computeSlipWindow` 要的是
 * "源窗口平移多少秒"，其合法域是**任意符号、任意大小**（向左滑出媒体起点 =
 * 前导静音，向右滑出末端 = 尾部静音，二者对称无界）。
 *
 * 复用被钳的 `deltaSec` 会让 Slip 单向卡死：clip 在 `startSec = 5` 时向左最多滑
 * 5 秒，在 `startSec = 0` 时**完全无法向左滑**；而向右因为早已没有上界，是不受限的
 * ——现场表现正是"Slip 只能往一个方向拖"。
 *
 * @param args 换算参数（`startSec` 不参与本函数）。
 * @returns 未钳制的位移（秒）；`pxPerSec` 非法时退化为 0（不产生 NaN）。
 */
export function resolveDragDeltaSec(args: DragDeltaArgs): number {
    const pxPerSec = Number.isFinite(args.pxPerSec) && args.pxPerSec > 0 ? args.pxPerSec : 0;
    return pxPerSec > 0 ? args.deltaContentXPx / pxPerSec : 0;
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

/**
 * 拖放几何：把「屏幕坐标」换算为「轨道下标 + 内容坐标」。
 *
 * 【为什么单独抽出来】素材拖入的落点由三个坐标源合成：内核容器的屏幕矩形、
 * 内核自绘的 scrollLeft/scrollTop、以及当前缩放。面板侧过去直接写在同一段
 * 事件回调里，测试只能靠肉眼；更糟的是，只要其中任一项在每个 render 期换成
 * 新闭包而监听没有重注册，就会悄悄回到**首次挂载时**的行高 / 缩放。
 *
 * 本函数只做几何，不碰 React / DOM；调用方负责提供量测值。
 *
 * 【边界语义】
 * - 指针在容器矩形之外：仍然换算（拖放事件在边缘可能略超出 1px），由纵向
 *   下标决定是否落到哨兵行 / 返回 `null`；
 * - 纵向落在全部轨道之下：`trackIndex = tracks.length`，表示「新建轨道」。
 *   这与旧实现的 `trackIdFromClientY` 返回 null 表达同一件事，但保留了下标，
 *   便于预览画在幽灵行而不是第一行；
 * - 纵向落在轨道上方（负值）：`trackIndex = 0`、`sec` 钳到 0。这是"拖到标尺
 *   上"的常见情形，不能把文件丢到不存在的前一行。
 */
export interface DropGeometryArgs {
    /** 指针屏幕坐标 x（CSS px）。 */
    readonly clientX: number;
    /** 指针屏幕坐标 y（CSS px）。 */
    readonly clientY: number;
    /** 内核容器的屏幕矩形。 */
    readonly containerRect: {
        readonly left: number;
        readonly top: number;
        readonly right?: number;
        readonly bottom?: number;
    };
    /** 内核当前横向滚动（CSS px）。 */
    readonly scrollLeft: number;
    /** 内核当前纵向滚动（CSS px）。 */
    readonly scrollTop: number;
    /** 当前横向缩放（px/秒）。 */
    readonly pxPerSec: number;
    /** 当前轨道行高（CSS px）。 */
    readonly rowHeight: number;
    /** 轨道总数。 */
    readonly trackCount: number;
}

/** 拖放几何结果。 */
export interface DropGeometryResult {
    /** 工程时间（秒，恒 >= 0）。 */
    readonly sec: number;
    /** 指针的内容坐标 x（CSS px）。 */
    readonly contentX: number;
    /** 指针的内容坐标 y（CSS px）；落在轨道上方时为 0。 */
    readonly contentY: number;
    /**
     * 命中的轨道下标；`trackCount` 表示全部轨道之下的「新建轨道」哨兵行。
     *
     * 只有 `trackCount === 0` 时可能为 0 但无对应轨道，调用方应结合
     * `trackCount` 判断。
     */
    readonly trackIndex: number;
    /** 是否落在全部轨道之下（新建轨道）。 */
    readonly belowTracks: boolean;
    /**
     * 指针在目标行内的纵向比例 `[0, 1]`。
     *
     * 用于「末轨底边附近的边界」类判定：调用方按自己的业务决定阈值（时间轴
     * 素材拖放取 1.0，即越过底边才算新建轨道），几何层不藏业务常量。
     */
    readonly rowFractionY: number;
}

/**
 * 把屏幕坐标换算为拖放落点几何。
 *
 * @param args 量测值与指针坐标。
 * @returns 落点几何；数值非法时退化为安全值（不产生 NaN）。
 */
export function resolveDropGeometry(args: DropGeometryArgs): DropGeometryResult {
    const pxPerSec = Number.isFinite(args.pxPerSec) && args.pxPerSec > 0 ? args.pxPerSec : 1;
    const rowHeight = Number.isFinite(args.rowHeight) && args.rowHeight > 0 ? args.rowHeight : 1;
    const rectLeft = Number.isFinite(args.containerRect.left) ? args.containerRect.left : 0;
    const rectTop = Number.isFinite(args.containerRect.top) ? args.containerRect.top : 0;
    const scrollLeft = Number.isFinite(args.scrollLeft) ? Math.max(0, args.scrollLeft) : 0;
    const scrollTop = Number.isFinite(args.scrollTop) ? Math.max(0, args.scrollTop) : 0;
    const clientX = Number.isFinite(args.clientX) ? args.clientX : rectLeft;
    const clientY = Number.isFinite(args.clientY) ? args.clientY : rectTop;
    const trackCount = Number.isFinite(args.trackCount)
        ? Math.max(0, Math.floor(args.trackCount))
        : 0;

    const rawContentX = scrollLeft + (clientX - rectLeft);
    const rawContentY = scrollTop + (clientY - rectTop);
    const contentX = Math.max(0, rawContentX);
    const contentY = Math.max(0, rawContentY);
    const sec = contentX / pxPerSec;
    const rawTrackIndex = Math.floor(contentY / rowHeight);
    const trackIndex = trackCount === 0 ? 0 : Math.max(0, Math.min(trackCount, rawTrackIndex));
    const rowFractionY =
        rowHeight > 0 ? Math.min(1, Math.max(0, (contentY % rowHeight) / rowHeight)) : 0;

    return {
        sec,
        contentX,
        contentY,
        trackIndex,
        belowTracks: trackCount > 0 && trackIndex >= trackCount,
        rowFractionY,
    };
}
