/**
 * 参数编辑器内核 · 拖拽算术（纯函数）
 *
 * 【主要内容】
 * 参数编辑器里重复出现的两处坐标换算：
 * 1. 选区的帧区间 → 采样下标（`selectionIndexRange`，含 clamp）；
 * 2. 帧号 → 采样下标（`frameToIndex`）。
 *
 * 【单位】选区本身已经是**帧**（见 `paramSelection.ts`），所以这里不再有
 * beat → 帧的换算：那一层（`selectionFrameRange` / `beatToFrameDelta` /
 * `frameDeltaToBeat`）随单位改造一并删除。附带好处是"拍 → 帧两端取整不对称"这类
 * 半格错位风险整体消失 —— 选区给过来的就是整数帧，这里只做帧 → 下标投影。
 *
 * 【作用】
 * 这段算术此前在 `usePianoRollInteractions`（3,875 行）里被**抄了 3 遍**：
 * 选区拉伸的 `buildDense`、拉伸起点快照、morph 叠加层构造。三处必须对"选区覆盖
 * 哪些采样点"给出**完全一致**的答案——否则拉伸预览与最终提交会错位，而错位量
 * 只有半格、肉眼难察，属于最难排查的一类缺陷。抽成单一实现后由单测钉住语义。
 *
 * 【与其他模块的关系】
 * - 上游：`usePianoRollInteractions` 的手势路径（选区拉伸 / 形变 / 复制剪切）。
 * - 下游：`selectionEditData`（按 slice 取全分辨率数据）、`polylineGeometry`
 *   （不直接消费，但绘制路径的帧↔下标换算必须与这里同一规则）。
 * - 独立性：纯函数，不依赖 DOM / WebGL / React。
 *
 * 【设计约束（逐条都有单测）】
 * 1. **末帧取 `frameCount - 1`**。选区是半开区间 `[startFrame, start+count)`，
 *    而采样下标是闭区间，因此右端必须退一帧 —— 不退会把选区外的一帧拉进窗口，
 *    正是旧实现（beat 端点 `ceil` 后再投影）那半格错位的来源。
 * 2. **下标必须 clamp 到 `[0, len-1]` 且保证 `startIdx <= endIdx`**。曲线数据的
 *    覆盖窗口由后端按请求给出，选区可能超出它；不 clamp 会 slice 出空数组，让整个
 *    功能**静默失效**（不报错，只是没反应）。
 * 3. **`stride` 归一为 ≥1**。与既有实现的 `Math.max(1, stride)` 一致：
 *    `stride = 0` 会让下标除零得 Infinity，clamp 后落到末位——看似"无害"，语义却
 *    完全错误。
 * 4. **非法输入返回 `null`**，而不是空范围。调用方据此提前退出；返回空范围会让
 *    "没有数据"与"选区为空"混为一谈。
 */

/** 选区覆盖的采样下标范围。 */
export interface SelectionIndexRange {
    /** 起始采样下标（含），已 clamp 到 `[0, len-1]`。 */
    readonly startIdx: number;
    /** 结束采样下标（含），`>= startIdx` 且 `< len`。 */
    readonly endIdx: number;
}

/** 采样段的几何元信息（只需要这几个字段，便于单测构造）。 */
export interface ParamViewLike {
    /** 首个采样值对应的帧号。 */
    readonly startFrame: number;
    /** 采样步长（帧）。 */
    readonly stride: number;
    /** 采样值数组（只用其长度）。 */
    readonly edit: readonly number[];
}

/**
 * 把 `stride` 归一为 ≥1 的整数。
 *
 * 【为什么取整方向要明确写死】`stride` 在本工程恒为整数（取数侧写死 `stride = 1`，
 * 见 `usePianoRollData`），因此取整与否在当前数据下没有实际差别。但**取整方向
 * 必须与渲染路径一致**：`render.ts`、`kernel/scene/curvePoints`、
 * `selectionEditData` 都用 `Math.max(1, Math.floor(stride))`，而
 * `usePianoRollInteractions` 的三处内联实现用的是 `Math.max(1, pv.stride)`（不取整）。
 *
 * 这里取**渲染路径的规则**：一旦真的出现小数 stride，不取整会让"命中/切片用的
 * 下标"与"绘制用的下标"差一格——正是本模块要消除的那类静默错位。单测显式钉住
 * 了这个选择，避免后人当成随手写的。
 */
function normalizeStride(stride: number): number {
    return Math.max(1, Math.floor(stride));
}

/** 把值限制在 `[lo, hi]`。 */
function clampTo(value: number, lo: number, hi: number): number {
    return Math.min(hi, Math.max(lo, value));
}

/**
 * 把帧号换算为采样下标。
 *
 * 流程：`round((frame − startFrame) / stride)`。
 *
 * 特殊说明：用**四舍五入**而不是取整。采样点是离散的，取整会让"帧落在两个采样点
 * 中间"时偏向左侧，与绘制路径（相邻点直线连接）产生半格的系统性偏移。这与
 * `gestureHitTest.curveValueAtPointerFrame` 的规则一致。
 *
 * @param args 换算参数。
 * @returns 采样下标；输入非法时为 `null`（**不 clamp**，由调用方决定边界）。
 */
export function frameToIndex(args: {
    readonly frame: number;
    readonly startFrame: number;
    readonly stride: number;
}): number | null {
    const { frame, startFrame, stride } = args;
    if (!Number.isFinite(frame) || !Number.isFinite(startFrame)) return null;
    if (!Number.isFinite(stride)) return null;
    const idx = Math.round((frame - startFrame) / normalizeStride(stride));
    return Number.isFinite(idx) ? idx : null;
}

/**
 * 把选区的帧区间投影为采样下标范围。
 *
 * 入参是选区的**半开**帧区间（`{startFrame, frameCount}`，即 `ParamSelection` 的元素）；
 * 右端按约束 1 退一帧后各自 `frameToIndex` → clamp 到 `[0, len-1]`，并保证
 * `startIdx <= endIdx`。
 *
 * 特殊说明：`frameCount === 0`（单击留下的零长选区）退化为"只覆盖起点那一帧"，
 * 与旧实现把零长选区投影为单帧的行为一致。
 *
 * @param args 换算参数。
 * @returns 下标范围；输入非法或数据为空时为 `null`。
 */
export function selectionIndexRange(args: {
    /** 选区起点帧（含）。 */
    readonly startFrame: number;
    /** 选区帧数（`>= 0`）。 */
    readonly frameCount: number;
    readonly paramView: ParamViewLike;
}): SelectionIndexRange | null {
    const { startFrame, frameCount, paramView } = args;
    if (!Number.isFinite(startFrame) || !Number.isFinite(frameCount)) return null;
    if (!Array.isArray(paramView.edit) || paramView.edit.length === 0) return null;
    if (!Number.isFinite(paramView.startFrame) || !Number.isFinite(paramView.stride)) return null;

    const last = paramView.edit.length - 1;
    const lastFrame = startFrame + Math.max(0, frameCount - 1);
    const rawStart = frameToIndex({
        frame: startFrame,
        startFrame: paramView.startFrame,
        stride: paramView.stride,
    });
    const rawEnd = frameToIndex({
        frame: lastFrame,
        startFrame: paramView.startFrame,
        stride: paramView.stride,
    });
    if (rawStart === null || rawEnd === null) return null;

    const startIdx = clampTo(rawStart, 0, last);
    const endIdx = clampTo(rawEnd, startIdx, last);
    return { startIdx, endIdx };
}

/** 边缘自动滚动的边缘带宽（CSS px）。 */
export const EDGE_SCROLL_BAND_PX = 32;

/** 边缘自动滚动的单帧最大步长（CSS px）。 */
export const EDGE_SCROLL_MAX_STEP_PX = 18;

/**
 * 把「指针离画布边缘的距离」换算为每帧滚动像素。
 *
 * 流程：指针落在左/右边缘带内时，按「进入带宽的比例」线性加速，
 * 比例上限 1.5（即带外仍可加速到 1.5 倍），再乘以单帧最大步长；
 * 带外返回 0。
 *
 * 特殊说明 1：**方向符号**——左缘返回负值（内容向左滚 = 看到更早的时间），
 * 右缘返回正值。与 `scrollLeft` 增大方向一致，调用方直接相加即可。
 *
 * 特殊说明 2：指针被 pointer capture 拖到视口外时会超出 `[left, right]`，
 * 此时比例被 clamp 到 1.5，因此**不会**继续加速；这是刻意的上限，
 * 否则拖得越远滚得越快，用户很难停在想看的位置。
 *
 * @param args 指针与画布横向边界。
 * @returns 每帧滚动像素（左负右正）；非有限输入返回 0。
 */
export function edgeAutoScrollDeltaPx(args: {
    /** 指针的视口 x（clientX）。 */
    readonly clientX: number;
    /** 画布左边界（getBoundingClientRect().left）。 */
    readonly leftPx: number;
    /** 画布右边界（getBoundingClientRect().right）。 */
    readonly rightPx: number;
}): number {
    const { clientX, leftPx, rightPx } = args;
    if (!Number.isFinite(clientX) || !Number.isFinite(leftPx) || !Number.isFinite(rightPx)) {
        return 0;
    }
    if (clientX < leftPx + EDGE_SCROLL_BAND_PX) {
        const ratio = (leftPx + EDGE_SCROLL_BAND_PX - clientX) / EDGE_SCROLL_BAND_PX;
        return -clampTo(ratio, 0, 1.5) * EDGE_SCROLL_MAX_STEP_PX;
    }
    if (clientX > rightPx - EDGE_SCROLL_BAND_PX) {
        const ratio = (clientX - (rightPx - EDGE_SCROLL_BAND_PX)) / EDGE_SCROLL_BAND_PX;
        return clampTo(ratio, 0, 1.5) * EDGE_SCROLL_MAX_STEP_PX;
    }
    return 0;
}
