/**
 * 参数编辑器内核 · 拖拽算术（纯函数）
 *
 * 【主要内容】
 * 参数编辑器里三处重复出现的坐标换算：
 * 1. 选区（beat）→ 帧号（`selectionFrameRange`）；
 * 2. 选区（beat）→ 采样下标（`selectionIndexRange`，含 clamp）；
 * 3. 帧号 → 采样下标（`frameToIndex`）。
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
 * 1. **起点 `floor`、终点 `ceil`**。这个不对称是"选区覆盖"的语义要求：都取 floor
 *    会漏掉右端不足一帧的部分，都取 ceil 会把左端选区外的一帧拉进来。
 * 2. **下标必须 clamp 到 `[0, len-1]` 且保证 `startIdx <= endIdx`**。曲线数据的
 *    覆盖窗口由后端按请求给出，选区可能超出它；不 clamp 会 slice 出空数组，让整个
 *    功能**静默失效**（不报错，只是没反应）。
 * 3. **`stride` 归一为 ≥1、`framePeriodMs` 归一为 ≥1e-6**。与既有实现的
 *    `Math.max(1, stride)` / `Math.max(1e-6, fp)` 一致：`stride = 0` 会让下标除零
 *    得 Infinity，clamp 后落到末位——看似"无害"，语义却完全错误。
 * 4. **非法输入返回 `null`**，而不是空范围。调用方据此提前退出；返回空范围会让
 *    "没有数据"与"选区为空"混为一谈。
 */

/** 选区覆盖的帧范围。 */
export interface SelectionFrameRange {
    /** 起始帧（含），已 clamp 到 ≥0。 */
    readonly startFrame: number;
    /** 结束帧（含），不小于 `startFrame`。 */
    readonly endFrame: number;
}

/** 选区覆盖的采样下标范围。 */
export interface SelectionIndexRange extends SelectionFrameRange {
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

/** 把帧周期归一为 ≥1e-6 的正数（与既有 `Math.max(1e-6, fp)` 一致）。 */
function normalizeFramePeriod(framePeriodMs: number): number {
    return Math.max(1e-6, framePeriodMs);
}

/** 把值限制在 `[lo, hi]`。 */
function clampTo(value: number, lo: number, hi: number): number {
    return Math.min(hi, Math.max(lo, value));
}

/**
 * 把选区（beat）换算为帧范围。
 *
 * 流程：beat → 秒（`beat × secPerBeat`）→ 帧（`秒 × 1000 / framePeriodMs`）；
 * 起点 `floor` 并 clamp 到 ≥0，终点 `ceil` 且不小于起点。
 *
 * 特殊说明 1：**两端取整方向不同是刻意的**（见文件头约束 1）。
 *
 * 特殊说明 2：beat 次序颠倒时自动归一（取 min/max），不依赖调用方先排序——
 * 拖拽过程中 aBeat/bBeat 会随方向互换。
 *
 * @param args 换算参数。
 * @returns 帧范围；输入非法时为 `null`。
 */
export function selectionFrameRange(args: {
    readonly aBeat: number;
    readonly bBeat: number;
    readonly secPerBeat: number;
    readonly framePeriodMs: number;
}): SelectionFrameRange | null {
    const { aBeat, bBeat, secPerBeat, framePeriodMs } = args;
    if (!Number.isFinite(aBeat) || !Number.isFinite(bBeat)) return null;
    if (!Number.isFinite(secPerBeat) || secPerBeat <= 0) return null;
    if (!Number.isFinite(framePeriodMs)) return null;

    const fp = normalizeFramePeriod(framePeriodMs);
    const loBeat = Math.min(aBeat, bBeat);
    const hiBeat = Math.max(aBeat, bBeat);
    const startFrame = Math.max(0, Math.floor((loBeat * secPerBeat * 1000) / fp));
    const endFrame = Math.max(startFrame, Math.ceil((hiBeat * secPerBeat * 1000) / fp));
    if (!Number.isFinite(startFrame) || !Number.isFinite(endFrame)) return null;
    return { startFrame, endFrame };
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
 * 把选区（beat）换算为采样下标范围。
 *
 * 流程：`selectionFrameRange` 取帧范围 → 各自 `frameToIndex` → clamp 到
 * `[0, len-1]`，并保证 `startIdx <= endIdx`。
 *
 * 特殊说明：返回的 `startFrame` / `endFrame` 是**未被下标 clamp 影响**的原始帧范围
 * （拉伸路径要用它算"选区移动了多少帧"），下标只是对数据窗口的投影。
 *
 * @param args 换算参数。
 * @returns 下标范围；输入非法或数据为空时为 `null`。
 */
export function selectionIndexRange(args: {
    readonly aBeat: number;
    readonly bBeat: number;
    readonly secPerBeat: number;
    readonly framePeriodMs: number;
    readonly paramView: ParamViewLike;
}): SelectionIndexRange | null {
    const { aBeat, bBeat, secPerBeat, framePeriodMs, paramView } = args;
    const frames = selectionFrameRange({ aBeat, bBeat, secPerBeat, framePeriodMs });
    if (frames === null) return null;
    if (!Array.isArray(paramView.edit) || paramView.edit.length === 0) return null;
    if (!Number.isFinite(paramView.startFrame) || !Number.isFinite(paramView.stride)) return null;

    const last = paramView.edit.length - 1;
    const rawStart = frameToIndex({
        frame: frames.startFrame,
        startFrame: paramView.startFrame,
        stride: paramView.stride,
    });
    const rawEnd = frameToIndex({
        frame: frames.endFrame,
        startFrame: paramView.startFrame,
        stride: paramView.stride,
    });
    if (rawStart === null || rawEnd === null) return null;

    const startIdx = clampTo(rawStart, 0, last);
    const endIdx = clampTo(rawEnd, startIdx, last);
    return { ...frames, startIdx, endIdx };
}
