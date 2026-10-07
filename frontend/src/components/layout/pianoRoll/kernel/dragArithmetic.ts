/**
 * 参数编辑器内核 · 拖拽算术（纯函数）
 *
 * 【主要内容】
 * 参数编辑器里重复出现的两处坐标换算：
 * 1. 选区的帧区间 → 采样下标（`selectionIndexRange`，含 clamp）；
 * 2. 帧号 → 采样下标（`frameToIndex`）。
 *
 * （边缘自动滚屏的几何曾在这里，现已与时间轴内核共用
 * `components/layout/shared/edgeAutoScroll.ts`，见文件末尾的说明。）
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

// ─── 边缘自动滚屏 ─────────────────────────────────────────────────────────────
//
// 步长公式、边缘带宽与比例上限在 `components/layout/shared/edgeAutoScroll.ts`，
// 与时间轴内核共用同一份实现（此前两边各写一份副本）。
// 本面板的**速度**留在这里：它是有意与时间轴不同的手感参数，见下方常量说明。

/**
 * 参数编辑器边缘自动滚屏的最大速度（CSS px/秒）。
 *
 * 【为什么比时间轴**慢**】时间轴是 720 px/秒（clip 更宽，全览时一个 clip 可能横跨
 * 半个视口，滚太快难以精确落点）。参数编辑器更需要精度 —— 用户在**画曲线**，视图
 * 一下跳过一大段就会把笔画带歪，因此取约 0.6 倍 = **420 px/秒**：
 * - 60Hz 下每帧约 7px、120Hz 下 3.5px —— 单个像素级的推进，画出来的曲线不会因为
 *   "视图一下跳过一大段"而失真；
 * - 边缘带内线性加速、比例上限 1.5，因此"拖出窗外"满速 630 px/秒（约 1.5 屏/秒）。
 *   够用来让开边界，但不至于失控。
 *
 * 【为什么不是此前的 1080 px/秒】那个值是从旧实现的"每帧 18px"按 60Hz 折算来的
 *（目的是换算时手感零变化）。但旧实现按 pointermove **事件**计步，事件频率越高越快，
 * "18px/帧"只是在 60Hz 参考帧率下的表观值 —— 它继承了本就不适合编辑场景的量级，
 * 比时间轴还快 50%。
 *
 * 【为什么也不必更快】"快速跨越工程"有别的入口（拖滚动条 / 滚轮 / 播放跟随）。
 * 边缘自动滚屏的定位是"笔画快到边界时，让视图以可控的速度让开一点"。
 */
export const EDGE_SCROLL_MAX_SPEED_PX_PER_SEC = 420;
