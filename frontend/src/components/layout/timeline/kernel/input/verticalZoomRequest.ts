/**
 * 时间轴内核 · 竖直缩放的「请求」解析
 *
 * 【主要内容】
 * - `resolveVerticalZoomStep()`：把一次竖直缩放手势解析成一条**请求**
 *   （目标行高 + 锚点不变式），或 `null`（本步不产生缩放）；
 * - `resolveVerticalZoomScrollTop()`：请求落地时，按已提交的行高反算竖直位置。
 *
 * 【作用：为什么竖直缩放要变成"请求"而不是"内核立即生效"】
 * 行高的真值源在 React（左侧轨道头与内核必须同源）。若内核**先行**改行高，镜像要等
 * 下一次提交才落地，于是必然存在一帧"内核已是新行高、而 React 侧（波形行几何、标尺
 * 派生量、滚动条口径）还是旧行高"的窗口；用户看到的就是"竖直缩放之后竖直方向抽动
 * 一下"。水平缩放早已按同一理由改成请求式（见宿主 `applyHorizontalWheelZoom` 的说明：
 * 标尺是 DOM、用 React 的 `pxPerSec` 布局，不同帧切缩放就会抽动），轨道头路径同理
 * （`TrackList` 的 `pendingVerticalZoomRef` + 行高落地后的 layout effect）。
 * 本模块把"要缩放多少、锚在哪"算成一条可传递、可延迟落地的请求，让画布路径也收编到
 * 同一条契约上。
 *
 * 【锚点为什么记"行号"而不是"像素"】
 * 请求提出时还不知道 React 最终会提交哪个行高（连续滚轮会被合并成最后一次提交），
 * 因此不能先把 scrollTop 算死。记下"指针处的行位置（行单位）"这个**缩放不变式**，
 * 落地时再乘以真正生效的行高即可 —— 这样连续手势与合并提交天然正确。
 *
 * 【与其他模块的关系】
 * - 上游：宿主 `onWheel` 的竖直缩放分支（手势解析与方向判定在 `wheelZoomIntent`）。
 * - 下游：宿主把请求交给 React（`onRowHeightChange`），行高落地后由
 *   `applyPendingVerticalZoom` 调 `resolveVerticalZoomScrollTop` 反算位置并原子提交。
 * - 独立性：纯函数，不依赖 DOM / React / GL，可在 node 环境直接单测。
 */

/** 一次竖直缩放的请求（可延迟落地）。 */
export interface VerticalZoomRequest {
    /** 目标行高（CSS px，已钳制并取整）。 */
    readonly rowHeight: number;
    /** 锚点不变式：指针所在的**行位置**（行单位），缩放前后保持不变。 */
    readonly anchorRowUnit: number;
    /** 锚点视口 y（CSS px，指针相对容器顶缘）。 */
    readonly anchorScreenY: number;
}

/** {@link resolveVerticalZoomStep} 的入参。 */
export interface ResolveVerticalZoomStepArgs {
    /** 本方向的缩放倍率（放大 > 1、缩小 < 1）。 */
    readonly factor: number;
    /**
     * **内核当前**的行高（CSS px）。
     *
     * 锚点不变式用它换算：内容此刻真的是按它布局的（未落地的请求还没改过任何东西）。
     */
    readonly kernelRowHeight: number;
    /** 内核当前的竖直位置（CSS px）。 */
    readonly scrollTop: number;
    /** 指针相对容器顶缘的视口 y（CSS px）。 */
    readonly pointerY: number;
    /**
     * 行高的**累积基准**（CSS px）。
     *
     * 取"尚未落地的请求行高"（若有），否则取内核当前行高：React 落地之前连续滚轮时，
     * 若每步都从内核的旧行高重算，中间步进会被丢掉（与水平缩放的 `pendingZoom` 同一理由）。
     */
    readonly baseRowHeight: number;
    /** 行高下限（CSS px）。 */
    readonly minRowHeight: number;
    /** 行高上限（CSS px）。 */
    readonly maxRowHeight: number;
}

/** 数值夹取（本模块自持，避免为一次夹取引入依赖）。 */
function clamp(value: number, min: number, max: number): number {
    return Math.min(max, Math.max(min, value));
}

/**
 * 解析一次竖直缩放手势。
 *
 * 流程：基准 × 倍率 → 按上下限夹取 → 取整 → 与基准相同则返回 `null`（不产生缩放）；
 * 否则给出目标行高，并用**内核当前**的行高与位置算出锚点行号。
 *
 * 特殊说明 1：取整后再比较。行高以整数像素落地（与 `constants.ts` 的上下限同一口径），
 * 到限后 `round(clamp(...))` 会等于基准，据此收敛，不会出现"缩不动却每格都提请求"。
 *
 * 特殊说明 2：`factor` 非法（NaN / Infinity / ≤ 0）时返回 `null`：缩放是高频输入，
 * 单个脏事件不应把行高写成 NaN。
 *
 * @param args 见 {@link ResolveVerticalZoomStepArgs}。
 * @returns 请求；本步不产生缩放时为 `null`。
 */
export function resolveVerticalZoomStep(
    args: ResolveVerticalZoomStepArgs,
): VerticalZoomRequest | null {
    if (!Number.isFinite(args.factor) || args.factor <= 0) return null;
    const base = Number.isFinite(args.baseRowHeight) ? Math.max(1, args.baseRowHeight) : 1;
    const min = Number.isFinite(args.minRowHeight) ? args.minRowHeight : base;
    const max = Number.isFinite(args.maxRowHeight) ? args.maxRowHeight : base;
    const rowHeight = Math.round(clamp(base * args.factor, Math.min(min, max), Math.max(min, max)));
    if (rowHeight === base) return null;
    // 锚点用内核当前值换算（不是基准）：未落地的请求尚未改变任何真实布局，
    // 因此"指针处的行位置"此刻就是 `(scrollTop + pointerY) / kernelRowHeight`。
    const kernelRowHeight = Number.isFinite(args.kernelRowHeight)
        ? Math.max(1e-9, args.kernelRowHeight)
        : 1;
    const anchorRowUnit = (args.scrollTop + args.pointerY) / kernelRowHeight;
    return { rowHeight, anchorRowUnit, anchorScreenY: args.pointerY };
}

/**
 * 按**已提交**的行高反算竖直位置（锚点不变式的落地）。
 *
 * 特殊说明：**不做钳制** —— 钳制由 `ScrollKernel` 统一负责（单一职责，见
 * `setRowHeightAndScrollTop`）。这里再夹一次会出现两份上限来源。
 *
 * @param request 请求（提供锚点不变式）。
 * @param committedRowHeight React 已提交的行高（CSS px）。
 * @returns 目标竖直位置（CSS px，可能为负，由内核夹到 0）。
 */
export function resolveVerticalZoomScrollTop(
    request: VerticalZoomRequest,
    committedRowHeight: number,
): number {
    return request.anchorRowUnit * committedRowHeight - request.anchorScreenY;
}
