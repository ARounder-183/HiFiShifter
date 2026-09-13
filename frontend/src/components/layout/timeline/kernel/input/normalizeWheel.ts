/**
 * 时间轴渲染内核 · wheel 输入归一化
 *
 * 【主要内容】
 * 把 `WheelEvent` 的原始增量归一化为 CSS 像素：
 * - `deltaMode = 0`（DOM_DELTA_PIXEL）原样返回；
 * - `deltaMode = 1`（DOM_DELTA_LINE）按行高换算；
 * - `deltaMode = 2`（DOM_DELTA_PAGE）按视口高度换算。
 * 另提供 `readWheelPixels()` 一次性取出水平 / 竖直两轴。
 *
 * 【作用】
 * 自绘滚动不再有浏览器的默认滚动行为兜底，滚轮增量必须由内核自己解释。
 * 部分浏览器（如 Firefox）对鼠标滚轮派发 `deltaMode = 1` 的事件：不换算会让
 * 「滚一格」在时间轴上只移动几个像素（或反之过冲），与触摸板手感差异巨大。
 * 归一化后所有输入源产出同一量纲（CSS px），`ScrollKernel` 无需关心来源。
 *
 * 【与其他模块的关系】
 * - 上游：输入层（宿主视图的 wheel 监听）在事件回调内调用；
 * - 下游：归一化结果交给 `ScrollKernel.setScrollLeft` / `setScrollTop` / `setZoom`
 *   （缩放路径只需竖直增量乘以缩放因子，同样以本模块输出为基准）。
 * - 独立性：纯函数，不依赖 DOM 类型与 React；`WheelEvent` 仅以结构化最小字段接收，
 *   便于单测直接传字面量。
 */

/** `DOM_DELTA_PIXEL`：增量已是像素。 */
const DOM_DELTA_PIXEL = 0;
/** `DOM_DELTA_LINE`：增量以行为单位。 */
const DOM_DELTA_LINE = 1;
/** `DOM_DELTA_PAGE`：增量以页为单位。 */
const DOM_DELTA_PAGE = 2;

/**
 * 行高回退值（CSS px）。
 *
 * 用于调用方未提供有效行高时：宁可退化为「一行 ≈ 16px」这一浏览器常规行高，
 * 也不要因换算系数为 0 / NaN 让滚轮完全失效。
 */
const DEFAULT_LINE_HEIGHT_PX = 16;

/**
 * 页高回退值（CSS px）。
 *
 * 用于调用方未提供有效页高时；取值接近常见时间轴宿主高度，保证手感不失真。
 */
const DEFAULT_PAGE_HEIGHT_PX = 800;

/** 归一化所需的量测上下文。 */
export interface WheelNormalizeContext {
    /** 一行的像素高度（用于 `deltaMode = 1`）。 */
    lineHeightPx: number;
    /** 一页的像素高度（用于 `deltaMode = 2`），通常取宿主视口高度。 */
    pageHeightPx: number;
}

/** 归一化后的双轴增量（CSS px）。 */
export interface WheelPixels {
    /** 水平增量（CSS px，正值 = 向右滚动）。 */
    x: number;
    /** 竖直增量（CSS px，正值 = 向下滚动）。 */
    y: number;
}

/**
 * 读取一个正数换算系数，非法值回退到默认。
 *
 * 规则：只接受有限且 > 0 的值；0 / 负数 / NaN / Infinity 一律回退——换算系数为 0
 * 会让该输入源静默失效，比"用默认行高略微不精确"更糟。
 *
 * @param value 候选系数（CSS px）。
 * @param fallback 回退值（CSS px，调用方保证为正）。
 * @returns 可安全用于乘法的正数。
 */
function resolvePositiveScale(value: number, fallback: number): number {
    return Number.isFinite(value) && value > 0 ? value : fallback;
}

/**
 * 把单个轴的滚轮增量归一化为 CSS 像素。
 *
 * 流程：非法增量直接归零 → 按 `deltaMode` 选择换算系数 → 乘法换算。
 *
 * 特殊说明：
 * - 未知 `deltaMode` 按像素处理（保守策略：宁可少动也不要误放大到不可控的滚动量）；
 * - 非法增量返回 0 而非抛错：滚轮是高频输入，单个脏事件不应中断手势链。
 *
 * @param delta 原始增量（`WheelEvent.deltaX` 或 `deltaY`）。
 * @param deltaMode 增量单位（`WheelEvent.deltaMode`）。
 * @param ctx 量测上下文（行高 / 页高）。
 * @returns 归一化后的增量（CSS px，可能为负）。
 */
export function normalizeWheelDelta(
    delta: number,
    deltaMode: number,
    ctx: WheelNormalizeContext,
): number {
    if (!Number.isFinite(delta)) return 0;
    switch (deltaMode) {
        case DOM_DELTA_LINE:
            return delta * resolvePositiveScale(ctx.lineHeightPx, DEFAULT_LINE_HEIGHT_PX);
        case DOM_DELTA_PAGE:
            return delta * resolvePositiveScale(ctx.pageHeightPx, DEFAULT_PAGE_HEIGHT_PX);
        case DOM_DELTA_PIXEL:
        default:
            return delta;
    }
}

/**
 * 从滚轮事件读取双轴增量并归一化。
 *
 * 流程：分别归一化 `deltaX` 与 `deltaY`。
 *
 * 特殊说明：只读取结构化最小字段（`deltaX` / `deltaY` / `deltaMode`），
 * 因此单测可直接传字面量对象，无需构造真实 `WheelEvent`。
 *
 * @param event 含双轴增量与单位的事件对象。
 * @param ctx 量测上下文（行高 / 页高）。
 * @returns 归一化后的双轴增量（CSS px）。
 */
export function readWheelPixels(
    event: { deltaX: number; deltaY: number; deltaMode: number },
    ctx: WheelNormalizeContext,
): WheelPixels {
    return {
        x: normalizeWheelDelta(event.deltaX, event.deltaMode, ctx),
        y: normalizeWheelDelta(event.deltaY, event.deltaMode, ctx),
    };
}
