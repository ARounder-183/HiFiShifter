/**
 * 时间轴渲染内核 · GL 光栅化参数
 *
 * 【主要内容】
 * 由「CSS 尺寸 + devicePixelRatio」算出画布物理尺寸与绘制坐标系尺寸：
 * 物理尺寸 = `round(css × dpr)`（画布真实像素），绘制坐标系尺寸 = `物理 / dpr`。
 *
 * 【作用】
 * 与既有 `runtime/canvasRaster.rasterize` 保持同一契约（向上取整到物理像素），
 * 但把「绘制坐标系尺寸」显式回算为 `物理 / dpr`：WebGL 的 `u_resolution` 必须用
 * 这个值，才能保证顶点坐标（CSS px）与物理像素 1:1 对应。若直接用传入的 CSS 尺寸，
 * 在分数 DPR 下会出现半个物理像素的累积偏移（网格线相位漂移）。
 *
 * 【与其他模块的关系】
 * - 上游：宿主视图在 `ResizeObserver` 回调里调用；
 * - 下游：`glContext` 用它设置 `canvas.width/height` 与 `gl.viewport`，
 *   program 用它设置 `u_resolution`。
 * - 独立性：纯函数，不依赖 DOM / WebGL，可直接单测。
 */

/** 光栅化目标：画布物理尺寸与绘制坐标系尺寸。 */
export interface GlRasterTarget {
    /** 画布物理宽（= `canvas.width`）。 */
    readonly physicalWidthPx: number;
    /** 画布物理高（= `canvas.height`）。 */
    readonly physicalHeightPx: number;
    /** 绘制坐标系宽（CSS px，= 物理宽 / dpr），供 `u_resolution`。 */
    readonly cssWidthPx: number;
    /** 绘制坐标系高（CSS px，= 物理高 / dpr），供 `u_resolution`。 */
    readonly cssHeightPx: number;
    /** 生效的 DPR（非法输入已回退为 1）。 */
    readonly dpr: number;
}

/**
 * 计算光栅化目标。
 *
 * 流程：归一化 DPR（非法 → 1）与 CSS 尺寸（非法 → 1，且不小于 1）→ 物理尺寸取整
 * → 绘制坐标系尺寸由物理尺寸回算。
 *
 * 特殊说明：物理尺寸恒 >= 1——`canvas.width = 0` 会让 WebGL 上下文报错并丢弃后续
 * 绘制调用，宿主尚未量出尺寸时也不能让画布进入非法状态。
 *
 * @param cssWidthPx 宿主 CSS 宽（px）。
 * @param cssHeightPx 宿主 CSS 高（px）。
 * @param dpr 设备像素比。
 * @returns 光栅化目标；`cssWidthPx` / `cssHeightPx` 为**回算值**（物理 / dpr）。
 */
export function resolveGlRasterTarget(
    cssWidthPx: number,
    cssHeightPx: number,
    dpr: number,
): GlRasterTarget {
    const safeDpr = Number.isFinite(dpr) && dpr > 0 ? dpr : 1;
    const cssWidth = Number.isFinite(cssWidthPx) ? Math.max(1, cssWidthPx) : 1;
    const cssHeight = Number.isFinite(cssHeightPx) ? Math.max(1, cssHeightPx) : 1;
    const physicalWidthPx = Math.max(1, Math.round(cssWidth * safeDpr));
    const physicalHeightPx = Math.max(1, Math.round(cssHeight * safeDpr));
    return {
        physicalWidthPx,
        physicalHeightPx,
        cssWidthPx: physicalWidthPx / safeDpr,
        cssHeightPx: physicalHeightPx / safeDpr,
        dpr: safeDpr,
    };
}
