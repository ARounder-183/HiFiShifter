/**
 * 画布 DPR 光栅化契约。
 *
 * 【主要内容】提供 `rasterize()` 一个函数：把 CSS 尺寸 + 设备像素比换算成画布
 * 的物理尺寸，并给出 WebGL 着色器所需的 `u_resolution`。
 *
 * 【作用】消除各画布自行处理 DPR 造成的**缩放比不一致**。历史问题：
 * - Clip 体画布用 `Math.floor(w*dpr)` + `setTransform(dpr,…)`，缩放比严格 = dpr；
 * - 波形 WebGL2 用 `Math.round(w*dpr)` 做 `gl.viewport`，却把 CSS 尺寸传进
 *   `u_resolution`，于是 NDC 被拉伸到 `round(w*dpr)` 个物理像素，实际缩放比
 *   变成 `round(w*dpr)/w`，**不等于 dpr**，在 dpr=1.25/1.5 或宽度为奇数时
 *   边缘偏差可达半像素，且随窗口宽度变化跳动；
 * - 参数编辑器主画布又用第三种取整（`Math.floor`）。
 *
 * 【契约（唯一）】
 * 1. 绘制坐标一律是 **CSS 像素**；
 * 2. 物理尺寸一律 `Math.round(css * dpr)`；
 * 3. WebGL 的 `u_resolution` 必须传 `physical / dpr`，**不是** CSS 尺寸。
 *
 * 推导：顶点在 CSS 空间 → NDC = `pos / (physical/dpr) * 2 - 1` → 物理像素
 * = `pos * dpr`，与 Canvas2D 的 `setTransform(dpr,…)` 严格等价。
 *
 * 【与其他模块的关系】
 * 被时间线 Clip 体画布、波形 WebGL2 与 Canvas2D 回退、参数编辑器主画布与轴
 * 画布共同使用；不依赖任何业务逻辑，可独立测试。
 */

/** 一次光栅化的结果：调用方据此设置变换或 uniform。 */
export interface RasterTarget {
    /** CSS 像素宽（绘制坐标系的宽度）。 */
    readonly cssWidthPx: number;
    /** CSS 像素高（绘制坐标系的高度）。 */
    readonly cssHeightPx: number;
    /** 物理像素宽 = `Math.round(cssWidthPx * dpr)`。 */
    readonly physicalWidth: number;
    /** 物理像素高 = `Math.round(cssHeightPx * dpr)`。 */
    readonly physicalHeight: number;
    /** 实际使用的设备像素比。 */
    readonly dpr: number;
    /**
     * WebGL `u_resolution` 的 x 分量 = `physicalWidth / dpr`。
     * 传 CSS 尺寸会导致缩放比变成 `physicalWidth / cssWidthPx`（≠ dpr）。
     */
    readonly resolutionWidth: number;
    /** WebGL `u_resolution` 的 y 分量 = `physicalHeight / dpr`。 */
    readonly resolutionHeight: number;
}

/**
 * 按统一契约调整画布的物理尺寸与 CSS 尺寸。
 *
 * 流程：
 * 1. 夹取 CSS 尺寸到 >= 1，dpr 到 >= 1；
 * 2. 物理尺寸取 `Math.round(css * dpr)`；
 * 3. **仅在变化时**写回 `canvas.width/height` 与 `style.width/height`
 *    （赋值会触发画布清屏与样式重算，热路径上必须避免无谓写入）；
 * 4. 返回给调用方设置变换 / uniform 所需的全部尺寸。
 *
 * 特殊说明：物理尺寸取整后与 `css * dpr` 最多差半个物理像素，这是栅格对齐的
 * 固有代价；关键是**所有画布使用同一个取整规则**，否则层与层之间会差一整个
 * 物理像素。
 *
 * @param canvas 目标画布。
 * @param cssWidthPx CSS 像素宽。
 * @param cssHeightPx CSS 像素高。
 * @param dpr 设备像素比，非法值回退为 1。
 * @returns 光栅化结果。
 */
export function rasterize(
    canvas: HTMLCanvasElement,
    cssWidthPx: number,
    cssHeightPx: number,
    dpr: number,
): RasterTarget {
    const cssWidth = Number.isFinite(cssWidthPx) ? Math.max(1, cssWidthPx) : 1;
    const cssHeight = Number.isFinite(cssHeightPx) ? Math.max(1, cssHeightPx) : 1;
    const effectiveDpr = Number.isFinite(dpr) && dpr > 0 ? dpr : 1;

    const physicalWidth = Math.max(1, Math.round(cssWidth * effectiveDpr));
    const physicalHeight = Math.max(1, Math.round(cssHeight * effectiveDpr));

    if (canvas.width !== physicalWidth) canvas.width = physicalWidth;
    if (canvas.height !== physicalHeight) canvas.height = physicalHeight;

    const cssWidthStyle = `${cssWidth}px`;
    const cssHeightStyle = `${cssHeight}px`;
    if (canvas.style.width !== cssWidthStyle) canvas.style.width = cssWidthStyle;
    if (canvas.style.height !== cssHeightStyle) canvas.style.height = cssHeightStyle;

    return {
        cssWidthPx: cssWidth,
        cssHeightPx: cssHeight,
        physicalWidth,
        physicalHeight,
        dpr: effectiveDpr,
        resolutionWidth: physicalWidth / effectiveDpr,
        resolutionHeight: physicalHeight / effectiveDpr,
    };
}
