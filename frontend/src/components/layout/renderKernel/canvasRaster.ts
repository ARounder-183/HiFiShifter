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
 * 4. 清屏必须覆盖整个物理 backing store（`clearCanvasPhysical`），不能沿用
 *    `clearRect(0,0,cssW,cssH)`：当 `round(cssH*dpr)` 向上取整时，底部
 *    0~0.5 物理行永远不被清除，贴底绘制的内容会永久残留在画布最底边。
 * 5. **CSS 布局尺寸必须回算为 `physical / dpr`**，不能沿用入参的原始 CSS 尺寸。
 *
 * 推导：顶点在 CSS 空间 → NDC = `pos / (physical/dpr) * 2 - 1` → 物理像素
 * = `pos * dpr`，与 Canvas2D 的 `setTransform(dpr,…)` 严格等价。
 *
 * 【第 5 条为什么是硬要求（历史 bug 的根因）】
 * 物理尺寸取整后若仍把**未取整**的 CSS 尺寸写回 `style.width`，浏览器会把
 * `physical` 个物理像素的 backing store 铺到 `css * dpr` 个物理像素的布局盒上，
 * 合成器据此做一次**非整数倍重采样**：
 * ```
 * 实际缩放比 = (css * dpr) / round(css * dpr)  ≠  1
 * ```
 * 误差最大 0.5 物理像素，且随 `frac(css * dpr)` 变化 —— 表现为"窗口拖到某些
 * 宽度波形发虚、另一些宽度清晰"，dpr=1.25/1.5 时按宽度奇偶交替，dpr=2 时
 * 恰好消失（用户报告过的现象）。回写 `physical / dpr` 后 `style × dpr` 严格
 * 等于 `physical`，缩放比恒为 1，重采样与模糊一并消失。
 *
 * 代价是 CSS 布局尺寸与入参最多差 `0.5 / dpr` CSS px；调用方的画布均为
 * 绝对定位且父容器 `overflow: hidden`，该亚像素差被裁掉，不产生滚动条、
 * 不挤压兄弟元素。
 *
 * 【与其他模块的关系】
 * 被时间线 Clip 体画布、波形 WebGL2 与 Canvas2D 回退、参数编辑器主画布与轴
 * 画布共同使用；不依赖任何业务逻辑，可独立测试。
 */

/**
 * 判定「CSS 尺寸是否需要重写」的容差（CSS px）。
 *
 * 远小于 1 个物理像素（最小 dpr 下也有 1 px），因此不会漏掉任何有视觉意义的
 * 变化；同时吸收浏览器对 `style.width` 序列化时的精度截断，避免每帧重复写
 * 样式触发重算。用数值比较而非字符串比较正是为了这个。
 */
const CSS_SIZE_EPSILON_PX = 1e-4;

/** 单条边的「按数值比较、仅在变化时写入」。 */
function writeCssSizeIfChanged(
    canvas: HTMLCanvasElement,
    key: "width" | "height",
    valuePx: number,
): void {
    const current = Number.parseFloat(canvas.style[key]);
    if (Number.isFinite(current) && Math.abs(current - valuePx) < CSS_SIZE_EPSILON_PX) return;
    canvas.style[key] = `${valuePx}px`;
}

/**
 * 仅在数值确实变化时写回画布的 CSS 尺寸。
 *
 * 热路径上必须避免无谓写入：`style.width` 赋值会触发样式重算。字符串比较在
 * 浏览器把 `1000.6666666666666px` 规范化成别的写法时会失效（每帧都判定为
 * "变了"），故按数值比较。
 *
 * 导出供 GL 路径（`gl/glContext.ts`）复用，保证两条路径的写回策略完全一致。
 */
export function writeCanvasCssSize(
    canvas: HTMLCanvasElement,
    widthPx: number,
    heightPx: number,
): void {
    writeCssSizeIfChanged(canvas, "width", widthPx);
    writeCssSizeIfChanged(canvas, "height", heightPx);
}

/**
 * 把 CSS 尺寸吸附到「整数物理像素 / dpr」的精确落点。
 *
 * 返回值 `v` 满足 `v * dpr` 为整数（在浮点精度内），即它对应的布局盒恰好覆盖
 * 整数个设备像素。调用方用它作为**绘制坐标系宽度**时，画布内容与设备像素严格
 * 1:1，不会出现半像素采样。
 *
 * 与 `rasterize` 内部回算的 `physical / dpr` 是同一个数；此处独立导出供不持有
 * 画布、只需要"稳定尺寸键"的调用方使用（例如波形几何缓存锚点：直接把分数宽度
 * 当缓存键会让亚像素抖动反复击穿缓存，吸附后键随设备像素稳定）。
 *
 * 非法输入回退 1（与 `rasterize` 的夹取规则一致）。
 */
export function fitCssPxToDevicePx(cssPx: number, dpr: number): number {
    const ratio = Number.isFinite(dpr) && dpr > 0 ? dpr : 1;
    const css = Number.isFinite(cssPx) ? Math.max(1, cssPx) : 1;
    return Math.max(1, Math.round(css * ratio)) / ratio;
}

/** 一次光栅化的结果：调用方据此设置变换或 uniform。 */
export interface RasterTarget {
    /**
     * CSS 像素宽（绘制坐标系的宽度）**回算值** = `physicalWidth / dpr`。
     *
     * 它同时是写回 `canvas.style.width` 的值：`cssWidthPx * dpr` 严格等于
     * `physicalWidth`，因此布局盒与 backing store 一一对应（契约第 5 条）。
     * 与入参的原始 CSS 宽最多差 `0.5 / dpr`。
     */
    readonly cssWidthPx: number;
    /** CSS 像素高（绘制坐标系的高度）**回算值** = `physicalHeight / dpr`。 */
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
 * 3. **仅在变化时**写回 `canvas.width/height`（赋值会触发画布清屏与样式重算，
 *    热路径上必须避免无谓写入）；
 * 4. CSS 尺寸回算为 `physical / dpr` 并写回（契约第 5 条，见文件头推导）；
 * 5. 返回给调用方设置变换 / uniform 所需的全部尺寸。
 *
 * @param canvas 目标画布。
 * @param cssWidthPx 期望的 CSS 像素宽（布局量测值，可为小数）。
 * @param cssHeightPx 期望的 CSS 像素高（布局量测值，可为小数）。
 * @param dpr 设备像素比，非法值回退为 1。
 * @returns 光栅化结果；`cssWidthPx` / `cssHeightPx` 为**回算值**。
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

    // 回算：style × dpr 必须严格等于物理尺寸，否则合成器重采样（见文件头）。
    const layoutWidthPx = physicalWidth / effectiveDpr;
    const layoutHeightPx = physicalHeight / effectiveDpr;
    writeCanvasCssSize(canvas, layoutWidthPx, layoutHeightPx);

    return {
        cssWidthPx: layoutWidthPx,
        cssHeightPx: layoutHeightPx,
        physicalWidth,
        physicalHeight,
        dpr: effectiveDpr,
        resolutionWidth: layoutWidthPx,
        resolutionHeight: layoutHeightPx,
    };
}

/**
 * 按契约清空画布的整个物理 backing store。
 *
 * 历史教训：在 `setTransform(dpr,…)` 下用 `clearRect(0,0,cssW,cssH)` 只会
 * 清除 `cssH*dpr` 行；当 `Math.round(cssH*dpr)` 向上取整时，底部
 * `round(cssH*dpr) − cssH*dpr`（0~0.5）行物理像素永远不会被清除，贴底绘制的
 * 曲线/网格/选区带/波形颜色会永久残留在画布最底边——表现为一条无法去除的
 * 彩色残影。因此统一在设备坐标（单位变换）下按 physical 尺寸清除，事后再由
 * 调用方设置/恢复业务变换（本函数内部 save/restore，不改变调用方变换状态）。
 */
export function clearCanvasPhysical(ctx: CanvasRenderingContext2D, target: RasterTarget): void {
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, target.physicalWidth, target.physicalHeight);
    ctx.restore();
}
