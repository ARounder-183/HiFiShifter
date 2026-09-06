/**
 * 设备像素栅格对齐工具（播放光标等 1px 竖线的"粗细不一"修复）。
 *
 * 【问题根源】与 56238d45 修的网格线同根：分数 devicePixelRatio
 * （Windows 125%/150% 缩放、浏览器缩放）下，1px CSS 线覆盖 1.25/1.5 个
 * 物理像素；落点的小数相位不同，取整后有的渲染成 1 物理像素、有的 2 物理
 * 像素 —— 静止时只是有的线粗有的线细，而播放光标每帧都被重写位置，相位
 * 随移动连续变化，肉眼就是"粗细不停变化"。
 *
 * 【修法】与网格修复同一套规则：
 * 1. 位置吸附到设备像素边界（`snapToDevicePx`）——让落点相位恒为整数；
 * 2. 线宽取整物理像素（`wholeDevicePxLength`）——让覆盖量恒为整数个物理
 *    像素，任何缩放下边缘不再被抗锯齿摊开。
 *
 * 播放光标的写入点分散在多处命令式热路径（rAF / 滚动同步 / 拖拽），不持有
 * TimelineAxis，故这里提供只依赖 dpr 的最小原语；画布侧的居中描边仍走
 * `timelineAxis.strokePx`（语义：snap + 奇数宽度补半个设备像素）。
 */

/** 读取当前设备像素比；`window` 缺失（测试/SSR 环境）时回退 1。 */
export function readDevicePixelRatio(): number {
    if (typeof window === "undefined") return 1;
    return window.devicePixelRatio || 1;
}

/**
 * 把 CSS 像素坐标吸附到设备像素边界。
 *
 * `Math.round(x * dpr) / dpr`：任何输入的落点相位都归零，配合整数物理
 * 像素线宽即可做到"任意缩放下粗细恒定"。非法输入（NaN/∞）回退 0。
 */
export function snapToDevicePx(cssX: number, dpr: number): number {
    if (!Number.isFinite(cssX)) return 0;
    const ratio = Number.isFinite(dpr) && dpr > 0 ? dpr : 1;
    return Math.round(cssX * ratio) / ratio;
}

/**
 * 把 CSS 像素长度取整到**整数个物理像素**。
 *
 * 规则：取最接近原意图（`cssLen` CSS 像素）的整数物理像素数。
 * - dpr=1 → 1 物理像素（与旧 `w-px` 完全一致）；
 * - dpr=1.25 → 1 物理像素（旧行为是 1.25，边缘必发虚）；
 * - dpr=1.5 → 2 物理像素（四舍五入）；
 * - dpr=2 → 2 物理像素（= 1 CSS px，与旧 `w-px` 一致）。
 *
 * 下限 1：防止极端 dpr 下取整到 0 导致线消失。
 */
export function wholeDevicePxLength(cssLen: number, dpr: number): number {
    const ratio = Number.isFinite(dpr) && dpr > 0 ? dpr : 1;
    const len = Number.isFinite(cssLen) && cssLen > 0 ? cssLen : 1;
    return Math.max(1, Math.round(len * ratio)) / ratio;
}
