/** 设备像素栅格对齐工具（播放光标等 1px 竖线的"粗细不一"修复）。 */

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
