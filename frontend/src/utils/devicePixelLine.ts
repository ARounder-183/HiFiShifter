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

/**
 * 竖直细线的**设备像素几何**：位置吸附到设备像素栅格、宽度取整数个物理像素，
 * 并以 `x` 为中心。
 *
 * 【为什么必须成对做这两件事】只吸附位置、宽度仍是 `1px` CSS：在 dpr=1.25 下
 * 1px CSS = 1.25 物理像素，边缘必然落在半个物理像素上被抗锯齿。只取整宽度、不
 * 吸附位置：线跨在两个物理像素之间，覆盖度随位置的小数部分变化 —— 同一排竖线
 * **粗细不一**（系统缩放率 > 1 时尤其明显，用户报告过）。
 *
 * 两者都做之后，任何 dpr 下线体都覆盖**恰好** `physicalWidthPx` 个物理像素，
 * 且左右边缘都落在设备像素边界上，粗细恒定。
 *
 * @param x 线中心的 CSS 像素位置（内容坐标，可为小数）。
 * @param physicalWidthPx 期望的物理像素宽度（1 = 发丝线，2 = 小节线）。
 * @param dpr 设备像素比。
 */
export function verticalHairlineGeometry(
    x: number,
    physicalWidthPx: number,
    dpr: number,
): { left: number; width: number } {
    const ratio = Number.isFinite(dpr) && dpr > 0 ? dpr : 1;
    // 取整数个物理像素宽（与 `wholeDevicePxLength` 同一规则）。
    const physical = Math.max(1, Math.round((physicalWidthPx > 0 ? physicalWidthPx : 1) * ratio));
    // 【左缘必须在**设备空间**里算】曾经先吸附中心、再减去 `width/2`：当宽度是奇数个
    // 物理像素时（如 1），两条边都落在半像素边界上 —— 线体横跨相邻两列、各覆盖一半，
    // 恰恰是最该避免的抗锯齿。改为在设备空间取整左缘，线体因此**恰好覆盖整数列**。
    const leftDevice = Math.round(x * ratio - physical / 2);
    return { left: leftDevice / ratio, width: physical / ratio };
}
