/**
 * 变化点拖拽的位移换算。
 *
 * 【要解决的问题】双击变化点标签进入编辑时，第一下点击的按下与抬起之间几乎总会
 * 有 1-2px 抖动。此前任何位移都会立刻进入"拖拽"并应用吸附 —— 变化点被挪到最近的
 * 吸附位，随后输入框出现在**被挪动过**的位置，看起来就是"双击后输入框偏移了"；
 * 而挪到哪里取决于相邻变化点（它们决定吸附候选集），于是症状表现为"和前一个变化点
 * 是否接近有关"。
 *
 * 因此拖拽要有**启动阈值**：阈值内视为点击（位移 0，不改变任何东西）；越过阈值后
 * 从**阈值处**起算（而不是从起点），避免越过瞬间跳一格。
 */

/** 拖拽启动阈值（CSS px）。 */
export const TEMPO_DRAG_THRESHOLD_PX = 3;

/**
 * 把指针位移换算成"应当施加到变化点上的位移"。
 *
 * @param dx 指针相对拖拽起点的水平位移（CSS px，可负）。
 * @returns 阈值内的位移返回 0（此时调用方应当直接返回，不产生任何变更）；
 *   越过阈值后返回扣除阈值后的位移（同号）。
 */
export function resolveTempoDragOffsetPx(dx: number): number {
    if (!Number.isFinite(dx)) return 0;
    if (Math.abs(dx) <= TEMPO_DRAG_THRESHOLD_PX) return 0;
    return dx - Math.sign(dx) * TEMPO_DRAG_THRESHOLD_PX;
}

/**
 * 由"标签屏幕左缘 + 光标位置"换算光标所在的时间。
 *
 * 【用途】从内联输入框拖出去时，变化点应当**落到光标处**，而不是保留"光标与标签
 * 之间的初始偏移"。拖拽是在指针离开输入框边界之后才接管的，此时光标已离标签几十
 * 像素；沿用初始偏移会让变化点永远落后光标那一段距离（用户反馈"拖到哪儿都不是我
 * 指的位置"）。
 *
 * 标签的屏幕左缘即该变化点当前的时间位置，因此只需把像素差换算成秒。
 *
 * @param pointSec 变化点当前时间（秒）。
 * @param flagLeftPx 标签左缘的屏幕 x（CSS px）。
 * @param clientX 光标屏幕 x（CSS px）。
 * @param pxPerSec 当前水平缩放。
 * @returns 光标所在的时间（秒）；输入非法时回退 `pointSec`。
 */
export function tempoSecUnderCursor(args: {
    pointSec: number;
    flagLeftPx: number;
    clientX: number;
    pxPerSec: number;
}): number {
    const { pointSec, flagLeftPx, clientX, pxPerSec } = args;
    if (!Number.isFinite(flagLeftPx) || !Number.isFinite(clientX) || !(pxPerSec > 0)) {
        return pointSec;
    }
    return pointSec + (clientX - flagLeftPx) / pxPerSec;
}
