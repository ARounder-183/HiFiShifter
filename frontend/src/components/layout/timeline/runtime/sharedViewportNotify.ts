/**
 * 共享水平视口 · 「本帧是否要通知面板」判定。
 *
 * 【主要内容】
 * 时间轴内核每个绘制帧把水平视口交给面板（`onScrollLeftFrame`），本函数决定这一帧
 * 是否需要通知——去重键是 **`{scrollLeft, pxPerSec}` 整对**，而不是只看位置。
 *
 * 【为什么必须看整对（这是一个真实缺陷的根因）】
 * 同步「参数编辑器的水平位置与缩放到时间轴」时，共享视口是一对真值，参数编辑器按
 * **整对**消费：位置决定它画到哪一段，缩放决定它的网格/标尺密度。时间轴内核若只按
 * `scrollLeft` 去重，就会漏掉**缩放变化而位置不变**的那一步：
 *
 * ```
 * 光标停在工程起点附近滚轮缩小
 *   → resolveHorizontalWheelZoom 的锚点位置被范围钳回 0（与当前位置相同）
 *   → scrollLeft 未变 → 不通知 → 共享视口仍写着旧缩放
 *   → 参数编辑器停在旧缩放：两个面板的网格/标尺从此不同密度
 * ```
 *
 * 实测（浏览器 1920×1200，mock 工程 120s，光标 x=800）：连续三次滚轮缩小，时间轴
 * 150 → 135 → 121.5 → 109.35 px/s，参数编辑器**始终保持 150 px/s**（`__hsPianoRollKernel`
 * 的 `getViewport().pxPerSec`），画面上标尺/网格密度明显不同。放大则没有这个问题：
 * 同一锚点放大时位置会右移（0 → 55.6），`scrollLeft` 变了、于是照常通知。
 *
 * 【契约】
 * - 任一分量变化 → 通知（`scrollLeft` 与 `pxPerSec` 必须同帧一起送达，不能拆成两次）；
 * - 两个分量都没变 → 不通知（避免空转：绘制帧是热的，通知会连带 `emit` 一次）；
 * - 首次通知（`last*` 为 `NaN`）→ 必然通知：`NaN !== NaN` 恒真，无需特判。
 */
export function shouldNotifySharedViewport(args: {
    /** 本帧的水平位置（内核真值，CSS px）。 */
    scrollLeftPx: number;
    /** 本帧的水平缩放（内核真值，CSS px/秒）。 */
    pxPerSec: number;
    /** 上一次已通知的水平位置（`NaN` = 从未通知）。 */
    lastScrollLeftPx: number;
    /** 上一次已通知的水平缩放（`NaN` = 从未通知）。 */
    lastPxPerSec: number;
}): boolean {
    return args.scrollLeftPx !== args.lastScrollLeftPx || args.pxPerSec !== args.lastPxPerSec;
}
