/**
 * 视口提交判据（内核 → React）。
 *
 * 【为什么必须单独成模块并可测】标尺的**刻度窗口**由 React 侧
 * `(pxPerSec, scrollLeft)` 生成（`useTimelineState` 的 `tickAxis`），而**屏幕位置**
 * 由内核视口决定（标尺内容层的 transform 与 GL 都按它画）。两者必须是**同一个**
 * 视口对：
 *
 * - 只提交位置、不提交缩放 ⇒ React 拿到"新缩放 + 旧像素位置"。同一像素值在新缩放
 *   下对应的时间完全不同，刻度窗口的锚点因此错位 —— 标尺某段既没有刻度线也没有
 *   文本（用户报告的"缩放时标尺文本闪烁 / 消失"）。
 * - 只提交缩放、不提交位置 ⇒ 同上，方向相反。
 *
 * 因此判据把 `pxPerSec` 与 `scrollLeft` 放在**同一个提交**里：任一发生变化就提交
 * 整对。这样"内核换了缩放 ⇒ React 在同一帧收到成对视口"是**结构性**保证，新增
 * 视口变更路径不可能再漏掉其中一项。
 */

/** 可提交的视口对。 */
export interface CommittableViewport {
    /** 水平滚动位置（CSS px）。 */
    scrollLeftPx: number;
    /** 每秒像素数。 */
    pxPerSec: number;
}

/**
 * 判定本次绘制是否需要把视口提交给 React。
 *
 * @param view 本帧的视口真值。
 * @param last 上次提交的视口（首次调用时两字段为 `NaN`，必然判为需要提交）。
 * @param scrollStepPx 水平位置的量化步长（死区）。
 * @returns 位置跨过死区**或**缩放发生变化时为 true。
 */
export function shouldCommitViewport(
    view: CommittableViewport,
    last: CommittableViewport,
    scrollStepPx: number,
): boolean {
    // 取反写法：`NaN`（"从未提交"）必须判为"需要提交"——直接写
    // `Math.abs(a - b) > eps` 会把首次调用判成"无需提交"，标尺刻度便永远停在
    // 初始视口（与宿主 `shouldWrite` 同一口径）。
    const scrollMoved = !(Math.abs(view.scrollLeftPx - last.scrollLeftPx) <= scrollStepPx);
    // 缩放是比例量，用相对容差（与 `scrollKernel` 的 `zoomEquals` 同一口径）；
    // `last.pxPerSec` 为 NaN 时容差也是 NaN，取反写法同样判为"需要提交"。
    const zoomEpsilon = Math.abs(last.pxPerSec) * 1e-9;
    const zoomChanged = !(Math.abs(view.pxPerSec - last.pxPerSec) <= zoomEpsilon);
    return scrollMoved || zoomChanged;
}
