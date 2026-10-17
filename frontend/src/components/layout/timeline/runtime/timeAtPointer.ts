/**
 * 标尺交互用的指针 → 工程秒换算。
 *
 * 【为什么单独成文件】时间轴与参数编辑器都必须在**同一套实时视口**上做交互换算：
 * 标尺内容层的平移用的是内核逐帧精确的 `scrollLeft`，而 React state 是按
 * `REACT_SCROLL_STEP_PX` 量化的提交值（残差可能永不提交）。两处若各写一遍
 * `(scrollLeft + x) / pxPerSec`，迟早会有一处又拿回量化值。
 *
 * 调用方契约：`scrollLeft` 必须是**绘制视口的实时值**，不是 React 的量化 state。
 */
export function timeAtPointerX(args: {
    clientX: number;
    boundsLeft: number;
    scrollLeft: number;
    pxPerSec: number;
}): number {
    const x = args.clientX - args.boundsLeft;
    return Math.max(0, (args.scrollLeft + x) / Math.max(1e-9, args.pxPerSec));
}
