/**
 * 原生滚动条命中测试（所有自定义滚轮面共用）。
 *
 * 悬停在滚动条上时，滚轮语义应归属该滚动条的轴（滚动 / 缩放），而不是
 * 落入画布的全局滚轮绑定（时间轴默认滚轮 = 水平缩放）。宽度取
 * offset - client 的实测差值，对浏览器缩放 / 分数 DPI / 滚动条样式变化
 * 自适应。
 */

export type NativeScrollbarZone = "vertical" | "horizontal";

/**
 * 指针所在的原生滚动条区域；不在任何滚动条上时返回 null。
 *
 * 水平条与竖直条的角落重叠区按竖直处理（与 Chromium 对角落部件的
 * 命中归属一致，且滚轮语境下竖直滚动的意图更常见）。
 */
export function nativeScrollbarZoneAt(
    scroller: HTMLElement,
    clientX: number,
    clientY: number,
): NativeScrollbarZone | null {
    const bounds = scroller.getBoundingClientRect();
    if (
        clientX < bounds.left ||
        clientX > bounds.right ||
        clientY < bounds.top ||
        clientY > bounds.bottom
    ) {
        return null;
    }
    const horizontalBarHeight = scroller.offsetHeight - scroller.clientHeight;
    const verticalBarWidth = scroller.offsetWidth - scroller.clientWidth;
    if (verticalBarWidth > 0 && clientX > bounds.right - verticalBarWidth) {
        return "vertical";
    }
    if (horizontalBarHeight > 0 && clientY > bounds.bottom - horizontalBarHeight) {
        return "horizontal";
    }
    return null;
}
