/**
 * 原生滚动写入的“写后回读”辅助。
 *
 * 原生 scroller 的 scrollLeft 是时间轴水平坐标的唯一事实源：DOM 内容层
 * （Clip 选中框、播放头、网格）直接跟随它，sticky 画布层（Clip 体/波形）
 * 经视口总线跟随同步出去的值。写入的请求值并不可靠——浏览器会把它钳制到
 * 实际可滚动范围、量化到物理像素，scroll anchoring 还可能在布局后再次修正
 * 偏移。若把“请求值”当作已生效值广播，画布层就会与原生 DOM 层错位，表现为
 * Clip 偏离其选中框。
 *
 * 因此所有程序化写入都必须经此函数：写入后立即回读浏览器实际接受的偏移，
 * 后续同步（标尺变换 / 视口总线 / React state）一律以回读值为准。
 */
export function applyNativeScrollLeft(scroller: HTMLDivElement, requested: number): number {
    scroller.scrollLeft = requested;
    return scroller.scrollLeft;
}
