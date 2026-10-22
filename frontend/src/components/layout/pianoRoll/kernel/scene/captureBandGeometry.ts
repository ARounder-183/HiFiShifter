/**
 * 参数编辑器标尺上「捕获区间带」的几何（纯函数）。
 *
 * 【为什么几何必须由内核算】这条带子此前是一个 React 组件，读的是面板 state 里的
 * `scrollLeft`。而那个 state 每 256px 才提交一次（见宿主里的量化提交），真值在内核
 * 视口里 —— 于是滚动时带子几乎不动、最多偏 255px；再叠加"绘制坐标 = 原生 − 同步偏移"
 * 这一层，开启同步时又偏一个偏移量。用户看到的就是"伪影和标尺完全分离"。
 *
 * 现在改由内核在帧提交里用**同一个视口**算（与标尺平移、播放头线同源），
 * 这个文件只负责"给定视口算矩形"，便于脱离 DOM 单测。
 */

export interface CaptureBandRect {
    /** 左缘（CSS px，相对绘制区左缘）。 */
    readonly left: number;
    /** 宽度（CSS px，至少一个设备像素）。 */
    readonly width: number;
    /** 是否落在视口内（false 时调用方应把它藏起来）。 */
    readonly visible: boolean;
}

/** 把 CSS px 吸附到设备像素栅格（两端都要吸，见 devicePixelLine 的纪律）。 */
function snap(value: number, dpr: number): number {
    return Math.round(value * dpr) / dpr;
}

/**
 * 算出捕获带的矩形。
 *
 * 两端都吸附到设备像素：只吸右缘时，左缘会落在半像素上，dpr 1.25 / 1.5 下整条边被
 * 抗锯齿成 1~2 物理像素的灰边。
 */
export function captureBandGeometry(args: {
    startSec: number;
    endSec: number;
    pxPerSec: number;
    /** **绘制坐标**（原生滚动 − 同步偏移），与标尺内容层同一口径。 */
    scrollLeftPx: number;
    viewportWidthPx: number;
    dpr: number;
}): CaptureBandRect {
    const dpr = args.dpr > 0 ? args.dpr : 1;
    const safeRate = args.pxPerSec > 0 ? args.pxPerSec : 0;
    const left = snap(args.startSec * safeRate - args.scrollLeftPx, dpr);
    const right = snap(args.endSec * safeRate - args.scrollLeftPx, dpr);
    const width = Math.max(1 / dpr, right - left);
    const visible = right >= 0 && left <= args.viewportWidthPx && width > 0;
    return { left, width, visible };
}
