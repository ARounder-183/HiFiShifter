/**
 * Timeline 视口事件总线
 *
 * TimelineSurface 的两个 sticky 画布层（TimelineCanvasViewport / 波形面）与
 * MidiPitchTrackCanvas 的视口锚定绘制都从这里取 scrollLeft。它们不随滚动
 * 容器原生移动，必须与原生 DOM 内容层在**同一帧**内提交位移，否则滚动
 * 过程中会出现“Clip/波形与网格分离”的分层漂移。
 *
 * 因此 emit 采用【同步派发】：滚动事件处理器内 emit 时，所有画布层在
 * 浏览器绘制前完成重绘（scroll 事件在 paint 前触发）。任何 rAF/ state
 * 延迟都会造成可见的一帧以上错位，严禁改回异步派发。
 *
 * 数据流：
 *   滚动/缩放 → useTimelineState.syncScrollLeft()
 *     → scrollLeftRef/pxPerSecRef 立即更新
 *     → timelineViewportBus.emit(scrollLeft, pxPerSec, viewportWidth)（同步）
 *       → 各画布层立即以新视口重绘
 *     → rAF 合并 setScrollLeft(state)（仅驱动裁剪/窗口化等非视觉更新）
 */

type ViewportListener = (scrollLeft: number, pxPerSec: number, viewportWidth: number) => void;

const _listeners = new Set<ViewportListener>();
let _snapshot = { scrollLeft: 0, pxPerSec: 150, viewportWidth: 1, revision: 0 };

function dispatch(): void {
    const listeners = Array.from(_listeners);
    for (const fn of listeners) {
        fn(_snapshot.scrollLeft, _snapshot.pxPerSec, _snapshot.viewportWidth);
    }
}

export const timelineViewportBus = {
    /**
     * 发送视口更新事件（同步派发，见文件头注释）
     * 由 useTimelineState.syncScrollLeft() 在每次滚动/缩放时调用
     */
    emit(scrollLeft: number, pxPerSec: number, viewportWidth: number): void {
        _snapshot = {
            scrollLeft,
            pxPerSec,
            viewportWidth,
            revision: _snapshot.revision + 1,
        };
        dispatch();
    },

    /** Current value lets newly mounted virtual rows render without waiting for another event. */
    getSnapshot(): typeof _snapshot {
        return _snapshot;
    },

    /**
     * 订阅视口更新事件
     * @returns 取消订阅函数
     */
    subscribe(fn: ViewportListener): () => void {
        _listeners.add(fn);
        return () => {
            _listeners.delete(fn);
        };
    },
};
