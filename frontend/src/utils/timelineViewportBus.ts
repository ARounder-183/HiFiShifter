/**
 * Timeline 视口事件总线
 *
 * 解决 WaveformTrackCanvas 在滚动/缩放时的渲染延迟问题。
 *
 * 问题背景：
 *   PianoRoll 是单一组件，syncScrollLeft() 可以直接调 invalidate() 触发 Canvas 重绘（~16ms）。
 *   WaveformTrackCanvas 是子组件，滚动信息必须经过 React state → props 链路传递：
 *     syncScrollLeft → setTimeout 50ms → setScrollLeft(state) → re-render → useEffect → invalidate
 *   这导致 Canvas 重绘延迟 ~80ms+，加上 N 条轨道全部 re-render，造成明显卡顿。
 *
 * 解决方案：
 *   用一个全局事件总线，让 syncScrollLeft 直接通知所有 WaveformTrackCanvas 实例 invalidate。
 *   Canvas 绘制完全绕过 React props 链路，与 PianoRoll 完全一致。
 *
 * 数据流（优化后）：
 *   滚动 → syncScrollLeft()
 *     → scrollLeftRef/pxPerSecRef 立即更新
 *     → timelineViewportBus.emit(scrollLeft, pxPerSec, viewportWidth) ← 新增
 *       → WaveformTrackCanvas 从自身 ref 读取最新值 → invalidate() → rAF → draw()
 *     → setTimeout 50ms → setScrollLeft(state)（仅更新 React UI）
 */

type ViewportListener = (scrollLeft: number, pxPerSec: number, viewportWidth: number) => void;

const _listeners = new Set<ViewportListener>();

export const timelineViewportBus = {
    /**
     * 发送视口更新事件
     * 由 TimelinePanel.syncScrollLeft() 在每次滚动/缩放时调用
     */
    emit(scrollLeft: number, pxPerSec: number, viewportWidth: number): void {
        for (const fn of _listeners) {
            fn(scrollLeft, pxPerSec, viewportWidth);
        }
    },

    /**
     * 订阅视口更新事件
     * 由 WaveformTrackCanvas 在挂载时调用
     * @returns 取消订阅函数
     */
    subscribe(fn: ViewportListener): () => void {
        _listeners.add(fn);
        return () => {
            _listeners.delete(fn);
        };
    },
};
