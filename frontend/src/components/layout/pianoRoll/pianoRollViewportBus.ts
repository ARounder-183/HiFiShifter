/**
 * 参数编辑器（Piano Roll）sticky 视口的事件总线。
 *
 * 参数编辑器的主画布/网格在 scroll 事件内经 applyScrollLayers() 同步绘制；
 * PianoRollWaveformSurface 若只依赖 React 的 scrollLeft state（rAF 提交），
 * 波形会晚主画布一帧。该总线与 timelineViewportBus 保持相同的同步契约：
 * 滚动处理函数内 emit，订阅方立即重绘，所有 sticky 层在同一 paint 帧提交。
 */

type ViewportListener = (scrollLeft: number, pxPerSec: number, viewportWidth: number) => void;

const listeners = new Set<ViewportListener>();

let snapshot = {
    scrollLeft: 0,
    pxPerSec: 150,
    viewportWidth: 1,
};

function dispatch(): void {
    for (const listener of Array.from(listeners)) {
        listener(snapshot.scrollLeft, snapshot.pxPerSec, snapshot.viewportWidth);
    }
}

export const pianoRollViewportBus = {
    emit(scrollLeft: number, pxPerSec: number, viewportWidth: number): void {
        snapshot = { scrollLeft, pxPerSec, viewportWidth };
        dispatch();
    },

    getSnapshot(): typeof snapshot {
        return snapshot;
    },

    subscribe(listener: ViewportListener): () => void {
        listeners.add(listener);
        return () => {
            listeners.delete(listener);
        };
    },
};
