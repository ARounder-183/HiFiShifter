/**
 * 轨道视图与参数编辑器共享的水平视口状态。
 *
 * 当“同步时间轴视图”启用时，两个面板共享同一份水平视口：
 * - 任一视图的滚动/缩放都会写入共享视口，另一个视图随之更新；
 * - 启用瞬间以轨道视图当前值为基准（参数编辑器对齐到轨道视图）。
 *
 * 存储中的 `scrollLeft` 采用“轨道视图原生坐标”；参数编辑器通过
 * `measureTimelineViewportOffsetPx()` 测得的左右偏移进行换算，
 * 使网格/时间轴在两个面板中按同一全局屏幕位置对齐。
 */

export interface TimelineViewportState {
    scrollLeft: number;
    pxPerSec: number;
}

const state: TimelineViewportState = {
    scrollLeft: 0,
    pxPerSec: 150,
};

const listeners = new Set<() => void>();

function emit(): void {
    for (const listener of listeners) {
        listener();
    }
}

export const timelineViewportSync = {
    get(): Readonly<TimelineViewportState> {
        return state;
    },
    /**
     * 一次写入缩放与滚动位置，并只广播一次。
     *
     * 这是同步模式下的推荐写入入口：缩放必然同时改变 pxPerSec 与
     * scrollLeft，分两次写入会让另一个视图先收到“新滚动、旧缩放”
     * 的中间状态，进而在内容宽度尚未更新时被浏览器钳制、反向写回。
     */
    setViewport(next: Partial<TimelineViewportState>): void {
        let changed = false;

        if (next.scrollLeft != null && Number.isFinite(next.scrollLeft)) {
            const value = next.scrollLeft;
            if (Math.abs(state.scrollLeft - value) >= 0.5) {
                state.scrollLeft = value;
                changed = true;
            }
        }

        if (next.pxPerSec != null && Number.isFinite(next.pxPerSec) && next.pxPerSec > 0) {
            const value = next.pxPerSec;
            if (Math.abs(state.pxPerSec - value) >= 1e-9) {
                state.pxPerSec = value;
                changed = true;
            }
        }

        if (changed) {
            emit();
        }
    },
    subscribe(listener: () => void): () => void {
        listeners.add(listener);
        return () => {
            listeners.delete(listener);
        };
    },
    reset(): void {
        state.scrollLeft = 0;
        state.pxPerSec = 150;
        emit();
    },
};

/**
 * 测量轨道视图时间线区域与参数编辑器绘制区域之间的全局水平偏移（像素）。
 *
 * 轨道视图左侧有“轨道头”区域，参数编辑器左侧只有较窄的钢琴卷帘/参数刻度，
 * 因此即使两者的 scrollLeft / pxPerSec 相同，网格线也不会在屏幕上垂直对齐。
 * 该偏移 = 轨道时间线区左缘 - 参数编辑器画布区左缘（恒为正）。
 */
export function measureTimelineViewportOffsetPx(): number {
    if (typeof document === "undefined") return 0;
    const track = document.querySelector<HTMLElement>("[data-timeline-scroller]");
    const param = document.querySelector<HTMLElement>("[data-piano-roll-scroller]");
    if (!track || !param) return 0;
    return track.getBoundingClientRect().left - param.getBoundingClientRect().left;
}

/**
 * 把参数编辑器的“绘制坐标”换算为“原生滚动坐标”。
 *
 * 同步时原生滚动位置 == 共享视口值（轨道坐标），绘制坐标 = 原生 - 偏移。
 */
export function timelineViewportStateToNative(stateScrollLeft: number, offsetPx: number): number {
    return stateScrollLeft + offsetPx;
}

/** 把参数编辑器的“原生滚动坐标”换算为“绘制坐标”（与上方互为逆运算）。 */
export function timelineViewportNativeToState(nativeScrollLeft: number, offsetPx: number): number {
    return nativeScrollLeft - offsetPx;
}
