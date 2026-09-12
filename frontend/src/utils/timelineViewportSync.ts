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

/**
 * 是否已被"拥有方"（时间轴视图）播种。
 *
 * 模块默认值 {0, 150} 只是占位，不是任何面板的当前位置；参数编辑器在
 * 未播种时必须**拒绝应用**共享视口，否则启动时会把一帧"默认缩放/位置"
 * 画出来再被纠正（"一闪"）。时间轴在挂载 layout effect 阶段（首帧绘制前）
 * 播种；任何 setViewport 调用都视为播种完成。
 */
let seeded = false;

/**
 * 最后一次写入的来源标识。
 *
 * 【为什么需要它——反馈环】两个面板都订阅同一个共享视口，而时间轴**自己也订阅**
 * （它要把参数编辑器推来的位置应用回来）。于是时间轴每次发布后都会立刻收到自己的
 * 回灌，再用 `setZoomAndScroll(store.scrollLeft)` 写回内核。
 *
 * 写入方在发布时登记来源，订阅方在回调里比对 `getOrigin()`：若来源就是自己，说明
 * 这次广播是自己造成的，应**跳过应用**。
 *
 * 【不这么做会怎样（实测）】时间轴的逐帧发布与自身的回灌在同一帧内竞争，回灌读到
 * 的是**上一次**的值 → 内核位置出现 `10 → 20 → 10` 的回退。连续拖拽时表现为
 * 每三帧回退一次（增量呈 `+30, +10, -10` 循环），即用户报告的"阶梯感 / 被吸附感"。
 *
 * 注意：原先的 `timelineSyncApplyingRef` 标志只在**同步调用栈内**为真，无法覆盖
 * 跨帧的回灌，所以它不能替代来源判定。
 */
let origin: string | null = null;

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
    /** 共享视口是否已由时间轴播种（见 seeded 注释）。 */
    isSeeded(): boolean {
        return seeded;
    },
    /**
     * 一次写入缩放与滚动位置，并只广播一次。
     *
     * 这是同步模式下的推荐写入入口：缩放必然同时改变 pxPerSec 与
     * scrollLeft，分两次写入会让另一个视图先收到“新滚动、旧缩放”
     * 的中间状态，进而在内容宽度尚未更新时被浏览器钳制、反向写回。
     */
    /**
     * 当前写入来源（由最近一次 `setViewport` 登记）。
     *
     * 订阅方在回调里读取它判断"这次广播是不是我自己造成的"，见 `origin` 说明。
     * 未登记时为 `null`（例如 `reset()` 或外部直接写入）。
     */
    getOrigin(): string | null {
        return origin;
    },
    setViewport(next: Partial<TimelineViewportState>, from?: string): void {
        origin = from ?? null;
        // 播种语义：拥有方已确定当前位置（即使值与模块默认一致，例如启动
        // 时恰为 scrollLeft=0 / pxPerSec=150），也要立即可被订阅方应用。
        seeded = true;
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
        seeded = false;
        origin = null;
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
