/**
 * Timeline 视口事件总线
 *
 * 时间线的 sticky 图层（背景网格 / clip 体画布 / 波形面 / 网格覆盖层）都不随
 * 滚动容器原生移动，必须与原生 DOM 内容层在**同一帧**内提交位移（水平与竖直
 * 两个轴），否则滚动过程中会出现"Clip/波形与网格分离"的分层漂移。
 *
 * 因此所有派发都是**同步**的：滚动事件处理器内提交时，所有图层在浏览器绘制
 * 前完成重绘（scroll 事件在 paint 前触发）。任何 rAF / state 延迟都会造成可见
 * 的一帧以上错位，严禁改回异步派发。
 *
 * 数据流：
 *   水平滚动/缩放 → useTimelineState.syncScrollLeft()
 *   竖直滚动     → useTimelineState.syncScrollTop()
 *     → scrollLeftRef/scrollTopPxRef/pxPerSecRef 立即更新
 *     → timelineViewportBus.emit(...)（同步）
 *       → 帧提交器按固定层序同步调用所有已注册图层（见 timelineFrameCommitter）
 *     → rAF/React state 仅驱动裁剪/窗口化等非视觉更新
 *
 * 迁移说明：
 * 新代码请用 `register(layer, order)` 注册图层并消费 `TimelineAxis`，坐标一律走
 * `timelineAxis.ts`。`subscribe()` / `getSnapshot()` 是为尚未迁移的旧订阅方保留的
 * 兼容层，它们排在 LAYER_ORDER.legacy，会在所有已迁移图层之后绘制。
 */

import {
    LAYER_ORDER,
    type TimelineLayer,
} from "../components/layout/timeline/runtime/timelineFrameCommitter.js";
import {
    createTimelineAxis,
    type TimelineAxis,
} from "../components/layout/timeline/runtime/timelineAxis.js";
import { createViewportBus } from "../components/layout/timeline/runtime/viewportBus.js";

/** 旧式监听器的参数签名（尚未迁移到 TimelineAxis 的订阅方使用）。 */
type ViewportListener = (
    scrollLeft: number,
    pxPerSec: number,
    viewportWidth: number,
    scrollTopPx: number,
    rowHeight: number,
) => void;

const bus = createViewportBus(createTimelineAxis({ pxPerSec: 150, viewportWidthPx: 1 }));

/** 不属于投影、但仍需透传给旧订阅方的上下文。 */
let rowHeight = 80;
let revision = 0;

function advance(next: {
    scrollLeft: number;
    pxPerSec: number;
    viewportWidth: number;
    scrollTopPx?: number;
    rowHeight?: number;
}): void {
    bus.patch({
        scrollLeftPx: next.scrollLeft,
        pxPerSec: next.pxPerSec,
        viewportWidthPx: next.viewportWidth,
        scrollTopPx: next.scrollTopPx,
    });
    if (next.rowHeight != null) rowHeight = next.rowHeight;
    revision += 1;
}

export const timelineViewportBus = {
    /**
     * 提交视口更新（同步派发，见文件头注释）。
     * 由 useTimelineState.syncScrollLeft() / syncScrollTop() 在每次滚动/缩放时调用。
     * scrollTopPx / rowHeight 可省略：沿用上一个值，便于只更新单一轴。
     */
    emit(
        scrollLeft: number,
        pxPerSec: number,
        viewportWidth: number,
        scrollTopPx?: number,
        rowHeight?: number,
    ): void {
        advance({ scrollLeft, pxPerSec, viewportWidth, scrollTopPx, rowHeight });
    },

    /** 当前投影。新挂载的图层应优先用它做首次对齐。 */
    getAxis(): TimelineAxis {
        return bus.getAxis();
    },

    /** 当前快照（旧字段形态，仅供未迁移的调用方使用）。 */
    getSnapshot(): {
        scrollLeft: number;
        pxPerSec: number;
        viewportWidth: number;
        scrollTopPx: number;
        rowHeight: number;
        revision: number;
    } {
        const axis = bus.getAxis();
        return {
            scrollLeft: axis.scrollLeftPx,
            pxPerSec: axis.pxPerSec,
            viewportWidth: axis.viewportWidthPx,
            scrollTopPx: axis.scrollTopPx,
            rowHeight,
            revision,
        };
    },

    /**
     * 注册一个参与统一帧提交的图层（推荐的新接入方式）。
     *
     * @param layer 图层实现，`paint(axis)` 必须同步完成。
     * @param order 绘制顺序，取 `LAYER_ORDER` 中的值。
     * @returns 注销函数。
     */
    register(layer: TimelineLayer, order: number): () => void {
        return bus.register(layer, order);
    },

    /** 直接提交一份完整投影（绕过逐字段 patch）。 */
    commit(axis: TimelineAxis): void {
        revision += 1;
        bus.commit(axis);
    },

    /**
     * 订阅视口更新（兼容层，新代码请用 `register`）。
     *
     * 说明：旧式监听器统一排在 LAYER_ORDER.legacy，即在所有已迁移图层之后
     * 执行，以保证主图层（网格/clip/波形）的顺序稳定。
     *
     * @returns 取消订阅函数
     */
    subscribe(fn: ViewportListener, order: number = LAYER_ORDER.legacy): () => void {
        return bus.register(
            {
                name: "legacy-listener",
                paint: (axis) =>
                    fn(
                        axis.scrollLeftPx,
                        axis.pxPerSec,
                        axis.viewportWidthPx,
                        axis.scrollTopPx,
                        rowHeight,
                    ),
            },
            order,
        );
    },
};
