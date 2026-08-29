/**
 * 参数编辑器（Piano Roll）sticky 视口的事件总线。
 *
 * 参数编辑器的主画布/网格在 scroll 事件内经 applyScrollLayers() 同步绘制；
 * PianoRollWaveformSurface 若只依赖 React 的 scrollLeft state（rAF 提交），
 * 波形会晚主画布一帧。本总线与时间线总线共用同一套实现与契约（见
 * `runtime/viewportBus.ts`）：滚动处理函数内提交，订阅方立即重绘，所有
 * sticky 层在同一 paint 帧提交。
 *
 * 迁移说明：新代码请用 `register(layer, order)` 并消费 `TimelineAxis`；
 * `subscribe()` / `getSnapshot()` 是为尚未迁移的旧订阅方保留的兼容层。
 */

import { LAYER_ORDER, type TimelineLayer } from "../timeline/runtime/timelineFrameCommitter.js";
import { createTimelineAxis, type TimelineAxis } from "../timeline/runtime/timelineAxis.js";
import { createViewportBus } from "../timeline/runtime/viewportBus.js";

/** 旧式监听器的参数签名（参数编辑器不涉及竖直滚动与行高）。 */
type ViewportListener = (scrollLeft: number, pxPerSec: number, viewportWidth: number) => void;

const bus = createViewportBus(createTimelineAxis({ pxPerSec: 150, viewportWidthPx: 1 }));

export const pianoRollViewportBus = {
    /** 提交视口更新（同步派发，见文件头注释）。 */
    emit(scrollLeft: number, pxPerSec: number, viewportWidth: number): void {
        bus.patch({
            scrollLeftPx: scrollLeft,
            pxPerSec,
            viewportWidthPx: viewportWidth,
        });
    },

    /** 当前投影。新挂载的图层应优先用它做首次对齐。 */
    getAxis(): TimelineAxis {
        return bus.getAxis();
    },

    /** 当前快照（旧字段形态，仅供未迁移的调用方使用）。 */
    getSnapshot(): { scrollLeft: number; pxPerSec: number; viewportWidth: number } {
        const axis = bus.getAxis();
        return {
            scrollLeft: axis.scrollLeftPx,
            pxPerSec: axis.pxPerSec,
            viewportWidth: axis.viewportWidthPx,
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
        bus.commit(axis);
    },

    /** 订阅视口更新（兼容层，新代码请用 `register`）。 */
    subscribe(fn: ViewportListener, order: number = LAYER_ORDER.legacy): () => void {
        return bus.register(
            {
                name: "piano-roll-legacy-listener",
                paint: (axis) => fn(axis.scrollLeftPx, axis.pxPerSec, axis.viewportWidthPx),
            },
            order,
        );
    },
};
