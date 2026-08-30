/**
 * 视口总线工厂。
 *
 * 【主要内容】把一个「投影 + 帧提交器」打包成总线：外部只管提交投影变更，
 * 总线负责广播给所有已注册图层。
 *
 * 【作用】时间线与参数编辑器此前各自复制了一份总线实现，且快照字段不一致
 * （时间线有 scrollTopPx / revision，参数编辑器没有，导致新挂载的波形面会读到
 * 上一个面板实例遗留的快照）。现在两侧共用同一个实现与同一份契约。
 *
 * 【契约】
 * - 提交是**同步**的：滚动事件在浏览器 paint 前触发，只有同步派发才能保证
 *   sticky 图层与原生滚动的 DOM 内容层落在同一帧。任何 rAF / 微任务延迟都会
 *   造成可见的一帧以上错位。
 * - 投影相同则跳过整轮重绘（由帧提交器去重）。
 *
 * 【与其他模块的关系】
 * - 门面：`utils/timelineViewportBus.ts`、`pianoRoll/pianoRollViewportBus.ts`
 *   各自持有一个实例，并保留本面板的兼容 API。
 * - 依赖：`timelineAxis.ts`（投影）、`timelineFrameCommitter.ts`（派发）。
 */

import {
    createTimelineFrameCommitter,
    type TimelineFrameCommitter,
    type TimelineLayer,
} from "./timelineFrameCommitter.js";
import { withAxis, type TimelineAxis } from "./timelineAxis.js";

/** 视口总线。 */
export interface ViewportBus {
    /** 当前投影（新挂载的图层可用它做首次对齐，不会读到别的面板的残留值）。 */
    getAxis(): TimelineAxis;
    /** 注册一个参与统一帧提交的图层，返回注销函数。 */
    register(layer: TimelineLayer, order: number): () => void;
    /** 直接提交一份新投影。 */
    commit(axis: TimelineAxis): void;
    /** 在现有投影上打补丁后提交（未给出的字段沿用旧值）。 */
    patch(next: Partial<TimelineAxis>): void;
    /** 已注册图层数，供测试与诊断使用。 */
    layerCount(): number;
}

/**
 * 创建视口总线。
 *
 * @param initial 初始投影。
 * @returns 总线实例；内部持有独立的帧提交器，因此两个面板的总线互不干扰。
 */
export function createViewportBus(initial: TimelineAxis): ViewportBus {
    let axis = initial;
    const committer: TimelineFrameCommitter = createTimelineFrameCommitter(initial);

    return {
        getAxis() {
            return axis;
        },

        register(layer, order) {
            return committer.register(layer, order);
        },

        commit(next) {
            axis = next;
            committer.commit(axis);
        },

        patch(next) {
            axis = withAxis(axis, next);
            committer.commit(axis);
        },

        layerCount() {
            return committer.layerCount();
        },
    };
}
