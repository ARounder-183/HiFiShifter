/**
 * 时间线帧提交器。
 *
 * 【主要内容】把一次视口变更（滚动 / 缩放）派发给所有已注册的图层，按固定
 * 顺序、同步执行，并对完全相同的投影做去重。
 *
 * 【作用】此前每个图层各有各的触发方式：clip 体画布与波形订阅视口总线、背景
 * 网格由 `useTimelineState` 手写调用命令式桥接、标尺与播放头直接改 DOM。只要
 * 新增一条视口变更路径而漏掉其中某一个，该图层就会与其它层分叉——而现在只能
 * 靠"每帧对账"在下一帧救回来，表现为一帧错位。
 *
 * 提交器把入口收敛为一个：任何视口变更只管 `commit(axis)`，所有图层同帧、
 * 按序重绘。
 *
 * 【与其他模块的关系】
 * - 生产方：`timelineViewportBus` / `pianoRollViewportBus` 在 emit 时调用。
 * - 消费方：背景网格、clip 体画布、波形面、网格覆盖层等注册为图层。
 * - 依赖：`timelineAxis.ts` 提供的投影与 `axisEquals` 去重。
 */

import { axisEquals, type TimelineAxis } from "./timelineAxis.js";

/** 一个可参与统一帧提交的图层。 */
export interface TimelineLayer {
    /** 图层名，仅用于调试与错误定位。 */
    readonly name: string;
    /**
     * 按给定投影重绘本图层。
     *
     * 约束：必须同步完成（滚动事件在浏览器 paint 前触发，只有同步绘制才能
     * 保证与其它图层落在同一帧）；不得在此发起 React 状态更新或 rAF。
     */
    paint(axis: TimelineAxis): void;
}

/** 帧提交器。 */
export interface TimelineFrameCommitter {
    /** 注册图层；返回注销函数。同一 order 内按注册顺序执行。 */
    register(layer: TimelineLayer, order: number): () => void;
    /** 同步提交一次投影变更；与上次完全相同时直接返回。 */
    commit(axis: TimelineAxis, options?: { force?: boolean }): void;
    /** 最近一次成功提交的投影（新挂载的图层可用它做首次对齐）。 */
    getSnapshot(): TimelineAxis;
    /** 已注册图层数，供测试与诊断使用。 */
    layerCount(): number;
}

/**
 * 图层的固定绘制顺序。
 *
 * 顺序即层叠顺序：网格在最底，clip 体与波形居中，网格覆盖层压在波形上（弱化
 * 波形区的网格），播放头置顶。任何新增图层都必须在这里登记，避免顺序散落在
 * 各个组件里。
 */
export const LAYER_ORDER = {
    gridBack: 10,
    clipBody: 20,
    waveform: 30,
    gridOverlay: 40,
    playhead: 50,
    /** 未迁移的历史订阅方（仍用总线的旧式 listener）。 */
    legacy: 100,
} as const;

/**
 * 创建帧提交器。
 *
 * @param initial 初始投影，通常取总线构造时的默认值。
 * @returns 提交器实例。
 */
export function createTimelineFrameCommitter(initial: TimelineAxis): TimelineFrameCommitter {
    const layers: Array<{ layer: TimelineLayer; order: number; seq: number }> = [];
    let lastCommitted: TimelineAxis = initial;
    let seq = 0;

    return {
        register(layer, order) {
            const entry = { layer, order, seq: seq++ };
            layers.push(entry);
            return () => {
                const index = layers.indexOf(entry);
                if (index >= 0) layers.splice(index, 1);
            };
        },

        /**
         * 同步提交。
         *
         * 流程：
         * 1. 与上次提交的投影做全等比较，相同则跳过（同一帧内多次 emit 常由
         *    水平轴与竖直轴分别触发，去重可省掉一轮全量重绘）；
         * 2. 按 order（同 order 按注册序）同步调用每个图层的 paint；
         * 3. 单个图层抛错不影响其它图层——一个图层的数据异常不应让整条时间线
         *    消失，错误会被记录到控制台以便定位。
         *
         * @param axis 本次视口投影。
         * @param options.force 忽略去重强制重绘（例如画布尺寸变化但投影未变）。
         */
        commit(axis, options) {
            if (!options?.force && axisEquals(lastCommitted, axis)) return;
            lastCommitted = axis;
            // 每次都排序会有开销，但图层数量是个位数且提交频率远低于逐帧，
            // 相比维护有序数组的插入成本更划算。
            const ordered = layers
                .slice()
                .sort((a, b) => (a.order === b.order ? a.seq - b.seq : a.order - b.order));
            // dev 帧率探针：经 globalThis 挂钩上报各图层绘制耗时（毫秒）。
            // 不做静态 import，生产构建零耦合；未启用探针时只有一次属性查找。
            const profiler = (globalThis as unknown as Record<string, unknown>)
                .__hfsFrameProfiler as
                | { recordLayer(name: string, ms: number): void; recordCommit(ms: number): void }
                | undefined;
            const commitStart = profiler ? performance.now() : 0;
            for (const entry of ordered) {
                try {
                    const paintStart = profiler ? performance.now() : 0;
                    entry.layer.paint(axis);
                    if (profiler)
                        profiler.recordLayer(entry.layer.name, performance.now() - paintStart);
                } catch (error) {
                    console.error(
                        `[timelineFrame] layer "${entry.layer.name}" paint failed`,
                        error,
                    );
                }
            }
            if (profiler) profiler.recordCommit(performance.now() - commitStart);
        },

        getSnapshot() {
            return lastCommitted;
        },

        layerCount() {
            return layers.length;
        },
    };
}
