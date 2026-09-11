/**
 * 时间轴渲染内核 · 特性开关
 *
 * 【主要内容】
 * 读取 `localStorage` 决定是否启用新内核（Spike）。默认关闭，只有显式写入 `"1"`
 * 才启用——与既有 PERF 开关（`hifishifter.glClipBodies` 的逃生门语义相反）不同，
 * 新内核是**未完成**的实验路径，必须显式开启。
 *
 * 【作用】
 * 落地策略要求「可一键回退」：Spike 与后续阶段 1 的切换都经本开关，关闭后时间轴
 * 完全走既有实现，零影响。开关值在模块加载时读取一次（切换需刷新页面，与既有
 * dev 开关行为一致）。
 *
 * 【与其他模块的关系】
 * - 上游：`TimelinePanel` 在模块加载时读取并决定是否渲染 `TimelineKernelSpikeView`。
 * - 独立性：纯函数，不依赖 React。
 */

/** 开关 key。 */
export const TIMELINE_KERNEL_FLAG_KEY = "hifishifter.timelineKernel";

/**
 * 是否启用时间轴渲染内核（Spike）。
 *
 * @returns 显式写入 `"1"` 时为 true；读取失败（隐私模式）或未设置时为 false。
 */
export function isTimelineKernelEnabled(): boolean {
    try {
        return localStorage.getItem(TIMELINE_KERNEL_FLAG_KEY) === "1";
    } catch {
        // 读不到 localStorage 时按默认关闭处理：新内核是实验路径，
        // 环境异常时不应意外启用。
        return false;
    }
}
