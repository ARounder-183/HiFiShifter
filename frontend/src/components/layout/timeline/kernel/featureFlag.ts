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
 * 规则（按优先级）：
 * 1. 显式写入 `"0"` → 关闭（逃生门，出问题时一键退回既有实现）；
 * 2. 显式写入 `"1"` → 开启；
 * 3. 未显式设置 → **dev 环境默认开启**（便于真机验证），生产构建默认关闭
 *    （Spike 尚未接入标尺 / 波形 / 交互，不能作为发布默认路径）。
 *
 * 特殊说明：macOS 上 Tauri 的 devtools 不易打开，因此验证不依赖控制台——
 * dev 环境的 PERF 悬浮面板提供了切换按钮（见 `dev/perfProject`），切换后自动刷新。
 *
 * @returns 当前是否启用内核。
 */
export function isTimelineKernelEnabled(): boolean {
    try {
        const override = localStorage.getItem(TIMELINE_KERNEL_FLAG_KEY);
        if (override === "0") return false;
        if (override === "1") return true;
        return import.meta.env.DEV;
    } catch {
        // 读不到 localStorage（隐私模式）时按默认关闭处理，避免意外启用。
        return false;
    }
}
