/**
 * 时间轴渲染内核 · 特性开关
 *
 * 【主要内容】
 * 读取 `localStorage` 决定时间轴是否走新渲染内核：未显式设置时 dev 环境开启、
 * 生产构建关闭；显式写入 `"0"` / `"1"` 可强制关闭 / 开启。
 *
 * 【作用】
 * 落地策略要求「可一键回退」：内核是**仍在做新旧对齐（parity）的 opt-in 路径**，
 * 因此本开关保留为逃生门——显式写入 `"0"` 即整体退回既有实现，零影响。开关值在
 * 模块加载时读取一次（切换需刷新页面，与既有 dev 开关行为一致）。
 *
 * 【与其他模块的关系】
 * - 上游：`TimelinePanel` 在模块加载时读取并决定是否渲染 `TimelineKernelView`。
 * - 独立性：纯函数，不依赖 React。
 */

/** 开关 key。 */
export const TIMELINE_KERNEL_FLAG_KEY = "hifishifter.timelineKernel";

/**
 * 是否启用时间轴渲染内核。
 *
 * 规则（按优先级）：
 * 1. 显式写入 `"0"` → 关闭（逃生门，出问题时一键退回既有实现）；
 * 2. 显式写入 `"1"` → 开启；
 * 3. 未显式设置 → **dev 环境默认开启**（便于真机验证），生产构建默认关闭
 *    （内核仍是 opt-in 路径，新旧对齐尚未完成，暂不作为发布默认路径）。
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
