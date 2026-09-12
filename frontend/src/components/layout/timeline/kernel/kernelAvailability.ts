/**
 * 时间轴内核 · 可用性判定
 *
 * 【主要内容】
 * 把"内核本次会话是否已失败"归约为一个可测的纯函数，供面板决定渲染内核还是渲染
 * 失败界面。
 *
 * 【作用：为什么值得单独成模块】
 * 内核是**唯一**渲染路径，失败时**没有回退**——只能显示失败界面。这条"失败必须
 * 报错而不是静默空白"的约束是用户可感知的行为，必须有测试守护；而面板是 `.tsx`，
 * 本工程 Vitest 跑在 node 环境（无 jsdom），组件无法被单测引用。因此把判定抽成
 * 纯函数。
 *
 * 【历史背景（勿删）】内核曾是 opt-in 路径，其默认值一度跟随
 * `import.meta.env.DEV`，导致打包后**静默退回旧渲染器**（Phase 3 计划 R8）——
 * Windows 真机上的卡顿报告全部来自旧实现。旧实现已移除，但"绝不静默降级"的结论
 * 保留在此：失败必须显式告知用户。
 *
 * 【与其他模块的关系】
 * - 上游：`TimelinePanel` 传入 `TimelineKernelView` 回报的失败状态。
 * - 下游：面板据此渲染 `TimelineKernelView` 或 `KernelUnavailableNotice`。
 * - 独立性：纯函数，不依赖 DOM / React。
 */

/**
 * 内核是否可用于渲染。
 *
 * @param unavailable 本次会话内内核是否已回报不可用（WebGL2 创建失败等）。
 * @returns 可用时为 true；失败时为 false（调用方必须渲染失败界面）。
 */
export function isKernelAvailable(unavailable: boolean): boolean {
    return !unavailable;
}
