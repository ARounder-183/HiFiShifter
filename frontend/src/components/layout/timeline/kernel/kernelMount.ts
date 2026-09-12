/**
 * 渲染内核 · 挂载决策
 *
 * 【主要内容】
 * 把"本次渲染是否该挂内核"的判据抽成一个纯函数，合并两条互不相干的条件：
 * 1. **静态意愿**（`enabled`）：调用方是否要求使用内核；
 * 2. **运行期可用性**（`unavailable`）：本次会话内内核是否已经失败过（WebGL2 创建失败等）。
 *
 * 【当前状态：本模块已无生产消费者（勿据此直接删除，见下）】
 * 内核收归唯一路径后，`TimelinePanel` 的判据简化成单条件——只问"运行期是否可用"
 * （`isKernelAvailable` + `KernelUnavailableNotice`，见 `kernelAvailability.ts`），
 * 静态意愿那一维随四个开关一起消失。因此本模块目前**只被自己的单测引用**。
 *
 * 【为什么暂时保留】它钉住的是"失败必须显式报错、绝不能静默空白"这一行为契约，
 * 以及"没试（用户不想要）"与"试了失败（环境不支持）"必须可区分这一更普遍的约束；
 * 这两条在唯一的调用点被简化掉之后依然成立，且日后若再出现第二条挂载分支就会重新
 * 需要它。是否随本次收尾一并删除，属于独立决策，不在"移除开关"这一任务的范围内。
 *
 * 【作用：为什么值得抽成独立模块（历史理由，仍然有效）】
 * 这个判据此前内联在 `TimelinePanel` 的 JSX 里。本工程 Vitest 跑在 **node** 环境
 * （无 jsdom、无 testing-library），任何 `.tsx` 都不被测试引用——**0 个**测试文件
 * import 组件。后果是：把该分支改成恒真或恒假（后者正是"内核永不挂载"这一回归）
 * **全部测试仍然通过**，日志里看不出任何异常。抽成纯函数后，判据本身可被单测覆盖。
 *
 * 【两条件的语义差别（不能合并成一个布尔量）】
 * - `enabled === false`：调用方**主动不要**内核 → 不挂，且**不该**尝试（尝试会白建
 *   一次 GL 上下文、并在控制台留下误导性的"不可用"告警）。
 * - `unavailable === true`：**想要**但环境不支持 → 不挂，且必须走到失败界面
 *   （绝不能让用户看到空白面板）。
 *
 * 两者结果相同（都不挂），但原因不同——因此返回值同时给出**两者**，供调用方
 * 决定是否记日志 / 是否提示用户。
 *
 * 【与其他模块的关系】
 * - 上游（曾）：`TimelinePanel` 在渲染期调用。当前无生产调用方。
 * - 横向：`kernelAvailability.isKernelAvailable` 是面板现在实际使用的可用性判据；
 *   可用性来自 `TimelineKernelView` 的 `onUnavailable` 回报。
 * - 独立性：纯函数，不依赖 DOM / React，可直接单测。
 */

/** 挂载决策入参。 */
export interface KernelMountDecisionArgs {
    /**
     * 静态意愿：调用方是否要求使用内核。
     *
     * 缺省视为 true——漏传不应静默退回既有实现。
     */
    readonly enabled?: boolean;
    /**
     * 本次会话内内核是否已回报不可用（WebGL2 创建失败 / 着色器编译失败）。
     *
     * 缺省视为 false（尚未失败）。
     */
    readonly unavailable?: boolean;
}

/** 挂载决策结果。 */
export interface KernelMountDecision {
    /** 最终是否挂载内核。 */
    readonly useKernel: boolean;
    /** 是否因为调用方主动关闭而不挂（用于区分"没试"与"试了失败"）。 */
    readonly disabledByUser: boolean;
    /** 是否因为运行期不可用而不挂。 */
    readonly disabledByFailure: boolean;
}

/**
 * 解析挂载决策。
 *
 * 流程：归一化两个可选入参（缺省 = 要求使用、未失败）→ 求 `useKernel` →
 * 给出两个互斥的"为何不挂"标记。
 *
 * 特殊说明 1：`useKernel` 是 `enabled && !unavailable`。**不做**"失败过一次就
 * 永久记住"的持久化——那是调用方（面板 state）的职责；本函数是无状态的。
 *
 * 特殊说明 2：两个 `disabledBy*` 标记**互不冒充**——`disabledByFailure` 只在
 * "要求使用却失败"时为 true。主动关闭时不报失败，避免在控制台留下误导性的
 * "内核不可用"日志（环境未必有问题，是调用方自己不要）。
 *
 * @param args 见 `KernelMountDecisionArgs`。
 * @returns 见 `KernelMountDecision`。
 */
export function resolveKernelMount(args: KernelMountDecisionArgs = {}): KernelMountDecision {
    const enabled = args.enabled ?? true;
    const unavailable = args.unavailable ?? false;
    const useKernel = enabled && !unavailable;
    return {
        useKernel,
        disabledByUser: !enabled,
        disabledByFailure: enabled && unavailable,
    };
}
