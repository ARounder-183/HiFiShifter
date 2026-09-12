/**
 * 渲染内核 · 挂载决策
 *
 * 【主要内容】
 * 决定"本次渲染该挂内核还是既有实现"的**唯一判据**，把两组互不相干的条件合并：
 * 1. **静态开关**：用户在设置界面 / localStorage 表达的意愿（默认开启）；
 * 2. **运行期可用性**：本次会话内内核是否已经失败过（WebGL2 创建失败等）。
 *
 * 【作用：为什么必须抽成独立模块】
 * 这个判据此前内联在 `TimelinePanel` 的 JSX 里（`TIMELINE_KERNEL_ENABLED && !kernelUnavailable`）。
 * 本工程 Vitest 跑在 **node** 环境（无 jsdom、无 testing-library），任何 `.tsx`
 * 都不被测试引用——**0 个**测试文件 import 组件。后果是：把该分支改成恒真或恒假
 * （后者正是"内核永不挂载"这一回归）**全部测试仍然通过**，日志里看不出任何异常。
 *
 * 抽成纯函数后，判据本身可被单测覆盖，包括"用户想用但内核失败"这种只能靠两条
 * 输入组合出来的状态。
 *
 * 【两条件的语义差别（不能合并成一个布尔量）】
 * - 静态开关为 false：用户**主动不要**内核 → 不挂，且**不该**尝试（尝试会白建
 *   一次 GL 上下文、并在控制台留下误导性的"不可用"告警）。
 * - 运行期不可用为 true：用户**想要**但环境不支持 → 不挂，且必须回退到既有实现
 *   （绝不能让用户看到空白面板）。
 *
 * 两者结果相同（都不挂），但原因不同——因此返回值同时给出**两者**，供调用方
 * 决定是否记日志 / 是否提示用户。
 *
 * 【与其他模块的关系】
 * - 上游：`TimelinePanel` 在渲染期调用；静态开关来自 `featureFlag`，
 *   可用性来自 `TimelineKernelView` 的 `onUnavailable` 回报。
 * - 下游：`TimelinePanel` 据 `useKernel` 决定渲染 `TimelineKernelView` 还是既有实现。
 * - 独立性：纯函数，不依赖 DOM / React，可直接单测。
 */

/** 挂载决策入参。 */
export interface KernelMountDecisionArgs {
    /**
     * 静态开关：用户是否要求使用内核（来自 `featureFlag`，模块加载时读一次）。
     *
     * 缺省视为 true（与开关的默认值一致）——漏传不应静默退回旧实现。
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
    /** 是否因为用户主动关闭而不挂（用于区分"没试"与"试了失败"）。 */
    readonly disabledByUser: boolean;
    /** 是否因为运行期不可用而不挂。 */
    readonly disabledByFailure: boolean;
}

/**
 * 解析挂载决策。
 *
 * 流程：归一化两个可选入参（缺省 = 开关默认开启、未失败）→ 求 `useKernel` →
 * 给出两个互斥的"为何不挂"标记。
 *
 * 特殊说明 1：`useKernel` 是 `enabled && !unavailable`。**不做**"失败过一次就
 * 永久记住"的持久化——那是调用方（面板 state）的职责；本函数是无状态的。
 *
 * 特殊说明 2：两个 `disabledBy*` 标记**互不冒充**——`disabledByFailure` 只在
 * "用户想用却失败"时为 true。用户主动关闭时不报失败，避免在控制台留下误导性的
 * "内核不可用"日志（环境未必有问题，是用户自己关的）。
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
