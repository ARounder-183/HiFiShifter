/**
 * 渲染内核 · 特性开关
 *
 * 【主要内容】
 * 读取 `localStorage` 决定各面板是否走新渲染内核。目前有两个开关：
 * - **时间轴**：未显式设置时 dev 环境开启、生产构建关闭；
 * - **参数编辑器（PianoRoll）**：未显式设置时**一律关闭**（连 dev 也关）。
 * 两者都支持显式写入 `"0"` / `"1"` 强制关闭 / 开启。
 *
 * 【作用】
 * 落地策略要求「可一键回退」：内核是**仍在做新旧对齐（parity）的 opt-in 路径**，
 * 因此本开关保留为逃生门——显式写入 `"0"` 即整体退回既有实现，零影响。开关值在
 * 模块加载时读取一次（切换需刷新页面，与既有 dev 开关行为一致）。
 *
 * 【与其他模块的关系】
 * - 上游：`TimelinePanel` 在模块加载时读取并决定是否渲染 `TimelineKernelView`；
 *   `PianoRollPanel` 读取参数编辑器开关决定是否走内核滚动/渲染路径。
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

/** 参数编辑器（PianoRoll）内核开关 key。 */
export const PIANO_ROLL_KERNEL_FLAG_KEY = "hifishifter.pianoRollKernel";

/**
 * 是否启用参数编辑器渲染内核。
 *
 * 规则与时间轴内核**刻意不同**：未显式设置时**默认关闭**（连 dev 也关）。
 *
 * 特殊说明 1：时间轴内核在 dev 默认开启，是为了让真机验证不必每次改 localStorage；
 * 参数编辑器分三个阶段落地，阶段 1 期间新路径尚不完整（绘制仍在 Canvas2D），
 * 默认开启会让日常开发一直跑在半迁移状态。需要验证时显式写 `"1"`。
 *
 * 特殊说明 2：必须经 `globalThis.localStorage` + `typeof` 守卫读取，不能直接引用
 * 裸 `localStorage`——本工程 Vitest 跑在 node 环境（无 jsdom），直接引用会让
 * **导入该模块的任何测试**在模块求值期就抛 ReferenceError。
 *
 * @returns 当前是否启用参数编辑器内核。
 */
export function isPianoRollKernelEnabled(): boolean {
    try {
        const storage = globalThis.localStorage;
        if (storage == null) return false;
        return storage.getItem(PIANO_ROLL_KERNEL_FLAG_KEY) === "1";
    } catch {
        return false;
    }
}

/** 参数编辑器 GL 场景层开关 key（在内核开关之上再分一层）。 */
export const PIANO_ROLL_KERNEL_GL_FLAG_KEY = "hifishifter.pianoRollKernel.gl";

/**
 * 是否启用参数编辑器的 GL 场景层（阶段 2：静态图层上 GL）。
 *
 * 规则：未显式设置时关闭；显式写入 `"1"` 开启。
 *
 * 特殊说明 1（为什么要独立于内核开关）：阶段 2 把网格 / 键盘 / 刻度 / 文字搬到
 * WebGL2，是**逐层替换**的过程。独立开关让"某一层迁移出问题"可以只回退 GL 层，
 * 而保留阶段 1 已经验证过的滚动内核（否则一次回退会丢掉两阶段的收益）。
 *
 * 特殊说明 2：本开关**不含** `isPianoRollKernelEnabled()` 的判断——两个开关各管
 * 各的，调用方按 `内核 && GL` 组合使用。把依赖写进这里会让"内核开着但 GL 单独关"
 * 这种合法组合无法表达。
 *
 * 特殊说明 3：与其它开关一样经 `globalThis.localStorage` + `typeof` 守卫读取
 * （本工程 Vitest 跑在 node 环境，无 jsdom）。
 *
 * @returns 当前是否启用 GL 场景层。
 */
export function isPianoRollGlSceneEnabled(): boolean {
    try {
        const storage = globalThis.localStorage;
        if (storage == null) return false;
        return storage.getItem(PIANO_ROLL_KERNEL_GL_FLAG_KEY) === "1";
    } catch {
        return false;
    }
}
