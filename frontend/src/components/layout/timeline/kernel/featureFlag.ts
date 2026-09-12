/**
 * 渲染内核 · 特性开关
 *
 * 【主要内容】
 * 读取 `localStorage` 决定各面板是否走新渲染内核。目前有四层开关：
 * - **时间轴** `timelineKernel`；
 * - **参数编辑器（PianoRoll）** `pianoRollKernel`；
 * - **参数编辑器 GL 场景层** `pianoRollKernel.gl`；
 * - **曲线 GL 层** `pianoRollKernel.curveGl`。
 * 四者规则统一：未显式设置时 **dev 环境开启、生产构建关闭**；显式写入
 * `"0"` / `"1"` 可强制关闭 / 开启（逃生门，刷新页面后生效）。
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
 * 规则与时间轴内核**一致**：未显式设置时 dev 默认开启、生产构建默认关闭。
 *
 * 特殊说明 1（为什么 dev 也要默认开启）：内核的价值是解决真机上的性能问题，
 * 而真机验证只能在 dev 里做。默认关闭时 `tauri dev` 跑的一直是旧实现，
 * 真机报告（如 Windows 上的卡顿）反映的是旧路径的现象，无法用于判断新内核
 * 是否达标。默认开启后，关掉内核只需写 `"0"`，成本很低。
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
        const override = storage.getItem(PIANO_ROLL_KERNEL_FLAG_KEY);
        if (override === "0") return false;
        if (override === "1") return true;
        // 未显式设置：dev 默认开启、生产构建默认关闭（与时间轴内核同一策略）。
        return import.meta.env.DEV;
    } catch {
        return false;
    }
}

/** 参数编辑器 GL 场景层开关 key（在内核开关之上再分一层）。 */
export const PIANO_ROLL_KERNEL_GL_FLAG_KEY = "hifishifter.pianoRollKernel.gl";

/**
 * 是否启用参数编辑器的 GL 场景层（阶段 2：静态图层上 GL）。
 *
 * 规则：未显式设置时跟随 `import.meta.env.DEV`；显式写入 `"0"` / `"1"` 可覆盖。
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
        const override = storage.getItem(PIANO_ROLL_KERNEL_GL_FLAG_KEY);
        if (override === "0") return false;
        if (override === "1") return true;
        return import.meta.env.DEV;
    } catch {
        return false;
    }
}

/** 参数编辑器曲线 GL 层开关 key（在 GL 场景层之上再分一层）。 */
export const PIANO_ROLL_CURVE_GL_FLAG_KEY = "hifishifter.pianoRollKernel.curveGl";

/**
 * 是否启用曲线 GL 层（阶段 3）。
 *
 * 规则：未显式设置时跟随 `import.meta.env.DEV`；显式写入 `"0"` / `"1"` 可覆盖。
 *
 * 特殊说明（为什么单独一层开关）：曲线与静态图层（网格/键盘/刻度）的风险完全不同
 * ——它有 miter 连接、非整数线宽、虚线相位与裁剪，是视觉保真度最难的一层。独立开关
 * 让"曲线迁移出问题"可以只回退曲线，保留阶段 2 已充分验证的静态层与叠加层。
 *
 * 实测依据（Phase 3 计划 R7）：最小缩放下 3 分钟曲线约 36000 个可见点，
 * Canvas2D 描边需 71.7ms/帧（≈14fps），GL 全路径 4.6ms（15.6×）；该结论在 GPU 与
 * 软件栅格化下一致，因此**不需要**按平台分别开关。
 *
 * @returns 当前是否启用曲线 GL 层。
 */
export function isPianoRollCurveGlEnabled(): boolean {
    try {
        const storage = globalThis.localStorage;
        if (storage == null) return false;
        const override = storage.getItem(PIANO_ROLL_CURVE_GL_FLAG_KEY);
        if (override === "0") return false;
        if (override === "1") return true;
        return import.meta.env.DEV;
    } catch {
        return false;
    }
}
