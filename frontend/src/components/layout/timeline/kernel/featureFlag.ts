/**
 * 渲染内核 · 特性开关
 *
 * 【主要内容】
 * 读取 `localStorage` 决定各面板是否走新渲染内核。目前有四层开关：
 * - **时间轴** `timelineKernel`；
 * - **参数编辑器（PianoRoll）** `pianoRollKernel`；
 * - **参数编辑器 GL 场景层** `pianoRollKernel.gl`；
 * - **曲线 GL 层** `pianoRollKernel.curveGl`。
 * 四者规则统一：未显式设置时**一律开启**（dev 与生产构建一致）；显式写入
 * `"0"` / `"1"` 可强制关闭 / 开启（逃生门，刷新页面后生效）。
 *
 * 【作用】
 * 内核已是**默认渲染路径**，不再是 opt-in。本开关因此只剩一个职责：**逃生门**
 * ——某个平台/驱动上出问题时，显式写 `"0"` 即退回既有实现，不需要重新发版。
 * 开关值在模块加载时读取一次（切换需刷新页面，与既有 dev 开关行为一致）。
 *
 * 【逃生门必须"够得到"——这是本文件同时提供 UI 助手的原因】
 * 只认 `localStorage` 的逃生门在**打包版里形同虚设**：正式构建没有 devtools
 * （`backend/src-tauri/Cargo.toml` 的 `tauri` 未启用 `devtools` feature，release 下
 * wry 亦默认关闭），用户没有控制台可写这个键；dev 环境的 PERF 面板又被
 * `import.meta.env.DEV` 挡住、根本不进产物。因此本文件额外导出
 * `isKernelRenderingEnabled` / `setKernelRenderingEnabled` 作为**设置界面**的
 * 后端，入口在「视图 → 时间轴显示设置」的总开关（见 `TimelineDisplaySettingsDialog`）。
 * 总开关一次写全四层，避免留下"时间轴退回旧实现、参数编辑器仍走内核"的混合状态。
 *
 * 【为什么总开关不自动刷新页面】工程可能含未保存编辑，而本工程没有脏标记或
 * `beforeunload` 保护，替用户决定丢弃是错的；界面只提示"重启后生效"。
 *
 * 【为什么默认值不能再跟随 `import.meta.env.DEV`】内核的价值是解决真机上的性能
 * 问题，而 build 模式（`TAURI_UI_MODE=build`）跑的是生产包：`DEV === false` 会让
 * 全部开关默认关闭，于是**打包后悄悄退回旧渲染器**。开发时看到新实现、打包后看到
 * 旧实现，是这条路径上最难察觉的一类问题（本项目实际踩过：Windows 上的卡顿报告
 * 全部来自旧实现，与新内核无关）。默认值必须与构建模式无关。
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
 * 3. 未显式设置 → **开启**（与构建模式无关）。
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
        // 未显式设置 → 开启（与 DEV / PROD 无关，见文件头）。
        return true;
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
 * 规则与时间轴内核**一致**：未显式设置时默认开启（与构建模式无关）。
 *
 * 特殊说明 1（为什么不能跟随 DEV）：`TAURI_UI_MODE=build` 跑生产包，其中
 * `import.meta.env.DEV === false`。若默认值跟随它，打包后会静默退回旧渲染器——
 * 开发与打包行为不一致，且只有上手操作才发现。
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
        // 未显式设置 → 开启（与时间轴内核同一策略）。
        return true;
    } catch {
        return false;
    }
}

/** 参数编辑器 GL 场景层开关 key（在内核开关之上再分一层）。 */
export const PIANO_ROLL_KERNEL_GL_FLAG_KEY = "hifishifter.pianoRollKernel.gl";

/**
 * 是否启用参数编辑器的 GL 场景层（阶段 2：静态图层上 GL）。
 *
 * 规则：未显式设置时开启；显式写入 `"0"` / `"1"` 可覆盖。
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
        return true;
    } catch {
        return false;
    }
}

/** 参数编辑器曲线 GL 层开关 key（在 GL 场景层之上再分一层）。 */
export const PIANO_ROLL_CURVE_GL_FLAG_KEY = "hifishifter.pianoRollKernel.curveGl";

/**
 * 是否启用曲线 GL 层（阶段 3）。
 *
 * 规则：未显式设置时开启；显式写入 `"0"` / `"1"` 可覆盖。
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
        return true;
    } catch {
        return false;
    }
}

/**
 * 四层开关的 key 全集（顺序无关）。
 *
 * 【为什么需要一份"全集"】设置界面的总开关要**一次写全四层**——只写其中一层会留下
 * 混合状态（例如时间轴退回旧实现、参数编辑器仍走内核），出问题时用户与支持者都
 * 难以判断当前到底跑的是哪套实现。
 */
export const KERNEL_FLAG_KEYS = [
    TIMELINE_KERNEL_FLAG_KEY,
    PIANO_ROLL_KERNEL_FLAG_KEY,
    PIANO_ROLL_KERNEL_GL_FLAG_KEY,
    PIANO_ROLL_CURVE_GL_FLAG_KEY,
] as const;

/**
 * 渲染内核总开关的当前值：任一层被显式关闭即视为关闭。
 *
 * 【为什么用"与"而不是只看时间轴】设置界面把它显示为一个复选框，用户勾上意味着
 * "我要用新实现"。混合状态下（部分层关闭）显示为未勾选更诚实——它确实没在跑完整
 * 的新实现。
 *
 * 特殊说明：读的是**四层开关本身**，因此与真实渲染路径同源，不会出现"界面显示开启
 * 但实际跑旧实现"的分叉。
 *
 * @returns 四层全部启用时为 true。
 */
export function isKernelRenderingEnabled(): boolean {
    return (
        isTimelineKernelEnabled() &&
        isPianoRollKernelEnabled() &&
        isPianoRollGlSceneEnabled() &&
        isPianoRollCurveGlEnabled()
    );
}

/**
 * 一次写全四层开关（设置界面的总开关）。
 *
 * 流程：对 [`KERNEL_FLAG_KEYS`] 逐个写入 `"1"` / `"0"`。
 *
 * 特殊说明 1：**必须四层一起写**，理由见 `KERNEL_FLAG_KEYS` 说明。
 *
 * 特殊说明 2：本函数**不会**让改动立即生效——四个开关都在各面板模块加载时读取一次
 * （见文件头），因此调用方必须在写完后提示用户**重启 / 刷新**。这里不自动
 * `location.reload()`：工程可能含未保存编辑，而本工程没有脏标记或 `beforeunload`
 * 保护，替用户决定丢弃是错的。
 *
 * 特殊说明 3：存储不可用（隐私模式 / 受限 WebView）时静默失败——调用方按"刷新后
 * 仍为默认值"处理即可，不应让设置界面抛错。
 *
 * @param enabled true = 写入 `"1"`（使用内核）；false = 写入 `"0"`（退回既有实现）。
 */
export function setKernelRenderingEnabled(enabled: boolean): void {
    try {
        const value = enabled ? "1" : "0";
        for (const key of KERNEL_FLAG_KEYS) {
            globalThis.localStorage?.setItem(key, value);
        }
    } catch {
        // 存储不可写：忽略（见特殊说明 3）。
    }
}
