/**
 * 工具栏控件的**唯一清单**：每个 `data-toolbar-control` 必须在这里出现一次。
 *
 * 【为什么单独一个文件】`ToolbarStrip.tsx` 只导出组件（React Fast Refresh 要求
 * 一个模块要么全是组件、要么全是常量，混在一起会让热更新退化成整页刷新）。
 *
 * 【为什么需要这份清单】移动端此前**完全没有**工具栏：外壳只渲染菜单 sheet 与动作
 * 抽屉，而 `ACTION_META` 的 135 条动作里没有任何一条叫 autoScroll / ripple / snap /
 * dragDirection。于是"自动滚屏"在手机上既没有按钮也没有动作 —— 没有报错，也没有日志。
 * `mobileToolbarCoverage.test.tsx` 拿这份清单与真实 DOM 的
 * `[data-toolbar-control]` 集合比对：新增一个按钮却忘了让移动端看到，会在 CI 里失败。
 */
export const TOOLBAR_CONTROL_IDS = [
    "capture",
    "crossfade",
    "splitTransition",
    "snap",
    "ripple",
    "playheadZoom",
    "autoScroll",
    "followPlayhead",
    "clickSelectTrack",
    "ignoreGrouping",
    "fileBrowser",
    "notebook",
] as const;

export type ToolbarControlId = (typeof TOOLBAR_CONTROL_IDS)[number];

/**
 * 只在**插件形态**渲染的控件（宿主才有可录的音频）。
 *
 * 【为什么要单列】覆盖测试在独立 App 里跑，这些控件不在 DOM 里；把它们从期望集合里
 * 显式减掉，比让它们"看起来永远缺失"要诚实。它们与形态无关 —— 桌面与移动用的是
 * 同一个 `ToolbarStrip`，因此不会出现"桌面有、手机没有"。
 */
export const PLUGIN_ONLY_TOOLBAR_CONTROLS: readonly ToolbarControlId[] = ["capture"];
