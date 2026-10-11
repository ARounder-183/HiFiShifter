// 宿主形态探测的**唯一底层**：只读显式 bootstrap 全局，零运行时依赖。
//
// 【为什么单独成文件】`hostCapabilities.ts` 为了 `dawControlledReason` 依赖 i18n
// Provider（运行时依赖），`uiStorage` 这类非 React 上下文不能 import 它 —— 此前
// 只能逐字复制一份 `isPluginMode()`（判据漂移风险）。判定集中在这一处：任何模块
// （组件、hook、service、无 React 的工具）都从这里拿形态，不自己摸全局。
//
// 【为什么是显式 bootstrap】不能把"所有 WebView2 都当插件"：独立 App 也跑在
// WebView2 里。插件由原生侧注入 `window.__HFS_PLUGIN_BOOTSTRAP__`（见
// `pluginHost.ts` 的 Window 接口声明）。

import type { PluginBootstrap } from "./pluginHost";

export type HostMode = "standalone" | "plugin";

/** 显式 bootstrap 快照；非插件环境返回 `null`。services 层之外请用下面的判据函数。 */
export function pluginBootstrap(): PluginBootstrap | null {
    if (typeof window === "undefined") return null;
    return window.__HFS_PLUGIN_BOOTSTRAP__ ?? null;
}

/** 当前宿主形态。`version === 1` 才认（字段存在但版本不符 = 旧原生壳，按独立 App 处理）。 */
export function hostMode(): HostMode {
    return pluginBootstrap()?.version === 1 ? "plugin" : "standalone";
}

export function isPluginMode(): boolean {
    return hostMode() === "plugin";
}

/** 独立 App 桌面形态（Tauri）。插件与纯浏览器预览均为 false。 */
export function isTauriRuntime(): boolean {
    if (typeof window === "undefined") return false;
    return Boolean((window as unknown as { __TAURI__?: unknown }).__TAURI__);
}
