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
import { surfaceProfile, type DeclaredForm } from "./surfaceProfileStore";

export type HostMode = "standalone" | "plugin";

/**
 * 应用形态：桌面 / 插件 / 移动。
 *
 * 【为什么是三值而不是布尔】"是不是移动端"与"是不是插件"是两个正交的问题：
 * 插件跑在桌面上（有键鼠、可多窗口），移动端是独立 App（无键鼠、单窗口）。
 * 合成一个布尔之后，任何新组合（移动端插件？平板上跑插件？）都要改枚举。
 *
 * 【为什么这个判断必须只有一个出口】外壳、停靠剖面、控件密度、输入剖面、CSS
 * 令牌层、手势层全都依赖它。散成多处之后，任何一次不一致都表现为"某个界面
 * 长成了另一种形态"，而这种故障只在特定窗口尺寸下才出现。
 * `scripts/checks/check-mobile-form-gate.py` 会拦下 `window.innerWidth <` 与
 * `navigator.userAgent` 这类"自己猜形态"的写法。
 */
export type AppForm = "desktop" | "plugin" | "mobile";

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

/**
 * 当前应用形态。**全仓唯一的形态闸门**。
 *
 * 【判据顺序】插件优先于移动：插件的形态差异（宿主接管、窗口下限 640×400）比
 * 视口大小更根本 —— 一个跑在平板上的插件仍然是插件。移动形态由后端剖面声明
 * （`surfaceProfileStore`），后端不可用时回落桌面。
 */
export function appForm(): AppForm {
    if (isPluginMode()) return "plugin";
    return declaredForm();
}

/**
 * 后端声明的形态（不含插件叠加）。
 *
 * 【为什么单独导出】需要回答"这台设备的输入/存储是什么"的地方（能力表、手势层）
 * 要的是这个；需要回答"界面长什么样"的地方要的是 `appForm()`。
 */
export function declaredForm(): DeclaredForm {
    return surfaceProfile().form;
}

/** 是否移动形态。等价于 `appForm() === "mobile"`，只是调用点更短。 */
export function isMobileForm(): boolean {
    return appForm() === "mobile";
}

/** 是否桌面形态（独立 App + 桌面）。插件与移动端都为 false。 */
export function isDesktopForm(): boolean {
    return appForm() === "desktop";
}

/** 独立 App 桌面形态（Tauri）。插件与纯浏览器预览均为 false。 */
export function isTauriRuntime(): boolean {
    if (typeof window === "undefined") return false;
    return Boolean((window as unknown as { __TAURI__?: unknown }).__TAURI__);
}
