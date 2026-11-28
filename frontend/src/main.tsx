/**
 * 入口：先让用户偏好就位，再加载并挂载应用。
 *
 * 【为什么是"先 hydrate 再 import"】偏好（语言 / 快捷键 / 外观 / 时间轴缩放…）
 * 由后端的共享配置文件持有，而它们的读取点大量是同步的、有些还发生在**模块加载期**
 * （`keybindingsSlice` 建 store 时读覆盖项、`themeStorage` 读外观、
 * `fileBrowserSlice` 读上次目录）。如果先静态 import 应用再 hydrate，那些读取已经
 * 按空值定型了 —— 界面会以默认外观与默认快捷键起手，用户会看到"设置没保存"。
 *
 * 因此本文件**只做两件事**：灌数据、动态加载 `./mount`。所有应用模块都在
 * `hydrateUiStorage()` 之后才被求值。顶层的 `await` 由构建目标支持
 * （`import.meta.env` 与 `await import` 已在既有代码里使用）。
 */

import { installGlobalErrorReporting, installFrontendLogPolicy } from "./services/frontendErrorLog";
import { hydrateUiStorage, installUiStorageFlushHooks } from "./services/uiStorage";
import { invoke } from "./services/invoke";
import { hydrateSurfaceProfile } from "./services/surfaceProfileStore";

// 日志策略先就位：Off 档下连错误监听器都不许跨 IPC 上报。
void installFrontendLogPolicy();
// 全局兜底：未捕获异常 / 未处理的 Promise rejection 回传到后端统一日志。
installGlobalErrorReporting();

// dev-only 后端替身：URL 带 `?mock=1` 时在灌数据之前安装假后端，使前端可在纯浏览器
// 里独立运行（本工程默认依赖 pywebview / Tauri 后端）。生产构建不打包该模块。
if (import.meta.env.DEV && new URLSearchParams(window.location.search).has("mock")) {
    const { installMockBackend } = await import("./dev/mockBackend");
    installMockBackend();
}

// 写入是去抖的：关窗前的最后一次变更必须在 `pagehide` 补交。
installUiStorageFlushHooks();

// 偏好就位（后端不可用时静默降级为本地缓存），此后加载的应用模块才能读到正确的值。
await hydrateUiStorage();

// ── 形态剖面就位 ──────────────────────────────────────────────────────────
//
// 【为什么必须在这里、而不是在 mount 里】形态决定的是一批**模块加载期**就要定型的
// 东西：停靠布局的出厂值（`dockSlice` 的初始 state）、控件密度默认值、输入剖面。
// 那些读取点在模块求值时就跑完了，晚一步灌入就等于没灌 —— 用户会看到"移动端以
// 桌面布局起手"，而重启一次又好了（因为第二次读到了缓存），是最难复现的一类。
//
// 与 `hydrateUiStorage` 同一时机、同一理由。失败时静默回落桌面剖面（浏览器预览）。
await hydrateSurfaceProfile(() => invoke("get_surface_profile"));

const { mountApp } = await import("./mount");
await mountApp();
