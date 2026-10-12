/**
 * 应用挂载点。
 *
 * 【为什么从 `main.tsx` 拆出来】偏好必须在**任何模块读取它们之前**就位：
 * `keybindingsSlice` 在模块加载期就读快捷键覆盖项、`themeStorage` 读外观、
 * `fileBrowserSlice` 读上次目录。这些读取是同步的（调用点太多，改成异步会迫使
 * 它们重排时序），所以启动顺序必须是「先灌数据 → 再加载这些模块」。
 * `main.tsx` 因此先 `await hydrateUiStorage()`，再动态 import 本文件。
 *
 * 本文件只在动态 import 之后被求值，因此它内部所有的静态 import 都发生在
 * 偏好就位之后 —— 这正是需要的保证。
 */

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { Provider } from "react-redux";
/*
 * Radix 的样式表**不在这里** import：它必须落在 `@layer radix` 里才能被 Tailwind
 * 工具类覆盖，而 CSS 的 `@import ... layer()` 只能写在 CSS 内（见 `index.css`
 * 顶部的层顺序说明）。在这里再 import 一次会得到一份**未分层**的副本，
 * `.rt-reset { all: unset }` 会重新压过所有工具类。
 */
import "./index.css";
import App from "./App.tsx";
import { store } from "./app/store";
import { getDockDragState } from "./features/dock/dockDragStore";
import { isFileBrowserDragActive } from "./features/fileBrowser/fileBrowserDragStore";
import { AppTooltipProvider } from "./components/AppTooltip";
import { GlobalGestureServices } from "./components/GlobalGestureServices";
import { AppRootErrorBoundary } from "./components/AppRootErrorBoundary";
import { fadeToolTipSuppress } from "./components/layout/timeline/FadeContextMenu";
import { I18nProvider } from "./i18n/I18nProvider";
import { AppThemeProvider } from "./theme/AppThemeProvider";
import { SessionUnavailable } from "./features/ara/SessionUnavailable";
import { isPluginMode } from "./services/hostCapabilities";
import { invoke, invokeErrorKind } from "./services/invoke";

/**
 * 挂载前的会话探活：后端编辑会话是否**已经**处于终态。
 *
 * 【为什么必须在挂载 App 之前问一次】会话在启动前就已经终结时（宿主从未把本插件接入
 * ARA），App 的初始化扇出会**并发**发出十几条命令，每条都失败、每条都进日志 —— 实测
 * Cubase 上那一秒有 15 条 `Invoke failed`。闸门拦不住它们（IPC 往返约 200ms，大部分
 * 命令在第一个响应回来前就发出去了），只有在扇出**之前**问一次才拦得住。
 *
 * 【为什么不能用 `ping`】`ping` 在 `editor/webview.rs` 里是**内联**处理的：它只回
 * `{"ok":true,"connected":link.owner().is_ok()}`，**不经过 `editor_session()`** ——
 * 也就是说"宿主从未绑定 ARA"时它照样回 `ok`。用它探活会得到一个假的健康信号，整条
 * 保护静默失效。必须用一个真正走命令队列的命令；`get_runtime_info` 正好合适（它本就
 * 在启动扇出里，返回值有用，这次往返不浪费）。
 *
 * 【为什么瞬时失败不算不可用】"编辑器尚未连上处理器"这类是**暂时**的：组件创建与绑定
 * 之间必然经历它，正常挂载后 App 自己会重试（见 `editor/mod.rs` 的分类测试）。
 */
async function probeSession(): Promise<"ready" | "unavailable"> {
    if (!isPluginMode()) return "ready";
    try {
        await invoke("get_runtime_info");
        return "ready";
    } catch (error) {
        return invokeErrorKind(error) === "terminal" ? "unavailable" : "ready";
    }
}

export async function mountApp(): Promise<void> {
    // dev-only 性能工程脚手架：动态 import 保证生产构建完全不打包该模块。
    // 用法：`?perf=400`（clip 总数，按 10 轨均分）或 `?perf=10x40`（轨数 × 每轨
    // clip 数）冷启动即全览；运行时也可用控制台 `window.__hsPerf({...})` 重生成。
    if (import.meta.env.DEV) {
        void import("./dev/perfProject").then((module) => module.installPerfProjectDevtools());
        // dev-only 调试出口：把 store 挂到 window，便于在浏览器里读取**真实**运行期
        // 状态（选区来源标记 `multiSelectionIntentional` 这类"必须有 Redux 上下文
        // 才能验证"的事实，从内核句柄读不到）。生产构建不挂载。
        (window as unknown as { __hfsStore?: typeof store }).__hfsStore = store;
    }

    // 探活必须在扇出之前完成（见 `probeSession` 的说明）。
    const session = await probeSession();

    createRoot(document.getElementById("root")!).render(
        <StrictMode>
            <Provider store={store}>
                <I18nProvider>
                    <AppThemeProvider>
                        <AppTooltipProvider
                            isSuppressedExternal={() =>
                                // 停靠拖拽期间必须抑制悬停提示：它不再是原生 tooltip，
                                // 不会自己消失，会正好盖住拖拽时给用户看的落点提示。
                                //
                                // 文件浏览器拖拽同理：行既带 `data-tooltip` 又是拖拽源，
                                // 不抑制的话气泡会被钉住并一路跟着鼠标飘到时间轴上方。
                                fadeToolTipSuppress.isSuppressed ||
                                getDockDragState()?.started === true ||
                                isFileBrowserDragActive()
                            }
                        >
                            <GlobalGestureServices />
                            {/*
                              根级错误边界：兜住面板子树之外抛出的渲染异常。
                              没有它时，任何非面板位置抛错都会卸载整棵树、留下空白窗口。
                              放在 Provider 内侧以便用上主题与语言，包住 App 整体。
                            */}
                            <AppRootErrorBoundary>
                                {session === "unavailable" ? <SessionUnavailable /> : <App />}
                            </AppRootErrorBoundary>
                        </AppTooltipProvider>
                    </AppThemeProvider>
                </I18nProvider>
            </Provider>
        </StrictMode>,
    );
}
