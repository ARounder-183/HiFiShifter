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
import "@radix-ui/themes/styles.css";
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

export function mountApp(): void {
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
                                <App />
                            </AppRootErrorBoundary>
                        </AppTooltipProvider>
                    </AppThemeProvider>
                </I18nProvider>
            </Provider>
        </StrictMode>,
    );
}
