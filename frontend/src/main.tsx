/* eslint-disable react-refresh/only-export-components -- 文件同时导出组件与 Hook/常量（刷新边界按文件粒度接受） */
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { useEffect } from "react";
import { Provider } from "react-redux";
import "@radix-ui/themes/styles.css";
import "./index.css";
import App from "./App.tsx";
import { store } from "./app/store";
import { AppTooltipProvider } from "./components/AppTooltip";
import { fadeToolTipSuppress } from "./components/layout/timeline/FadeContextMenu";
import { I18nProvider } from "./i18n/I18nProvider";
import { AppThemeProvider } from "./theme/AppThemeProvider";
import { initModifierWatcher } from "./components/layout/timeline/hooks/modifierWatcher";
import { installGlobalErrorReporting } from "./services/frontendErrorLog";

// 全局兜底：未捕获异常 / 未处理的 Promise rejection 回传到后端统一日志。
installGlobalErrorReporting();

/** 进程级全局手势基建：自愈式修饰键跟踪（淡化曲率等 modifierOnly 键位）。 */
function GlobalGestureServices() {
    useEffect(() => initModifierWatcher() ?? undefined, []);
    return null;
}

createRoot(document.getElementById("root")!).render(
    <StrictMode>
        <Provider store={store}>
            <I18nProvider>
                <AppThemeProvider>
                    <AppTooltipProvider
                        isSuppressedExternal={() => fadeToolTipSuppress.isSuppressed}
                    >
                        <GlobalGestureServices />
                        <App />
                    </AppTooltipProvider>
                </AppThemeProvider>
            </I18nProvider>
        </Provider>
    </StrictMode>,
);
