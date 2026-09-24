/**
 * 独立窗口的 React 入口（`detached.html` 的 JS 入口）。
 *
 * 与主入口的区别：这里只挂载**一个**被拆出去的面板，且必须自带 Provider
 * （I18n / 主题 / Tooltip）与 Redux Provider —— 独立窗口是另一个 JS 上下文，主窗口
 * 的那一套在这里都不存在。状态经 `detachBridge` 从主窗口取一次快照后靠动作复制
 * 保持同步（见 `app/store.ts` 与 `features/dock/detachBridge.ts`）。
 */

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { Provider } from "react-redux";
import "@radix-ui/themes/styles.css";
import "./index.css";

import { store } from "./app/store";
import { DetachedRoot } from "./components/dock/DetachedRoot";
import { AppTooltipProvider } from "./components/AppTooltip";
import { I18nProvider } from "./i18n/I18nProvider";
import { AppThemeProvider } from "./theme/AppThemeProvider";

createRoot(document.getElementById("root")!).render(
    <StrictMode>
        <Provider store={store}>
            <I18nProvider>
                <AppThemeProvider>
                    <AppTooltipProvider>
                        <DetachedRoot />
                    </AppTooltipProvider>
                </AppThemeProvider>
            </I18nProvider>
        </Provider>
    </StrictMode>,
);
