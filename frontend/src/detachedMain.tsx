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
import { registerBuiltinPanels } from "./components/dock/registerBuiltinPanels";
import { attachBuiltinPanelComponents } from "./components/dock/attachBuiltinPanelComponents";
import { AppTooltipProvider } from "./components/AppTooltip";
import { I18nProvider } from "./i18n/I18nProvider";
import { AppThemeProvider } from "./theme/AppThemeProvider";

/*
 * 独立窗口是**另一个 JS 上下文**：主窗口 `App.tsx` 里的模块级注册不会执行到这里，
 * 因此必须自己注册一次 —— 否则 `DetachedRoot` 查注册中心查不到任何面板，
 * 表现为"拆出去的面板是一片空白"。注册是幂等的（同 id 覆盖），两处调用无冲突。
 */
registerBuiltinPanels();
attachBuiltinPanelComponents();

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
