/**
 * 外观设置独立窗口的 React 入口
 *
 * 该文件作为 appearance.html 的 JS 入口，创建独立的 React 树，
 * 包含 I18nProvider 和 AppThemeProvider 以确保国际化和主题正常工作。
 * 不需要 Redux Provider，因为外观设置窗口不依赖 Redux 状态。
 */

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "@radix-ui/themes/styles.css";
import "./index.css";
import { I18nProvider } from "./i18n/I18nProvider";
import { AppThemeProvider } from "./theme/AppThemeProvider";
import { AppearanceWindow } from "./components/layout/AppearanceWindow";
import { AppRootErrorBoundary } from "./components/AppRootErrorBoundary";

createRoot(document.getElementById("root")!).render(
    <StrictMode>
        <I18nProvider>
            <AppThemeProvider>
                {/* 外观窗口也需要根级边界：它没有停靠层，抛错即整窗空白。 */}
                <AppRootErrorBoundary>
                    <AppearanceWindow />
                </AppRootErrorBoundary>
            </AppThemeProvider>
        </I18nProvider>
    </StrictMode>,
);
