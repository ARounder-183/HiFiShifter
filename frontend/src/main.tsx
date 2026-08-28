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
                    <AppTooltipProvider isSuppressedExternal={() => fadeToolTipSuppress.isSuppressed}>
                        <GlobalGestureServices />
                        <App />
                    </AppTooltipProvider>
                </AppThemeProvider>
            </I18nProvider>
        </Provider>
    </StrictMode>,
);
