import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { Provider } from "react-redux";
import "@radix-ui/themes/styles.css";
import "./index.css";
import App from "./App.tsx";
import { store } from "./app/store";
import { AppTooltipProvider } from "./components/AppTooltip";
import { I18nProvider } from "./i18n/I18nProvider";
import { AppThemeProvider } from "./theme/AppThemeProvider";

createRoot(document.getElementById("root")!).render(
    <StrictMode>
        <Provider store={store}>
            <I18nProvider>
                <AppThemeProvider>
                    <AppTooltipProvider>
                        <App />
                    </AppTooltipProvider>
                </AppThemeProvider>
            </I18nProvider>
        </Provider>
    </StrictMode>,
);
