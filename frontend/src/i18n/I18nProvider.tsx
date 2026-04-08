import {
    createContext,
    useContext,
    useEffect,
    useMemo,
    useState,
    type PropsWithChildren,
} from "react";
import { messages, type Locale, type MessageKey } from "./messages";
import { coreApi } from "../services/api/core";

interface I18nContextValue {
    locale: Locale;
    setLocale: (locale: Locale) => void;
    t: (key: MessageKey) => string;
}

const I18nContext = createContext<I18nContextValue | null>(null);

const STORAGE_KEY = "hifishifter.locale";

function getDefaultLocale(): Locale {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored && stored in messages) {
        return stored as Locale;
    }
    const lang = navigator.language.toLowerCase();
    if (lang.startsWith("zh")) {
        // 區分繁體中文地區（台灣、香港、澳門）與簡體中文
        if (
            lang === "zh-tw" ||
            lang === "zh-hk" ||
            lang === "zh-mo" ||
            lang === "zh-hant" ||
            lang.includes("hant")
        ) {
            return "zh-TW";
        }
        return "zh-CN";
    }
    if (lang.startsWith("ja")) return "ja-JP";
    if (lang.startsWith("ko")) return "ko-KR";
    return "en-US";
}

export function I18nProvider({ children }: PropsWithChildren) {
    const [localeState, setLocaleState] = useState<Locale>(getDefaultLocale);

    useEffect(() => {
        // Native close-confirmation dialog lives in Rust (Tauri). Keep backend locale in sync
        // so the dialog follows the user's in-app language.
        const tauriInvoke = window.__TAURI__?.core?.invoke ?? window.__TAURI__?.invoke;
        if (typeof tauriInvoke !== "function") return;

        void coreApi.setUiLocale(localeState).catch(() => {
            // Best-effort: ignore failures (e.g. during early boot).
        });
    }, [localeState]);

    const value = useMemo<I18nContextValue>(() => {
        return {
            locale: localeState,
            setLocale: (nextLocale: Locale) => {
                setLocaleState(nextLocale);
                localStorage.setItem(STORAGE_KEY, nextLocale);
            },
            t: (key: MessageKey) => {
                const localeMessages = messages[localeState] as Record<MessageKey, string>;
                return localeMessages[key] ?? messages["en-US"][key];
            },
        };
    }, [localeState]);

    return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>;
}

export function useI18n() {
    const context = useContext(I18nContext);
    if (!context) {
        throw new Error("useI18n must be used within I18nProvider");
    }
    return context;
}
