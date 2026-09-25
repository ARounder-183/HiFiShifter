/* eslint-disable react-refresh/only-export-components -- 文件同时导出组件与 Hook/常量（刷新边界按文件粒度接受） */
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

/**
 * 组件之外按当前语言取文案（命令式 API、窗口创建等）。
 *
 * 【为什么不能直接用 `useI18n`】这些调用点不是组件：例如创建独立窗口时要把面板
 * 标题交给**系统标题栏**，而 `titleKey` 是 i18n 键 —— 直接塞过去就会显示成
 * "undo_history_title"（用户报告过）。语言取自与 Provider 初值**同一来源**
 * （localStorage，`setLocale` 会写；缺省回落到浏览器语言），两边因此不会分叉。
 *
 * 查不到的键原样返回：宁可显示键名，也不要显示一个空白标题。
 */
export function translateOutsideReact(key: string): string {
    // 无浏览器环境（单测、SSR 探针）时 `localStorage` 不存在：回落到英文词典，
    // 而不是抛错 —— 取一条文案不该让调用方崩掉。
    const locale: Locale = typeof localStorage === "undefined" ? "en-US" : getDefaultLocale();
    const dict = messages[locale] as Record<string, string | undefined>;
    return dict[key] ?? (messages["en-US"] as Record<string, string | undefined>)[key] ?? key;
}
