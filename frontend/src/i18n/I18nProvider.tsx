/* eslint-disable react-refresh/only-export-components -- 文件同时导出组件与 Hook/常量（刷新边界按文件粒度接受） */
import {
    createContext,
    useContext,
    useEffect,
    useMemo,
    useState,
    useSyncExternalStore,
    type PropsWithChildren,
} from "react";
import { messages, type Locale, type MessageKey } from "./messages";
import {
    formatNumber,
    formatShortcutLabel,
    formatTemplate,
    formatUnit,
    selectPluralForm,
} from "./format";
import {
    getExtensionMessagesVersion,
    lookupExtensionMessage,
    subscribeExtensionMessages,
} from "./extensionMessages";
import { coreApi } from "../services/api/core";

interface I18nContextValue {
    locale: Locale;
    setLocale: (locale: Locale) => void;
    t: (key: MessageKey) => string;
    /** `{name}` 插值。取代调用点手写的 `.replace("{name}", …)`（全仓约 40 处）。 */
    tVars: (key: MessageKey, vars: Record<string, string | number>) => string;
    /**
     * 复数形态。词典里用 `"clip|clips"` 写单复数，单形态表示该语言不区分复数。
     *
     * 选中形态后会把 `{count}` 回填为传入的数量 —— 因此带计数的复数文案
     * **一律用 `{count}` 占位符**（不要用 `{n}` / `{c}` 等同义词，
     * 否则会原样渲染出花括号）。
     */
    plural: (key: MessageKey, count: number) => string;
    /** 把文案里的 `{modifier}` 换成当前平台的主修饰键（macOS `⌘`，其余 `Ctrl`）。 */
    shortcut: (key: MessageKey) => string;
    /** 按语系格式化数字。 */
    number: (value: number, options?: Intl.NumberFormatOptions) => string;
    /** 按语系格式化「数字 + 单位」，空格与符号由引擎决定。 */
    unit: (value: number, unit: string, options?: Intl.NumberFormatOptions) => string;
    /**
     * **无类型**翻译，供扩展使用。
     *
     * 内置的 `t()` 受 `MessageKey` 联合约束，第三方键编译期不存在于其中，
     * 因此单独给一个字符串键的入口 —— 而不是像历史上 71 处那样
     * `tf` 把内置的类型安全也一起丢掉。
     *
     * 查不到时返回键名本身（便于定位漏注册的文案）。
     */
    tf: (key: string) => string;
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

    /*
     * 订阅扩展文案注册表：第三方面板注册文案后，已经挂载的组件需要重新取值。
     * 用 `useSyncExternalStore` 而不是把版本号塞进 `useMemo` 依赖 —— 后者会让
     * 每次注册都重建整个 context value，把所有消费方一起重渲染。
     */
    useSyncExternalStore(
        subscribeExtensionMessages,
        getExtensionMessagesVersion,
        getExtensionMessagesVersion,
    );

    const value = useMemo<I18nContextValue>(() => {
        const dict = messages[localeState] as Record<MessageKey, string>;
        /*
         * 解析顺序：静态词典（当前语系）→ 扩展层（当前语系）→ 静态词典（en-US）。
         *
         * 扩展层排在静态词典**之后**是有意的：内置键不可被第三方覆盖，
         * 否则一个面板注册 `ok` 就能改掉全应用的「确定」按钮。
         */
        const lookup = (key: MessageKey): string =>
            dict[key] ?? lookupExtensionMessage(localeState, key) ?? messages["en-US"][key];
        return {
            locale: localeState,
            setLocale: (nextLocale: Locale) => {
                setLocaleState(nextLocale);
                localStorage.setItem(STORAGE_KEY, nextLocale);
            },
            t: lookup,
            tVars: (key, vars) => formatTemplate(lookup(key), vars),
            plural: (key, count) =>
                formatTemplate(selectPluralForm(localeState, count, lookup(key)), { count }),
            shortcut: (key) => formatShortcutLabel(lookup(key)),
            number: (value, options) => formatNumber(localeState, value, options),
            unit: (value, unit, options) => formatUnit(localeState, value, unit, options),
            tf: (key) =>
                lookupExtensionMessage(localeState, key) ??
                (messages[localeState] as Record<string, string>)[key] ??
                (messages["en-US"] as Record<string, string>)[key] ??
                key,
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
    return (
        dict[key] ??
        lookupExtensionMessage(locale, key) ??
        (messages["en-US"] as Record<string, string | undefined>)[key] ??
        key
    );
}
