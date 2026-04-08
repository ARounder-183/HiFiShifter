/**
 * 应用主题提供者
 *
 * 管理全局主题状态，包括：
 * - Radix Theme 的动态配置（accentColor、grayColor、radius）
 * - 自定义 --qt-* 颜色变量注入
 * - 字体动态切换
 * - 预览/回退机制（外观设置对话框使用）
 */

import {
    createContext,
    useCallback,
    useContext,
    useEffect,
    useMemo,
    useRef,
    useState,
    type PropsWithChildren,
} from "react";
import { Theme } from "@radix-ui/themes";
import type {
    RadixAccentColor,
    RadixGrayColor,
    RadixRadius,
    AppearanceSettings,
    QtColorToken,
} from "./themeTypes";
import { QT_COLOR_TOKENS } from "./themeTypes";
import { loadAppearance, saveAppearance, loadCustomThemes } from "./themeStorage";

export type ThemeMode = "dark" | "light";
const PREVIEW_SETTINGS_KEY = "hifishifter.appearance.preview";
const PREVIEW_COLORS_KEY = "hifishifter.appearance.preview.colors";

interface ThemeContextValue {
    mode: ThemeMode;
    setMode: (mode: ThemeMode) => void;
    toggleMode: () => void;

    /* ── Radix Theme 动态属性 ── */
    accentColor: RadixAccentColor;
    setAccentColor: (color: RadixAccentColor) => void;
    grayColor: RadixGrayColor;
    setGrayColor: (color: RadixGrayColor) => void;
    radius: RadixRadius;
    setRadius: (radius: RadixRadius) => void;

    /* ── 字体 ── */
    fontFamily: string;
    setFontFamily: (font: string) => void;

    /* ── 自定义主题 ── */
    activeCustomThemeId: string | null;

    /* ── 外观设置：批量应用 & 预览回退 ── */
    applySettings: (settings: AppearanceSettings) => void;
    revertPreview: () => void;
}

const ThemeContext = createContext<ThemeContextValue | null>(null);

function loadInitialAppearance(): AppearanceSettings {
    return loadAppearance();
}

export function AppThemeProvider({ children }: PropsWithChildren) {
    const [appearance, setAppearance] = useState<AppearanceSettings>(loadInitialAppearance);

    // 快照：用于打开外观设置对话框时保存当前状态，关闭时如果未保存则回退
    const snapshotRef = useRef<AppearanceSettings | null>(null);

    /* ── 独立 state（方便子组件直接控制） ── */
    const mode = appearance.mode;
    const accentColor = appearance.accentColor;
    const grayColor = appearance.grayColor;
    const radius = appearance.radius;
    const fontFamily = appearance.fontFamily;
    const activeCustomThemeId = appearance.activeCustomThemeId;

    /* ── Setter helpers ── */
    const updateField = useCallback(
        <K extends keyof AppearanceSettings>(key: K, value: AppearanceSettings[K]) => {
            setAppearance((prev) => ({ ...prev, [key]: value }));
        },
        [],
    );

    const setMode = useCallback(
        (next: ThemeMode) => {
            updateField("mode", next);
        },
        [updateField],
    );

    const toggleMode = useCallback(() => {
        setAppearance((prev) => ({
            ...prev,
            mode: prev.mode === "dark" ? "light" : "dark",
        }));
    }, []);

    const setAccentColor = useCallback(
        (color: RadixAccentColor) => {
            updateField("accentColor", color);
        },
        [updateField],
    );

    const setGrayColor = useCallback(
        (color: RadixGrayColor) => {
            updateField("grayColor", color);
        },
        [updateField],
    );

    const setRadius = useCallback(
        (r: RadixRadius) => {
            updateField("radius", r);
        },
        [updateField],
    );

    const setFontFamily = useCallback(
        (font: string) => {
            updateField("fontFamily", font);
        },
        [updateField],
    );

    /* ── 批量应用（保存到 localStorage） ── */
    const applySettings = useCallback((settings: AppearanceSettings) => {
        setAppearance(settings);
        saveAppearance(settings);
        snapshotRef.current = null; // 清除快照

        // 应用自定义主题颜色
        applyCustomThemeColors(settings.activeCustomThemeId);
    }, []);

    /* ── 预览回退 ── */
    const revertPreview = useCallback(() => {
        if (snapshotRef.current) {
            const saved = snapshotRef.current;
            setAppearance(saved);
            snapshotRef.current = null;

            // 恢复自定义主题颜色
            applyCustomThemeColors(saved.activeCustomThemeId);
        }
    }, []);

    /* ── 注入自定义主题的 --qt-* 覆盖 ── */
    function applyCustomThemeColors(themeId: string | null) {
        const root = document.documentElement;

        // 先清除所有自定义覆盖
        for (const token of QT_COLOR_TOKENS) {
            root.style.removeProperty(`--${token}`);
        }

        if (!themeId) return;

        const themes = loadCustomThemes();
        const active = themes.find((t) => t.id === themeId);
        if (!active) return;

        for (const [token, value] of Object.entries(active.colors)) {
            if (value) {
                root.style.setProperty(`--${token as QtColorToken}`, value);
            }
        }
    }

    /* ── 副作用：同步 data-theme & font CSS variable ── */
    useEffect(() => {
        document.documentElement.dataset.theme = mode;
    }, [mode]);

    useEffect(() => {
        document.documentElement.style.setProperty("--qt-font-family", fontFamily);
        document.documentElement.style.setProperty("--default-font-family", fontFamily);
        document.body.style.fontFamily = fontFamily;
    }, [fontFamily]);

    useEffect(() => {
        const applyPreviewFromStorage = () => {
            try {
                const settingsRaw = localStorage.getItem(PREVIEW_SETTINGS_KEY);
                const colorsRaw = localStorage.getItem(PREVIEW_COLORS_KEY);
                if (!settingsRaw && !colorsRaw) return;

                if (settingsRaw) {
                    const settings = JSON.parse(settingsRaw) as Pick<
                        AppearanceSettings,
                        "mode" | "accentColor" | "grayColor" | "radius" | "fontFamily"
                    >;
                    setAppearance((prev) => ({
                        ...prev,
                        mode: settings.mode,
                        accentColor: settings.accentColor,
                        grayColor: settings.grayColor,
                        radius: settings.radius,
                        fontFamily: settings.fontFamily,
                        activeCustomThemeId: null,
                    }));
                }

                const root = document.documentElement;
                const colors = colorsRaw
                    ? (JSON.parse(colorsRaw) as Partial<Record<QtColorToken, string>>)
                    : {};
                for (const token of QT_COLOR_TOKENS) {
                    const val = colors[token];
                    if (val) {
                        root.style.setProperty(`--${token}`, val);
                    } else {
                        root.style.removeProperty(`--${token}`);
                    }
                }
            } catch {
                // ignore malformed preview payload
            }
        };

        const onStorage = (e: StorageEvent) => {
            if (e.key === PREVIEW_SETTINGS_KEY || e.key === PREVIEW_COLORS_KEY) {
                applyPreviewFromStorage();
            }
        };

        window.addEventListener("storage", onStorage);
        return () => window.removeEventListener("storage", onStorage);
    }, []);

    /* ── 初始化时保存快照（用于第一次打开设置对话框） ── */
    useEffect(() => {
        // 首次挂载时保存快照
        snapshotRef.current = appearance;

        // 初始化时应用自定义主题颜色
        applyCustomThemeColors(appearance.activeCustomThemeId);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    /* ── 保存快照（每当 appearance 稳定保存后更新快照供下次使用） ── */
    // 注意：只有通过 applySettings 才会真正保存到 localStorage
    // snapshotRef 在 revertPreview 中回退到上一次 apply 的状态
    useEffect(() => {
        // 每当 appearance 变化时，如果没有快照就创建一个
        if (!snapshotRef.current) {
            snapshotRef.current = appearance;
        }
    }, [appearance]);

    const value = useMemo<ThemeContextValue>(
        () => ({
            mode,
            setMode,
            toggleMode,
            accentColor,
            setAccentColor,
            grayColor,
            setGrayColor,
            radius,
            setRadius,
            fontFamily,
            setFontFamily,
            activeCustomThemeId,
            applySettings,
            revertPreview,
        }),
        [
            mode,
            setMode,
            toggleMode,
            accentColor,
            setAccentColor,
            grayColor,
            setGrayColor,
            radius,
            setRadius,
            fontFamily,
            setFontFamily,
            activeCustomThemeId,
            applySettings,
            revertPreview,
        ],
    );

    return (
        <ThemeContext.Provider value={value}>
            <Theme
                appearance={mode}
                accentColor={accentColor}
                grayColor={grayColor}
                radius={radius}
                className={`qt-theme ${mode}`}
            >
                {children}
            </Theme>
        </ThemeContext.Provider>
    );
}

export function useAppTheme() {
    const ctx = useContext(ThemeContext);
    if (!ctx) {
        throw new Error("useAppTheme must be used within AppThemeProvider");
    }
    return ctx;
}
