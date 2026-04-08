/**
 * 外观设置窗口启动器
 *
 * 当 open 变为 true 时，通过 Tauri WebviewWindow API 创建一个 OS 级独立窗口，
 * 加载 appearance.html 入口。该窗口可以自由拖动到主窗口外部。
 *
 * 当独立窗口应用主题后，通过 Tauri 事件 "appearance-applied" 通知主窗口
 * 重新从 localStorage 加载外观设置。
 */

import { useCallback, useEffect, useRef } from "react";
import { useAppTheme } from "../../theme/AppThemeProvider";
import { loadAppearance, loadCustomThemes } from "../../theme/themeStorage";
import type { QtColorToken } from "../../theme/themeTypes";
import { QT_COLOR_TOKENS } from "../../theme/themeTypes";
import type { RadixAccentColor, RadixGrayColor, RadixRadius } from "../../theme/themeTypes";

/* ═══════════════════════════════════════════════════════════
 * Props
 * ═══════════════════════════════════════════════════════════ */

interface AppearanceSettingsDialogProps {
    open: boolean;
    onOpenChange: (open: boolean) => void;
}

export const AppearanceSettingsDialog = ({ open, onOpenChange }: AppearanceSettingsDialogProps) => {
    const theme = useAppTheme();
    const prevOpenRef = useRef(false);
    const windowCreatedRef = useRef(false);

    const applyPreviewFromStorage = useCallback(() => {
        try {
            const settingsRaw = localStorage.getItem("hifishifter.appearance.preview");
            const colorsRaw = localStorage.getItem("hifishifter.appearance.preview.colors");
            if (!settingsRaw && !colorsRaw) return;

            if (settingsRaw) {
                const settings = JSON.parse(settingsRaw) as {
                    mode: "dark" | "light";
                    accentColor: RadixAccentColor;
                    grayColor: RadixGrayColor;
                    radius: RadixRadius;
                    fontFamily: string;
                };
                theme.applySettings({
                    mode: settings.mode,
                    accentColor: settings.accentColor,
                    grayColor: settings.grayColor,
                    radius: settings.radius,
                    fontFamily: settings.fontFamily,
                    activeCustomThemeId: null,
                });
            }

            const root = document.documentElement;
            const previewColors = colorsRaw
                ? (JSON.parse(colorsRaw) as Partial<Record<QtColorToken, string>>)
                : {};
            for (const token of QT_COLOR_TOKENS) {
                const val = previewColors[token];
                if (val) {
                    root.style.setProperty(`--${token}`, val);
                } else {
                    root.style.removeProperty(`--${token}`);
                }
            }
        } catch {
            // ignore malformed preview payload
        }
    }, [theme]);

    /* ── open 变化时创建独立窗口 ── */
    useEffect(() => {
        const justOpened = open && !prevOpenRef.current;
        prevOpenRef.current = open;
        if (!justOpened) return;

        // 避免重复创建
        if (windowCreatedRef.current) return;

        void createAppearanceWindow();
    }, [open]);

    const createAppearanceWindow = useCallback(async () => {
        try {
            const { WebviewWindow } = await import("@tauri-apps/api/webviewWindow");

            // 如果已经存在同 label 的窗口，聚焦它而不是创建新的
            const existing = await WebviewWindow.getByLabel("appearance");
            if (existing) {
                await existing.setFocus();
                onOpenChange(false);
                return;
            }

            windowCreatedRef.current = true;

            const win = new WebviewWindow("appearance", {
                url: "appearance.html",
                title: "Appearance Settings",
                width: 580,
                height: 680,
                center: true,
                resizable: true,
                decorations: true,
                focus: true,
                minWidth: 480,
                minHeight: 400,
            });

            // 窗口关闭时重置状态
            await win.once("tauri://close-requested", () => {
                windowCreatedRef.current = false;
                onOpenChange(false);
            });

            // 也立即重置 open 状态（创建窗口后不再需要保持 open=true）
            onOpenChange(false);
        } catch (err) {
            console.error("Failed to create appearance window:", err);
            windowCreatedRef.current = false;
            onOpenChange(false);
        }
    }, [onOpenChange]);

    /* ── 监听独立窗口发来的 "appearance-applied" 事件 ── */
    useEffect(() => {
        let unlisten: (() => void) | undefined;
        let unlistenPreview: (() => void) | undefined;
        let unlistenReverted: (() => void) | undefined;

        void (async () => {
            try {
                const { listen } = await import("@tauri-apps/api/event");
                unlisten = await listen("appearance-applied", () => {
                    // 从 localStorage 重新加载外观设置并应用到主窗口
                    const settings = loadAppearance();
                    theme.applySettings(settings);

                    // 重新应用自定义颜色 CSS 变量
                    applyCustomColorsFromStorage(settings.activeCustomThemeId);

                    windowCreatedRef.current = false;
                });

                unlistenPreview = await listen<{
                    settings: {
                        mode: "dark" | "light";
                        accentColor: RadixAccentColor;
                        grayColor: RadixGrayColor;
                        radius: RadixRadius;
                        fontFamily: string;
                    };
                    colors: Partial<Record<QtColorToken, string>>;
                }>("appearance-preview", (event) => {
                    const payload = event.payload;
                    theme.applySettings({
                        mode: payload.settings.mode,
                        accentColor: payload.settings.accentColor,
                        grayColor: payload.settings.grayColor,
                        radius: payload.settings.radius,
                        fontFamily: payload.settings.fontFamily,
                        activeCustomThemeId: null,
                    });

                    const root = document.documentElement;
                    for (const token of QT_COLOR_TOKENS) {
                        const val = payload.colors[token];
                        if (val) {
                            root.style.setProperty(`--${token}`, val);
                        } else {
                            root.style.removeProperty(`--${token}`);
                        }
                    }
                });

                unlistenReverted = await listen("appearance-reverted", () => {
                    const settings = loadAppearance();
                    theme.applySettings(settings);
                    applyCustomColorsFromStorage(settings.activeCustomThemeId);
                });
            } catch {
                // Tauri event API 不可用时忽略
            }
        })();

        return () => {
            unlisten?.();
            unlistenPreview?.();
            unlistenReverted?.();
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    useEffect(() => {
        const onStorage = (e: StorageEvent) => {
            if (
                e.key === "hifishifter.appearance.preview" ||
                e.key === "hifishifter.appearance.preview.colors"
            ) {
                applyPreviewFromStorage();
            }
        };
        window.addEventListener("storage", onStorage);
        return () => window.removeEventListener("storage", onStorage);
    }, [applyPreviewFromStorage]);

    // 不渲染任何 DOM，独立窗口由 Tauri 管理
    return null;
};

/** 从 localStorage 重新加载自定义主题颜色并注入 CSS 变量 */
function applyCustomColorsFromStorage(themeId: string | null) {
    const root = document.documentElement;

    // 先清除所有自定义覆盖
    for (const token of QT_COLOR_TOKENS) {
        root.style.removeProperty(`--${token as QtColorToken}`);
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
