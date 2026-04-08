/**
 * 外观设置独立窗口根组件
 *
 * 在 Tauri 独立窗口中运行，拥有自己的 React 树。
 * 复用了 AppearanceSettingsDialog 中的全部业务逻辑：
 * - 主题模式 / 强调色 / 灰阶 / 圆角选择
 * - 自定义颜色编辑（CSS 变量实时预览）
 * - 字体选择 / 系统字体检测
 * - 主题导入导出
 *
 * 与主窗口通过 localStorage（天然共享）+ Tauri 事件通信。
 * 点击「应用」时：保存到 localStorage → 通知主窗口刷新 → 关闭自身。
 * 点击「关闭」时：恢复预览 → 关闭自身。
 */

import React, { useCallback, useEffect, useMemo, useRef, useState, type ChangeEvent } from "react";
import { Tooltip } from "@radix-ui/themes";
import { MagnifyingGlassIcon } from "@radix-ui/react-icons";
import { useI18n } from "../../i18n/I18nProvider";
import { useAppTheme } from "../../theme/AppThemeProvider";
import {
    RADIX_ACCENT_COLORS,
    RADIX_RADIUS_OPTIONS,
    QT_COLOR_TOKENS,
    QT_COLOR_TOKEN_LABELS,
    DEFAULT_FONT_FAMILY,
    type RadixAccentColor,
    type RadixGrayColor,
    type RadixRadius,
    type CustomTheme,
    type QtColorToken,
} from "../../theme/themeTypes";
import { getBuiltinThemeColors } from "../../theme/defaultThemes";
import {
    loadCustomThemes,
    loadAppearance,
    saveCustomThemes,
    exportThemeAsJson,
    importThemeFromJson,
} from "../../theme/themeStorage";

/* ═══════════════════════════════════════════════════════════
 * 常量与数据
 * ═══════════════════════════════════════════════════════════ */

/** Radix 强调色 hex（用于色块显示） */
const RADIX_ACCENT_HEX: Record<RadixAccentColor, string> = {
    gray: "#8b8d98",
    gold: "#978365",
    bronze: "#a18072",
    brown: "#ad7f58",
    yellow: "#ffe16a",
    amber: "#ffc53d",
    orange: "#f76b15",
    tomato: "#e54d2e",
    red: "#e5484d",
    ruby: "#e54666",
    crimson: "#e93d82",
    pink: "#d6409f",
    plum: "#ab4aba",
    purple: "#8e4ec6",
    violet: "#6e56cf",
    iris: "#5b5bd6",
    indigo: "#3e63dd",
    blue: "#0090ff",
    cyan: "#00a2c7",
    teal: "#12a594",
    jade: "#29a383",
    green: "#30a46c",
    grass: "#46a758",
    lime: "#bdee63",
    mint: "#86ead4",
    sky: "#7ce2fe",
};

/** 强调色 → 推荐灰阶自动映射 */
const ACCENT_TO_GRAY: Partial<Record<RadixAccentColor, RadixGrayColor>> = {
    crimson: "mauve",
    pink: "mauve",
    plum: "mauve",
    purple: "mauve",
    violet: "mauve",
    iris: "mauve",
    ruby: "mauve",
    indigo: "slate",
    blue: "slate",
    sky: "slate",
    cyan: "slate",
    teal: "sage",
    jade: "sage",
    mint: "sage",
    green: "sage",
    grass: "olive",
    lime: "olive",
    gold: "sand",
    bronze: "sand",
    brown: "sand",
    orange: "sand",
    amber: "sand",
    yellow: "sand",
    tomato: "mauve",
    red: "mauve",
};

function getAutoGray(accent: RadixAccentColor): RadixGrayColor {
    return ACCENT_TO_GRAY[accent] ?? "auto";
}

/* Tab 类型 */
type SettingsTab = "theme" | "font";

const PALETTE_GROUPS: Array<{ labelKey: string; tokens: QtColorToken[] }> = [
    {
        labelKey: "appearance_color_group_base",
        tokens: ["qt-window", "qt-base", "qt-panel", "qt-surface"],
    },
    { labelKey: "appearance_color_group_text", tokens: ["qt-text", "qt-text-muted"] },
    {
        labelKey: "appearance_color_group_ui",
        tokens: [
            "qt-highlight",
            "qt-playhead",
            "qt-border",
            "qt-clip-bg",
            "qt-clip-border",
            "qt-clip-selected-border",
        ],
    },
];

const CARD_CLASS = "rounded-md border border-qt-border bg-qt-panel";
const SECTION_LABEL_CLASS = "text-[11px] font-semibold text-qt-text";
const SECONDARY_BUTTON_CLASS =
    "px-3 py-1.5 text-[11px] font-medium rounded border border-qt-border bg-qt-surface text-qt-text-muted hover:bg-qt-hover hover:text-qt-text transition-colors cursor-pointer select-none";
const PRIMARY_BUTTON_CLASS =
    "px-4 py-1.5 text-[11px] font-semibold rounded bg-qt-highlight text-white hover:brightness-110 transition-colors cursor-pointer select-none";
const PREVIEW_SETTINGS_KEY = "hifishifter.appearance.preview";
const PREVIEW_COLORS_KEY = "hifishifter.appearance.preview.colors";

const COMMON_SYSTEM_FONT_CANDIDATES = [
    "Segoe UI",
    "Segoe UI Variable",
    "Arial",
    "Arial Nova",
    "Verdana",
    "Tahoma",
    "Trebuchet MS",
    "Calibri",
    "Cambria",
    "Corbel",
    "Candara",
    "Constantia",
    "Consolas",
    "Courier New",
    "Georgia",
    "Times New Roman",
    "Palatino Linotype",
    "Impact",
    "Franklin Gothic Medium",
    "Bahnschrift",
    "Yu Gothic UI",
    "Yu Gothic",
    "Meiryo",
    "MS Gothic",
    "MS UI Gothic",
    "Microsoft YaHei",
    "Microsoft JhengHei",
    "Malgun Gothic",
    "SimSun",
    "SimHei",
    "KaiTi",
    "PingFang SC",
    "PingFang TC",
    "PingFang HK",
    "Hiragino Sans",
    "Hiragino Kaku Gothic ProN",
    "Noto Sans",
    "Noto Sans CJK SC",
    "Noto Sans CJK TC",
    "Noto Sans CJK JP",
    "Noto Sans CJK KR",
    "Noto Serif",
    "Roboto",
    "Roboto Flex",
    "Roboto Condensed",
    "Open Sans",
    "Lato",
    "Inter",
    "Ubuntu",
    "Ubuntu Sans",
    "Cantarell",
    "Fira Sans",
    "Fira Code",
    "JetBrains Mono",
    "Source Sans 3",
    "Source Han Sans SC",
    "Source Han Sans TC",
    "Source Han Sans JP",
    "Source Han Sans KR",
    "Helvetica",
    "Helvetica Neue",
    "SF Pro Text",
    "SF Pro Display",
    "Avenir",
    "Avenir Next",
    "Menlo",
    "Monaco",
    "Geneva",
    "Marker Felt",
    "Optima",
    "Apple SD Gothic Neo",
    "Apple Color Emoji",
].sort((a, b) => a.localeCompare(b, undefined, { sensitivity: "base" }));

function uniqueSortedStrings(values: Iterable<string>): string[] {
    return [...new Set([...values].map((value) => value.trim()).filter(Boolean))].sort((a, b) =>
        a.localeCompare(b, undefined, { sensitivity: "base" }),
    );
}

function normalizeFontName(value: string): string {
    return value
        .trim()
        .replace(/^['"]|['"]$/g, "")
        .trim();
}

function extractFontFamilies(value: string): string[] {
    return uniqueSortedStrings(
        value
            .split(",")
            .map((part) => normalizeFontName(part))
            .filter(
                (part) =>
                    part &&
                    !["sans-serif", "serif", "monospace", "system-ui"].includes(part.toLowerCase()),
            ),
    );
}

function detectInstalledFontsFromCandidates(candidates: string[]): string[] {
    if (typeof document === "undefined") return [];

    const canvas = document.createElement("canvas");
    const context = canvas.getContext("2d");
    if (!context) return [];

    const sample = "mmmmmmmmmmlliWWWW1234567890AaBbCc中한あ";
    const baseFamilies = ["monospace", "sans-serif", "serif"] as const;
    const baselines = new Map<string, number>();

    for (const base of baseFamilies) {
        context.font = `72px ${base}`;
        baselines.set(base, context.measureText(sample).width);
    }

    return candidates.filter((candidate) => {
        const family = normalizeFontName(candidate);
        if (!family) return false;
        return baseFamilies.some((base) => {
            context.font = `72px "${family}", ${base}`;
            return context.measureText(sample).width !== baselines.get(base);
        });
    });
}

/* ═══════════════════════════════════════════════════════════
 * 系统字体检测 Hook
 * ═══════════════════════════════════════════════════════════ */

interface FontInfo {
    family: string;
    fullName: string;
    postscriptName: string;
    style: string;
}

function useSystemFonts() {
    const [fonts, setFonts] = useState<string[]>([]);
    const [loading, setLoading] = useState(false);
    const [supported, setSupported] = useState(true);
    const [source, setSource] = useState<"native" | "fallback" | null>(null);
    const loadedRef = useRef(false);

    const detect = useCallback(async (force = false) => {
        if (loadedRef.current && !force) {
            return;
        }

        setLoading(true);
        setSupported(true);
        try {
            if ("queryLocalFonts" in window) {
                const fontData: FontInfo[] = await (
                    window as unknown as {
                        queryLocalFonts: () => Promise<FontInfo[]>;
                    }
                ).queryLocalFonts();
                const families = uniqueSortedStrings(fontData.map((f) => f.family));
                if (families.length > 0) {
                    setFonts(families);
                    setSource("native");
                    loadedRef.current = true;
                    return;
                }
            }

            const fallbackFamilies = detectInstalledFontsFromCandidates(
                COMMON_SYSTEM_FONT_CANDIDATES,
            );
            setFonts(fallbackFamilies);
            setSource("fallback");
            setSupported(fallbackFamilies.length > 0);
            loadedRef.current = true;
        } catch {
            const fallbackFamilies = detectInstalledFontsFromCandidates(
                COMMON_SYSTEM_FONT_CANDIDATES,
            );
            setFonts(fallbackFamilies);
            setSource("fallback");
            setSupported(fallbackFamilies.length > 0);
            loadedRef.current = true;
        } finally {
            setLoading(false);
        }
    }, []);

    return { fonts, loading, supported, source, detect };
}

/* ═══════════════════════════════════════════════════════════
 * 小组件
 * ═══════════════════════════════════════════════════════════ */

/** 顶部 Segmented Control */
const SegmentedControl: React.FC<{
    tabs: { id: string; label: string }[];
    active: string;
    onChange: (id: string) => void;
}> = ({ tabs, active, onChange }) => (
    <div className="flex gap-1 rounded border border-qt-border bg-qt-panel p-1">
        {tabs.map((tab) => {
            const isActive = active === tab.id;
            return (
                <button
                    key={tab.id}
                    className={
                        "flex-1 px-3 py-1.5 text-xs font-semibold rounded transition-colors duration-150 " +
                        "cursor-pointer select-none " +
                        (isActive
                            ? "bg-qt-highlight text-white"
                            : "text-qt-text-muted hover:bg-qt-hover hover:text-qt-text")
                    }
                    onClick={() => onChange(tab.id)}
                >
                    {tab.label}
                </button>
            );
        })}
    </div>
);

/** 单个颜色 token 行 */
const ColorTokenRow: React.FC<{
    label: string;
    color: string;
    onChange: (value: string) => void;
    disableNativePicker?: boolean;
}> = ({ label, color, onChange, disableNativePicker = false }) => {
    const validHex = /^#[0-9a-fA-F]{6}$/.test(color);
    const validHexAlpha = /^#[0-9a-fA-F]{8}$/.test(color);
    const hasPicker = (validHex || validHexAlpha) && !disableNativePicker;
    const previewColor = validHex || validHexAlpha ? color : "#000000";
    return (
        <div
            className={
                "flex items-center gap-2 px-2 py-2 rounded border border-qt-border bg-qt-panel group"
            }
        >
            <label
                className={`relative shrink-0 ${hasPicker ? "cursor-pointer" : "cursor-default"}`}
            >
                <div
                    className={`w-5 h-5 rounded ${hasPicker ? "transition-transform duration-100 group-hover:scale-110" : ""}`}
                    style={{
                        backgroundColor: previewColor,
                        boxShadow: "inset 0 0 0 1px rgba(255,255,255,0.1)",
                    }}
                />
                {hasPicker && (
                    <input
                        type="color"
                        value={validHex ? color : `#${color.slice(1, 7)}`}
                        onInput={(e) => onChange((e.target as HTMLInputElement).value)}
                        onChange={(e) => onChange(e.target.value)}
                        className="absolute inset-0 opacity-0 cursor-pointer w-full h-full"
                    />
                )}
            </label>
            <span className="text-[10px] text-qt-text flex-1 truncate">{label}</span>
            <input
                type="text"
                value={color}
                onChange={(e) => onChange(e.target.value)}
                className="w-[92px] px-2 py-1 text-[10px] bg-qt-panel text-qt-text-muted font-mono text-right rounded border border-qt-border focus:text-qt-text focus:outline-none focus:ring-1 focus:ring-qt-highlight/30 transition-all"
                spellCheck={false}
            />
        </div>
    );
};

/* ═══════════════════════════════════════════════════════════
 * 关闭窗口辅助
 * ═══════════════════════════════════════════════════════════ */

async function closeThisWindow() {
    try {
        const mod = await import("@tauri-apps/api/window");
        const win = mod.getCurrentWindow();
        await win.close();
    } catch {
        // 如果 Tauri API 不可用（如开发模式下直接在浏览器打开），尝试关闭标签
        window.close();
    }
}

/** 通知主窗口刷新主题 */
async function emitThemeApplied() {
    try {
        const { emit } = await import("@tauri-apps/api/event");
        await emit("appearance-applied");
    } catch {
        // fallback: 不做任何事，主窗口重新 focus 时会从 localStorage 重新读取
    }
}

async function emitThemePreview(payload: {
    settings: {
        mode: "dark" | "light";
        accentColor: RadixAccentColor;
        grayColor: RadixGrayColor;
        radius: RadixRadius;
        fontFamily: string;
    };
    colors: Partial<Record<QtColorToken, string>>;
}) {
    try {
        const { emit } = await import("@tauri-apps/api/event");
        await emit("appearance-preview", payload);
    } catch {
        // ignore when Tauri API unavailable
    }
}

async function emitThemeReverted() {
    try {
        const { emit } = await import("@tauri-apps/api/event");
        await emit("appearance-reverted");
    } catch {
        // ignore when Tauri API unavailable
    }
}

/* ═══════════════════════════════════════════════════════════
 * 主组件
 * ═══════════════════════════════════════════════════════════ */

export const AppearanceWindow: React.FC = () => {
    const { t } = useI18n();
    const tAny = t as (key: string) => string;
    const theme = useAppTheme();

    /* ── Tab ── */
    const [activeTab, setActiveTab] = useState<SettingsTab>("theme");

    /* ── 本地编辑状态 ── */
    const [accentColor, setAccentColor] = useState<RadixAccentColor>(theme.accentColor);
    const [grayColor, setGrayColor] = useState<RadixGrayColor>(theme.grayColor);
    const [radius, setRadius] = useState<RadixRadius>(theme.radius);
    const [fontFamily, setFontFamily] = useState(theme.fontFamily);

    const [customThemes, setCustomThemes] = useState<CustomTheme[]>([]);
    const [activeThemeId, setActiveThemeId] = useState<string | null>(theme.activeCustomThemeId);
    const [editColors, setEditColors] = useState<Partial<Record<QtColorToken, string>>>({});
    const [editWaveform, setEditWaveform] = useState<{ fill: string; stroke: string } | undefined>(
        undefined,
    );
    const [editThemeName, setEditThemeName] = useState("");
    const [activePaletteGroup, setActivePaletteGroup] = useState(PALETTE_GROUPS[0].labelKey);

    const [fontSearch, setFontSearch] = useState("");
    const systemFonts = useSystemFonts();
    const fileInputRef = useRef<HTMLInputElement>(null);

    /* ── 初始化 ── */
    useEffect(() => {
        const themes = loadCustomThemes();
        setCustomThemes(themes);

        const active = themes.find((ct) => ct.id === theme.activeCustomThemeId);
        if (active) {
            setEditColors(active.colors);
            setEditWaveform(active.waveformColors);
            setEditThemeName(active.name);
        }

        systemFonts.detect();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    useEffect(() => {
        const syncFromStorage = () => {
            const latest = loadAppearance();
            theme.applySettings(latest);
            setAccentColor(latest.accentColor);
            setGrayColor(latest.grayColor);
            setRadius(latest.radius);
            setFontFamily(latest.fontFamily);
            setActiveThemeId(latest.activeCustomThemeId);

            const themes = loadCustomThemes();
            setCustomThemes(themes);
            const active = themes.find((ct) => ct.id === latest.activeCustomThemeId);
            if (active) {
                setEditColors(active.colors);
                setEditWaveform(active.waveformColors);
                setEditThemeName(active.name);
            } else {
                setEditColors({});
                setEditWaveform(undefined);
                setEditThemeName("");
            }
        };

        const onWindowFocus = () => syncFromStorage();
        const onVisibilityChange = () => {
            if (document.visibilityState === "visible") syncFromStorage();
        };
        const onStorage = (e: StorageEvent) => {
            if (!e.key || e.key.startsWith("hifishifter.")) syncFromStorage();
        };

        window.addEventListener("focus", onWindowFocus);
        document.addEventListener("visibilitychange", onVisibilityChange);
        window.addEventListener("storage", onStorage);

        let unlistenPromise: Promise<(() => void) | undefined> | null = null;
        try {
            unlistenPromise = import("@tauri-apps/api/event")
                .then((mod) => mod.listen("appearance-applied", syncFromStorage))
                .catch(() => undefined);
        } catch {
            unlistenPromise = null;
        }

        return () => {
            window.removeEventListener("focus", onWindowFocus);
            document.removeEventListener("visibilitychange", onVisibilityChange);
            window.removeEventListener("storage", onStorage);
            if (unlistenPromise) {
                void unlistenPromise.then((unlisten) => unlisten?.());
            }
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    /* ── 强调色 → 灰阶自动映射 ── */
    useEffect(() => {
        setGrayColor(getAutoGray(accentColor));
    }, [accentColor]);

    /* ── 内置颜色 ── */
    const builtinColors = useMemo(() => getBuiltinThemeColors(theme.mode), [theme.mode]);

    const getDisplayColor = useCallback(
        (token: QtColorToken) => editColors[token] ?? builtinColors[token] ?? "#000000",
        [editColors, builtinColors],
    );

    /* ── 实时预览（Radix 属性） ── */
    useEffect(() => {
        theme.setAccentColor(accentColor);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [accentColor]);

    useEffect(() => {
        theme.setGrayColor(grayColor);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [grayColor]);

    useEffect(() => {
        theme.setRadius(radius);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [radius]);

    useEffect(() => {
        theme.setFontFamily(fontFamily);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [fontFamily]);

    /* ── 实时预览（自定义颜色 CSS 变量） ── */
    useEffect(() => {
        const root = document.documentElement;
        for (const token of QT_COLOR_TOKENS) {
            const val = editColors[token];
            if (val) {
                root.style.setProperty(`--${token}`, val);
            } else {
                root.style.removeProperty(`--${token}`);
            }
        }
    }, [editColors, theme.mode]);

    useEffect(() => {
        localStorage.setItem(
            PREVIEW_SETTINGS_KEY,
            JSON.stringify({
                mode: theme.mode,
                accentColor,
                grayColor,
                radius,
                fontFamily,
            }),
        );
        localStorage.setItem(PREVIEW_COLORS_KEY, JSON.stringify(editColors));
        void emitThemePreview({
            settings: {
                mode: theme.mode,
                accentColor,
                grayColor,
                radius,
                fontFamily,
            },
            colors: editColors,
        });
    }, [theme.mode, accentColor, grayColor, radius, fontFamily, editColors]);

    /* ── 应用 & 关闭 ── */
    const handleApply = useCallback(() => {
        const hasCustom = Object.keys(editColors).length > 0 || editWaveform;
        let themeId: string | null = null;

        if (hasCustom) {
            const existing = customThemes.find((ct) => ct.id === activeThemeId);
            const id = existing?.id ?? crypto.randomUUID();
            const newTheme: CustomTheme = {
                id,
                name: editThemeName || tAny("appearance_custom_theme"),
                base: theme.mode,
                colors: editColors,
                waveformColors: editWaveform,
                accentColor,
                grayColor,
                radius,
            };
            const updated = existing
                ? customThemes.map((ct) => (ct.id === id ? newTheme : ct))
                : [...customThemes, newTheme];
            setCustomThemes(updated);
            saveCustomThemes(updated);
            themeId = id;
            setActiveThemeId(id);
        } else {
            setActiveThemeId(null);
            themeId = null;
        }

        theme.applySettings({
            mode: theme.mode,
            accentColor,
            grayColor,
            radius,
            fontFamily,
            activeCustomThemeId: themeId,
        });
        localStorage.removeItem(PREVIEW_SETTINGS_KEY);
        localStorage.removeItem(PREVIEW_COLORS_KEY);

        // 通知主窗口刷新 → 关闭自身
        void emitThemeApplied().then(() => closeThisWindow());
    }, [
        accentColor,
        grayColor,
        radius,
        fontFamily,
        editColors,
        editWaveform,
        editThemeName,
        activeThemeId,
        customThemes,
        theme,
        tAny,
    ]);

    const handleClose = useCallback(() => {
        theme.revertPreview();
        localStorage.removeItem(PREVIEW_SETTINGS_KEY);
        localStorage.removeItem(PREVIEW_COLORS_KEY);
        void emitThemeReverted().then(() => closeThisWindow());
    }, [theme]);

    const paletteTokens = useMemo(() => PALETTE_GROUPS.flatMap((group) => group.tokens), []);
    const paletteTokenSet = useMemo(() => new Set<QtColorToken>(paletteTokens), [paletteTokens]);

    /* ── 颜色操作 ── */
    const handleResetColors = useCallback(() => {
        setEditColors((prev) => {
            const next = { ...prev };
            for (const token of paletteTokens) {
                delete next[token];
            }
            return next;
        });
        setEditWaveform(undefined);
        setEditThemeName("");
        setActiveThemeId(null);
    }, [paletteTokens]);

    const handleSelectTheme = useCallback((item: CustomTheme) => {
        setActiveThemeId(item.id);
        setEditColors(item.colors);
        setEditWaveform(item.waveformColors);
        setEditThemeName(item.name);
        if (item.accentColor) setAccentColor(item.accentColor);
        if (item.grayColor) setGrayColor(item.grayColor);
        if (item.radius) setRadius(item.radius);
    }, []);

    const handleDeleteTheme = useCallback(
        (id: string) => {
            const updated = customThemes.filter((ct) => ct.id !== id);
            setCustomThemes(updated);
            saveCustomThemes(updated);
            if (activeThemeId === id) {
                setActiveThemeId(null);
                setEditColors({});
                setEditWaveform(undefined);
                setEditThemeName("");
            }
        },
        [customThemes, activeThemeId],
    );

    const handleExportTheme = useCallback(() => {
        const themeData: CustomTheme = {
            id: activeThemeId ?? crypto.randomUUID(),
            name: editThemeName || `${theme.mode === "dark" ? "Dark" : "Light"} Theme`,
            base: theme.mode,
            colors: editColors,
            waveformColors: editWaveform,
            accentColor,
            grayColor,
            radius,
        };
        const json = exportThemeAsJson(themeData, { accentColor, grayColor, radius });
        const blob = new Blob([json], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `${themeData.name.replace(/\s+/g, "_")}.json`;
        a.click();
        URL.revokeObjectURL(url);
    }, [
        activeThemeId,
        editThemeName,
        theme.mode,
        editColors,
        editWaveform,
        accentColor,
        grayColor,
        radius,
    ]);

    const handleImportTheme = useCallback(
        (e: React.ChangeEvent<HTMLInputElement>) => {
            const file = e.target.files?.[0];
            if (!file) return;
            const reader = new FileReader();
            reader.onload = () => {
                const result = importThemeFromJson(reader.result as string);
                if (result) {
                    const updated = [...customThemes, result.theme];
                    setCustomThemes(updated);
                    saveCustomThemes(updated);
                    setActiveThemeId(result.theme.id);
                    setEditColors(result.theme.colors);
                    setEditWaveform(result.theme.waveformColors);
                    setEditThemeName(result.theme.name);
                    if (result.accentColor) setAccentColor(result.accentColor);
                    if (result.grayColor) setGrayColor(result.grayColor);
                    if (result.radius) setRadius(result.radius);
                }
            };
            reader.readAsText(file);
            e.target.value = "";
        },
        [customThemes],
    );

    const updateColorToken = useCallback((token: QtColorToken, value: string) => {
        setEditColors((prev) => ({ ...prev, [token]: value }));
    }, []);

    const modifiedColorCount = useMemo(
        () =>
            Object.keys(editColors).filter((token) => paletteTokenSet.has(token as QtColorToken))
                .length,
        [editColors, paletteTokenSet],
    );
    const hasCustomColors = modifiedColorCount > 0;

    /* ── 字体过滤 ── */
    const availableFonts = useMemo(
        () =>
            uniqueSortedStrings([
                ...systemFonts.fonts,
                ...extractFontFamilies(fontFamily),
                ...extractFontFamilies(DEFAULT_FONT_FAMILY),
            ]),
        [fontFamily, systemFonts.fonts],
    );

    const filteredFonts = useMemo(() => {
        if (!fontSearch) return availableFonts;
        const q = fontSearch.toLowerCase();
        return availableFonts.filter((f) => f.toLowerCase().includes(q));
    }, [availableFonts, fontSearch]);

    const tabItems = useMemo(
        () => [
            { id: "theme", label: tAny("appearance_tab_theme") },
            { id: "font", label: tAny("appearance_tab_font") },
        ],
        [tAny],
    );

    /* ═══════════════════════════════════════════════════════════
     * 渲染 — 直接作为窗口内容，不需要 portal
     * ═══════════════════════════════════════════════════════════ */
    return (
        <div className="flex h-screen flex-col overflow-hidden select-none bg-qt-window">
            {/* ═══════ 头部：标题栏 + Tab 切换 ═══════ */}
            <div
                className={`mx-auto w-full max-w-[920px] ${CARD_CLASS} space-y-2 px-4 py-3 shrink-0`}
                data-tauri-drag-region
            >
                <div className="flex items-start justify-between gap-3">
                    <div className="space-y-1">
                        <h2 className="m-0 text-base font-semibold text-qt-text">
                            {tAny("appearance_title")}
                        </h2>
                    </div>
                    <div className="flex items-center gap-2 pt-1">
                        {modifiedColorCount > 0 && (
                            <span className="text-[10px] text-qt-highlight bg-qt-highlight/10 px-2 py-0.5 rounded font-semibold">
                                {tAny("appearance_modified_count").replace(
                                    "{count}",
                                    String(modifiedColorCount),
                                )}
                            </span>
                        )}
                    </div>
                </div>
                <SegmentedControl
                    tabs={tabItems}
                    active={activeTab}
                    onChange={(id) => setActiveTab(id as SettingsTab)}
                />
            </div>

            {/* ═══════ 内容区 ═══════ */}
            <div className="mx-auto mt-1 flex-1 w-full max-w-[920px] overflow-y-auto custom-scrollbar">
                <div className="pb-1">
                    {/* ═══════ Tab: 主题 ═══════ */}
                    {activeTab === "theme" && (
                        <div className="space-y-2">
                            {/* ── 工具栏 ── */}
                            <div
                                className={`${CARD_CLASS} flex items-center gap-2 flex-wrap px-3 py-2`}
                            >
                                <button
                                    className={SECONDARY_BUTTON_CLASS}
                                    onClick={() => fileInputRef.current?.click()}
                                >
                                    {tAny("appearance_import_theme")}
                                </button>
                                <input
                                    ref={fileInputRef}
                                    type="file"
                                    accept=".json"
                                    className="hidden"
                                    onChange={handleImportTheme}
                                />
                                <button
                                    className={SECONDARY_BUTTON_CLASS}
                                    onClick={handleExportTheme}
                                >
                                    {tAny("appearance_export_theme")}
                                </button>
                                {hasCustomColors && (
                                    <>
                                        <div className="flex-1" />
                                        <button
                                            className="px-3 py-1.5 text-[11px] font-medium rounded border border-qt-danger-border/40 bg-qt-danger-bg/20 text-qt-danger-text hover:bg-qt-danger-bg/35 transition-colors cursor-pointer select-none"
                                            onClick={handleResetColors}
                                        >
                                            {tAny("appearance_reset_all_colors")}
                                        </button>
                                    </>
                                )}
                            </div>

                            {/* ── 已保存主题 ── */}
                            {customThemes.length > 0 && (
                                <div className={`${CARD_CLASS} p-3 space-y-2`}>
                                    <span className={SECTION_LABEL_CLASS}>
                                        {tAny("appearance_saved_themes")}
                                    </span>
                                    <div className="flex flex-wrap gap-1.5">
                                        {customThemes.map((ct) => {
                                            const isActive = activeThemeId === ct.id;
                                            return (
                                                <div
                                                    key={ct.id}
                                                    className={
                                                        "inline-flex items-center gap-1 px-2 py-1 text-[10px] rounded cursor-pointer " +
                                                        "transition-all duration-100 select-none " +
                                                        (isActive
                                                            ? "bg-qt-highlight/20 text-qt-highlight font-semibold"
                                                            : "bg-qt-surface text-qt-text hover:bg-qt-hover")
                                                    }
                                                    onClick={() => handleSelectTheme(ct)}
                                                >
                                                    <span>{ct.name}</span>
                                                    <button
                                                        className="text-[9px] opacity-30 hover:opacity-100 hover:text-qt-danger-text transition-opacity cursor-pointer"
                                                        onClick={(e) => {
                                                            e.stopPropagation();
                                                            handleDeleteTheme(ct.id);
                                                        }}
                                                    >
                                                        ×
                                                    </button>
                                                </div>
                                            );
                                        })}
                                    </div>

                                    {/* ── 主题名称（内嵌在已保存主题卡片内） ── */}
                                    {hasCustomColors && (
                                        <div className="flex items-center gap-2 pt-2 border-t border-[color:var(--qt-divider)]">
                                            <span className="text-[11px] text-qt-text-muted shrink-0">
                                                {tAny("appearance_theme_name")}
                                            </span>
                                            <input
                                                type="text"
                                                value={editThemeName}
                                                onChange={(e) => setEditThemeName(e.target.value)}
                                                className="flex-1 rounded border border-qt-border bg-qt-base px-2 py-1.5 text-[11px] text-qt-text focus:outline-none focus:ring-1 focus:ring-qt-highlight/30 transition-all"
                                                placeholder={tAny("appearance_custom_theme")}
                                            />
                                        </div>
                                    )}
                                </div>
                            )}

                            {/* ── 主题模式 ── */}
                            <div className={`${CARD_CLASS} p-3 space-y-2`}>
                                <span className={SECTION_LABEL_CLASS}>
                                    {tAny("appearance_mode")}
                                </span>
                                <div className="grid grid-cols-2 gap-2">
                                    {(["dark", "light"] as const).map((mode) => {
                                        const isSelected = theme.mode === mode;
                                        const isDarkMode = mode === "dark";
                                        return (
                                            <button
                                                key={mode}
                                                className={
                                                    "flex flex-col items-center gap-2 p-2.5 rounded border transition-all duration-150 " +
                                                    "cursor-pointer select-none active:scale-[0.98] " +
                                                    (isSelected
                                                        ? "border-qt-highlight bg-qt-highlight/12"
                                                        : "border-qt-border bg-qt-panel hover:bg-qt-hover")
                                                }
                                                onClick={() => theme.setMode(mode)}
                                            >
                                                <div
                                                    className="w-full h-10 rounded-lg overflow-hidden relative"
                                                    style={{
                                                        backgroundColor: isDarkMode
                                                            ? "#2d2d2d"
                                                            : "#f0f0f0",
                                                    }}
                                                >
                                                    <div
                                                        className="absolute inset-x-0 top-0 h-3"
                                                        style={{
                                                            backgroundColor: isDarkMode
                                                                ? "#353535"
                                                                : "#fff",
                                                        }}
                                                    />
                                                    <div className="absolute bottom-1 left-1.5 right-1.5 flex gap-0.5">
                                                        <div
                                                            className="h-1.5 flex-1 rounded-sm"
                                                            style={{
                                                                backgroundColor: isDarkMode
                                                                    ? "rgba(59,130,246,0.4)"
                                                                    : "rgba(91,91,214,0.3)",
                                                            }}
                                                        />
                                                        <div
                                                            className="h-1.5 flex-1 rounded-sm"
                                                            style={{
                                                                backgroundColor: isDarkMode
                                                                    ? "#404040"
                                                                    : "#d9d9e0",
                                                            }}
                                                        />
                                                    </div>
                                                </div>
                                                <span
                                                    className={`text-[10px] font-medium ${isSelected ? "text-qt-highlight" : "text-qt-text-muted"}`}
                                                >
                                                    {tAny(
                                                        isDarkMode ? "theme_dark" : "theme_light",
                                                    )}
                                                </span>
                                            </button>
                                        );
                                    })}
                                </div>
                            </div>

                            {/* ── 强调色 ── */}
                            <div className={`${CARD_CLASS} p-3 space-y-2`}>
                                <div className="flex items-center justify-between">
                                    <span className={SECTION_LABEL_CLASS}>
                                        {tAny("appearance_accent")}
                                    </span>
                                    <span className="text-[9px] text-qt-text-muted/50 font-mono">
                                        {accentColor} {RADIX_ACCENT_HEX[accentColor]}
                                    </span>
                                </div>
                                <div className="flex flex-wrap gap-1">
                                    {RADIX_ACCENT_COLORS.map((c) => {
                                        const isSelected = accentColor === c;
                                        return (
                                            <Tooltip key={c} content={c}>
                                                <button
                                                    className={
                                                        "w-6 h-6 rounded-md transition-all duration-100 cursor-pointer relative " +
                                                        "active:scale-90 " +
                                                        (isSelected
                                                            ? "ring-2 ring-offset-1 ring-qt-text scale-105"
                                                            : "hover:scale-110 ring-1 ring-transparent hover:ring-white/20")
                                                    }
                                                    style={{
                                                        backgroundColor: RADIX_ACCENT_HEX[c],
                                                        ["--tw-ring-offset-color" as string]:
                                                            "var(--qt-panel)",
                                                    }}
                                                    onClick={() => {
                                                        setAccentColor(c);
                                                    }}
                                                >
                                                    {isSelected && (
                                                        <span className="absolute inset-0 flex items-center justify-center text-white text-[10px] font-bold drop-shadow-sm">
                                                            ✓
                                                        </span>
                                                    )}
                                                </button>
                                            </Tooltip>
                                        );
                                    })}
                                </div>
                            </div>

                            {/* ── 圆角 ── */}
                            <div className={`${CARD_CLASS} p-3 space-y-2`}>
                                <span className={SECTION_LABEL_CLASS}>
                                    {tAny("appearance_radius")}
                                </span>
                                <div className="flex gap-1.5">
                                    {RADIX_RADIUS_OPTIONS.map((r) => {
                                        const isSelected = radius === r;
                                        const px: Record<string, string> = {
                                            none: "0",
                                            small: "3px",
                                            medium: "6px",
                                            large: "10px",
                                            full: "9999px",
                                        };
                                        return (
                                            <button
                                                key={r}
                                                className={
                                                    "flex-1 flex flex-col items-center gap-1 py-2 rounded " +
                                                    "transition-all duration-100 cursor-pointer select-none active:scale-[0.97] " +
                                                    (isSelected
                                                        ? "bg-qt-highlight/12"
                                                        : "bg-qt-base hover:bg-qt-hover")
                                                }
                                                onClick={() => setRadius(r)}
                                            >
                                                <div
                                                    className="w-7 h-5 border-2 transition-colors duration-100"
                                                    style={{
                                                        borderRadius: px[r],
                                                        borderColor: isSelected
                                                            ? "var(--qt-highlight)"
                                                            : "var(--qt-text-muted)",
                                                        opacity: isSelected ? 0.8 : 0.25,
                                                    }}
                                                />
                                                <span className="sr-only">{r}</span>
                                            </button>
                                        );
                                    })}
                                </div>
                            </div>

                            {/* ── 颜色编辑（单套色卡） ── */}
                            <div className={`${CARD_CLASS} space-y-3 p-3`}>
                                <div className="flex items-center justify-between">
                                    <span className={SECTION_LABEL_CLASS}>
                                        {tAny("appearance_tab_colors")}
                                    </span>
                                    <span className="text-[10px] text-qt-text-muted">
                                        {tAny("appearance_modified_count").replace(
                                            "{count}",
                                            String(modifiedColorCount),
                                        )}
                                    </span>
                                </div>

                                <div className="flex items-center gap-1 p-1 rounded border border-qt-border bg-qt-panel">
                                    {PALETTE_GROUPS.map((group) => {
                                        const active = activePaletteGroup === group.labelKey;
                                        return (
                                            <button
                                                key={group.labelKey}
                                                className={
                                                    "px-2.5 py-1 text-[10px] rounded transition-colors " +
                                                    (active
                                                        ? "bg-qt-highlight text-white"
                                                        : "text-qt-text-muted hover:bg-qt-hover hover:text-qt-text")
                                                }
                                                onClick={() =>
                                                    setActivePaletteGroup(group.labelKey)
                                                }
                                            >
                                                {tAny(group.labelKey)}
                                            </button>
                                        );
                                    })}
                                </div>

                                <div className="space-y-1">
                                    {(
                                        PALETTE_GROUPS.find(
                                            (group) => group.labelKey === activePaletteGroup,
                                        )?.tokens ?? PALETTE_GROUPS[0].tokens
                                    ).map((token) => (
                                        <ColorTokenRow
                                            key={token}
                                            label={tAny(QT_COLOR_TOKEN_LABELS[token])}
                                            color={getDisplayColor(token)}
                                            onChange={(v) => updateColorToken(token, v)}
                                        />
                                    ))}
                                </div>
                            </div>
                        </div>
                    )}

                    {/* ═══════ Tab: 字体 ═══════ */}
                    {activeTab === "font" && (
                        <div className="space-y-2">
                            {/* 字体输入 */}
                            <div className={`${CARD_CLASS} p-3 space-y-2`}>
                                <span className={SECTION_LABEL_CLASS}>
                                    {tAny("appearance_font")}
                                </span>
                                <div className="flex items-center gap-2">
                                    <input
                                        type="text"
                                        value={fontFamily}
                                        onChange={(e: ChangeEvent<HTMLInputElement>) =>
                                            setFontFamily(e.target.value)
                                        }
                                        className="flex-1 rounded-xl border border-[color:var(--qt-divider)] bg-qt-surface/40 px-3 py-2 text-[11px] text-qt-text font-mono focus:outline-none focus:ring-1 focus:ring-qt-highlight/30 transition-all"
                                        placeholder={DEFAULT_FONT_FAMILY}
                                        spellCheck={false}
                                    />
                                    <button
                                        className={SECONDARY_BUTTON_CLASS}
                                        onClick={() => setFontFamily(DEFAULT_FONT_FAMILY)}
                                    >
                                        {tAny("appearance_reset")}
                                    </button>
                                    <button
                                        className={SECONDARY_BUTTON_CLASS}
                                        onClick={() => {
                                            setFontFamily(DEFAULT_FONT_FAMILY);
                                            setFontSearch("");
                                        }}
                                    >
                                        {tAny("appearance_font_restore_default")}
                                    </button>
                                </div>

                                {/* 字体预览 */}
                                <div
                                    className="rounded border border-qt-border bg-qt-base px-3 py-3 text-qt-text"
                                    style={{ fontFamily }}
                                >
                                    <div className="text-sm mb-1.5 leading-relaxed">
                                        The quick brown fox jumps over the lazy dog.
                                    </div>
                                    <div className="text-sm mb-1.5 leading-relaxed">
                                        中文字体预览：你好世界 1234567890
                                    </div>
                                    <div className="text-[10px] text-qt-text-muted">
                                        ABCDEFG abcdefg !@#$%^&*()
                                    </div>
                                </div>
                            </div>

                            {/* 系统字体列表 */}
                            <div className={`${CARD_CLASS} p-3 space-y-2`}>
                                <div className="flex items-center justify-between gap-3">
                                    <div className="flex items-center gap-2">
                                        <span className={SECTION_LABEL_CLASS}>
                                            {tAny("appearance_font_system")}
                                        </span>
                                        {availableFonts.length > 0 && (
                                            <span className="rounded-full bg-qt-surface/60 px-2 py-0.5 text-[10px] text-qt-text-muted">
                                                {tAny("appearance_font_count").replace(
                                                    "{count}",
                                                    String(availableFonts.length),
                                                )}
                                            </span>
                                        )}
                                    </div>
                                    <button
                                        className={SECONDARY_BUTTON_CLASS}
                                        onClick={() => void systemFonts.detect(true)}
                                    >
                                        {systemFonts.loading
                                            ? tAny("appearance_font_detecting")
                                            : tAny("appearance_font_detect")}
                                    </button>
                                </div>

                                {/* 加载中 */}
                                {availableFonts.length === 0 && (
                                    <div className="flex items-center gap-2 rounded-xl border border-dashed border-[color:var(--qt-divider)] bg-qt-base/35 px-3 py-3 text-[11px] text-qt-text-muted">
                                        {systemFonts.loading ? (
                                            <>
                                                <span className="animate-spin inline-block w-3 h-3 border-2 border-qt-text-muted/20 border-t-qt-highlight rounded-full" />
                                                {tAny("appearance_font_detecting")}
                                            </>
                                        ) : systemFonts.supported ? (
                                            tAny("appearance_font_detect")
                                        ) : (
                                            tAny("appearance_font_not_supported")
                                        )}
                                    </div>
                                )}

                                {/* 搜索 + 列表 */}
                                {availableFonts.length > 0 && (
                                    <>
                                        <div className="relative">
                                            <MagnifyingGlassIcon className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3 h-3 text-qt-text-muted/40" />
                                            <input
                                                type="text"
                                                value={fontSearch}
                                                onChange={(e: ChangeEvent<HTMLInputElement>) =>
                                                    setFontSearch(e.target.value)
                                                }
                                                className="w-full rounded-xl border border-[color:var(--qt-divider)] bg-qt-surface/40 py-2 pl-8 pr-8 text-[11px] text-qt-text focus:outline-none focus:ring-1 focus:ring-qt-highlight/30 transition-all"
                                                placeholder={tAny(
                                                    "appearance_font_search_placeholder",
                                                )}
                                                spellCheck={false}
                                            />
                                            {fontSearch && (
                                                <button
                                                    className="absolute right-2 top-1/2 -translate-y-1/2 text-[9px] text-qt-text-muted hover:text-qt-text cursor-pointer"
                                                    onClick={() => setFontSearch("")}
                                                >
                                                    ×
                                                </button>
                                            )}
                                        </div>

                                        <div className="max-h-[320px] overflow-y-auto rounded border border-qt-border bg-qt-base p-1 custom-scrollbar">
                                            {filteredFonts.length > 0 ? (
                                                filteredFonts.map((f) => {
                                                    const isActive = extractFontFamilies(
                                                        fontFamily,
                                                    ).includes(normalizeFontName(f));
                                                    return (
                                                        <button
                                                            key={f}
                                                            className={
                                                                "w-full rounded-xl border px-3 py-2 text-left transition-colors duration-100 " +
                                                                "cursor-pointer select-none text-[11px] flex items-center gap-3 " +
                                                                (isActive
                                                                    ? "border-qt-highlight/25 bg-qt-highlight/12 text-qt-highlight"
                                                                    : "border-transparent text-qt-text hover:border-[color:var(--qt-divider)] hover:bg-qt-surface/40")
                                                            }
                                                            onClick={() => {
                                                                setFontFamily(f);
                                                                setFontSearch("");
                                                            }}
                                                        >
                                                            <span
                                                                className={`shrink-0 w-[150px] truncate ${isActive ? "font-semibold" : ""}`}
                                                            >
                                                                {f}
                                                            </span>
                                                            <span
                                                                className={`flex-1 truncate ${isActive ? "text-qt-highlight/60" : "text-qt-text-muted/60"}`}
                                                                style={{ fontFamily: f }}
                                                            >
                                                                AaBbCc 你好 123
                                                            </span>
                                                            {isActive && (
                                                                <span className="text-qt-highlight text-[10px] shrink-0 font-bold">
                                                                    ✓
                                                                </span>
                                                            )}
                                                        </button>
                                                    );
                                                })
                                            ) : (
                                                <div className="px-3 py-4 text-[10px] text-qt-text-muted/40 italic text-center">
                                                    {tAny("appearance_font_no_results")}
                                                </div>
                                            )}
                                        </div>
                                    </>
                                )}
                            </div>
                        </div>
                    )}
                </div>
            </div>

            {/* ═══════ 底部按钮 ═══════ */}
            <div className="mx-auto mt-1 flex w-full max-w-[920px] items-center justify-end gap-2 border-t border-qt-border px-1 pt-1.5 shrink-0">
                <button className={SECONDARY_BUTTON_CLASS} onClick={handleClose}>
                    {tAny("close")}
                </button>
                <button className={PRIMARY_BUTTON_CLASS} onClick={handleApply}>
                    {tAny("appearance_apply")}
                </button>
            </div>
        </div>
    );
};
