/**
 * 外观设置面板。
 *
 * 【为什么从独立 OS 窗口改成停靠面板】它此前是一个单独的 Tauri 窗口
 * （`appearance.html` + 独立 React 根），于是完全落在 UI 系统之外：自带 15px 标题、
 * 自造按钮类（11px 字、`bg-qt-highlight`），而对话框标题是 20px、按钮是 32px
 * `--qt-accent`。它是全仓库**唯一**还有自造按钮类的文件。
 *
 * 现在它复用停靠机制，并声明为：
 * - `openAsFloating: { anchor: "center" }` —— 默认在主窗口**正中**浮出
 *   （不是右下角：它不是"瞥一眼"的辅助面板，而是接下来一段时间的主焦点）；
 * - `dockable: false` —— 拖得动，但停不进去。设置界面被编入工作布局占一格
 *   既无意义又会污染用户排好的布局；
 * - `excludeFromWindowMenu: true` —— 不进「视图 → 窗口」。那个菜单列的是日常
 *   切换的工作面板，低频设置入口混进去只会稀释常用项。
 *
 * 【为什么不再需要 Tauri 事件】主窗口与面板现在是**同一个 React 树**，
 * `AppThemeProvider` 是同一个实例：预览直接改它，应用直接落盘。原先那套
 * `appearance-preview / -applied / -reverted` 事件是为跨窗口通信存在的，
 * 跨窗口消失了，事件也就没有存在理由。
 */

import React, { useCallback, useEffect, useMemo, useRef, useState, type ChangeEvent } from "react";
import { MagnifyingGlassIcon } from "@radix-ui/react-icons";
import { AppFileInput } from "../../ui/FileInput";
import { useAppDispatch, useAppSelector } from "../../app/hooks";
import type { RootState } from "../../app/store";
import { closeForm } from "../../features/dock/dockSlice";
import { broadcastAppearanceToSatellites } from "../../features/dock/detachBridge";
import { useI18n } from "../../i18n/I18nProvider";
import type { MessageKey } from "../../i18n/messages";
import { useAppTheme } from "../../theme/AppThemeProvider";
import type { AppearanceSettings } from "../../theme/themeTypes";
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
    type ThemeModeSetting,
} from "../../theme/themeTypes";
import { getBuiltinThemeColors } from "../../theme/defaultThemes";
import {
    AppButton,
    AppConfirmDialog,
    AppField,
    AppFormSection,
    AppNoticeDialog,
    AppSegmentedControl,
    AppStatusChip,
} from "../../ui";
import { effectiveSearchMode } from "../../features/search/searchSettings";
import { useTranslitIndex } from "../../features/search/useTranslitIndex";
import { buildQuery, fallbackTranslit, matchTranslit } from "../../features/search/translit";
import { exportThemeJson } from "../../services/api/jsonExport";
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

/** 稳定的空数组引用：避免每次渲染都触发转写索引的重建判定。 */
const NO_FONT_TEXTS: readonly string[] = [];

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

/**
 * 圆角档位的可见文字。此前磁贴只有图形、标签是 `sr-only` —— 用户看到的是
 * 五个无字磁贴，只能靠猜。键名与 `RadixRadius` 一一对应，错位是编译错误。
 */
const RADIUS_LABEL_KEYS: Record<RadixRadius, MessageKey> = {
    none: "appearance_radius_none",
    small: "appearance_radius_small",
    medium: "appearance_radius_medium",
    large: "appearance_radius_large",
    full: "appearance_radius_full",
};

/** 主题模式卡片的迷你预览配色：auto = 深浅各半（示意跟随系统）。 */
const MODE_PREVIEW: Record<
    ThemeModeSetting,
    { bg: string; top: string; bar1: string; bar2: string }
> = {
    auto: {
        bg: "linear-gradient(90deg, #2d2d2d 50%, #f0f0f0 50%)",
        top: "linear-gradient(90deg, #353535 50%, #ffffff 50%)",
        bar1: "rgba(59,130,246,0.4)",
        bar2: "#404040",
    },
    dark: {
        bg: "#2d2d2d",
        top: "#353535",
        bar1: "rgba(59,130,246,0.4)",
        bar2: "#404040",
    },
    light: {
        bg: "#f0f0f0",
        top: "#ffffff",
        bar1: "rgba(91,91,214,0.3)",
        bar2: "#d9d9e0",
    },
};
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

/**
 * 由主题名得到"可安全落盘的文件主名"。
 *
 * 【为什么要过滤】主题名是用户随便起的，可能含路径分隔符（`/`、`\`）或 Windows
 * 保留字符（`:` `*` `?` `"` `<` `>` `|`）。旧实现只把空白换成下划线，于是叫
 * 「a/b」的主题会让原生保存对话框把 `b.json` 当成另一个目录下的文件。
 * 空名/全非法字符时回退到一个固定名（不返回空串，否则后缀会变成纯 `.json`）。
 */
function exportFileBaseName(themeName: string): string {
    const cleaned = themeName
        .replace(/[\\/:*?"<>|]/g, "_")
        .replace(/\s+/g, "_")
        .trim();
    return cleaned.length > 0 ? cleaned : "hifishifter-theme";
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

/** 单个颜色 token 行：标签列对齐 `AppField` 的表单网格，值区 = 色块 + hex 输入。 */
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
        <AppField label={label}>
            <div className="flex items-center justify-end gap-2">
                <label
                    className={`relative shrink-0 ${hasPicker ? "cursor-pointer" : "cursor-default"}`}
                >
                    <div
                        className="w-5 h-5 rounded"
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
                <input
                    type="text"
                    value={color}
                    onChange={(e) => onChange(e.target.value)}
                    className="w-[92px] px-2 py-1 text-qt-micro bg-qt-base text-qt-text-muted font-mono text-right rounded border border-qt-border focus:text-qt-text focus:outline-none focus:ring-1 focus:ring-qt-highlight/30 transition-all"
                    spellCheck={false}
                />
            </div>
        </AppField>
    );
};

/* ═══════════════════════════════════════════════════════════
 * 主组件
 * ═══════════════════════════════════════════════════════════ */

export interface AppearanceSettingsPanelProps {
    /** 本面板所在的停靠窗体 id —— 关闭时用它从布局里移除自己。 */
    formId: string;
}

export const AppearanceSettingsPanel: React.FC<AppearanceSettingsPanelProps> = ({ formId }) => {
    const { t, tf, plural } = useI18n();
    const theme = useAppTheme();
    // 解构出稳定引用供下面的实时预览 effect 使用：effect 只依赖具体成员、
    // 不依赖整个 context 对象（对象身份随 provider 任意更新翻转，见该 effect 处说明）。
    const applyThemeSettings = theme.applySettings;
    const themeModeSetting = theme.modeSetting;
    const dispatch = useAppDispatch();
    /*
     * 字体列表的过滤要读搜索设置（转写开关与宽严）。设置本身在**独立的
     * 「搜索与匹配设置」对话框**里改 —— 它不是外观的一部分。
     */
    const searchSettings = useAppSelector((state: RootState) => state.session.searchSettings);

    /**
     * 关闭本面板。
     *
     * 【为什么不是 `window.close()`】它不再是独立 OS 窗口，而是停靠系统里的一个
     * 浮窗：关闭 = 从布局里移除该窗体，由停靠内核收尾（浮窗层卸载、布局落盘）。
     */
    const onRequestClose = useCallback(() => {
        dispatch(closeForm(formId));
    }, [dispatch, formId]);

    /**
     * 打开面板那一刻的外观。关闭**未应用的草稿**时回滚到它。
     *
     * 【为什么自己存一份，而不用 `theme.revertPreview()`】provider 的快照语义是
     * "上一次 `applySettings` 之前的状态"，而**预览本身就调 `applySettings`** ——
     * 于是第一次预览之后 provider 的快照已经变成预览值，`revertPreview()` 会
     * "回退到预览"，等于没回退。面板自己记下打开前的样子，语义才准确。
     *
     * 【为什么必须有回滚】预览会落盘（`applySettings` 会 `saveAppearance`）。
     * 于是用户点浮动窗的 X 关掉面板时，如果他只是随手试了试颜色，那些颜色会
     * **变成已保存的设置** —— 用户以为取消了，实际生效了。这是必须堵住的。
     */
    const openedWithRef = useRef<AppearanceSettings | null>(null);
    const themeRef = useRef(theme);
    useEffect(() => {
        themeRef.current = theme;
    }, [theme]);

    useEffect(() => {
        // 只在挂载时抓一次：这是"打开面板前的样子"。
        const current = themeRef.current;
        openedWithRef.current = {
            mode: current.modeSetting,
            accentColor: current.accentColor,
            grayColor: current.grayColor,
            radius: current.radius,
            fontFamily: current.fontFamily,
            activeCustomThemeId: current.activeCustomThemeId,
        };
        return () => {
            // 卸载 = 关闭（浮动窗的 X、菜单开关、布局重置都会走到这里）。
            // 草稿未应用就回滚，否则预览会被当成用户的选择留下来。
            if (!draftDirtyRef.current) return;
            draftDirtyRef.current = false;
            const opened = openedWithRef.current;
            if (opened) themeRef.current.applySettings(opened);
            localStorage.removeItem(PREVIEW_SETTINGS_KEY);
            localStorage.removeItem(PREVIEW_COLORS_KEY);
            void broadcastAppearanceToSatellites();
        };
    }, []);

    /* ── Tab ── */
    const [activeTab, setActiveTab] = useState<SettingsTab>("theme");
    /** 「重置全部颜色」确认框：会丢弃当前所有自定义色覆盖，先确认再执行。 */
    const [resetColorsConfirmOpen, setResetColorsConfirmOpen] = useState(false);
    /**
     * 导入 / 导出失败的通知。
     *
     * 【为什么必须有】这个面板此前**没有任何提示通道**：导入坏文件静默、导出被
     * WebView 拦下也静默，用户只知道"不起作用"。提示走 `AppNoticeDialog`（与
     * 全应用其它通知同一壳），不弹浏览器原生 alert。
     */
    const [notice, setNotice] = useState("");

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

    /**
     * 未保存草稿标记：任何用户编辑（调色板/强调色/圆角/字体/主题名/主题
     * 选择/导入）置位，focus / storage 同步据此跳过对编辑态的覆盖。
     * 应用（handleApply）会落盘并关闭窗口，无需复位。
     */
    const draftDirtyRef = useRef(false);
    const markDraftDirty = useCallback(() => {
        draftDirtyRef.current = true;
    }, []);

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
            // 有未保存的草稿（用户正在编辑颜色/主题名/字体等，尚未点“应用”）
            // 时，focus / storage / appearance-applied 同步一律跳过：否则在
            // 编辑过程中点一下主窗口再切回来（或收到外部应用事件），全部
            // 草稿会被 localStorage 旧值静默覆盖。
            if (draftDirtyRef.current) return;
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
            unlistenPromise = import("../../services/hostEvents")
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

    /* ── 强调色 → 灰阶自动映射 ──
     * 注意：映射只应在“用户点击强调色块”时执行一次（见强调色点击处理器）。
     * 不能用 effect 监听 accentColor —— 任何来源的 accentColor 变化
     * （focus 同步、主题选择带出的 grayColor）都会被自动映射覆盖，用户
     * 手动选择的不同灰阶会被静默改掉。 */

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

    /*
     * 实时预览：直接作用到**共享的** `AppThemeProvider`。
     *
     * 此前这里把设置写进 localStorage 再 `emit("appearance-preview")`，由主窗口
     * 收到事件后应用 —— 那是跨窗口通信的必需品。现在主窗口与面板在同一个 React 树里，
     * 直接调用即可：少一次序列化、少一次 IPC，也少一个"事件没送到就不同步"的失败面。
     *
     * localStorage 仍然写：`draftDirtyRef` 的同步逻辑与"预览在重载后仍生效"
     * 都依赖它（见 `applyPreviewFromStorage` 的语义）。
     */
    useEffect(() => {
        /*
         * 用户没编辑过就什么都不做。
         *
         * 这个 effect 的每一次执行都会**落盘**（`applySettings` → `saveAppearance`）。
         * 它此前在挂载的第一次 pass 就无条件跑一遍：`activeCustomThemeId` 写死为
         * `null`，于是"只是打开面板看一眼"就会把正在启用的自定义主题停用并落盘、
         * 同时清掉 `<html>` 上所有 `--qt-*` 覆盖。此时若用户不点「应用」直接关
         * （浮动窗 X / 布局重置），卸载清理又因 `draftDirtyRef` 仍为 false 而跳过
         * 回滚 —— 停用就永久留下来了。
         *
         * 所有用户编辑都在改 state 之前先 `markDraftDirty()`，因此以它为闸门：
         * 首次挂载（干净）被跳过，任何真实编辑照旧预览 + 落盘。
         */
        if (!draftDirtyRef.current) return;
        localStorage.setItem(
            PREVIEW_SETTINGS_KEY,
            JSON.stringify({
                mode: themeModeSetting,
                accentColor,
                grayColor,
                radius,
                fontFamily,
            }),
        );
        localStorage.setItem(PREVIEW_COLORS_KEY, JSON.stringify(editColors));
        applyThemeSettings({
            mode: themeModeSetting,
            accentColor,
            grayColor,
            radius,
            fontFamily,
            activeCustomThemeId: null,
        });
        // 依赖只列具体成员、不含 `theme` 对象：context 对象身份随 provider 的任何
        // 更新翻转（如 auto 模式下的系统深浅色切换会重建 toggleMode），把它放进
        // 依赖会让本 effect 在用户未触碰面板时重跑，把**正激活的自定义主题静默
        // 写回为已停用**（applySettings 会持久化）。`theme.applySettings` 是
        // useCallback([]) 的稳定引用，列进来只为满足 lint，不引入额外触发。
    }, [
        applyThemeSettings,
        themeModeSetting,
        accentColor,
        grayColor,
        radius,
        fontFamily,
        editColors,
    ]);

    /* ── 应用 & 关闭 ── */
    const handleApply = useCallback(() => {
        const hasCustom = Object.keys(editColors).length > 0 || editWaveform;
        let themeId: string | null = null;

        if (hasCustom) {
            const existing = customThemes.find((ct) => ct.id === activeThemeId);
            const id = existing?.id ?? crypto.randomUUID();
            const newTheme: CustomTheme = {
                id,
                name: editThemeName || tf("appearance_custom_theme"),
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
            mode: theme.modeSetting,
            accentColor,
            grayColor,
            radius,
            fontFamily,
            activeCustomThemeId: themeId,
        });
        localStorage.removeItem(PREVIEW_SETTINGS_KEY);
        localStorage.removeItem(PREVIEW_COLORS_KEY);
        // 草稿已应用：卸载时的回滚逻辑据此跳过。
        draftDirtyRef.current = false;

        /*
         * 拆到独立窗口的面板不共享主题（另一个 JS 上下文），必须显式通知。
         * 此前这条链是"外观窗口发事件 → 主窗口的桥收到 → 转发给卫星"；现在外观设置
         * 就在主窗口里，直接调用即可。
         */
        void broadcastAppearanceToSatellites();

        // 主题已经是共享实例上的最新值（`applySettings` 刚刚落盘），直接关面板。
        onRequestClose();
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
        tf,
        onRequestClose,
    ]);

    const handleClose = useCallback(() => {
        /*
         * 直接调 `revertPreview()` 在这里**不够**（见 `openedWithRef` 的说明：
         * provider 的快照会被预览覆盖）。改为显式回到打开前的设置。
         */
        const opened = openedWithRef.current;
        if (opened) theme.applySettings(opened);
        localStorage.removeItem(PREVIEW_SETTINGS_KEY);
        localStorage.removeItem(PREVIEW_COLORS_KEY);
        // 清掉草稿标记：卸载时的回滚逻辑据此跳过（这里已经回滚过了）。
        draftDirtyRef.current = false;
        // 回滚也改变了外观 —— 卫星窗口同样要跟上。
        void broadcastAppearanceToSatellites();
        onRequestClose();
    }, [theme, onRequestClose]);

    const paletteTokens = useMemo(() => PALETTE_GROUPS.flatMap((group) => group.tokens), []);
    const paletteTokenSet = useMemo(() => new Set<QtColorToken>(paletteTokens), [paletteTokens]);

    /* ── 颜色操作 ── */
    const handleResetColors = useCallback(() => {
        markDraftDirty();
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
    }, [paletteTokens, markDraftDirty]);

    const handleSelectTheme = useCallback(
        (item: CustomTheme) => {
            markDraftDirty();
            setActiveThemeId(item.id);
            setEditColors(item.colors);
            setEditWaveform(item.waveformColors);
            setEditThemeName(item.name);
            if (item.accentColor) setAccentColor(item.accentColor);
            if (item.grayColor) setGrayColor(item.grayColor);
            if (item.radius) setRadius(item.radius);
        },
        [markDraftDirty],
    );

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

    /**
     * 导出当前外观为主题 JSON。
     *
     * 【为什么不是浏览器下载】此前这里是 `Blob` + 游离 `<a download>` —— 而 wry
     * （Tauri 的 WebView）默认**拦截页面发起的下载**，于是在应用壳里点「导出」什么
     * 都不会发生（也没有任何报错）。现在与布局导出共用后端命令：原生保存对话框 +
     * 后端写文件（见 `services/api/jsonExport.ts`）。
     */
    const handleExportTheme = useCallback(async () => {
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
        setNotice("");
        try {
            const result = await exportThemeJson(
                json,
                `${exportFileBaseName(themeData.name)}.json`,
            );
            // 取消不是错误；只有真正失败才提示。
            if (!result.ok && !result.canceled) {
                setNotice(result.error || tf("appearance_export_failed"));
            }
        } catch {
            setNotice(tf("appearance_export_failed"));
        }
    }, [
        activeThemeId,
        editThemeName,
        theme.mode,
        editColors,
        editWaveform,
        accentColor,
        grayColor,
        radius,
        tf,
    ]);

    /**
     * 导入主题 JSON。
     *
     * 【为什么要提示失败】`importThemeFromJson` 对坏文件返回 `null`，旧实现在那种
     * 情况下**什么都不做**（静默）—— 用户点完「导入...」选了个文件，界面毫无反应，
     * 于是"导入不起作用"。失败必须说出来。
     */
    const handleImportTheme = useCallback(
        async (files: File[]) => {
            const file = files[0];
            if (!file) return;
            setNotice("");
            let text: string;
            try {
                text = await file.text();
            } catch {
                setNotice(tf("appearance_import_failed"));
                return;
            }
            const result = importThemeFromJson(text);
            if (!result) {
                setNotice(tf("appearance_import_failed"));
                return;
            }
            markDraftDirty();
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
        },
        [customThemes, markDraftDirty, tf],
    );

    const updateColorToken = useCallback(
        (token: QtColorToken, value: string) => {
            markDraftDirty();
            setEditColors((prev) => ({ ...prev, [token]: value }));
        },
        [markDraftDirty],
    );

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

    /*
     * 字体过滤也走同一套匹配：中文字体名（「微软雅黑」）能被打成 `yahei` 搜到，
     * 而拉丁字体名的行为与改动前一致（字面匹配）。
     *
     * 转写索引只在**字体页签可见时**才取：字体列表可能有几百项，没打开这一页
     * 就发起一次 IPC 是白费。
     */
    const fontTranslitTexts = useMemo(
        () => (activeTab === "font" ? availableFonts : NO_FONT_TEXTS),
        [activeTab, availableFonts],
    );
    const fontTranslitIndex = useTranslitIndex(fontTranslitTexts, searchSettings);

    const filteredFonts = useMemo(() => {
        if (!fontSearch) return availableFonts;
        const query = buildQuery(fontSearch);
        const mode = effectiveSearchMode(searchSettings);
        return availableFonts.filter((font) => {
            const forms = fontTranslitIndex.get(font) ?? fallbackTranslit(font);
            return matchTranslit(forms, query, mode) !== null;
        });
    }, [availableFonts, fontSearch, fontTranslitIndex, searchSettings]);

    const tabItems = useMemo(
        () => [
            { id: "theme", label: tf("appearance_tab_theme") },
            { id: "font", label: tf("appearance_tab_font") },
        ],
        [tf],
    );

    /* ═══════════════════════════════════════════════════════════
     * 渲染
     * ═══════════════════════════════════════════════════════════ */
    return (
        /*
         * `h-full` 而不是 `h-screen`：面板高度由停靠宿主决定。`h-screen` 在浮窗里
         * 会撑到整屏高，超出浮窗自身的框。
         *
         * 不再渲染自带标题栏：浮动窗已经有一条标题栏（`.hs-dock-float-title`），
         * 再画一条会出现**两个标题**。标题由注册表的 `titleKey` 提供。
         *
         * 【为什么没有卡片】此前内容是六张同色圆角卡片的堆叠（实测间距 8px、
         * 页边距 4px），卡中卡让层级消失。分组交给 `AppFormSection` 的留白 +
         * 节标题 —— 与本应用其它设置表单同一套语言。
         */
        <div className="flex h-full flex-col overflow-hidden">
            {/* ═══════ 页眉行：Tab 切换 + 修改计数（唯一一处，不再在颜色节重复） ═══════ */}
            <div className="flex shrink-0 items-center justify-between gap-3 border-b border-qt-border px-3 py-2">
                <AppSegmentedControl
                    size="md"
                    value={activeTab}
                    options={tabItems.map((tab) => ({ value: tab.id, label: tab.label }))}
                    onChange={(id) => setActiveTab(id as SettingsTab)}
                    ariaLabel={tf("appearance_title")}
                />
                {modifiedColorCount > 0 ? (
                    <AppStatusChip tone="accent">
                        {plural("appearance_modified_count", modifiedColorCount)}
                    </AppStatusChip>
                ) : null}
            </div>

            {/* ═══════ 内容区 ═══════ */}
            <div className="hs-scroll-gutter-flush min-h-0 flex-1 overflow-y-auto custom-scrollbar px-3">
                <div className="pb-3">
                    {/* ═══════ Tab: 主题 ═══════ */}
                    {activeTab === "theme" && (
                        <>
                            {/* ── 已保存主题 ──
                                节**常显**：导入/导出在节头动作区，不能随"还没有
                                主题"一起消失 —— 否则第一次导入无处可点。 */}
                            <AppFormSection
                                title={tf("appearance_saved_themes")}
                                action={
                                    <div className="flex items-center gap-2">
                                        <AppButton
                                            size="sm"
                                            onClick={() => fileInputRef.current?.click()}
                                        >
                                            {tf("appearance_import_theme")}
                                        </AppButton>
                                        <AppFileInput
                                            inputRef={fileInputRef}
                                            accept=".json"
                                            onFiles={handleImportTheme}
                                        />
                                        <AppButton size="sm" onClick={handleExportTheme}>
                                            {tf("appearance_export_theme")}
                                        </AppButton>
                                    </div>
                                }
                            >
                                {customThemes.length > 0 ? (
                                    <div className="flex flex-wrap gap-1.5">
                                        {customThemes.map((ct) => {
                                            const isActive = activeThemeId === ct.id;
                                            return (
                                                <div
                                                    key={ct.id}
                                                    className={
                                                        "inline-flex items-center gap-1 px-2 py-1 text-qt-micro rounded cursor-pointer " +
                                                        "transition-all duration-100 select-none " +
                                                        (isActive
                                                            ? "bg-qt-highlight/20 text-qt-highlight font-semibold"
                                                            : "bg-qt-surface text-qt-text hover:bg-qt-hover")
                                                    }
                                                    onClick={() => handleSelectTheme(ct)}
                                                >
                                                    <span>{ct.name}</span>
                                                    <button
                                                        className="text-qt-3xs opacity-30 hover:opacity-100 hover:text-qt-danger-text transition-opacity cursor-pointer"
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
                                ) : (
                                    <p className="hs-type-caption m-0">
                                        {tf("appearance_saved_themes_empty")}
                                    </p>
                                )}

                                {/* ── 主题名称：有自定义颜色才需要命名 ── */}
                                {hasCustomColors && (
                                    <AppField label={tf("appearance_theme_name")}>
                                        <input
                                            type="text"
                                            value={editThemeName}
                                            onChange={(e) => {
                                                markDraftDirty();
                                                setEditThemeName(e.target.value);
                                            }}
                                            className="w-full rounded border border-qt-border bg-qt-base px-2 py-1.5 text-qt-xs text-qt-text transition-colors focus:outline-none focus:ring-1 focus:ring-qt-highlight/30"
                                            placeholder={tf("appearance_custom_theme")}
                                        />
                                    </AppField>
                                )}
                            </AppFormSection>

                            {/* ── 主题模式 ── */}
                            <AppFormSection title={tf("appearance_mode")}>
                                <div className="grid grid-cols-3 gap-2">
                                    {(["auto", "dark", "light"] as const).map((mode) => {
                                        const isSelected = theme.modeSetting === mode;
                                        const preview = MODE_PREVIEW[mode];
                                        return (
                                            <button
                                                key={mode}
                                                className={
                                                    "flex flex-col items-center gap-2 p-2.5 rounded border transition-colors duration-150 " +
                                                    "cursor-pointer select-none " +
                                                    (isSelected
                                                        ? "border-qt-highlight bg-qt-highlight/12"
                                                        : "border-qt-border bg-qt-base hover:bg-qt-hover")
                                                }
                                                onClick={() => {
                                                    // 模式改动也是草稿：标记后上面的预览 effect
                                                    // 才会落盘，卸载清理也才会回滚。
                                                    markDraftDirty();
                                                    theme.setMode(mode);
                                                }}
                                            >
                                                <div
                                                    className="w-full h-10 rounded-lg overflow-hidden relative"
                                                    style={{ backgroundColor: preview.bg }}
                                                >
                                                    <div
                                                        className="absolute inset-x-0 top-0 h-3"
                                                        style={{ backgroundColor: preview.top }}
                                                    />
                                                    <div className="absolute bottom-1 left-1.5 right-1.5 flex gap-0.5">
                                                        <div
                                                            className="h-1.5 flex-1 rounded-sm"
                                                            style={{
                                                                backgroundColor: preview.bar1,
                                                            }}
                                                        />
                                                        <div
                                                            className="h-1.5 flex-1 rounded-sm"
                                                            style={{
                                                                backgroundColor: preview.bar2,
                                                            }}
                                                        />
                                                    </div>
                                                </div>
                                                <span
                                                    className={`text-qt-micro font-medium ${isSelected ? "text-qt-highlight" : "text-qt-text-muted"}`}
                                                >
                                                    {tf(`theme_${mode}`)}
                                                </span>
                                            </button>
                                        );
                                    })}
                                </div>
                            </AppFormSection>

                            {/* ── 强调色 ── */}
                            <AppFormSection
                                title={tf("appearance_accent")}
                                action={
                                    <span className="hs-type-mono shrink-0 text-qt-text-muted">
                                        {accentColor} {RADIX_ACCENT_HEX[accentColor]}
                                    </span>
                                }
                            >
                                <div className="flex flex-wrap gap-1">
                                    {RADIX_ACCENT_COLORS.map((c) => {
                                        const isSelected = accentColor === c;
                                        return (
                                            <button
                                                key={c}
                                                data-tooltip={c}
                                                className={
                                                    "w-6 h-6 rounded-md transition-all duration-100 cursor-pointer relative " +
                                                    (isSelected
                                                        ? "ring-2 ring-offset-1 ring-qt-text"
                                                        : "ring-1 ring-transparent hover:ring-white/20")
                                                }
                                                style={{
                                                    backgroundColor: RADIX_ACCENT_HEX[c],
                                                    ["--tw-ring-offset-color" as string]:
                                                        "var(--qt-panel)",
                                                }}
                                                onClick={() => {
                                                    markDraftDirty();
                                                    setAccentColor(c);
                                                    // 强调色 → 灰阶自动映射只在用户
                                                    // 主动点选强调色时执行一次（主题
                                                    // 自带/存储带入的 grayColor 不被覆盖）。
                                                    setGrayColor(getAutoGray(c));
                                                }}
                                            >
                                                {isSelected && (
                                                    <span className="absolute inset-0 flex items-center justify-center text-white text-qt-micro font-bold drop-shadow-sm">
                                                        ✓
                                                    </span>
                                                )}
                                            </button>
                                        );
                                    })}
                                </div>
                            </AppFormSection>

                            {/* ── 圆角 ── */}
                            <AppFormSection title={tf("appearance_radius")}>
                                <div className="flex gap-1.5">
                                    {RADIX_RADIUS_OPTIONS.map((r) => {
                                        const isSelected = radius === r;
                                        /*
                                         * 磁贴里的样本圆角是**示意**值，刻意不是真实像素：
                                         * 真实档位（small 3 / medium 4 / large 6px）在
                                         * 28×20 的方块上看不出差别，五个选项就会变成"五个
                                         * 一样的方块"。这里按比例夸张到 0/3/6/10/9999，让
                                         * "更圆"这件事可见；实际生效的圆角由 `--qt-radius-*`
                                         * 从同一档位推导（见 src/index.css）。
                                         */
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
                                                    "flex flex-1 flex-col items-center gap-1 py-2 rounded " +
                                                    "transition-colors duration-100 cursor-pointer select-none " +
                                                    (isSelected
                                                        ? "bg-qt-highlight/12"
                                                        : "bg-qt-base hover:bg-qt-hover")
                                                }
                                                onClick={() => {
                                                    markDraftDirty();
                                                    setRadius(r);
                                                }}
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
                                                <span
                                                    className={
                                                        isSelected
                                                            ? "text-qt-micro text-qt-highlight"
                                                            : "text-qt-micro text-qt-text-muted"
                                                    }
                                                >
                                                    {t(RADIUS_LABEL_KEYS[r])}
                                                </span>
                                            </button>
                                        );
                                    })}
                                </div>
                            </AppFormSection>

                            {/* ── 颜色编辑（单套色卡） ── */}
                            <AppFormSection
                                title={tf("appearance_tab_colors")}
                                action={
                                    hasCustomColors ? (
                                        <AppButton
                                            size="sm"
                                            intent="danger"
                                            onClick={() => setResetColorsConfirmOpen(true)}
                                        >
                                            {tf("appearance_reset_all_colors")}
                                        </AppButton>
                                    ) : undefined
                                }
                            >
                                <AppSegmentedControl
                                    size="md"
                                    className="w-full"
                                    value={activePaletteGroup}
                                    options={PALETTE_GROUPS.map((group) => ({
                                        value: group.labelKey,
                                        label: tf(group.labelKey),
                                    }))}
                                    onChange={(id) => setActivePaletteGroup(id)}
                                    ariaLabel={tf("appearance_tab_colors")}
                                />

                                <div className="flex flex-col gap-1">
                                    {(
                                        PALETTE_GROUPS.find(
                                            (group) => group.labelKey === activePaletteGroup,
                                        )?.tokens ?? PALETTE_GROUPS[0].tokens
                                    ).map((token) => (
                                        <ColorTokenRow
                                            key={token}
                                            label={tf(QT_COLOR_TOKEN_LABELS[token])}
                                            color={getDisplayColor(token)}
                                            onChange={(v) => updateColorToken(token, v)}
                                        />
                                    ))}
                                </div>
                            </AppFormSection>
                        </>
                    )}

                    {/* ═══════ Tab: 字体 ═══════ */}
                    {activeTab === "font" && (
                        <>
                            <AppFormSection
                                title={tf("appearance_font")}
                                action={
                                    <div className="flex items-center gap-2">
                                        <AppButton
                                            size="sm"
                                            onClick={() => {
                                                markDraftDirty();
                                                setFontFamily(DEFAULT_FONT_FAMILY);
                                            }}
                                        >
                                            {tf("appearance_reset")}
                                        </AppButton>
                                        <AppButton
                                            size="sm"
                                            onClick={() => {
                                                markDraftDirty();
                                                setFontFamily(DEFAULT_FONT_FAMILY);
                                                setFontSearch("");
                                            }}
                                        >
                                            {tf("appearance_font_restore_default")}
                                        </AppButton>
                                    </div>
                                }
                            >
                                <input
                                    type="text"
                                    value={fontFamily}
                                    onChange={(e: ChangeEvent<HTMLInputElement>) => {
                                        markDraftDirty();
                                        setFontFamily(e.target.value);
                                    }}
                                    className="w-full rounded-md border border-[color:var(--qt-divider)] bg-qt-surface/40 px-3 py-2 text-qt-xs text-qt-text font-mono focus:outline-none focus:ring-1 focus:ring-qt-highlight/30 transition-all"
                                    placeholder={DEFAULT_FONT_FAMILY}
                                    spellCheck={false}
                                />

                                {/* 字体预览 */}
                                <div
                                    className="rounded border border-qt-border bg-qt-base px-3 py-3 text-qt-text"
                                    style={{ fontFamily }}
                                >
                                    <div className="hs-type-body mb-1.5">
                                        The quick brown fox jumps over the lazy dog.
                                    </div>
                                    <div className="hs-type-body mb-1.5">
                                        {/* hs-text-exempt: 字体预览样本必须含中文字形，否则预览不出中文字体 */}
                                        中文字体预览：你好世界 1234567890
                                    </div>
                                    <div className="hs-type-caption">
                                        ABCDEFG abcdefg !@#$%^&*()
                                    </div>
                                </div>
                            </AppFormSection>

                            <AppFormSection
                                title={tf("appearance_font_system")}
                                description={
                                    availableFonts.length > 0
                                        ? tf("appearance_font_count").replace(
                                              "{count}",
                                              String(availableFonts.length),
                                          )
                                        : undefined
                                }
                                action={
                                    <AppButton
                                        size="sm"
                                        onClick={() => void systemFonts.detect(true)}
                                    >
                                        {systemFonts.loading
                                            ? tf("appearance_font_detecting")
                                            : tf("appearance_font_detect")}
                                    </AppButton>
                                }
                            >
                                {/* 加载中 */}
                                {availableFonts.length === 0 && (
                                    <div className="flex items-center gap-2 rounded-md border border-dashed border-[color:var(--qt-divider)] bg-qt-base/35 px-3 py-3 text-qt-xs text-qt-text-muted">
                                        {systemFonts.loading ? (
                                            <>
                                                <span className="animate-spin inline-block w-3 h-3 border-2 border-qt-text-muted/20 border-t-qt-highlight rounded-full" />
                                                {tf("appearance_font_detecting")}
                                            </>
                                        ) : systemFonts.supported ? (
                                            tf("appearance_font_detect")
                                        ) : (
                                            tf("appearance_font_not_supported")
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
                                                className="w-full rounded-md border border-[color:var(--qt-divider)] bg-qt-surface/40 py-2 pl-8 pr-8 text-qt-xs text-qt-text focus:outline-none focus:ring-1 focus:ring-qt-highlight/30 transition-all"
                                                placeholder={tf(
                                                    "appearance_font_search_placeholder",
                                                )}
                                                spellCheck={false}
                                            />
                                            {fontSearch && (
                                                <button
                                                    className="absolute right-2 top-1/2 -translate-y-1/2 text-qt-3xs text-qt-text-muted hover:text-qt-text cursor-pointer"
                                                    onClick={() => setFontSearch("")}
                                                >
                                                    ×
                                                </button>
                                            )}
                                        </div>

                                        {/*
                                         * 【这里的两层滚动是**有意**的】本页是设置页：
                                         * 外层是页面滚动（Tab 内容比视口高），内层是字体
                                         * 列表的有界滚动（字体可能有几百项，展开会把这
                                         * 一页撑到几千像素，把下面的颜色设置挤到很远处）。
                                         * 两条滚动条各自都有用，用户不会"滚下去看不到
                                         * 东西"。
                                         *
                                         * 这与对话框里那种"内层滚下去什么也看不到"的嵌套
                                         * 不同 —— 那种已按"外层不滚、内层滚"改掉（见
                                         * AppDialog 的滚动契约）。这里保留。
                                         */}
                                        <div className="hs-scroll-gutter max-h-[320px] overflow-y-auto rounded border border-qt-border bg-qt-base p-1 custom-scrollbar">
                                            {filteredFonts.length > 0 ? (
                                                filteredFonts.map((f) => {
                                                    const isActive = extractFontFamilies(
                                                        fontFamily,
                                                    ).includes(normalizeFontName(f));
                                                    return (
                                                        <button
                                                            key={f}
                                                            className={
                                                                "w-full rounded-md border px-3 py-2 text-left transition-colors duration-100 " +
                                                                "cursor-pointer select-none text-qt-xs flex items-center gap-3 " +
                                                                (isActive
                                                                    ? "border-qt-highlight/25 bg-qt-highlight/12 text-qt-highlight"
                                                                    : "border-transparent text-qt-text hover:border-[color:var(--qt-divider)] hover:bg-qt-surface/40")
                                                            }
                                                            onClick={() => {
                                                                markDraftDirty();
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
                                                                {/* hs-text-exempt: 字体预览样本必须含中文字形 */}
                                                                AaBbCc 你好 123
                                                            </span>
                                                            {isActive && (
                                                                <span className="text-qt-highlight text-qt-micro shrink-0 font-bold">
                                                                    ✓
                                                                </span>
                                                            )}
                                                        </button>
                                                    );
                                                })
                                            ) : (
                                                <div className="px-3 py-4 text-qt-micro text-qt-text-muted/40 italic text-center">
                                                    {tf("appearance_font_no_results")}
                                                </div>
                                            )}
                                        </div>
                                    </>
                                )}
                            </AppFormSection>
                        </>
                    )}
                </div>
            </div>

            {/* ═══════ 底部按钮 ═══════ */}
            <div className="flex shrink-0 items-center justify-end gap-2 border-t border-qt-border px-3 py-2">
                <AppButton onClick={handleClose}>{tf("close")}</AppButton>
                <AppButton intent="primary" onClick={handleApply}>
                    {tf("appearance_apply")}
                </AppButton>
            </div>

            {/* 重置全部颜色确认：丢弃当前所有自定义色覆盖。 */}
            <AppConfirmDialog
                open={resetColorsConfirmOpen}
                onOpenChange={setResetColorsConfirmOpen}
                title={tf("appearance_reset_all_colors")}
                message={tf("appearance_reset_all_colors_confirm")}
                confirmLabel={tf("appearance_reset_all_colors")}
                cancelLabel={tf("cancel")}
                intent="danger"
                onConfirm={handleResetColors}
            />

            {/* 导入 / 导出失败：纯通知（用户唯一能做的就是关闭）。 */}
            <AppNoticeDialog
                open={notice.length > 0}
                onOpenChange={(open) => {
                    if (!open) setNotice("");
                }}
                title={tf("status_error_prefix")}
                message={notice}
                closeLabel={tf("ok")}
            />
        </div>
    );
};
