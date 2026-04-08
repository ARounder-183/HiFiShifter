/**
 * 主题系统类型定义
 *
 * 定义 Radix 颜色、Qt 颜色 token、自定义主题、外观设置等类型。
 * 支持统一的主题导出格式，可包含通用设置（强调色、灰阶、圆角）和全部颜色覆盖。
 */

/* ─────────────────── Radix Theme 颜色类型 ─────────────────── */

/** Radix 支持的 26 种强调色 */
export type RadixAccentColor =
    | "gray"
    | "gold"
    | "bronze"
    | "brown"
    | "yellow"
    | "amber"
    | "orange"
    | "tomato"
    | "red"
    | "ruby"
    | "crimson"
    | "pink"
    | "plum"
    | "purple"
    | "violet"
    | "iris"
    | "indigo"
    | "blue"
    | "cyan"
    | "teal"
    | "jade"
    | "green"
    | "grass"
    | "lime"
    | "mint"
    | "sky";

/** Radix 支持的 7 种灰阶色系 */
export type RadixGrayColor = "auto" | "gray" | "mauve" | "slate" | "sage" | "olive" | "sand";

/** Radix 圆角风格 */
export type RadixRadius = "none" | "small" | "medium" | "large" | "full";

/** 所有 Radix 强调色列表（用于 UI 渲染） */
export const RADIX_ACCENT_COLORS: RadixAccentColor[] = [
    "gray",
    "gold",
    "bronze",
    "brown",
    "yellow",
    "amber",
    "orange",
    "tomato",
    "red",
    "ruby",
    "crimson",
    "pink",
    "plum",
    "purple",
    "violet",
    "iris",
    "indigo",
    "blue",
    "cyan",
    "teal",
    "jade",
    "green",
    "grass",
    "lime",
    "mint",
    "sky",
];

/** 所有 Radix 灰阶色列表（用于 UI 渲染） */
export const RADIX_GRAY_COLORS: RadixGrayColor[] = [
    "auto",
    "gray",
    "mauve",
    "slate",
    "sage",
    "olive",
    "sand",
];

/** 所有 Radix 圆角选项（用于 UI 渲染） */
export const RADIX_RADIUS_OPTIONS: RadixRadius[] = ["none", "small", "medium", "large", "full"];

/* ─────────────────── Qt 颜色 Token ─────────────────── */

/** 项目中所有 --qt-* CSS 变量 token 名 */
export type QtColorToken =
    | "qt-window"
    | "qt-base"
    | "qt-panel"
    | "qt-surface"
    | "qt-clip-bg"
    | "qt-clip-border"
    | "qt-clip-selected-border"
    | "qt-text"
    | "qt-text-muted"
    | "qt-highlight"
    | "qt-playhead"
    | "qt-button"
    | "qt-button-hover"
    | "qt-border"
    | "qt-hover"
    | "qt-danger-bg"
    | "qt-danger-text"
    | "qt-danger-border"
    | "qt-warning-bg"
    | "qt-warning-text"
    | "qt-warning-border"
    | "qt-graph-bg"
    | "qt-graph-grid-strong"
    | "qt-graph-grid-weak"
    | "qt-scrollbar-thumb"
    | "qt-scrollbar-thumb-hover"
    | "qt-overlay"
    | "qt-divider"
    | "qt-subtle-1"
    | "qt-subtle-2"
    | "qt-subtle-3"
    | "qt-meter-rail"
    | "qt-meter-well";

/** 所有 Qt 颜色 token（有序列表，用于 UI 渲染） */
export const QT_COLOR_TOKENS: QtColorToken[] = [
    "qt-window",
    "qt-base",
    "qt-panel",
    "qt-surface",
    "qt-clip-bg",
    "qt-clip-border",
    "qt-clip-selected-border",
    "qt-text",
    "qt-text-muted",
    "qt-highlight",
    "qt-playhead",
    "qt-button",
    "qt-button-hover",
    "qt-border",
    "qt-hover",
    "qt-danger-bg",
    "qt-danger-text",
    "qt-danger-border",
    "qt-warning-bg",
    "qt-warning-text",
    "qt-warning-border",
    "qt-graph-bg",
    "qt-graph-grid-strong",
    "qt-graph-grid-weak",
    "qt-scrollbar-thumb",
    "qt-scrollbar-thumb-hover",
    "qt-overlay",
    "qt-divider",
    "qt-subtle-1",
    "qt-subtle-2",
    "qt-subtle-3",
    "qt-meter-rail",
    "qt-meter-well",
];

/** Qt 颜色 token 的显示名（i18n key） */
export const QT_COLOR_TOKEN_LABELS: Record<QtColorToken, string> = {
    "qt-window": "appearance_color_window",
    "qt-base": "appearance_color_base",
    "qt-panel": "appearance_color_panel",
    "qt-surface": "appearance_color_surface",
    "qt-clip-bg": "appearance_color_clip_bg",
    "qt-clip-border": "appearance_color_clip_border",
    "qt-clip-selected-border": "appearance_color_clip_selected_border",
    "qt-text": "appearance_color_text",
    "qt-text-muted": "appearance_color_text_muted",
    "qt-highlight": "appearance_color_highlight",
    "qt-playhead": "appearance_color_playhead",
    "qt-button": "appearance_color_button",
    "qt-button-hover": "appearance_color_button_hover",
    "qt-border": "appearance_color_border",
    "qt-hover": "appearance_color_hover",
    "qt-danger-bg": "appearance_color_danger_bg",
    "qt-danger-text": "appearance_color_danger_text",
    "qt-danger-border": "appearance_color_danger_border",
    "qt-warning-bg": "appearance_color_warning_bg",
    "qt-warning-text": "appearance_color_warning_text",
    "qt-warning-border": "appearance_color_warning_border",
    "qt-graph-bg": "appearance_color_graph_bg",
    "qt-graph-grid-strong": "appearance_color_graph_grid_strong",
    "qt-graph-grid-weak": "appearance_color_graph_grid_weak",
    "qt-scrollbar-thumb": "appearance_color_scrollbar_thumb",
    "qt-scrollbar-thumb-hover": "appearance_color_scrollbar_thumb_hover",
    "qt-overlay": "appearance_color_overlay",
    "qt-divider": "appearance_color_divider",
    "qt-subtle-1": "appearance_color_subtle_1",
    "qt-subtle-2": "appearance_color_subtle_2",
    "qt-subtle-3": "appearance_color_subtle_3",
    "qt-meter-rail": "appearance_color_meter_rail",
    "qt-meter-well": "appearance_color_meter_well",
};

/* ─────────────────── 自定义主题 ─────────────────── */

/** 自定义颜色主题 */
export interface CustomTheme {
    id: string;
    name: string;
    /** 基于哪个内置主题 */
    base: "dark" | "light";
    /** 覆盖的 --qt-* 颜色值 */
    colors: Partial<Record<QtColorToken, string>>;
    /** 自定义波形颜色 */
    waveformColors?: { fill: string; stroke: string };
    /** 通用设置（Radix 强调色、灰阶、圆角），用于完整导出 */
    accentColor?: RadixAccentColor;
    grayColor?: RadixGrayColor;
    radius?: RadixRadius;
}

/* ─────────────────── 主题导出格式 ─────────────────── */

/** 统一主题导出 JSON 格式（v2），包含通用设置 + 所有颜色覆盖 */
export interface ThemeExportData {
    /** 格式版本号 */
    version: 2;
    name: string;
    base: "dark" | "light";
    /** Radix 通用设置 */
    accentColor?: RadixAccentColor;
    grayColor?: RadixGrayColor;
    radius?: RadixRadius;
    /** --qt-* 颜色覆盖 */
    colors: Partial<Record<QtColorToken, string>>;
    /** 波形颜色 */
    waveformColors?: { fill: string; stroke: string };
}

/* ─────────────────── 外观设置 ─────────────────── */

/** 用户外观偏好 */
export interface AppearanceSettings {
    /** 主题模式 */
    mode: "dark" | "light";
    /** Radix 强调色 */
    accentColor: RadixAccentColor;
    /** Radix 灰阶色系 */
    grayColor: RadixGrayColor;
    /** Radix 圆角风格 */
    radius: RadixRadius;
    /** 字体 */
    fontFamily: string;
    /** 当前激活的自定义主题 ID（null 表示使用内置主题） */
    activeCustomThemeId: string | null;
}

/** 默认外观设置 */
export const DEFAULT_APPEARANCE: AppearanceSettings = {
    mode: "dark",
    accentColor: "iris",
    grayColor: "mauve",
    radius: "medium",
    fontFamily: "'Segoe UI', Roboto, Helvetica, Arial, sans-serif",
    activeCustomThemeId: null,
};

/** 默认字体（用于重置） */
export const DEFAULT_FONT_FAMILY = "'Segoe UI', Roboto, Helvetica, Arial, sans-serif";
