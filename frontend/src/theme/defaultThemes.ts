/**
 * 内置主题颜色常量
 *
 * 从 index.css 中的 .qt-theme.dark / .qt-theme.light 提取为 TypeScript 对象，
 * 用于外观设置中的「重置为默认」功能。
 */

import type { QtColorToken } from "./themeTypes";

/** 深色内置主题的所有 --qt-* 颜色值 */
export const DARK_THEME_COLORS: Record<QtColorToken, string> = {
    "qt-window": "#353535",
    "qt-base": "#2d2d2d",
    "qt-panel": "#2a2a2a",
    "qt-surface": "#404040",
    "qt-clip-bg": "#343434",
    "qt-clip-border": "rgba(255,255,255,0.14)",
    "qt-clip-selected-border": "rgba(255,255,255,0.55)",
    "qt-text": "#d0d0d0",
    "qt-text-muted": "#909090",
    "qt-highlight": "#3b82f6",
    "qt-playhead": "#f05a5a",
    "qt-button": "#3d3d3d",
    "qt-button-hover": "#484848",
    "qt-border": "#505050",
    "qt-hover": "#484848",
    "qt-danger-bg": "#4a1a1a",
    "qt-danger-text": "#fca5a5",
    "qt-danger-border": "#ef4444",
    "qt-warning-bg": "#3d2a0a",
    "qt-warning-text": "#fcd34d",
    "qt-warning-border": "#f59e0b",
    "qt-graph-bg": "#232323",
    "qt-graph-grid-strong": "#4a4a4a",
    "qt-graph-grid-weak": "#373737",
    "qt-scrollbar-thumb": "#555555",
    "qt-scrollbar-thumb-hover": "#707070",
    "qt-overlay": "rgba(0, 0, 0, 0.35)",
    "qt-divider": "rgba(255,255,255,0.04)",
    "qt-subtle-1": "rgba(255,255,255,0.025)",
    "qt-subtle-2": "rgba(255,255,255,0.045)",
    "qt-subtle-3": "rgba(255,255,255,0.07)",
    "qt-meter-rail": "#2a2a2a",
    "qt-meter-well": "#1d1d1d",
};

/**
 * 浅色内置主题 — 使用 Radix var() 引用，不是固定 hex。
 * 在外观设置 UI 中不能直接用作 <input type="color"> 的默认值，
 * 所以这里存放的是「等效 hex」近似值，方便 UI 展示和编辑。
 */
export const LIGHT_THEME_COLORS: Record<QtColorToken, string> = {
    "qt-window": "#f2f3f6",
    "qt-base": "#ebedf1",
    "qt-panel": "#f8f9fb",
    "qt-surface": "#e1e4ea",
    "qt-clip-bg": "#dde1e8",
    "qt-clip-border": "rgba(32,38,52,0.18)",
    "qt-clip-selected-border": "rgba(32,38,52,0.42)",
    "qt-text": "#1f232b",
    "qt-text-muted": "#666d79",
    "qt-highlight": "#4b68d1",
    "qt-playhead": "#de4f5d",
    "qt-button": "#e3e6ec",
    "qt-button-hover": "#d7dbe4",
    "qt-border": "#b8bdc9",
    "qt-hover": "#dde1e8",
    "qt-danger-bg": "#ffe6ea",
    "qt-danger-text": "#b53a49",
    "qt-danger-border": "#df6674",
    "qt-warning-bg": "#fff1dc",
    "qt-warning-text": "#9b5f13",
    "qt-warning-border": "#d59140",
    "qt-graph-bg": "#f6f7fa",
    "qt-graph-grid-strong": "#aab1bf",
    "qt-graph-grid-weak": "#cfd4df",
    "qt-scrollbar-thumb": "#b4bac7",
    "qt-scrollbar-thumb-hover": "#969ead",
    "qt-overlay": "rgba(20, 24, 32, 0.22)",
    "qt-divider": "rgba(32,38,52,0.1)",
    "qt-subtle-1": "rgba(28,36,52,0.025)",
    "qt-subtle-2": "rgba(28,36,52,0.04)",
    "qt-subtle-3": "rgba(28,36,52,0.06)",
    "qt-meter-rail": "#d6dbe5",
    "qt-meter-well": "#c8ceda",
};

/** 根据模式获取内置主题颜色 */
export function getBuiltinThemeColors(mode: "dark" | "light"): Record<QtColorToken, string> {
    return mode === "dark" ? DARK_THEME_COLORS : LIGHT_THEME_COLORS;
}
