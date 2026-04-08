/**
 * 主题持久化存储工具
 *
 * 使用 localStorage 存储用户的外观设置和自定义主题列表。
 * 导出格式 v2 包含通用设置（强调色、灰阶、圆角）+ 颜色覆盖，
 * 导入时向后兼容 v1（仅颜色覆盖）。
 */

import type {
    AppearanceSettings,
    CustomTheme,
    ThemeExportData,
    RadixAccentColor,
    RadixGrayColor,
    RadixRadius,
} from "./themeTypes";
import {
    DEFAULT_APPEARANCE,
    RADIX_ACCENT_COLORS,
    RADIX_GRAY_COLORS,
    RADIX_RADIUS_OPTIONS,
} from "./themeTypes";

/* ─────────── Storage Keys ─────────── */

const APPEARANCE_KEY = "hifishifter.appearance";
const CUSTOM_THEMES_KEY = "hifishifter.customThemes";
/** 兼容旧版的主题模式 key */
const LEGACY_THEME_KEY = "hifishifter.theme";

/* ─────────── 外观设置 ─────────── */

/** 读取外观设置（带旧版兼容） */
export function loadAppearance(): AppearanceSettings {
    try {
        const raw = localStorage.getItem(APPEARANCE_KEY);
        if (raw) {
            const parsed = JSON.parse(raw) as Partial<AppearanceSettings>;
            return { ...DEFAULT_APPEARANCE, ...parsed };
        }
    } catch {
        // fallthrough
    }

    // 兼容旧版 hifishifter.theme
    const legacyMode = localStorage.getItem(LEGACY_THEME_KEY);
    if (legacyMode === "light" || legacyMode === "dark") {
        return { ...DEFAULT_APPEARANCE, mode: legacyMode };
    }

    // 检测系统偏好
    const prefersDark =
        typeof window !== "undefined" &&
        window.matchMedia?.("(prefers-color-scheme: dark)").matches;
    return { ...DEFAULT_APPEARANCE, mode: prefersDark ? "dark" : "light" };
}

/** 保存外观设置 */
export function saveAppearance(settings: AppearanceSettings): void {
    localStorage.setItem(APPEARANCE_KEY, JSON.stringify(settings));
    // 同步旧版 key（兼容其他可能直接读取的代码）
    localStorage.setItem(LEGACY_THEME_KEY, settings.mode);
}

/* ─────────── 自定义主题列表 ─────────── */

/** 读取自定义主题列表 */
export function loadCustomThemes(): CustomTheme[] {
    try {
        const raw = localStorage.getItem(CUSTOM_THEMES_KEY);
        if (raw) {
            return JSON.parse(raw) as CustomTheme[];
        }
    } catch {
        // fallthrough
    }
    return [];
}

/** 保存自定义主题列表 */
export function saveCustomThemes(themes: CustomTheme[]): void {
    localStorage.setItem(CUSTOM_THEMES_KEY, JSON.stringify(themes));
}

/* ─────────── 主题导入/导出 (v2) ─────────── */

/** 类型守卫 */
function isValidAccentColor(v: unknown): v is RadixAccentColor {
    return typeof v === "string" && (RADIX_ACCENT_COLORS as string[]).includes(v);
}
function isValidGrayColor(v: unknown): v is RadixGrayColor {
    return typeof v === "string" && (RADIX_GRAY_COLORS as string[]).includes(v);
}
function isValidRadius(v: unknown): v is RadixRadius {
    return typeof v === "string" && (RADIX_RADIUS_OPTIONS as string[]).includes(v);
}

/**
 * 导出主题为 v2 JSON 字符串（包含通用设置 + 颜色覆盖）
 */
export function exportThemeAsJson(
    theme: CustomTheme,
    settings?: { accentColor?: RadixAccentColor; grayColor?: RadixGrayColor; radius?: RadixRadius },
): string {
    const data: ThemeExportData = {
        version: 2,
        name: theme.name,
        base: theme.base,
        accentColor: settings?.accentColor ?? theme.accentColor,
        grayColor: settings?.grayColor ?? theme.grayColor,
        radius: settings?.radius ?? theme.radius,
        colors: theme.colors,
        waveformColors: theme.waveformColors,
    };
    return JSON.stringify(data, null, 2);
}

/**
 * 从 JSON 字符串导入主题（兼容 v1 和 v2 格式）
 * 返回 CustomTheme + 可选的通用设置
 */
export function importThemeFromJson(json: string): {
    theme: CustomTheme;
    accentColor?: RadixAccentColor;
    grayColor?: RadixGrayColor;
    radius?: RadixRadius;
} | null {
    try {
        const parsed = JSON.parse(json);
        if (
            parsed &&
            typeof parsed === "object" &&
            typeof parsed.name === "string" &&
            (parsed.base === "dark" || parsed.base === "light") &&
            typeof parsed.colors === "object"
        ) {
            const theme: CustomTheme = {
                id: parsed.id ?? crypto.randomUUID(),
                name: parsed.name,
                base: parsed.base,
                colors: parsed.colors,
                waveformColors: parsed.waveformColors ?? undefined,
                accentColor: isValidAccentColor(parsed.accentColor)
                    ? parsed.accentColor
                    : undefined,
                grayColor: isValidGrayColor(parsed.grayColor) ? parsed.grayColor : undefined,
                radius: isValidRadius(parsed.radius) ? parsed.radius : undefined,
            };
            return {
                theme,
                accentColor: theme.accentColor,
                grayColor: theme.grayColor,
                radius: theme.radius,
            };
        }
    } catch {
        // invalid JSON
    }
    return null;
}
