/**
 * 主题持久化存储工具
 *
 * 使用 localStorage 存储用户的外观设置和自定义主题列表。
 * 导出格式 v2 包含通用设置（强调色、灰阶、圆角）+ 颜色覆盖，
 * 导入时向后兼容 v1（仅颜色覆盖）。
 */

import { readUiValue, writeUiValue } from "../services/uiStorage";
import type {
    AppearanceSettings,
    CustomTheme,
    ThemeExportData,
    RadixAccentColor,
    RadixGrayColor,
    RadixRadius,
    QtColorToken,
} from "./themeTypes";
import {
    DEFAULT_APPEARANCE,
    QT_COLOR_TOKENS,
    RADIX_ACCENT_COLORS,
    RADIX_GRAY_COLORS,
    RADIX_RADIUS_OPTIONS,
    THEME_MODE_SETTINGS,
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
        const raw = readUiValue(APPEARANCE_KEY);
        if (raw) {
            const parsed = JSON.parse(raw) as Partial<AppearanceSettings>;
            const settings = { ...DEFAULT_APPEARANCE, ...parsed };
            // mode 只接受合法值（旧数据 dark/light 仍有效；脏数据回退默认）。
            if (!THEME_MODE_SETTINGS.includes(settings.mode)) {
                settings.mode = DEFAULT_APPEARANCE.mode;
            }
            return settings;
        }
    } catch {
        // fallthrough
    }

    // 兼容旧版 hifishifter.theme
    const legacyMode = readUiValue(LEGACY_THEME_KEY);
    if (legacyMode === "light" || legacyMode === "dark") {
        return { ...DEFAULT_APPEARANCE, mode: legacyMode };
    }

    // 默认：自动跟随系统深浅色（由 AppThemeProvider 实时解析系统偏好）。
    return { ...DEFAULT_APPEARANCE };
}

/**
 * 保存外观设置（内存中的外观状态照常更新；配额 / 隐私模式等持久化失败
 * 只记录告警，不向调用方抛出 —— 否则外观切换会在点击处理器里以未处理
 * 异常中断后续 UI 更新）。
 */
export function saveAppearance(settings: AppearanceSettings): void {
    try {
        writeUiValue(APPEARANCE_KEY, JSON.stringify(settings));
        // 同步旧版 key（兼容其他可能直接读取的代码）
        writeUiValue(LEGACY_THEME_KEY, settings.mode);
    } catch (err) {
        console.warn("[theme] failed to persist appearance settings", err);
    }
}

/* ─────────── 自定义主题列表 ─────────── */

/** 读取自定义主题列表 */
export function loadCustomThemes(): CustomTheme[] {
    try {
        const raw = readUiValue(CUSTOM_THEMES_KEY);
        if (raw) {
            return JSON.parse(raw) as CustomTheme[];
        }
    } catch {
        // fallthrough
    }
    return [];
}

/** 保存自定义主题列表（持久化失败只记录告警，见 saveAppearance）。 */
export function saveCustomThemes(themes: CustomTheme[]): void {
    try {
        writeUiValue(CUSTOM_THEMES_KEY, JSON.stringify(themes));
    } catch (err) {
        console.warn("[theme] failed to persist custom themes", err);
    }
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
 * 颜色值白名单正则（导入主题用）。
 *
 * 接受本应用颜色选择器会产出的全部形态：#hex（3/4/6/8 位）、rgb()/rgba()、
 * hsl()/hsla() 与 CSS 命名色。命名字用有界的字母段近似（完整命名色表 148 个
 * 不值得内置；不认识的词在 CSS 里本就解析失败、回落继承值，不会崩）。
 *
 * 【为什么要校验】导入的 colors 此后会逐条 `setProperty` 到 `--qt-*` 上——
 * 主题文件是用户手写/外部来源的 JSON，不校验就把任意字符串（分号、
 * `url(...)`、几 KB 的乱码）写进全局自定义属性，轻则界面花掉，重则把
 * 恶意内容持久化进 localStorage 并随每次启动生效。
 */
const CSS_COLOR_RE =
    /^(?:#[0-9a-f]{3,4}|#[0-9a-f]{6}|#[0-9a-f]{8}|rgba?\(\s*[\d.]+%?\s*,\s*[\d.]+%?\s*,\s*[\d.]+%?\s*(?:,\s*[\d.]+%?\s*)?\)|hsla?\(\s*[\d.]+(?:deg|rad|turn)?\s*,\s*[\d.]+%\s*,\s*[\d.]+%\s*(?:,\s*[\d.]+%?\s*)?\)|[a-z]{3,30})$/i;

const KNOWN_COLOR_TOKENS = new Set<string>(QT_COLOR_TOKENS);

/** 单个颜色值是否为可安全写入 CSS 变量的形态。 */
function isValidCssColor(v: unknown): v is string {
    return typeof v === "string" && CSS_COLOR_RE.test(v.trim()) && v.trim().length <= 64;
}

/**
 * 清洗导入的 colors 映射：只保留**已知 token** 且**值合法**的条目。
 * 非法条目静默丢弃（导入的应是可用的主题，报错列表对用户没有行动价值）。
 */
function sanitizeImportedColors(input: unknown): Partial<Record<QtColorToken, string>> {
    if (input === null || typeof input !== "object") return {};
    const out: Partial<Record<QtColorToken, string>> = {};
    for (const [key, value] of Object.entries(input as Record<string, unknown>)) {
        if (!KNOWN_COLOR_TOKENS.has(key)) continue;
        if (isValidCssColor(value)) out[key as QtColorToken] = value.trim();
    }
    return out;
}

/** 清洗导入的波形颜色（fill/stroke 都合法才保留，缺省交回默认值）。 */
function sanitizeWaveformColors(input: unknown): { fill: string; stroke: string } | undefined {
    if (input === null || typeof input !== "object") return undefined;
    const record = input as Record<string, unknown>;
    if (isValidCssColor(record.fill) && isValidCssColor(record.stroke)) {
        return { fill: record.fill.trim(), stroke: record.stroke.trim() };
    }
    return undefined;
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
                colors: sanitizeImportedColors(parsed.colors),
                waveformColors: sanitizeWaveformColors(parsed.waveformColors),
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
