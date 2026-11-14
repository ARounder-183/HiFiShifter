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
import { frontendLog } from "../services/frontendLog";

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
            if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
                return { ...DEFAULT_APPEARANCE };
            }
            const settings = { ...DEFAULT_APPEARANCE, ...parsed };
            if (typeof settings.fontFamily !== "string" || settings.fontFamily.length > 512) {
                settings.fontFamily = DEFAULT_APPEARANCE.fontFamily;
            }
            if (
                settings.activeCustomThemeId !== null &&
                typeof settings.activeCustomThemeId !== "string"
            ) {
                settings.activeCustomThemeId = DEFAULT_APPEARANCE.activeCustomThemeId;
            }
            // mode 只接受合法值（旧数据 dark/light 仍有效；脏数据回退默认）。
            if (!THEME_MODE_SETTINGS.includes(settings.mode)) {
                settings.mode = DEFAULT_APPEARANCE.mode;
            }
            // accent / gray / radius 与 mode 同一批直喂 `<Theme>` 的取值，必须同样
            // 过白名单：手改坏的存储值会生成一条没有 CSS 规则支撑的 Radix 类，
            // 主题无声劣化。导入路径（importThemeFromJson）对这三项已有同样的
            // 校验 —— 运行时存储路径不能是唯一漏网的入口。
            if (!isValidAccentColor(settings.accentColor)) {
                settings.accentColor = DEFAULT_APPEARANCE.accentColor;
            }
            if (!isValidGrayColor(settings.grayColor)) {
                settings.grayColor = DEFAULT_APPEARANCE.grayColor;
            }
            if (!isValidRadius(settings.radius)) {
                settings.radius = DEFAULT_APPEARANCE.radius;
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
        frontendLog("warn", "[theme] failed to persist appearance settings", err);
    }
}

/* ─────────── 自定义主题列表 ─────────── */

/** 读取自定义主题列表（外部 / 手改存储值时按导入路径的同一白名单清洗）。 */
export function loadCustomThemes(): CustomTheme[] {
    try {
        const raw = readUiValue(CUSTOM_THEMES_KEY);
        if (!raw) return [];
        const parsed: unknown = JSON.parse(raw);
        if (!Array.isArray(parsed)) return [];
        return parsed.flatMap((entry): CustomTheme[] => {
            if (!entry || typeof entry !== "object") return [];
            const record = entry as Record<string, unknown>;
            if (
                typeof record.id !== "string" ||
                typeof record.name !== "string" ||
                (record.base !== "dark" && record.base !== "light")
            ) {
                return [];
            }
            return [
                {
                    id: record.id,
                    name: record.name,
                    base: record.base,
                    colors: sanitizeImportedColors(record.colors),
                    waveformColors: sanitizeWaveformColors(record.waveformColors),
                    accentColor: isValidAccentColor(record.accentColor)
                        ? record.accentColor
                        : undefined,
                    grayColor: isValidGrayColor(record.grayColor) ? record.grayColor : undefined,
                    radius: isValidRadius(record.radius) ? record.radius : undefined,
                },
            ];
        });
    } catch {
        return [];
    }
}

/** 保存自定义主题列表（持久化失败只记录告警，见 saveAppearance）。 */
export function saveCustomThemes(themes: CustomTheme[]): void {
    try {
        writeUiValue(CUSTOM_THEMES_KEY, JSON.stringify(themes));
    } catch (err) {
        frontendLog("warn", "[theme] failed to persist custom themes", err);
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

/** 单个颜色值是否为可安全写入 CSS 变量的形态（导入、运行时预览共用）。 */
export function isValidCssColor(v: unknown): v is string {
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
        const parsed: unknown = JSON.parse(json);
        if (
            parsed &&
            typeof parsed === "object" &&
            !Array.isArray(parsed) &&
            typeof (parsed as { name?: unknown }).name === "string" &&
            ((parsed as { base?: unknown }).base === "dark" ||
                (parsed as { base?: unknown }).base === "light") &&
            typeof (parsed as { colors?: unknown }).colors === "object"
        ) {
            const record = parsed as Record<string, unknown>;
            const theme: CustomTheme = {
                id: typeof record.id === "string" && record.id ? record.id : crypto.randomUUID(),
                name: record.name as string,
                base: record.base as "dark" | "light",
                colors: sanitizeImportedColors(record.colors),
                waveformColors: sanitizeWaveformColors(record.waveformColors),
                accentColor: isValidAccentColor(record.accentColor)
                    ? record.accentColor
                    : undefined,
                grayColor: isValidGrayColor(record.grayColor) ? record.grayColor : undefined,
                radius: isValidRadius(record.radius) ? record.radius : undefined,
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
