/**
 * 本地化格式化层 —— 插值、复数、单位、快捷键的**唯一**实现。
 *
 * 【为什么需要它】审查发现文案层缺少三件基础设施，于是每处调用点各自退化：
 *
 * 1. **没有插值引擎**：`{name}` 靠手写 `.replace("{name}", value)`，全仓约 40 处。
 *    拼错占位符不会报错，只会把 `{name}` 原样显示给用户。
 * 2. **没有复数机制**：14 处用字面 `(s)` 糊过去，渲染出 `"1 clip(s)"`。
 * 3. **没有单位格式化**：`8kHz` 与 `1 kHz` 在同一个文件里并存（中文/日文/韩文
 *    都有这种混用），因为每处都是手写字符串。
 *
 * 另外快捷键文本在 5 个语系里硬编码 `Ctrl+O/E/B/I/K`，macOS 上全错 —— 而平台
 * 感知的格式化早已存在于 `formatKeybinding`（它用 `IS_MAC ? "⌘" : "Ctrl"`）。
 *
 * 本模块把这四件事各做一次，`I18nProvider` 把它们绑上当前语系后暴露给组件。
 */
import { IS_MAC } from "../utils/platform";

/**
 * 主修饰键的展示名。
 *
 * 与 `features/keybindings/keybindingsSlice.ts` 的 `formatKeybinding` 取同一规则：
 * macOS 的快捷键主修饰键是 Command（`⌘`），Windows/Linux 是 Control。
 * 两处必须一致，否则同一个快捷键在键位设置里显示 `⌘O`、在工具提示里显示 `Ctrl+O`。
 */
export function primaryModifierLabel(): string {
    return IS_MAC ? "⌘" : "Ctrl";
}

/**
 * `{name}` 插值。
 *
 * 未提供的占位符原样保留（而不是替换成 `undefined`）：宁可让缺失可见，
 * 也不要静默显示一个错误的句子。
 */
export function formatTemplate(template: string, vars: Record<string, string | number>): string {
    return template.replace(/\{(\w+)\}/g, (match, name: string) =>
        Object.prototype.hasOwnProperty.call(vars, name) ? String(vars[name]) : match,
    );
}

/** 把文案里的 `{modifier}` 替换为当前平台的主修饰键。 */
export function formatShortcutLabel(template: string): string {
    return formatTemplate(template, { modifier: primaryModifierLabel() });
}

/**
 * 复数形态分隔符。
 *
 * 词典里用 `"clip|clips"` 表示「单数|复数」，单形态（无 `|`）表示该语言不区分
 * 复数 —— 中文/日文/韩文都是这种，因此不需要为它们新增键。
 *
 * 【为什么不用 `_one`/`_other` 后缀键】那会为 3 个计数单位凭空增加 15 个键，
 * 且每个语系都要补齐（`MessageKey` 是封闭联合，缺一个就编译失败）。用 `|`
 * 把两种形态放在同一个值里，既保持键数不变，又让「这是一个复数文案」在
 * 字面上可见。
 */
export const PLURAL_SEPARATOR = "|";

/**
 * 按数量选形态。
 *
 * 用 `Intl.PluralRules` 而不是 `n === 1`：英语的 `one` 只覆盖 1，但其他语言
 * 的规则各不相同（例如俄语有 3 种形态）。当前五个语系里只有英语需要区分，
 * 但用引擎规则意味着新增语系时不必改这里。
 *
 * @param locale 语系
 * @param count 数量
 * @param raw 词典原文，可含一个 `|` 分隔单复数
 */
export function selectPluralForm(locale: string, count: number, raw: string): string {
    const separatorAt = raw.indexOf(PLURAL_SEPARATOR);
    if (separatorAt === -1) return raw;

    const singular = raw.slice(0, separatorAt);
    const plural = raw.slice(separatorAt + 1);
    let category: Intl.LDMLPluralRule = "other";
    try {
        category = new Intl.PluralRules(locale).select(count);
    } catch {
        // 语系标识不被引擎识别时退回英语规则（1 为单数）。
        category = count === 1 ? "one" : "other";
    }
    return category === "one" ? singular : plural;
}

/**
 * 数字 + 单位。
 *
 * 用 `Intl.NumberFormat` 的 `unit` 样式，空格与符号交给引擎按语系决定 ——
 * 这正是手写字符串做不到的地方（同一个 `kHz` 在中文里该不该留空格，
 * 引擎知道，作者不知道）。
 *
 * @param unit CLDR 单位标识（`kilohertz` / `decibel` / `millisecond` / `pixel` …）
 */
export function formatUnit(
    locale: string,
    value: number,
    unit: string,
    options: Intl.NumberFormatOptions = {},
): string {
    try {
        return new Intl.NumberFormat(locale, {
            style: "unit",
            unit,
            unitDisplay: "short",
            ...options,
        }).format(value);
    } catch {
        return `${value} ${unit}`;
    }
}

/** 按语系格式化数字（千分位等）。 */
export function formatNumber(
    locale: string,
    value: number,
    options: Intl.NumberFormatOptions = {},
): string {
    try {
        return new Intl.NumberFormat(locale, options).format(value);
    } catch {
        return String(value);
    }
}
