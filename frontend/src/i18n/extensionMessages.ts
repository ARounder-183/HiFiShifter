/**
 * 扩展文案注册层。
 *
 * 【为什么需要它】内置词典是一个**封闭联合**：`MessageKey` 由 `en-US` 的键推导，
 * 五个语系缺任何一个键都会让 `tsc` 失败。这个设计对内置文案是正确的（编译期就
 * 保证不漏译），但对第三方是死路 —— 面板拿不到任何键可以放自己的文字。
 *
 * 本模块提供一条运行时通道：扩展注册 `{ locale: { key: text } }`，`t()` 在静态
 * 词典查不到时来这里找。
 *
 * ## 解析顺序
 *
 * ```
 * 静态词典（当前语系） → 扩展层（当前语系） → 静态词典（en-US） → 原样返回键名
 * ```
 *
 * 【为什么扩展层排在静态词典之后】内置键必须不可被覆盖。若扩展层优先，
 * 一个第三方面板注册 `ok: "…"` 就能改掉全应用的「确定」按钮 —— 那不是扩展，
 * 是劫持。扩展只能**新增**键，不能改写既有键。
 *
 * 【为什么不做类型安全】第三方键在编译期不存在于 `MessageKey` 里，因此必然走
 * 字符串键。边界处用 `useExtensionTranslate()` 取一个 `(key: string) => string`，
 * 与内置的 `t()` 分开，避免像历史上 71 处 `tf` 那样
 * 把内置的类型安全一起丢掉。
 */
import type { Locale } from "./messages";

type LocaleMessages = Partial<Record<Locale, Record<string, string>>>;

const layers: LocaleMessages[] = [];
const listeners = new Set<() => void>();
let version = 0;

function notify(): void {
    version += 1;
    for (const listener of listeners) listener();
}

/**
 * 注册一组扩展文案。
 *
 * @returns 注销函数（扩展被卸载时必须调用，否则文案会一直留着）。
 *
 * @example
 * // 面板注册时
 * const dispose = registerExtensionMessages({
 *     "en-US": { "panel.myThing.title": "My Thing" },
 *     "zh-CN": { "panel.myThing.title": "我的面板" },
 * });
 */
export function registerExtensionMessages(messages: LocaleMessages): () => void {
    layers.push(messages);
    notify();
    let disposed = false;
    return () => {
        if (disposed) return;
        disposed = true;
        const at = layers.indexOf(messages);
        if (at !== -1) layers.splice(at, 1);
        notify();
    };
}

/** 按当前语系查扩展文案；查不到返回 `undefined`（交由上层继续回退）。 */
export function lookupExtensionMessage(locale: Locale, key: string): string | undefined {
    // 后注册的优先：后加载的扩展可以覆盖先前扩展的同名键（但仍不能碰内置键）。
    for (let i = layers.length - 1; i >= 0; i -= 1) {
        const value = layers[i][locale]?.[key];
        if (value !== undefined) return value;
    }
    return undefined;
}

/** 订阅注册变化（`I18nProvider` 据此重渲染）。 */
export function subscribeExtensionMessages(listener: () => void): () => void {
    listeners.add(listener);
    return () => listeners.delete(listener);
}

/** 快照：注册表版本号。 */
export function getExtensionMessagesVersion(): number {
    return version;
}

/** 仅供测试：清空所有扩展层。 */
export function resetExtensionMessagesForTests(): void {
    layers.length = 0;
    notify();
}
