// @vitest-environment jsdom
/*
 * 语言存储值的健壮性。
 *
 * 【要钉死什么】`localStorage` 里一个损坏/伪造的值不能让 `locale` 变成一个
 * 查不到任何文案的"语言"（例如 `"toString"` —— `in` 会命中 `Object.prototype`
 * 的继承键），否则整个界面静默回落到 en-US。这里挂真实 Provider 读出 locale
 * 状态来验证。
 */

import { act } from "react";
import { createRoot } from "react-dom/client";
import { expect, test } from "vitest";

import { I18nProvider, useI18n } from "./I18nProvider";

// React 19 要求显式声明这是 act() 环境。
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const STORAGE_KEY = "hifishifter.locale";

function LocaleProbe() {
    const { locale } = useI18n();
    return <span data-locale>{locale}</span>;
}

async function renderLocale(): Promise<string> {
    const host = document.createElement("div");
    document.body.append(host);
    const root = createRoot(host);
    await act(async () => {
        root.render(
            <I18nProvider>
                <LocaleProbe />
            </I18nProvider>,
        );
    });
    const locale = host.querySelector("[data-locale]")?.textContent ?? "";
    await act(async () => root.unmount());
    host.remove();
    return locale;
}

test("原型链上的存储值不会被当成语言", async () => {
    localStorage.setItem(STORAGE_KEY, "toString");
    try {
        const locale = await renderLocale();
        expect(locale).not.toBe("toString");
        expect(["en-US", "zh-CN", "zh-TW", "ja-JP", "ko-KR"]).toContain(locale);
    } finally {
        localStorage.removeItem(STORAGE_KEY);
    }
});

test("合法的存储值仍被采用", async () => {
    localStorage.setItem(STORAGE_KEY, "ja-JP");
    try {
        expect(await renderLocale()).toBe("ja-JP");
    } finally {
        localStorage.removeItem(STORAGE_KEY);
    }
});
