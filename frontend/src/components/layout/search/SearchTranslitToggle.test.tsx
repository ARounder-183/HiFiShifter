/**
 * 「拼音匹配」开关。
 *
 * 【为什么需要这个文件】它替代了原来的三选一菜单，理由是「搜索行上三个按钮的点击
 * 语义必须一致」。这条承诺有两个可见后果，纯逻辑测试盖不住：
 *   1. 图标不能再是 `Aa`（会被读成「区分大小写」）；
 *   2. 点击就是开/关，且关掉之后 `mode` 不被清掉。
 */
// @vitest-environment jsdom
import { describe, expect, it, vi } from "vitest";
import { act } from "react";
import { createRoot } from "react-dom/client";

import { SearchTranslitToggle } from "./SearchTranslitToggle";
import { I18nProvider } from "../../../i18n/I18nProvider";
import {
    DEFAULT_SEARCH_SETTINGS,
    type SearchSettings,
} from "../../../features/search/searchSettings";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

function mount(settings: Partial<SearchSettings>, regexActive = false) {
    const onChange = vi.fn();
    const container = document.createElement("div");
    document.body.appendChild(container);
    const root = createRoot(container);
    act(() => {
        root.render(
            <I18nProvider>
                <SearchTranslitToggle
                    settings={{ ...DEFAULT_SEARCH_SETTINGS, ...settings }}
                    onChange={onChange}
                    regexActive={regexActive}
                />
            </I18nProvider>,
        );
    });
    const button = container.querySelector<HTMLButtonElement>(".search-translit-toggle");
    expect(button, "开关按钮未渲染").not.toBeNull();
    return { button: button!, onChange, root };
}

describe("SearchTranslitToggle", () => {
    it("图标是「文A」而不是会被读成「区分大小写」的 Aa", () => {
        const { button } = mount({});
        expect(button.textContent).toBe("文A");
        expect(button.textContent).not.toContain("Aa");
    });

    it("点击即开/关（与同行的正则、仅媒体按钮同一套交互）", () => {
        const { button, onChange } = mount({ translit: true });
        act(() => {
            button.click();
        });
        expect(onChange).toHaveBeenCalledWith({ translit: false });
    });

    it("关闭时只翻总开关，不带 mode（再打开回到上次的宽严）", () => {
        const { button, onChange } = mount({ translit: true, mode: "fuzzy" });
        act(() => {
            button.click();
        });
        expect(onChange).toHaveBeenCalledWith({ translit: false });
        // 关键：补丁里没有 mode —— 用户的「模糊」不会被清成默认。
        expect(onChange.mock.calls[0][0]).not.toHaveProperty("mode");
    });

    it("激活态直接反映设置，不被正则模式改写", () => {
        expect(mount({ translit: true }).button.getAttribute("aria-pressed")).toBe("true");
        // 正则开着时它仍然显示「开」：点它切换的确实是那个设置，显示成「关」会是假话。
        expect(mount({ translit: true }, true).button.getAttribute("aria-pressed")).toBe("true");
        expect(mount({ translit: false }).button.getAttribute("aria-pressed")).toBeNull();
    });

    it("悬停提示里带当前档位（开关本身表达不了「智能还是模糊」）", () => {
        expect(mount({ translit: true, mode: "smart" }).button.dataset.tooltip).toContain(
            "Smart (pinyin and romaji)",
        );
        expect(mount({ translit: true, mode: "fuzzy" }).button.dataset.tooltip).toContain(
            "Fuzzy (allows skipped characters)",
        );
        expect(mount({ translit: false, mode: "fuzzy" }).button.dataset.tooltip).toContain(
            "Off (literal only)",
        );
    });

    it("正则模式下提示说明此刻不生效", () => {
        const { button } = mount({ translit: true, mode: "fuzzy" }, true);
        expect(button.dataset.tooltip).toContain("Transliteration is off in regex mode");
    });
});
