/**
 * 「拼音匹配」开关。
 *
 * 【为什么需要这个文件】它替代了原来的三选一菜单，理由是「搜索行上三个按钮的点击
 * 语义必须一致」。这条承诺有两个可见后果，纯逻辑测试盖不住：
 *   1. 图标不能再是 `Aa`（会被读成「区分大小写」）；
 *   2. 点击就是开/关，且关掉之后 `mode` 不被清掉。
 */
// @vitest-environment jsdom
import { beforeEach, describe, expect, it, vi } from "vitest";
import { act } from "react";
import { createRoot } from "react-dom/client";

import { SearchTranslitToggle } from "./SearchTranslitToggle";
import { I18nProvider } from "../../../i18n/I18nProvider";
import {
    DEFAULT_SEARCH_SETTINGS,
    type SearchSettings,
} from "../../../features/search/searchSettings";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

// 每个用例都往 body 里挂一个新的根，不清掉的话上一次留下的菜单会先被查到。
beforeEach(() => {
    document.body.replaceChildren();
});

function mount(
    settings: Partial<SearchSettings>,
    options: { regexActive?: boolean; onOpenSettings?: () => void } = {},
) {
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
                    regexActive={options.regexActive ?? false}
                    onOpenSettings={options.onOpenSettings}
                />
            </I18nProvider>,
        );
    });
    const button = container.querySelector<HTMLButtonElement>(".search-translit-toggle");
    expect(button, "开关按钮未渲染").not.toBeNull();

    /** 在按钮上右键，返回渲染出来的菜单项（按可见文本）。 */
    const openMenu = () => {
        act(() => {
            button!.dispatchEvent(
                new MouseEvent("contextmenu", { bubbles: true, clientX: 40, clientY: 40 }),
            );
        });
        // 只在本次挂载的容器里找：菜单不是 portal，就在按钮旁边。
        const menu = container.querySelector('[data-hs-context-menu="1"]');
        expect(menu, "右键未打开菜单").not.toBeNull();
        const items = Array.from(menu!.querySelectorAll<HTMLButtonElement>("button"));
        const byLabel = (label: string) => {
            const found = items.find((item) => (item.textContent ?? "").includes(label));
            expect(found, `菜单里没有「${label}」`).not.toBeUndefined();
            return found!;
        };
        return { menu: menu!, items, byLabel };
    };

    return { button: button!, onChange, root, openMenu };
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
        expect(
            mount({ translit: true }, { regexActive: true }).button.getAttribute("aria-pressed"),
        ).toBe("true");
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

    it("悬停提示只承载档位信息（右键提示已按设计移除，右键菜单行为另行覆盖）", () => {
        expect(mount({}).button.dataset.tooltip).not.toContain("Right-click");
    });

    it("右键打开完整菜单：三档宽严 + 三个子开关 + 匹配原因", () => {
        const mounted = mount({ translit: true, mode: "smart" });
        const { byLabel } = mounted.openMenu();
        byLabel("Off (literal only)");
        byLabel("Smart (pinyin and romaji)");
        byLabel("Fuzzy (allows skipped characters)");
        byLabel("Heteronyms");
        byLabel("Japanese long vowels");
        byLabel("Korean choseong");
        byLabel("Show why a result matched");
    });

    it("右键菜单里选档位走的是同一套补丁规则", () => {
        const fuzzy = mount({ translit: true, mode: "smart" });
        // 注意：`openMenu()` 自带 `act`，必须在它**外面**再开一层 act 才点得到 ——
        // 套在里面时 React 还没把菜单刷进 DOM。
        const fuzzyItems = fuzzy.openMenu();
        act(() => {
            fuzzyItems.byLabel("Fuzzy (allows skipped characters)").click();
        });
        expect(fuzzy.onChange).toHaveBeenCalledWith({ translit: true, mode: "fuzzy" });

        const off = mount({ translit: true, mode: "smart" });
        const offItems = off.openMenu();
        act(() => {
            offItems.byLabel("Off (literal only)").click();
        });
        // 选「关闭」同样只关总开关、保留 mode。
        expect(off.onChange).toHaveBeenCalledWith({ translit: false });
    });

    it("右键菜单里切换子开关", () => {
        const mounted = mount({ translit: true, heteronym: true });
        const { byLabel } = mounted.openMenu();
        act(() => {
            byLabel("Heteronyms").click();
        });
        expect(mounted.onChange).toHaveBeenCalledWith({ heteronym: false });
    });

    it("右键菜单里可以进设置（传了回调才有这一项）", () => {
        const onOpenSettings = vi.fn();
        const withSettings = mount({}, { onOpenSettings });
        const { byLabel } = withSettings.openMenu();
        act(() => {
            byLabel("Search and matching").click();
        });
        expect(onOpenSettings).toHaveBeenCalled();

        // 没传回调时不出现这一项（菜单里不该有指向空气的入口）。
        const { items } = mount({}).openMenu();
        expect(items.some((item) => (item.textContent ?? "").includes("Search and matching"))).toBe(
            false,
        );
    });

    it("正则模式下提示说明此刻不生效", () => {
        const { button } = mount({ translit: true, mode: "fuzzy" }, { regexActive: true });
        expect(button.dataset.tooltip).toContain("Transliteration is off in regex mode");
    });
});
