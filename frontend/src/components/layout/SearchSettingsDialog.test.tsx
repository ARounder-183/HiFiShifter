/**
 * 「搜索与匹配设置」对话框的渲染冒烟测试。
 *
 * 【为什么需要】设置项的**取值规则**由 `searchSettings.test.ts` 覆盖，但「对话框
 * 真的渲染出来了、文案键都存在、开关真的接在设置上」属于接线，纯函数测不到。
 * 上一轮把这份设置放进外观面板时，正是因为接线（面板页签 / 菜单入口）出问题才
 * 被退回来 —— 接线要有断言守着。
 */
// @vitest-environment jsdom
import { describe, expect, it } from "vitest";
import { act } from "react";
import { createRoot } from "react-dom/client";
import { Provider } from "react-redux";
import { configureStore } from "@reduxjs/toolkit";

import { SearchSettingsDialog } from "./SearchSettingsDialog";
import { I18nProvider } from "../../i18n/I18nProvider";
import sessionReducer from "../../features/session/sessionSlice";

/*
 * jsdom 没有 `ResizeObserver`，而 Radix 的下拉（`AppSelect`）在布局副作用里会用它。
 * 与 `KeybindingsDialog.test.tsx` 同一处处理。
 */
class ResizeObserverStub {
    observe() {}
    unobserve() {}
    disconnect() {}
}
(globalThis as { ResizeObserver?: unknown }).ResizeObserver ??= ResizeObserverStub;

// React 19 要求显式声明这是 act() 环境，否则每次 act 都会打印一条警告。
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

function mount() {
    const store = configureStore({ reducer: { session: sessionReducer } });
    const container = document.createElement("div");
    document.body.appendChild(container);
    const root = createRoot(container);
    act(() => {
        root.render(
            <Provider store={store}>
                <I18nProvider>
                    <SearchSettingsDialog open onOpenChange={() => {}} />
                </I18nProvider>
            </Provider>,
        );
    });
    return { store, root };
}

/** 对话框内容 portal 到 `<body>`，容器里始终是空的。 */
function bodyText(): string {
    return document.body.textContent ?? "";
}

describe("SearchSettingsDialog", () => {
    it("渲染标题、说明与全部开关", () => {
        mount();
        const text = bodyText();
        // 标题（词典键解析成功才会有这段文字；键缺失时界面显示的是键名本身）。
        expect(text).toContain("Search and matching");
        expect(text).not.toContain("search_settings_title");
        expect(text).not.toContain("search_translit_heteronym");
        // 四个开关的标签都在。
        expect(text).toContain("Heteronyms");
        expect(text).toContain("Japanese long vowels");
        expect(text).toContain("Korean choseong");
        expect(text).toContain("Show why a result matched");
    });

    it("模式下拉展示的是生效模式（总开关关闭时显示 Off）", () => {
        const { store } = mount();
        // 默认开启、smart。
        expect(store.getState().session.searchSettings.translit).toBe(true);
        expect(bodyText()).toContain("Smart (pinyin and romaji)");

        act(() => {
            store.dispatch({ type: "session/setSearchSettings", payload: { translit: false } });
        });
        // 总开关关掉后，即使 mode 仍是 smart，界面也必须读作 Off。
        expect(bodyText()).toContain("Off (literal only)");
    });
});
