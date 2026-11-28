// @vitest-environment jsdom
/**
 * 标尺右键菜单里的"读数位置"入口。
 *
 * 【为什么标尺也必须有】读数的右键菜单里可以选"隐藏"。一旦隐藏，读数本身就不再
 * 渲染 —— 用户若只从读数那里见过这个选项，就再也找不到开回来的地方。"时间轴显示
 * 设置"对话框里虽然有同一项，但那是设置页，与"我刚才在这儿关掉了它"建立不了直接
 * 联系。因此**两个入口都必须有**，且必须给出同一组选项（同一个构建器）。
 */
import { configureStore } from "@reduxjs/toolkit";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { Provider } from "react-redux";
import { afterEach, beforeEach, expect, test } from "vitest";

import keybindingsReducer from "../../../features/keybindings/keybindingsSlice";
import sessionReducer from "../../../features/session/sessionSlice";
import { I18nProvider } from "../../../i18n/I18nProvider";
import { TimeRuler } from "./TimeRuler";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
class ResizeObserverStub {
    observe(): void {}
    unobserve(): void {}
    disconnect(): void {}
}
(globalThis as { ResizeObserver?: unknown }).ResizeObserver ??= ResizeObserverStub;

let host: HTMLDivElement;
let root: Root;

beforeEach(() => {
    host = document.createElement("div");
    document.body.append(host);
    root = createRoot(host);
});

afterEach(() => {
    act(() => root.unmount());
    document.body.innerHTML = "";
});

/** 文案用恒等 `t`：断言直接对着 i18n 键，不依赖当前语言。 */
function renderRuler(): void {
    const store = configureStore({
        reducer: { session: sessionReducer, keybindings: keybindingsReducer },
    });
    act(() => {
        root.render(
            <Provider store={store}>
                <I18nProvider>
                    <TimeRuler
                        scrollLeft={0}
                        ticks={[]}
                        pxPerSec={100}
                        playheadSec={0}
                        onMouseDown={() => undefined}
                        timeContext={{ bpm: 120, beatsPerBar: 4, grid: "1/4", tempoMap: null }}
                        primaryUnit="barBeats"
                        secondaryUnit="clock"
                        t={(key) => key}
                    />
                </I18nProvider>
            </Provider>,
        );
    });
}

function openRulerMenu(): HTMLElement {
    const ruler = host.firstElementChild as HTMLElement | null;
    expect(ruler).toBeTruthy();
    act(() => {
        ruler?.dispatchEvent(
            new MouseEvent("contextmenu", { bubbles: true, clientX: 20, clientY: 10 }),
        );
    });
    const menu = document.querySelector<HTMLElement>('[data-hs-context-menu="1"]');
    expect(menu).toBeTruthy();
    return menu as HTMLElement;
}

test("标尺菜单同时给出速度映射与时间显示两组", () => {
    renderRuler();
    const menu = openRulerMenu();
    expect(menu.textContent).toContain("tempo_map");
    expect(menu.textContent).toContain("time_unit_primary");
    expect(menu.textContent).toContain("time_unit_secondary");
});

test("标尺菜单里的读数位置收在二级子菜单，五项与读数菜单一致", () => {
    renderRuler();
    const menu = openRulerMenu();
    // 一级菜单里不允许平铺那五个选项。
    expect(menu.textContent).not.toContain("playhead_time_placement_hidden");

    const trigger = menu.querySelector<HTMLElement>('[data-menu-key="playheadTimePlacement"]');
    expect(trigger).toBeTruthy();
    expect(trigger?.textContent).toContain("playhead_time_placement");
    act(() => {
        trigger?.click();
    });

    const panels = [...document.querySelectorAll<HTMLElement>('[data-hs-context-menu="1"]')];
    const entries = [
        ...(panels.at(-1)?.querySelectorAll<HTMLElement>('[role="menuitemcheckbox"]') ?? []),
    ];
    expect(entries).toHaveLength(5);
    ["auto", "toolbar", "status_bar", "track_header", "hidden"].forEach((key, index) => {
        expect(entries[index].textContent).toContain(`playhead_time_placement_${key}`);
    });
    // auto 的尾部标注说明它此刻落在哪一处（没有读数槽位挂载时是链首：工具栏）。
    expect(entries[0].textContent).toContain("playhead_time_placement_toolbar");
});
