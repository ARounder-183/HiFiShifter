// @vitest-environment jsdom
/**
 * 播放光标时间**放置槽位**的回归测试。
 *
 * 【为什么这些用例存在】读数在工具栏、状态栏、轨道头三处都是 chrome，而"此刻该在
 * 哪儿"是用户偏好。旧实现里工具栏无条件渲染一份、轨道头再渲染一份 —— 用户既挪不走
 * 也关不掉。这里钉住：**同一时刻只有一个主实例**，且解析结果与槽位严格对应。
 *
 * 【为什么还要钉"顺延"】`auto` 曾经被写死成状态栏 —— 而状态栏在主窗口里永远存在，
 * 这条链于是永远走不到第二级，"自动"退化成"状态栏"的别名。用例里**只挂状态栏槽位**
 * 的那一条就是这条链的可执行形式。
 */
import { configureStore } from "@reduxjs/toolkit";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { Provider } from "react-redux";
import { afterEach, beforeEach, expect, test } from "vitest";

import sessionReducer, { setPlayheadTimePlacement } from "../../../features/session/sessionSlice";
import type { PlayheadTimePlacement } from "../../../features/session/playheadTimePlacement";
import { I18nProvider } from "../../../i18n/I18nProvider";
import { PlayheadTimeReadoutSlot } from "./PlayheadTimeReadoutSlot";

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

type SlotName = "toolbar" | "statusBar" | "trackHeader";

function renderSlots(placement: PlayheadTimePlacement, slots: SlotName[]): void {
    const store = configureStore({ reducer: { session: sessionReducer } });
    store.dispatch(setPlayheadTimePlacement(placement));
    act(() => {
        root.render(
            <Provider store={store}>
                <I18nProvider>
                    {slots.map((slot) => (
                        <PlayheadTimeReadoutSlot key={slot} slot={slot} />
                    ))}
                </I18nProvider>
            </Provider>,
        );
    });
}

function renderAllSlots(placement: PlayheadTimePlacement): void {
    renderSlots(placement, ["toolbar", "statusBar", "trackHeader"]);
}

function mounted(): string[] {
    return Array.from(document.querySelectorAll<HTMLElement>("[data-playhead-time-readout]"))
        .map((node) => node.dataset.playheadTimeReadout ?? "")
        .sort();
}

test("auto 落在工具栏（不是永远存在的状态栏）", () => {
    renderAllSlots("auto");
    expect(mounted()).toEqual(["toolbar"]);
});

test("auto 在工具栏槽位不存在时顺延到状态栏", () => {
    renderSlots("auto", ["statusBar", "trackHeader"]);
    expect(mounted()).toEqual(["statusBar"]);
});

test("toolbar 只在工具栏槽渲染", () => {
    renderAllSlots("toolbar");
    expect(mounted()).toEqual(["toolbar"]);
});

test("statusBar 只在状态栏槽渲染", () => {
    renderAllSlots("statusBar");
    expect(mounted()).toEqual(["statusBar"]);
});

test("trackHeader 只在轨道头槽渲染", () => {
    renderAllSlots("trackHeader");
    expect(mounted()).toEqual(["trackHeader"]);
});

test("hidden 一个都不渲染", () => {
    renderAllSlots("hidden");
    expect(mounted()).toEqual([]);
});

/** 读数右键菜单 = 标尺菜单里"时间显示"那一组（速度映射那一组不适用于读数）。 */
test("右键读数弹出时间显示菜单，位置收在二级子菜单里", () => {
    renderAllSlots("toolbar");
    const readout = document.querySelector<HTMLElement>("[data-playhead-time-readout='toolbar']");
    expect(readout).toBeTruthy();
    act(() => {
        readout?.dispatchEvent(
            new MouseEvent("contextmenu", { bubbles: true, clientX: 12, clientY: 34 }),
        );
    });

    const menu = document.querySelector<HTMLElement>('[data-hs-context-menu="1"]');
    expect(menu).toBeTruthy();
    // 位置必须收在二级菜单：一级菜单里不允许直接出现那五个选项。
    expect(menu?.textContent).not.toContain("Automatic");
    expect(menu?.textContent).not.toContain("Tempo Map");
    expect(menu?.textContent).toContain("Primary");

    const trigger = menu?.querySelector<HTMLElement>('[data-menu-key="playheadTimePlacement"]');
    expect(trigger).toBeTruthy();
    act(() => {
        trigger?.click();
    });

    const panels = [...document.querySelectorAll<HTMLElement>('[data-hs-context-menu="1"]')];
    const submenu = panels.at(-1);
    expect(submenu).not.toBe(menu);
    const entries = [
        ...(submenu?.querySelectorAll<HTMLElement>('[role="menuitemcheckbox"]') ?? []),
    ];
    expect(entries).toHaveLength(5);
    // `auto` 的尾部标注说清它此刻落在哪一处。
    const auto = entries[0];
    expect(auto.textContent).toContain("Automatic");
    expect(auto.textContent).toContain("Toolbar");
});

test("从读数菜单选“隐藏”后读数消失（随后由标尺菜单开回来）", () => {
    renderAllSlots("toolbar");
    const readout = document.querySelector<HTMLElement>("[data-playhead-time-readout='toolbar']");
    act(() => {
        readout?.dispatchEvent(
            new MouseEvent("contextmenu", { bubbles: true, clientX: 12, clientY: 34 }),
        );
    });
    const trigger = document.querySelector<HTMLElement>('[data-menu-key="playheadTimePlacement"]');
    act(() => {
        trigger?.click();
    });
    const panels = [...document.querySelectorAll<HTMLElement>('[data-hs-context-menu="1"]')];
    const hidden = [
        ...(panels.at(-1)?.querySelectorAll<HTMLElement>('[role="menuitemcheckbox"]') ?? []),
    ].find((entry) => entry.textContent?.includes("Hidden"));
    expect(hidden).toBeTruthy();
    act(() => {
        hidden?.click();
    });
    expect(mounted()).toEqual([]);
});
