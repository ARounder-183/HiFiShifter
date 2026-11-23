// @vitest-environment jsdom
/**
 * 播放光标时间**放置槽位**的回归测试。
 *
 * 【为什么这些用例存在】读数在工具栏、状态栏、轨道头三处都是 chrome，而"此刻该在
 * 哪儿"是用户偏好。旧实现里工具栏无条件渲染一份、轨道头再渲染一份 —— 用户既挪不走
 * 也关不掉。这里钉住：**同一时刻只有一个主实例**，且解析结果与槽位严格对应。
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

function renderAllSlots(placement: PlayheadTimePlacement): void {
    const store = configureStore({ reducer: { session: sessionReducer } });
    store.dispatch(setPlayheadTimePlacement(placement));
    act(() => {
        root.render(
            <Provider store={store}>
                <I18nProvider>
                    <PlayheadTimeReadoutSlot slot="toolbar" />
                    <PlayheadTimeReadoutSlot slot="statusBar" />
                    <PlayheadTimeReadoutSlot slot="trackHeader" compact />
                </I18nProvider>
            </Provider>,
        );
    });
}

function mounted(): string[] {
    return Array.from(document.querySelectorAll<HTMLElement>("[data-playhead-time-readout]"))
        .map((node) => node.dataset.playheadTimeReadout ?? "")
        .sort();
}

test("auto 解析为状态栏：只有一个主实例", () => {
    renderAllSlots("auto");
    expect(mounted()).toEqual(["statusBar"]);
});

test("toolbar 只在工具栏槽渲染", () => {
    renderAllSlots("toolbar");
    expect(mounted()).toEqual(["toolbar"]);
});

test("trackHeader 只在轨道头槽渲染", () => {
    renderAllSlots("trackHeader");
    expect(mounted()).toEqual(["trackHeader"]);
});

test("hidden 一个都不渲染", () => {
    renderAllSlots("hidden");
    expect(mounted()).toEqual([]);
});
