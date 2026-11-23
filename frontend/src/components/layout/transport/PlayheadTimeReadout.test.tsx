// @vitest-environment jsdom
import { configureStore } from "@reduxjs/toolkit";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { Provider } from "react-redux";
import { afterEach, beforeEach, expect, test } from "vitest";

import sessionReducer from "../../../features/session/sessionSlice";
import { I18nProvider } from "../../../i18n/I18nProvider";
import { PlayheadTimeReadout } from "./PlayheadTimeReadout";

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

type Variant = "toolbar" | "statusBar" | "trackHeader" | "detached";

function renderReadout(variant: Variant, compact = false): void {
    const store = configureStore({ reducer: { session: sessionReducer } });
    act(() => {
        root.render(
            <Provider store={store}>
                <I18nProvider>
                    <PlayheadTimeReadout
                        variant={variant}
                        compact={compact}
                        interpolate={
                            variant === "detached" || variant === "trackHeader" ? false : undefined
                        }
                    />
                </I18nProvider>
            </Provider>,
        );
    });
}

test("toolbar 读数挂载并保留 data-playhead-sec", () => {
    renderReadout("toolbar");
    const readout = document.querySelector<HTMLElement>("[data-playhead-time-readout='toolbar']");
    expect(readout).toBeTruthy();
    expect(readout?.getAttribute("data-playhead-sec")).toBe("0.000000");
    expect(readout?.textContent?.length).toBeGreaterThan(0);
});

/**
 * 【为什么这条是用户报障的可执行形式】副单位此前用 `text-qt-text-muted`，被压到近
 * 背景色 —— 用户看到的是"副单位不见了"。次要由**分隔符**表达，不由降低对比度表达。
 */
test("状态栏读数：单行、副单位不再变淡", () => {
    renderReadout("statusBar");
    const readout = document.querySelector<HTMLElement>("[data-playhead-time-readout='statusBar']");
    expect(readout).toBeTruthy();
    // 单行：容器必须是横向 flex，不能是 flex-col。
    expect(readout?.className).toContain("items-center");
    expect(readout?.className).not.toContain("flex-col");
    expect(readout?.querySelector(".text-qt-text-muted")).toBeNull();
    const spans = readout?.querySelectorAll("span") ?? [];
    const classes = Array.from(spans).map((span) => span.className);
    expect(classes.some((value) => value.includes("text-qt-text-muted"))).toBe(false);
});

test("compact 轨道头读数只显示主单位，不渲染副单位", () => {
    renderReadout("trackHeader", true);
    const readout = document.querySelector<HTMLElement>(
        "[data-playhead-time-readout='trackHeader']",
    );
    expect(readout).toBeTruthy();
    expect(readout?.textContent).not.toContain("·");
});

test("detached 读数使用非插值模式也能渲染主单位", () => {
    renderReadout("detached", true);
    const readout = document.querySelector<HTMLElement>("[data-playhead-time-readout='detached']");
    expect(readout).toBeTruthy();
    expect(readout?.textContent?.length).toBeGreaterThan(0);
    expect(readout?.textContent).not.toContain("·");
});

test("ActionBar 通过槽位挂载工具栏读数（不再无条件渲染）", async () => {
    const source = await import("node:fs").then((fs) =>
        fs.readFileSync("src/components/layout/ActionBar.tsx", "utf8"),
    );
    expect(source).toContain('<PlayheadTimeReadoutSlot slot="toolbar" />');
    expect(source).not.toContain('<PlayheadTimeReadout variant="toolbar" />');
});

/**
 * 【回归】读数曾经**替换**掉"轨道"标签：开关一开，用户再也看不到这一列是什么。
 * 现在标签永远在，读数只是标签旁的附加槽位。
 */
test("TrackList 始终渲染“轨道”标签，读数作为附加槽位", async () => {
    const source = await import("node:fs").then((fs) =>
        fs.readFileSync("src/components/layout/timeline/TrackList.tsx", "utf8"),
    );
    expect(source).toContain('{t("common_tracks")}');
    expect(source).toContain('<PlayheadTimeReadoutSlot slot="trackHeader" compact />');
    expect(source).not.toContain('<PlayheadTimeReadout variant="trackHeader"');
});

test("状态栏通过槽位挂载读数", async () => {
    const source = await import("node:fs").then((fs) => fs.readFileSync("src/App.tsx", "utf8"));
    expect(source).toContain('<PlayheadTimeReadoutSlot slot="statusBar" />');
});
