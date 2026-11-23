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

function renderReadout(variant: "toolbar" | "trackHeader" | "detached", compact = false): void {
    const store = configureStore({ reducer: { session: sessionReducer } });
    act(() => {
        root.render(
            <Provider store={store}>
                <I18nProvider>
                    <PlayheadTimeReadout
                        variant={variant}
                        compact={compact}
                        interpolate={variant === "detached" ? false : undefined}
                    />
                </I18nProvider>
            </Provider>,
        );
    });
}

test("toolbar 读数始终挂载并保留 data-playhead-sec", () => {
    renderReadout("toolbar");
    const readout = document.querySelector<HTMLElement>("[data-playhead-time-readout='toolbar']");
    expect(readout).toBeTruthy();
    expect(readout?.getAttribute("data-playhead-sec")).toBe("0.000000");
    expect(readout?.textContent?.length).toBeGreaterThan(0);
});

test("compact 轨道头读数只显示主单位，不渲染副单位斜杠", () => {
    renderReadout("trackHeader", true);
    const readout = document.querySelector<HTMLElement>(
        "[data-playhead-time-readout='trackHeader']",
    );
    expect(readout).toBeTruthy();
    expect(readout?.textContent).not.toContain("/");
});

test("detached 读数使用非插值模式也能渲染主单位", () => {
    renderReadout("detached", true);
    const readout = document.querySelector<HTMLElement>("[data-playhead-time-readout='detached']");
    expect(readout).toBeTruthy();
    expect(readout?.textContent?.length).toBeGreaterThan(0);
    expect(readout?.textContent).not.toContain("/");
});

test("ActionBar 的 BPM & Time 组包含全局传输时间", async () => {
    const source = await import("node:fs").then((fs) =>
        fs.readFileSync("src/components/layout/ActionBar.tsx", "utf8"),
    );
    expect(source).toContain('<PlayheadTimeReadout variant="toolbar" />');
});

test("TrackList 显示时间时不再渲染“轨道”标签", async () => {
    const source = await import("node:fs").then((fs) =>
        fs.readFileSync("src/components/layout/timeline/TrackList.tsx", "utf8"),
    );
    expect(source).toContain("showPlayheadTime ?");
    expect(source).toContain('variant="trackHeader"');
    expect(source).toContain("interpolate={false}");
});
