// @vitest-environment jsdom
import { configureStore } from "@reduxjs/toolkit";
import { act, useEffect, useState } from "react";
import { createRoot, type Root } from "react-dom/client";
import { Provider } from "react-redux";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

import keybindingsReducer from "../../../features/keybindings/keybindingsSlice";
import { I18nProvider } from "../../../i18n/I18nProvider";
import { AppThemeProvider } from "../../../theme/AppThemeProvider";
import { FadeContextMenu } from "./FadeContextMenu";

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

function MenuHarness({ onDirChange }: { onDirChange: () => void }): React.ReactElement {
    const [ready, setReady] = useState(false);
    useEffect(() => setReady(true), []);
    return ready ? (
        <FadeContextMenu
            x={10}
            y={10}
            primary={{
                clipId: "clip-1",
                isOut: false,
                shape: 1,
                dir: 0.2,
                s: 0.3,
                lengthSec: 1,
                hostFades: {
                    curve_mode: "continuous",
                    in_curvature: 0.2,
                    out_curvature: 0,
                    in_s: 0.3,
                    out_s: 0,
                },
            }}
            onClose={() => undefined}
            onShapeChange={() => undefined}
            onDirChange={onDirChange}
            onSChange={() => undefined}
        />
    ) : (
        <span />
    );
}

function renderMenu(onDirChange = vi.fn()): void {
    const store = configureStore({ reducer: { keybindings: keybindingsReducer } });
    act(() => {
        root.render(
            <Provider store={store}>
                <I18nProvider>
                    <AppThemeProvider>
                        <MenuHarness onDirChange={onDirChange} />
                    </AppThemeProvider>
                </I18nProvider>
            </Provider>,
        );
    });
}

test("菜单挂在 .radix-themes 内，滑块头能拿到主题令牌", () => {
    renderMenu();
    const menu = document.querySelector<HTMLElement>("[data-hs-fade-context-menu]");
    expect(menu).toBeTruthy();
    expect(menu?.closest(".radix-themes")).not.toBeNull();
    expect(menu?.querySelector(".rt-SliderThumb")).toBeTruthy();
    expect(menu?.querySelector("input[type='range']")).toBeNull();
});

test("continuous 宿主同时渲染曲率与 S 两根 AppSlider", () => {
    renderMenu();
    const menu = document.querySelector<HTMLElement>("[data-hs-fade-context-menu]")!;
    expect(menu.querySelectorAll("[role='slider']").length).toBe(2);
    expect(menu.textContent).toContain("S");
});

test("方向键按 0.01 步长上报曲率", () => {
    const onDirChange = vi.fn();
    renderMenu(onDirChange);
    const thumb = document.querySelector<HTMLElement>("[role='slider']")!;
    act(() => {
        thumb.focus();
        thumb.dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowRight", bubbles: true }));
    });
    expect(onDirChange).toHaveBeenCalled();
    expect(onDirChange.mock.calls.at(-1)?.[2]).toBeCloseTo(0.21, 6);
});

test("源码不再保留裸 range 与豁免标记", async () => {
    const source = await import("node:fs").then((fs) =>
        fs.readFileSync("src/components/layout/timeline/FadeContextMenu.tsx", "utf8"),
    );
    expect(source).not.toContain('type="range"');
    expect(source).not.toContain("hs-interaction-exempt");
    expect(source).toContain("menuPortalContainer()");
});
