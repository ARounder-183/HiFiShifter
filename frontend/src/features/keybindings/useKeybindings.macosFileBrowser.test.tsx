// @vitest-environment jsdom
import { configureStore } from "@reduxjs/toolkit";
import { act } from "react";
import { createRoot } from "react-dom/client";
import { Provider } from "react-redux";
import { afterAll, afterEach, beforeEach, expect, test, vi } from "vitest";

import keybindingsReducer from "./keybindingsSlice";
import sessionReducer from "../session/sessionSlice";
import { useKeybindings } from "./useKeybindings";
import type { ActionId } from "./types";
import {
    installFocusSurfaceTracking,
    resetActiveSurfaceForTests,
    setActiveSurfaceExplicit,
} from "../uiFocus/focusSurface";

vi.mock("../../utils/platform", async (importOriginal) => {
    const actual = await importOriginal<typeof import("../../utils/platform")>();
    return { ...actual, IS_MAC: true };
});

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const fired: ActionId[] = [];
const disposeFocus = installFocusSurfaceTracking();

function Harness() {
    useKeybindings((actionId) => fired.push(actionId));
    return null;
}

let root: ReturnType<typeof createRoot> | null = null;

beforeEach(() => {
    fired.length = 0;
    resetActiveSurfaceForTests();
    const host = document.createElement("div");
    document.body.append(host);
    const store = configureStore({
        reducer: { keybindings: keybindingsReducer, session: sessionReducer },
    });
    root = createRoot(host);
    act(() => {
        root?.render(
            <Provider store={store}>
                <Harness />
            </Provider>,
        );
    });
});

afterEach(() => {
    act(() => root?.unmount());
    root = null;
    document.body.innerHTML = "";
});

test("macOS 文件列表拥有 ⌘+⌫（不能被 track.delete 截走）", () => {
    const row = document.createElement("div");
    row.setAttribute("data-hs-surface", "fileBrowser");
    row.setAttribute("data-hs-typeahead", "");
    row.tabIndex = 0;
    document.body.append(row);
    row.focus();
    setActiveSurfaceExplicit("fileBrowser");

    const event = new KeyboardEvent("keydown", {
        key: "Backspace",
        code: "Backspace",
        bubbles: true,
        cancelable: true,
        metaKey: true,
    });
    act(() => row.dispatchEvent(event));
    expect(fired).toEqual([]);
    expect(event.defaultPrevented).toBe(false);
});

test("macOS 文件列表拥有普通 ⌫（不能被 edit.initialize 截走）", () => {
    const row = document.createElement("div");
    row.setAttribute("data-hs-surface", "fileBrowser");
    row.setAttribute("data-hs-typeahead", "");
    row.tabIndex = 0;
    document.body.append(row);
    row.focus();
    setActiveSurfaceExplicit("fileBrowser");
    const event = new KeyboardEvent("keydown", {
        key: "Backspace",
        code: "Backspace",
        bubbles: true,
        cancelable: true,
    });
    act(() => row.dispatchEvent(event));
    expect(fired).toEqual([]);
    expect(event.defaultPrevented).toBe(false);
});

afterAll(() => disposeFocus());
