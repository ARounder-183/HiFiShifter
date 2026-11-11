// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import { Theme } from "@radix-ui/themes";
import { ParamBlockLayerToggle } from "./ParamBlockLayerToggle";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

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

const labels: Record<string, string> = {
    param_blocks_layer_show: "显示参数块",
    param_blocks_layer_hide: "隐藏参数块",
};

const tf = (key: string): string => labels[key] ?? key;

async function render(visible: boolean, toggle = vi.fn()) {
    await act(async () => {
        root.render(
            <Theme>
                <ParamBlockLayerToggle
                    m={{ paramBlocksVisible: visible, toggleParamBlocksVisible: toggle, tf }}
                />
            </Theme>,
        );
    });
    return toggle;
}

test("眼睛的 aria-pressed 与 tooltip 跟随图层可见性", async () => {
    await render(false);
    const hidden = host.querySelector<HTMLButtonElement>('[aria-label="显示参数块"]');
    expect(hidden).not.toBeNull();
    expect(hidden?.getAttribute("aria-pressed")).toBe("false");

    await render(true);
    const shown = host.querySelector<HTMLButtonElement>('[aria-label="隐藏参数块"]');
    expect(shown).not.toBeNull();
    expect(shown?.getAttribute("aria-pressed")).toBe("true");
});

test("点击眼睛调用图层开关", async () => {
    const toggle = await render(true);
    const eye = host.querySelector<HTMLButtonElement>('[aria-label="隐藏参数块"]');
    if (eye === null) throw new Error("layer eye missing");
    await act(async () => eye.click());
    expect(toggle).toHaveBeenCalledTimes(1);
});
