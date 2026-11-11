// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import { Theme } from "@radix-ui/themes";
import { ParamBlockAlwaysShowToggle } from "./ParamBlockAlwaysShowToggle";

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
    param_blocks_always_show: "永远显示参数块",
    param_blocks_always_show_tooltip: "参数块选择模式下一律显示；启用后，其他工具下也显示参数块。",
};

const tf = (key: string): string => labels[key] ?? key;

async function render(value: boolean, toggle = vi.fn()) {
    await act(async () => {
        root.render(
            <Theme>
                <ParamBlockAlwaysShowToggle
                    m={{ alwaysShowParamBlocks: value, toggleAlwaysShowParamBlocks: toggle, tf }}
                />
            </Theme>,
        );
    });
    return toggle;
}

/**
 * `aria-pressed` 必须读**设置值**，不能读派生可见性。
 *
 * 块选择模式下图层总是可见 —— 若按钮跟着派生可见性亮起，用户会看到「已开启」，
 * 而点它时画面毫无变化（因为设置本来就是关的）。
 */
test("aria-pressed 跟随设置值，而不是图层是否可见", async () => {
    await render(false);
    const off = host.querySelector<HTMLButtonElement>('[aria-label="永远显示参数块"]');
    expect(off).not.toBeNull();
    expect(off?.getAttribute("aria-pressed")).toBe("false");

    await render(true);
    const on = host.querySelector<HTMLButtonElement>('[aria-label="永远显示参数块"]');
    expect(on?.getAttribute("aria-pressed")).toBe("true");
});

test("文案只有一句，不随状态切换", async () => {
    await render(false);
    expect(host.querySelectorAll('[aria-label="永远显示参数块"]').length).toBe(1);
    await render(true);
    expect(host.querySelectorAll('[aria-label="永远显示参数块"]').length).toBe(1);
});

test("点击调用常显开关一次", async () => {
    const toggle = await render(true);
    const button = host.querySelector<HTMLButtonElement>('[aria-label="永远显示参数块"]');
    if (button === null) throw new Error("always-show button missing");
    await act(async () => button.click());
    expect(toggle).toHaveBeenCalledTimes(1);
});
