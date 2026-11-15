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

test("图标随常显设置切换实心 / 空心", async () => {
    await render(false);
    const outline = host.querySelector<SVGSVGElement>(
        '[data-hs-param-blocks-always-show-icon="outline"]',
    );
    expect(outline, "关闭时应渲染空心图标").not.toBeNull();
    const outlineBars = Array.from(outline!.querySelectorAll("rect"));
    expect(outlineBars).toHaveLength(3);
    for (const bar of outlineBars) {
        expect(bar.getAttribute("fill")).toBe("none");
        expect(bar.getAttribute("stroke")).toBe("currentColor");
        // 细描边：1.1px 曾让空心条看起来像带粗框的实心块。
        expect(bar.getAttribute("stroke-width")).toBe("0.8");
    }

    await render(true);
    const solid = host.querySelector<SVGSVGElement>(
        '[data-hs-param-blocks-always-show-icon="solid"]',
    );
    expect(solid, "开启时应渲染实心图标").not.toBeNull();
    expect(host.querySelector('[data-hs-param-blocks-always-show-icon="outline"]')).toBeNull();
    const solidBars = Array.from(solid!.querySelectorAll("rect"));
    expect(solidBars).toHaveLength(3);
    for (const bar of solidBars) {
        expect(bar.getAttribute("fill")).toBe("currentColor");
        expect(bar.hasAttribute("stroke")).toBe(false);
    }
    // 同一份几何：切换只换填充形式，不让三条分段条发生位移或宽度漂移。
    expect(solidBars.map((bar) => bar.getAttribute("width"))).toEqual(
        outlineBars.map((bar) => bar.getAttribute("width")),
    );
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
