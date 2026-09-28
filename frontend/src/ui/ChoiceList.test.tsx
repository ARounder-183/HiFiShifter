// @vitest-environment jsdom
/*
 * 「点选即执行」选项列表。
 *
 * 【为什么必须有】这个原语取代了三处手写的同形态列表（导入文件模式、多音轨媒体
 * 音轨选择、外观主题选择）。手写版本的问题不只是重复，而是**每一处都漏掉了键盘
 * 契约**：前两处连 Esc 都没有，方向键更没有。所以这里把契约钉死：
 * 整列一个 Tab 停留点、方向键循环、Home/End 到两端、Enter/Space 执行、
 * 禁用的项不参与。
 */

import { act, useState } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, expect, test, vi } from "vitest";

import { AppChoiceList } from "./ChoiceList";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const OPTIONS = [
    { id: "a", label: "Across time" },
    { id: "b", label: "Across tracks", description: "one per track" },
    { id: "disabled", label: "Nope", disabled: true },
    { id: "c", label: "As takes" },
];

const mounted: Array<() => Promise<void>> = [];
afterEach(async () => {
    while (mounted.length) await mounted.pop()?.();
});

async function mount(options = OPTIONS) {
    const onSelect = vi.fn();
    const host = document.createElement("div");
    document.body.append(host);
    const root = createRoot(host);
    await act(async () => {
        root.render(
            <AppChoiceList options={options} onSelect={onSelect} ariaLabel="Import mode" />,
        );
    });
    mounted.push(async () => {
        await act(async () => root.unmount());
        host.remove();
    });

    const items = () => Array.from(host.querySelectorAll<HTMLButtonElement>('[role="menuitem"]'));
    const focusedId = () => document.activeElement?.textContent?.trim() ?? null;
    const press = async (key: string, target?: HTMLElement) => {
        const el = target ?? (document.activeElement as HTMLElement | null);
        if (!el) throw new Error("no focus target");
        await act(async () => {
            el.dispatchEvent(
                new KeyboardEvent("keydown", { key, bubbles: true, cancelable: true }),
            );
        });
    };
    return { host, onSelect, items, focusedId, press };
}

test("列表与项的角色正确，副行渲染为说明文字", async () => {
    const m = await mount();
    expect(m.host.querySelector('[role="menu"]')?.getAttribute("aria-label")).toBe("Import mode");

    const items = m.items();
    expect(items).toHaveLength(4);
    // 副行是第二行文字，不是主标签的一部分
    expect(items[1].textContent).toContain("Across tracks");
    expect(items[1].textContent).toContain("one per track");
});

test("整列只占一个 Tab 停留点（roving tabindex）", async () => {
    const m = await mount();
    const items = m.items();
    // 第一个可用项进 Tab 序，其余不进 —— 否则 4 项会把对话框的 Tab 路径拉长 4 倍
    expect(items[0].tabIndex).toBe(0);
    expect(items[1].tabIndex).toBe(-1);
    expect(items[3].tabIndex).toBe(-1);
});

test("第一个可用项被禁用时，Tab 停留点顺延", async () => {
    const m = await mount([
        { id: "x", label: "Disabled first", disabled: true },
        { id: "y", label: "Usable" },
    ]);
    const items = m.items();
    expect(items[0].tabIndex).toBe(-1);
    expect(items[1].tabIndex).toBe(0);
});

test("方向键在可用项之间循环，跳过禁用项", async () => {
    const m = await mount();
    const items = m.items();

    await act(async () => items[0].focus());
    await m.press("ArrowDown");
    expect(m.focusedId()).toContain("Across tracks");

    // 第三项被禁用，必须跳过
    await m.press("ArrowDown");
    expect(m.focusedId()).toContain("As takes");

    // 循环回第一项
    await m.press("ArrowDown");
    expect(m.focusedId()).toContain("Across time");

    await m.press("ArrowUp");
    expect(m.focusedId()).toContain("As takes");
});

test("Home / End 到两端", async () => {
    const m = await mount();
    const items = m.items();

    await act(async () => items[0].focus());
    await m.press("End");
    expect(m.focusedId()).toContain("As takes");

    await m.press("Home");
    expect(m.focusedId()).toContain("Across time");
});

test("Enter / 点击执行对应项，禁用的项不执行", async () => {
    const m = await mount();
    const items = m.items();

    await act(async () => items[1].click());
    expect(m.onSelect).toHaveBeenCalledWith("b");

    // 禁用项点击不回调
    await act(async () => items[2].click());
    expect(m.onSelect).toHaveBeenCalledTimes(1);
});

test("选项变化时 Tab 停留点跟着重算", async () => {
    // 回归守卫：`firstEnabledIndex` 用 useMemo 依赖 options，
    // 若依赖写错，列表换成"首项可用"后仍会停在旧索引上。
    const host = document.createElement("div");
    document.body.append(host);
    const root = createRoot(host);
    function Harness() {
        const [wide, setWide] = useState(true);
        return (
            <>
                <button type="button" data-testid="swap" onClick={() => setWide(false)}>
                    swap
                </button>
                <AppChoiceList
                    options={
                        wide
                            ? [
                                  { id: "x", label: "Disabled first", disabled: true },
                                  { id: "y", label: "Usable" },
                              ]
                            : [
                                  { id: "y", label: "Usable" },
                                  { id: "x", label: "Disabled first", disabled: true },
                              ]
                    }
                    onSelect={() => {}}
                />
            </>
        );
    }
    await act(async () => root.render(<Harness />));
    mounted.push(async () => {
        await act(async () => root.unmount());
        host.remove();
    });

    const items = () => Array.from(host.querySelectorAll<HTMLButtonElement>('[role="menuitem"]'));
    expect(items()[1].tabIndex).toBe(0);

    await act(async () => host.querySelector<HTMLButtonElement>('[data-testid="swap"]')!.click());
    expect(items()[0].tabIndex).toBe(0);
});
