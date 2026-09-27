// @vitest-environment jsdom
/*
 * 原语层契约测试。锁定的都是"此前不存在或此前漂移"的行为：
 *
 *   - `AppButton` 的语义 → Radix 变体映射（此前同一个语义有 6 种写法）；
 *   - `AppContextMenu` 的**键盘导航**（此前 12 个手写菜单没有一个支持方向键）；
 *   - `AppField` 的标签宽度只允许三档（此前有 12 个魔法值）。
 */

import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

import { AppButton } from "./Button";
import { AppField, AppForm } from "./Field";
import { AppContextMenu, type AppMenuItemSpec } from "./Menu";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let container: HTMLDivElement;
let root: Root;

beforeEach(() => {
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
});

afterEach(() => {
    act(() => root.unmount());
    document.body.innerHTML = "";
});

function render(node: React.ReactNode) {
    return act(async () => {
        root.render(node);
    });
}

test("AppButton 语义映射：primary 走 solid、默认走 soft+gray、danger 走 soft+red", async () => {
    await render(
        <div>
            <AppButton intent="primary">P</AppButton>
            <AppButton>D</AppButton>
            <AppButton intent="subtle">S</AppButton>
            <AppButton intent="danger">X</AppButton>
        </div>,
    );

    const buttons = Array.from(container.querySelectorAll("button"));
    const [primary, fallback, subtle, danger] = buttons;

    expect(primary.className).toContain("rt-variant-solid");
    expect(fallback.className).toContain("rt-variant-soft");
    expect(subtle.className).toContain("rt-variant-ghost");
    expect(danger.className).toContain("rt-variant-soft");
    // Radix 用 `data-accent-color` 承载颜色（而非 class）
    expect(fallback.getAttribute("data-accent-color")).toBe("gray");
    expect(danger.getAttribute("data-accent-color")).toBe("red");
});

test("AppButton 默认尺寸为 sm（对齐 Radix size=1），不写 size 也是 24px 档", async () => {
    await render(<AppButton>x</AppButton>);
    const button = container.querySelector("button")!;
    expect(button.className).toContain("rt-r-size-1");
});

test("AppField 的标签宽度只允许三档令牌值", async () => {
    await render(
        <AppForm labelWidth="lg">
            <AppField label="A">
                <input />
            </AppField>
        </AppForm>,
    );
    const label = container.querySelector("label")!;
    expect(label.style.minWidth).toBe("132px");
});

test("AppField 单行覆盖优先于表单级设置", async () => {
    await render(
        <AppForm labelWidth="lg">
            <AppField label="A" labelWidth="sm">
                <input />
            </AppField>
        </AppForm>,
    );
    expect(container.querySelector("label")!.style.minWidth).toBe("80px");
});

/** 让菜单拿到非零坐标，便于断言定位。 */
const ITEMS: AppMenuItemSpec[] = [
    { key: "a", label: "Alpha", onSelect: () => {} },
    { key: "b", label: "Beta", onSelect: () => {} },
    { key: "c", label: "Gamma", onSelect: () => {} },
];

function press(key: string) {
    return act(async () => {
        document.dispatchEvent(new KeyboardEvent("keydown", { key, bubbles: true, cancelable: true }));
    });
}

test("AppContextMenu：方向键改变高亮项，Enter 触发当前高亮项并关闭", async () => {
    const onSelectBeta = vi.fn();
    const onClose = vi.fn();

    await render(
        <AppContextMenu
            x={10}
            y={10}
            ariaLabel="test"
            items={[
                { key: "a", label: "Alpha", onSelect: () => {} },
                { key: "b", label: "Beta", onSelect: onSelectBeta },
                { key: "c", label: "Gamma", onSelect: () => {} },
            ]}
            onClose={onClose}
        />,
    );

    await press("ArrowDown"); // → Alpha
    await press("ArrowDown"); // → Beta
    await press("Enter");

    expect(onSelectBeta).toHaveBeenCalledTimes(1);
    expect(onClose).toHaveBeenCalledTimes(1);
});

test("AppContextMenu：ArrowUp 从末尾回绕，Home/End 跳到首尾", async () => {
    const onSelectGamma = vi.fn();
    await render(
        <AppContextMenu
            x={10}
            y={10}
            items={[
                { key: "a", label: "Alpha", onSelect: () => {} },
                { key: "b", label: "Beta", onSelect: () => {} },
                { key: "c", label: "Gamma", onSelect: onSelectGamma },
            ]}
            onClose={() => {}}
        />,
    );

    await press("End");
    await press("Enter");
    expect(onSelectGamma).toHaveBeenCalledTimes(1);
});

test("AppContextMenu：禁用项被方向键跳过", async () => {
    const onSelectGamma = vi.fn();
    await render(
        <AppContextMenu
            x={10}
            y={10}
            items={[
                { key: "a", label: "Alpha", disabled: true, onSelect: () => {} },
                { key: "c", label: "Gamma", onSelect: onSelectGamma },
            ]}
            onClose={() => {}}
        />,
    );

    await press("ArrowDown"); // 跳过 Alpha，落到 Gamma
    await press("Enter");
    expect(onSelectGamma).toHaveBeenCalledTimes(1);
});

test("AppContextMenu：Esc 关闭且不触发任何项", async () => {
    const onSelect = vi.fn();
    const onClose = vi.fn();
    await render(
        <AppContextMenu
            x={10}
            y={10}
            items={[{ key: "a", label: "Alpha", onSelect }]}
            onClose={onClose}
        />,
    );

    await press("Escape");
    expect(onClose).toHaveBeenCalledTimes(1);
    expect(onSelect).not.toHaveBeenCalled();
});

test("AppContextMenu：外部指针按下关闭，内部按下不关闭", async () => {
    const onClose = vi.fn();
    await render(
        <AppContextMenu
            x={10}
            y={10}
            items={[{ key: "a", label: "Alpha", onSelect: () => {} }]}
            onClose={onClose}
        />,
    );

    const menu = container.querySelector('[role="menu"]')!;
    await act(async () => {
        menu.dispatchEvent(new PointerEvent("pointerdown", { bubbles: true }));
    });
    expect(onClose).not.toHaveBeenCalled();

    await act(async () => {
        document.body.dispatchEvent(new PointerEvent("pointerdown", { bubbles: true }));
    });
    expect(onClose).toHaveBeenCalledTimes(1);
});

test("AppContextMenu 暴露 role=menu / menuitem，供屏幕阅读器识别", async () => {
    await render(<AppContextMenu x={10} y={10} items={ITEMS} onClose={() => {}} />);
    expect(container.querySelector('[role="menu"]')).not.toBeNull();
    expect(container.querySelectorAll('[role="menuitem"]')).toHaveLength(3);
});
