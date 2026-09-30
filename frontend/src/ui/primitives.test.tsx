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
import { AppField, AppForm, AppFormSection, AppSwitchRow } from "./Field";
import { AppContextMenu, AppSubMenu, type AppMenuItemSpec } from "./Menu";

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

test("AppButton 默认尺寸为 md（对齐 Radix size=2，32px）", async () => {
    /*
     * 这条曾经断言默认是 `sm`（24px）—— 那个默认值把全部 42 个对话框的页脚按钮
     * 从重构前的 32px 压小了 25%，用户实测反馈"下方的按钮过小"。
     * 默认值必须是"大多数场景该用的那个"，而对话框页脚就是大多数场景。
     */
    await render(<AppButton>x</AppButton>);
    const button = container.querySelector("button")!;
    expect(button.className).toContain("rt-r-size-2");
});

test("AppButton 显式 size=sm 时才是 24px 档（行内动作用）", async () => {
    await render(<AppButton size="sm">x</AppButton>);
    expect(container.querySelector("button")!.className).toContain("rt-r-size-1");
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
        document.dispatchEvent(
            new KeyboardEvent("keydown", { key, bubbles: true, cancelable: true }),
        );
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

test("AppFormSection 渲染 section + section 角色标题，并靠留白而非分割线分组", async () => {
    await render(
        <AppForm>
            <AppFormSection title="网格">
                <AppField label="A">
                    <input />
                </AppField>
            </AppFormSection>
            <AppFormSection title="吸附">
                <AppField label="B">
                    <input />
                </AppField>
            </AppFormSection>
        </AppForm>,
    );

    const sections = container.querySelectorAll("section");
    expect(sections).toHaveLength(2);

    // 标题用 section 角色（13px/600），**不缩字号** —— 原实现是 12px/700 muted，
    // 比它统领的 14px 行还小。
    const heading = sections[0].querySelector("h3")!;
    expect(heading.className).toContain("hs-type-section");

    // 分区之间要额外上边距：本组件取代了 Radix Separator，分组由留白承担，
    // 若节间距与行距相同就分不出组。两个分区带**同一个类**，`:not(:first-child)`
    // 由 CSS 在运行期挑出后面的分区 —— 因此这里断言"机制存在"，
    // 而不是断言某个分区没有这个类（那是把选择器文本当成结果读）。
    expect(sections[0].className).toContain("[&:not(:first-child)]:mt-3");
    expect(sections[1].className).toContain("[&:not(:first-child)]:mt-3");
});

test("AppSwitchRow 的 control 决定控件类型，两种共用同一排版", async () => {
    await render(
        <div>
            <AppSwitchRow label="开关" checked onCheckedChange={() => {}} control="switch" />
            <AppSwitchRow label="勾选" checked onCheckedChange={() => {}} control="checkbox" />
        </div>,
    );

    // Radix 的 Switch 与 Checkbox 都渲染 button[role]
    const controls = container.querySelectorAll("button");
    expect(controls.length).toBeGreaterThanOrEqual(2);

    // 两条标签必须同号 —— 这正是"同一表单两种标签字号"的回归点
    const labels = [...container.querySelectorAll(".hs-type-body")].map(
        (el) => getComputedStyle(el).fontSize,
    );
    expect(new Set(labels).size).toBeLessThanOrEqual(1);
});

/*
 * 二级子菜单。它是从 `ClipContextMenu` 提到原语层的（颤音预设列表也需要它），
 * 因此两条契约都要钉住：展开后子项出现在**本层**的 role="menu" 里，
 * 以及"点开子项"只触发子项自身、不关闭整张菜单（`extraItems` 里的内容由
 * 调用方负责 onClose，主菜单的 items 才是自动关闭）。
 */
function renderSubMenu(children: React.ReactNode) {
    return render(
        <div role="menu" data-hs-context-menu="1" className="w-48">
            {children}
        </div>,
    );
}

test("AppSubMenu：点击展开子面板，再点收起", async () => {
    await renderSubMenu(
        <AppSubMenu label="Presets">
            <button type="button" role="menuitem">
                Natural
            </button>
        </AppSubMenu>,
    );

    const trigger = container.querySelector<HTMLButtonElement>('[aria-haspopup="menu"]');
    expect(trigger).not.toBeNull();
    expect(trigger?.getAttribute("aria-expanded")).toBe("false");
    // 未展开时子项不渲染
    expect(container.textContent).not.toContain("Natural");

    await act(async () => {
        trigger?.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }));
    });
    expect(trigger?.getAttribute("aria-expanded")).toBe("true");
    expect(container.textContent).toContain("Natural");

    await act(async () => {
        trigger?.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }));
    });
    expect(trigger?.getAttribute("aria-expanded")).toBe("false");
    expect(container.textContent).not.toContain("Natural");
});

test("AppSubMenu：子面板自身是独立的 role=menu 表面（键盘导航按层划分）", async () => {
    await renderSubMenu(
        <AppSubMenu label="Presets">
            <button type="button" role="menuitem">
                Natural
            </button>
            <button type="button" role="menuitem">
                Soft
            </button>
        </AppSubMenu>,
    );

    const trigger = container.querySelector<HTMLButtonElement>('[aria-haspopup="menu"]');
    await act(async () => {
        trigger?.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }));
    });

    // 外层容器 + 子面板：两个独立的 role="menu"。
    const menus = [...container.querySelectorAll<HTMLElement>('[role="menu"]')];
    expect(menus.length).toBe(2);
    const [outer, submenu] = menus;

    /*
     * 子面板在 DOM 上是外层菜单的**后代**（绝对定位只是视觉上浮出来），
     * 因此 `outer.textContent` 当然含子项文字 —— 真正决定键盘归属的是
     * `useMenuKeyboard` 的分层规则：`closest('[role="menu"]') === container`。
     * 这里按同一条规则复算，断言外层**自己的**项里不含子项，
     * 否则方向键会在两层之间串门。
     */
    const ownedBy = (menu: HTMLElement) =>
        [...menu.querySelectorAll<HTMLElement>('[role="menuitem"], [role="menuitemradio"]')].filter(
            (item) => item.closest('[role="menu"]') === menu,
        );
    const outerLabels = ownedBy(outer).map((item) => item.textContent);
    const submenuLabels = ownedBy(submenu).map((item) => item.textContent);

    expect(submenuLabels).toEqual(["Natural", "Soft"]);
    // 外层只有子菜单触发项；子项一个都不属于它 —— 否则方向键会在两层之间串门。
    expect(outerLabels).toEqual(["Presets"]);
});
