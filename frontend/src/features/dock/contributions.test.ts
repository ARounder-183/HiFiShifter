/*
 * 贡献点注册中心的契约测试。
 *
 * 【为什么必须有】这套机制的存在理由是"第三方能碰到 chrome 的地方只有 Window
 * 菜单"。要让它真的成立，必须保证三件事，而这三件事都容易在重构里悄悄坏掉：
 *
 *   1. **作用域筛选**：面板自己的按钮不会跑到别的面板上；
 *   2. **注销真的生效**：扩展卸载后按钮不能留在界面上指向不存在的面板；
 *   3. **顺序稳定**：注册顺序不影响展示顺序（否则界面会随加载顺序抖动）。
 */
import { afterEach, expect, test } from "vitest";

import {
    getContributionVersion,
    listPanelTabMenuItems,
    listToolbarItems,
    registerPanelTabMenuItem,
    registerToolbarItem,
    resetContributionsForTests,
    subscribeContributions,
} from "./contributions";

afterEach(() => {
    resetContributionsForTests();
});

const noop = () => null;

test("按面板作用域筛选：别的面板的按钮不会出现", () => {
    registerToolbarItem({ id: "a.one", panelId: "panelA", render: noop });
    registerToolbarItem({ id: "b.one", panelId: "panelB", render: noop });

    expect(listToolbarItems({ panelId: "panelA" }).map((item) => item.id)).toEqual(["a.one"]);
    expect(listToolbarItems({ panelId: "panelB" }).map((item) => item.id)).toEqual(["b.one"]);
});

test("无 panelId 的贡献项是全局项，所有宿主都收得到", () => {
    registerToolbarItem({ id: "global.one", render: noop });
    expect(listToolbarItems({ panelId: "panelA" }).map((item) => item.id)).toEqual(["global.one"]);
    expect(listToolbarItems({ panelId: "panelB" }).map((item) => item.id)).toEqual(["global.one"]);
    expect(listToolbarItems().map((item) => item.id)).toEqual(["global.one"]);
});

test("注销后不再出现，且不影响同作用域的其他项", () => {
    const disposeA = registerToolbarItem({ id: "a.one", panelId: "panelA", render: noop });
    registerToolbarItem({ id: "a.two", panelId: "panelA", render: noop });

    disposeA();
    expect(listToolbarItems({ panelId: "panelA" }).map((item) => item.id)).toEqual(["a.two"]);
});

test("重复注销是幂等的", () => {
    const dispose = registerToolbarItem({ id: "a.one", render: noop });
    dispose();
    expect(() => dispose()).not.toThrow();
    expect(listToolbarItems()).toEqual([]);
});

test("同 id 重复注册时，旧项的注销不会误删新项", () => {
    const disposeOld = registerToolbarItem({ id: "dup", render: noop });
    registerToolbarItem({ id: "dup", render: noop });
    // 旧项的注销函数在此时不应生效 —— 表里已经是新项了
    disposeOld();
    expect(listToolbarItems().map((item) => item.id)).toEqual(["dup"]);
});

test("按 order 排序，order 相同时按 id 排序（与注册顺序无关）", () => {
    registerToolbarItem({ id: "z", order: 1, render: noop });
    registerToolbarItem({ id: "a", order: 1, render: noop });
    registerToolbarItem({ id: "m", order: 0, render: noop });
    expect(listToolbarItems().map((item) => item.id)).toEqual(["m", "a", "z"]);
});

test("未给 order 的项排在给了 order 的项之后", () => {
    registerToolbarItem({ id: "explicit", order: 50, render: noop });
    registerToolbarItem({ id: "default", render: noop });
    expect(listToolbarItems().map((item) => item.id)).toEqual(["explicit", "default"]);
});

test("注册与注销都会推进版本号（宿主据此重渲染）", () => {
    const before = getContributionVersion();
    const dispose = registerToolbarItem({ id: "a.one", render: noop });
    const afterRegister = getContributionVersion();
    expect(afterRegister).toBeGreaterThan(before);

    dispose();
    expect(getContributionVersion()).toBeGreaterThan(afterRegister);
});

test("订阅者在注册与注销时都被通知", () => {
    let calls = 0;
    const unsubscribe = subscribeContributions(() => {
        calls += 1;
    });
    const dispose = registerToolbarItem({ id: "a.one", render: noop });
    expect(calls).toBe(1);
    dispose();
    expect(calls).toBe(2);
    unsubscribe();
    registerToolbarItem({ id: "a.two", render: noop });
    expect(calls).toBe(2);
});

test("标签菜单项也按作用域筛选，并保留 danger 标记", () => {
    registerPanelTabMenuItem({
        id: "menu.a",
        panelId: "panelA",
        label: "Do the thing",
        danger: true,
        onSelect: () => {},
    });
    registerPanelTabMenuItem({
        id: "menu.b",
        panelId: "panelB",
        label: "Other",
        onSelect: () => {},
    });

    const items = listPanelTabMenuItems({ panelId: "panelA" });
    expect(items.map((item) => item.id)).toEqual(["menu.a"]);
    expect(items[0].danger).toBe(true);
    expect(items[0].label).toBe("Do the thing");
});
