// @vitest-environment jsdom
/*
 * 「点完一项就收起菜单 sheet」的判据。
 *
 * 【为什么值得单独测】这条判据决定三件用户可见的事：点普通项后 sheet 是否消失、
 * 点子菜单触发器后能不能继续选下一级、点禁用项后菜单会不会莫名消失。三者都只在
 * 触摸设备上出现（桌面不渲染 sheet），回归时没人会顺手发现 —— 只能靠这里锁住。
 *
 * 元素形状按 Radix 的真实输出构造：条目是 role="menuitem"，子菜单触发器多一个
 * aria-haspopup="menu"，禁用项带 data-disabled。
 */

import { expect, test } from "vitest";

import { menuSelectionClosesSheet } from "./mobileMenuSheet";

function mount(html: string): void {
    document.body.innerHTML = html;
}

/** 取一个真实的元素节点当事件目标。 */
function target(selector: string): Element {
    const element = document.querySelector(selector);
    if (element === null) throw new Error("missing " + selector);
    return element;
}

test("普通菜单项：点完收起 sheet", () => {
    mount('<div role="menuitem" data-id="a"><span data-target>导出音频…</span></div>');
    expect(menuSelectionClosesSheet(target("[data-target]"))).toBe(true);
});

test("直接点在条目本身也收起", () => {
    mount('<div role="menuitem" data-id="a"></div>');
    expect(menuSelectionClosesSheet(target('[role="menuitem"]'))).toBe(true);
});

test("单选项与复选项同样收起", () => {
    mount(
        '<div role="menuitemradio"><span data-radio>中文</span></div>' +
            '<div role="menuitemcheckbox"><span data-check>显示所有 take</span></div>',
    );
    expect(menuSelectionClosesSheet(target("[data-radio]"))).toBe(true);
    expect(menuSelectionClosesSheet(target("[data-check]"))).toBe(true);
});

test("子菜单触发器：不收起，否则刚展开的那一级会跟着消失", () => {
    mount('<div role="menuitem" aria-haspopup="menu"><span data-sub>窗口</span></div>');
    expect(menuSelectionClosesSheet(target("[data-sub]"))).toBe(false);
});

test("禁用项：不收起，点了本来就没有结果可看", () => {
    mount(
        '<div role="menuitem" data-disabled=""><span data-disabled-item>导出诊断信息</span></div>' +
            '<div role="menuitem" aria-disabled="true"><span data-aria-disabled>（空）</span></div>',
    );
    expect(menuSelectionClosesSheet(target("[data-disabled-item]"))).toBe(false);
    expect(menuSelectionClosesSheet(target("[data-aria-disabled]"))).toBe(false);
});

test("菜单之外的点击：不归这条判据管", () => {
    mount('<div class="somewhere"><span data-plain>画布</span></div>');
    expect(menuSelectionClosesSheet(target("[data-plain]"))).toBe(false);
});

test("非元素目标（文本节点 / null）一律不收起", () => {
    mount("<div>text</div>");
    expect(menuSelectionClosesSheet(null)).toBe(false);
    expect(menuSelectionClosesSheet(document.createTextNode("x"))).toBe(false);
});
