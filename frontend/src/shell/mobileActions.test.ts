/*
 * 移动抽屉的搜索过滤。
 *
 * 【为什么值得单独测】抽屉在移动端是唯一的功能入口，150 行条目里没有搜索就等于
 * "低频功能不存在"。而这条逻辑有两个容易写错的点：
 * 1. **空查询必须原样返回同一个数组**（每次返回新数组会让抽屉在输入框每次
 *    重渲染时都重建整份列表）；
 * 2. 过滤后**整段没有命中的分区要消失**（否则用户会看到一串空标题）。
 */

import { expect, test } from "vitest";

import { filterMobileSections, type MobileActionSection } from "./mobileActions";

const SECTIONS: MobileActionSection[] = [
    {
        id: "edit",
        title: "编辑",
        items: [
            { id: "edit.undo", label: "撤销", shortcut: "Ctrl+Z" },
            { id: "edit.redo", label: "重做", shortcut: "Ctrl+Y" },
        ],
    },
    {
        id: "project",
        title: "工程",
        items: [{ id: "project.save", label: "保存工程", shortcut: "Ctrl+S" }],
    },
];

test("空查询原样返回（保持引用，避免无谓重渲染）", () => {
    expect(filterMobileSections(SECTIONS, "")).toBe(SECTIONS);
    expect(filterMobileSections(SECTIONS, "   ")).toBe(SECTIONS);
});

test("按标签子串匹配", () => {
    const hit = filterMobileSections(SECTIONS, "保存");
    expect(hit).toHaveLength(1);
    expect(hit[0].id).toBe("project");
    expect(hit[0].items.map((item) => item.id)).toEqual(["project.save"]);
});

test("大小写无关（英文界面下的实际输入）", () => {
    const english: MobileActionSection[] = [
        {
            id: "project",
            title: "Project",
            items: [{ id: "project.export", label: "Export Audio...", shortcut: null }],
        },
    ];
    expect(filterMobileSections(english, "export")).toHaveLength(1);
    expect(filterMobileSections(english, "EXPORT")).toHaveLength(1);
    expect(filterMobileSections(english, "export audio")).toHaveLength(1);
});

test("整段没有命中时分段消失", () => {
    const hit = filterMobileSections(SECTIONS, "撤销");
    expect(hit.map((section) => section.id)).toEqual(["edit"]);
    expect(hit[0].items).toHaveLength(1);
});

test("没有命中时返回空数组", () => {
    expect(filterMobileSections(SECTIONS, "不存在的动作")).toEqual([]);
});
