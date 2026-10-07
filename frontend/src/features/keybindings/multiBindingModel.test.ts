/**
 * 多绑定模型（`KeybindingMap` 的值是 `readonly Keybinding[]`）的单测。
 *
 * 【要锁住的行为】一个功能可以绑多个快捷键；下标 0 是主绑定（菜单显示与长按
 * 重复的基准）。数组形状的不变式只在 `normalizeBindings` 里实现一次，这里把
 * 它连同"写入 / 去重 / 与默认一致即删覆盖"一起钉死。
 */
import { describe, expect, it } from "vitest";

import keybindingsReducer, {
    firstBinding,
    formatKeybinding,
    formatKeybindingList,
    hasDuplicateBinding,
    isNoneBindingList,
    keybindingsEqual,
    normalizeBindings,
    resetKeybinding,
    setKeybindings,
} from "./keybindingsSlice";
import { DEFAULT_KEYBINDINGS, ALL_ACTION_IDS } from "./defaultKeybindings";
import type { ActionId, Keybinding } from "./types";

const NONE: Keybinding = { key: "__none__" };

function reduce(
    state: { overrides: Record<string, readonly Keybinding[]> },
    action: ReturnType<typeof setKeybindings> | ReturnType<typeof resetKeybinding>,
) {
    return keybindingsReducer(state as never, action) as {
        overrides: Record<string, readonly Keybinding[]>;
    };
}

describe("默认值：重做绑两个键", () => {
    it("edit.redo 默认同时绑 Ctrl+Shift+Z 与 Ctrl+Y，主绑定是 Ctrl+Shift+Z", () => {
        const redo = DEFAULT_KEYBINDINGS["edit.redo"];
        expect(redo).toHaveLength(2);
        expect(redo[0]).toEqual({ key: "z", ctrl: true, shift: true });
        expect(redo[1]).toEqual({ key: "y", ctrl: true });
    });

    it("除重做外，所有动作的默认绑定都是单个键", () => {
        // 多绑定是能力，不是默认风格：默认表里只应有 edit.redo 用到它。
        const multi = ALL_ACTION_IDS.filter((id) => DEFAULT_KEYBINDINGS[id].length > 1);
        expect(multi).toEqual(["edit.redo"]);
    });
});

describe("normalizeBindings — 列表形状的不变式", () => {
    it("剔除空值", () => {
        expect(normalizeBindings([null, { key: "a" }, undefined])).toEqual([{ key: "a" }]);
    });

    it("按首次出现的位置去重（主绑定身份稳定）", () => {
        expect(normalizeBindings([{ key: "a" }, { key: "b" }, { key: "a", ctrl: false }])).toEqual([
            { key: "a" },
            { key: "b" },
        ]);
    });

    it("__none__ 只作为唯一元素保留，出现在其它位置的一律删除", () => {
        expect(normalizeBindings([{ key: "a" }, NONE])).toEqual([{ key: "a" }]);
        expect(normalizeBindings([NONE, { key: "a" }])).toEqual([{ key: "a" }]);
        expect(normalizeBindings([NONE])).toEqual([NONE]);
    });

    it("空列表回退为「无」", () => {
        expect(normalizeBindings([])).toEqual([NONE]);
    });

    it("修饰键手势的 modifierOnly 标志在去重后保留", () => {
        const alt: Keybinding = { key: "alt", modifierOnly: true, alt: true };
        expect(normalizeBindings([alt, { ...alt }])).toEqual([alt]);
    });
});

describe("列表读取辅助", () => {
    it("firstBinding 取下标 0，空列表退化为「无」", () => {
        expect(firstBinding([{ key: "a" }, { key: "b" }])).toEqual({ key: "a" });
        expect(firstBinding([])).toEqual(NONE);
        expect(firstBinding(undefined)).toEqual(NONE);
    });

    it("isNoneBindingList 覆盖空列表与「无」单元素", () => {
        expect(isNoneBindingList([])).toBe(true);
        expect(isNoneBindingList(undefined)).toBe(true);
        expect(isNoneBindingList([NONE])).toBe(true);
        expect(isNoneBindingList([{ key: "a" }])).toBe(false);
    });

    it("keybindingsEqual 顺序敏感（顺序即语义）", () => {
        const a = { key: "a" };
        const b = { key: "b" };
        expect(keybindingsEqual([a, b], [a, b])).toBe(true);
        expect(keybindingsEqual([a, b], [b, a])).toBe(false);
        expect(keybindingsEqual([a], [a, b])).toBe(false);
    });

    it("formatKeybindingList 用 ; 连接全部绑定，跳过「无」", () => {
        const text = formatKeybindingList([
            { key: "z", ctrl: true, shift: true },
            { key: "y", ctrl: true },
        ]);
        expect(text).toBe(
            `${formatKeybinding({ key: "z", ctrl: true, shift: true })};${formatKeybinding({
                key: "y",
                ctrl: true,
            })}`,
        );
        // 分隔符无空格，且不与组合键自身的 `+` 混淆。
        expect(text).toContain(";");
        expect(text).not.toContain(" / ");
        expect(formatKeybindingList([{ key: "a" }, NONE])).toBe(formatKeybinding({ key: "a" }));
        expect(formatKeybindingList([NONE], "none")).toBe("none");
        expect(formatKeybindingList([], "none")).toBe("none");
    });

    it("hasDuplicateBinding 可排除正在录入的槽位", () => {
        const list = [{ key: "a" }, { key: "b" }];
        expect(hasDuplicateBinding(list, { key: "a" })).toBe(true);
        expect(hasDuplicateBinding(list, { key: "a" }, 0)).toBe(false);
        expect(hasDuplicateBinding(list, { key: "c" })).toBe(false);
        // 「无」不算重复（它不占任何按键）。
        expect(hasDuplicateBinding([NONE], NONE)).toBe(false);
    });
});

describe("setKeybindings — 唯一的写入口", () => {
    it("写入新绑定会建立覆盖项", () => {
        const next = reduce(
            { overrides: {} },
            setKeybindings({ actionId: "clip.split", bindings: [{ key: "g" }] }),
        );
        expect(next.overrides["clip.split"]).toEqual([{ key: "g" }]);
    });

    it("改回默认即删除覆盖项（改回默认 == 没有自定义）", () => {
        const withOverride = reduce(
            { overrides: {} },
            setKeybindings({ actionId: "clip.split", bindings: [{ key: "g" }] }),
        );
        const backToDefault = reduce(
            withOverride,
            setKeybindings({
                actionId: "clip.split",
                bindings: DEFAULT_KEYBINDINGS["clip.split"],
            }),
        );
        expect(backToDefault.overrides["clip.split"]).toBeUndefined();
    });

    it("追加第二个绑定（重做式用法）保留两个槽位且顺序不变", () => {
        const next = reduce(
            { overrides: {} },
            setKeybindings({
                actionId: "edit.redo",
                bindings: [
                    { key: "z", ctrl: true, shift: true },
                    { key: "y", ctrl: true },
                    { key: "r", ctrl: true, alt: true },
                ],
            }),
        );
        expect(next.overrides["edit.redo"]).toEqual([
            { key: "z", ctrl: true, shift: true },
            { key: "y", ctrl: true },
            { key: "r", ctrl: true, alt: true },
        ]);
    });

    it("写入重复绑定会被规范化去重（不会出现两个同样的槽位）", () => {
        const next = reduce(
            { overrides: {} },
            setKeybindings({
                actionId: "clip.split",
                bindings: [{ key: "g" }, { key: "g" }],
            }),
        );
        expect(next.overrides["clip.split"]).toEqual([{ key: "g" }]);
    });

    it("清空所有槽位回退为「无」而非空列表", () => {
        const next = reduce(
            { overrides: {} },
            setKeybindings({ actionId: "clip.split", bindings: [] }),
        );
        expect(next.overrides["clip.split"]).toEqual([NONE]);
    });

    it("resetKeybinding 删除覆盖项", () => {
        const withOverride = reduce(
            { overrides: {} },
            setKeybindings({ actionId: "clip.split", bindings: [{ key: "g" }] }),
        );
        const reset = reduce(withOverride, resetKeybinding("clip.split" as ActionId));
        expect(reset.overrides["clip.split"]).toBeUndefined();
    });
});
