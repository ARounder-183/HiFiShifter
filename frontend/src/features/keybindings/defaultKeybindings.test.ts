/**
 * 默认快捷键表的整体不变式。
 *
 * 【为什么要一张"总表"测试】快捷键动作散落在四处：`ActionId` 联合、`DEFAULT_KEYBINDINGS`、
 * `ACTION_META`、以及各消费点（菜单 / 键盘分派 / 能力矩阵）。少写一处不会编译失败，
 * 表现是"设置面板里能看到，但按了没反应"或"菜单里根本没有这一项"。这里把
 * 「三者必须同时存在」钉成一条断言，加动作时只需改数据、不用再手工同步测试清单。
 */
import { describe, expect, it } from "vitest";

import {
    ACTION_GROUP_ORDER,
    ACTION_META,
    ALL_ACTION_IDS,
    DEFAULT_KEYBINDINGS,
} from "./defaultKeybindings";
import { KEYBINDING_PRESETS, KEYBINDING_PRESET_IDS } from "./keybindingPresets";
import { formatKeybindingList } from "./keybindingsSlice";
import {
    SAVE_PROJECT_AS_FORMATS,
    saveAsFormatForAction,
} from "../../features/session/saveProjectAsMenu";

describe("快捷键动作的三处登记", () => {
    it("每个动作都有元信息与默认绑定", () => {
        for (const id of ALL_ACTION_IDS) {
            expect(ACTION_META[id], `${id} 缺 ACTION_META`).toBeDefined();
            expect(DEFAULT_KEYBINDINGS[id]?.length ?? 0, `${id} 没有默认绑定`).toBeGreaterThan(0);
        }
    });

    it("元信息的分组都在展示顺序里（否则设置面板永远不显示该动作）", () => {
        const groups = new Set<string>(ACTION_GROUP_ORDER);
        for (const id of ALL_ACTION_IDS) {
            expect(groups.has(ACTION_META[id].group), `${id} 的分组不在 ACTION_GROUP_ORDER`).toBe(
                true,
            );
        }
    });

    it("动作名不重复", () => {
        expect(new Set(ALL_ACTION_IDS).size).toBe(ALL_ACTION_IDS.length);
    });
});

describe("另存为三种格式的动作", () => {
    it("三种格式各有一个动作，且默认不占键位", () => {
        expect(SAVE_PROJECT_AS_FORMATS.map((entry) => entry.format)).toEqual([
            "hshp",
            "json",
            "zip",
        ]);
        for (const entry of SAVE_PROJECT_AS_FORMATS) {
            // 「默认快捷键为空」是明确的产品要求：格式选择是低频决定，绑死默认键
            // 只会挤掉用户真正想要的组合。菜单右侧因此显示空串，不占位。
            expect(DEFAULT_KEYBINDINGS[entry.actionId], entry.actionId).toEqual([
                { key: "__none__" },
            ]);
            expect(formatKeybindingList(DEFAULT_KEYBINDINGS[entry.actionId], "")).toBe("");
            // 出现在「工程」分组，与「导入特定文件」那几个动作同一处。
            expect(ACTION_META[entry.actionId].group).toBe("project");
        }
    });

    it("动作 → 格式的映射与表格一致", () => {
        for (const entry of SAVE_PROJECT_AS_FORMATS) {
            expect(saveAsFormatForAction(entry.actionId)).toBe(entry.format);
        }
        // 非"指定格式"动作不得命中：`project.saveAs` 走的是"上次格式"。
        expect(saveAsFormatForAction("project.saveAs")).toBeNull();
        expect(saveAsFormatForAction("project.save")).toBeNull();
    });

    it("预设不覆盖这三个动作（应用预设不会把它们悄悄绑上默认键）", () => {
        for (const presetId of KEYBINDING_PRESET_IDS) {
            const preset = KEYBINDING_PRESETS[presetId];
            for (const entry of SAVE_PROJECT_AS_FORMATS) {
                expect(preset[entry.actionId], `${presetId}: ${entry.actionId}`).toBeUndefined();
            }
        }
    });
});
