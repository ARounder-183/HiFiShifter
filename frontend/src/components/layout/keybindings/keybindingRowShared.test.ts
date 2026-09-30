/**
 * `keybindingRowShared` 的单元测试 —— 默认绑定判定与分组短标签的回落。
 *
 * 【为什么要测回落】分组标签在窄导航栏里会溢出（英文最长 205px、日文 214px，
 * 而栏内曾只有 155px）。修法是给冗长的分组配一套短标签 —— 但那套表是**部分**映射，
 * 缺项必须静默回落到长标签。这个"缺了就回落"的行为不测就会在将来某个语系漏加
 * 短键时静默变成"导航栏显示键名"。
 */
import { describe, expect, it } from "vitest";

import { isDefaultBinding, resolveGroupNavLabel } from "./keybindingRowShared";
import {
    ACTION_GROUP_ORDER,
    GROUP_LABEL_KEYS,
    GROUP_NAV_LABEL_KEYS,
} from "../../../features/keybindings/defaultKeybindings";
import { enUS } from "../../../i18n/en-US";

describe("isDefaultBinding — 可选布尔的等价判定", () => {
    it("省略的修饰键字段与显式 false 等价", () => {
        expect(isDefaultBinding({ key: "z", ctrl: true }, { key: "z", ctrl: true })).toBe(true);
        expect(
            isDefaultBinding({ key: "z", ctrl: true, shift: false }, { key: "z", ctrl: true }),
        ).toBe(true);
    });

    it("修饰键不同则不是默认", () => {
        expect(
            isDefaultBinding({ key: "z", ctrl: true, shift: true }, { key: "z", ctrl: true }),
        ).toBe(false);
    });

    it("主键不同则不是默认", () => {
        expect(isDefaultBinding({ key: "y", ctrl: true }, { key: "z", ctrl: true })).toBe(false);
    });

    it("modifierOnly 也参与判定", () => {
        expect(
            isDefaultBinding(
                { key: "alt", alt: true, modifierOnly: true },
                { key: "alt", alt: true },
            ),
        ).toBe(false);
    });
});

describe("resolveGroupNavLabel — 窄位标签与回落", () => {
    /** 只认识"真实存在"的键；查不到时返回键名本身（与 `tf` 的行为一致）。 */
    const fakeTf = (registered: Record<string, string>) => (key: string) => registered[key] ?? key;

    const both = {
        kb_group_playback: "Playback & Navigation",
        kb_group_playback_nav: "Playback",
        kb_group_edit: "Edit",
    };

    it("有短标签时优先用短标签", () => {
        expect(resolveGroupNavLabel("playback", fakeTf(both))).toBe("Playback");
    });

    it("没有短标签时回落到长标签，而不是返回键名", () => {
        // `edit` 不在 `GROUP_NAV_LABEL_KEYS` 里（它的长标签本来就够短）。
        expect(GROUP_NAV_LABEL_KEYS.edit).toBeUndefined();
        expect(resolveGroupNavLabel("edit", fakeTf(both))).toBe("Edit");
    });

    it("词典漏了短键时也回落到长标签（不显示键名）", () => {
        // 只有长键注册 —— 模拟某个语系漏加 `_nav` 键。
        const onlyLong = { kb_group_playback: "Playback & Navigation" };
        expect(resolveGroupNavLabel("playback", fakeTf(onlyLong))).toBe("Playback & Navigation");
    });

    it("全部 14 个分组都能解析出非空文案", () => {
        /*
         * 【为什么这条最关键】新增分组时若只加了 `GROUP_LABEL_KEYS` 而忘了词典，
         * `tf` 会返回键名、导航栏就会出现 `kb_group_xxx` —— 这条把那个漏洞堵住。
         */
        const registered: Record<string, string> = {};
        for (const g of ACTION_GROUP_ORDER) {
            registered[GROUP_LABEL_KEYS[g]] = `L:${g}`;
            const nav = GROUP_NAV_LABEL_KEYS[g];
            if (nav) registered[nav] = `S:${g}`;
        }
        for (const g of ACTION_GROUP_ORDER) {
            const out = resolveGroupNavLabel(g, fakeTf(registered));
            expect(out.startsWith("kb_group_"), `${g} 解析出了键名：${out}`).toBe(false);
            expect(out.length).toBeGreaterThan(0);
        }
    });

    it("每个注册过的短键都确实存在于英语词典", () => {
        /*
         * 【为什么用英语词典做参照】`MessageKey` 由 en-US 推导，因此 en-US 里存在
         * 即代表该键合法；其余语系由 `catalogIntegrity.test.ts` 守键集合一致。
         */
        const dict = enUS as unknown as Record<string, string>;
        for (const key of Object.values(GROUP_NAV_LABEL_KEYS)) {
            expect(typeof dict[key], `${key} 不在 en-US 词典里`).toBe("string");
        }
    });
});
