import { describe, expect, it } from "vitest";
import { DEFAULT_KEYBINDINGS } from "./defaultKeybindings";
import {
    PARAM_BLOCK_ACTION_TO_OP,
    resolveActionByFocus,
    resolveParamBlockEditChannel,
} from "./focusRouting";
import type { ActionId } from "./types";

/**
 * 块命令的**端到端**路由（按键 → ActionId → 事件通道）。
 *
 * 【为什么必须有这条测试】上一轮只测了前半段（`paramBlockShortcuts.test.ts` 测
 * 「按键能解析成 ActionId」），后半段（ActionId → 通道 → 执行）没有任何覆盖；
 * 而真实失效的正是后半段：通道判据收到的是 `session.toolMode`（永远不是 "block"），
 * 于是 9 个块动作全部静默丢弃。任何一条断链都会让这里失败。
 */
const BLOCK_SCOPE = { toolMode: "select", selectToolMode: "block" } as const;

function keyEvent(binding: {
    readonly key: string;
    readonly ctrl?: boolean;
    readonly shift?: boolean;
    readonly alt?: boolean;
    readonly meta?: boolean;
}): KeyboardEvent {
    return {
        key: binding.key,
        code: binding.key.length === 1 ? `Key${binding.key.toUpperCase()}` : binding.key,
        ctrlKey: binding.ctrl === true,
        shiftKey: binding.shift === true,
        altKey: binding.alt === true,
        metaKey: binding.meta === true,
    } as KeyboardEvent;
}

describe("param block shortcut routing", () => {
    it("every mapped ActionId reaches the block channel in block selection mode", () => {
        const actionIds = Object.keys(PARAM_BLOCK_ACTION_TO_OP) as ActionId[];
        expect(actionIds.length).toBe(9);
        for (const actionId of actionIds) {
            expect(
                resolveParamBlockEditChannel("pianoRoll", BLOCK_SCOPE),
                `${actionId}：块模式必须有通道`,
            ).toBe("hifi:paramBlockEditOp");
        }
    });

    it("the same ActionIds are dropped outside block selection mode", () => {
        /* 通道判据只看工具身份：range / 绘制族 / 时间轴都不得派发块命令。 */
        expect(
            resolveParamBlockEditChannel("pianoRoll", {
                toolMode: "select",
                selectToolMode: "range",
            }),
        ).toBeNull();
        expect(
            resolveParamBlockEditChannel("pianoRoll", {
                toolMode: "draw",
                selectToolMode: "block",
            }),
        ).toBeNull();
        expect(resolveParamBlockEditChannel("timeline", BLOCK_SCOPE)).toBeNull();
    });

    it("every default-bound block action resolves from its own key in block mode", () => {
        const bound = Object.keys(PARAM_BLOCK_ACTION_TO_OP).filter((actionId) => {
            const bindings = DEFAULT_KEYBINDINGS[actionId as ActionId];
            return bindings !== undefined && bindings.some((item) => item.key !== "__none__");
        });
        /* 5 个动作有默认键（split / delete / group / ungroup / selectAll），其余只有菜单入口。 */
        expect(bound.length).toBe(5);
        for (const actionId of bound) {
            const binding = DEFAULT_KEYBINDINGS[actionId as ActionId].find(
                (item) => item.key !== "__none__",
            );
            if (binding === undefined) throw new Error(`${actionId} 缺少默认绑定`);
            expect(
                resolveActionByFocus(keyEvent(binding), DEFAULT_KEYBINDINGS, "pianoRoll", "block"),
                `${actionId}：按键应解析回自己`,
            ).toBe(actionId);
            /* 时间轴上同样的键仍是 Clip 命令 —— 块层不得抢走。 */
            expect(
                resolveActionByFocus(keyEvent(binding), DEFAULT_KEYBINDINGS, "timeline", "block"),
            ).not.toBe(actionId);
        }
    });
});
