import { describe, expect, test } from "vitest";

// 直接读后端源码文本（与 `invoke.wiring.test.ts` 同一手法）：这样后端增删键时
// 本测试自动跟随，不需要手工维护一份副本。
import rustStrokesSource from "../../../backend/hifishifter-plugin/src/editor/host_key_relay/strokes.rs?raw";

import { RELAYABLE_CODES } from "./hostKeyRelay";

/**
 * 可让路按键表的**跨语言**一致性。
 *
 * 【为什么必须有】这张表同时存在于两处：前端要先判掉"这个键根本不让"（否则每次按都要
 * 多一次 IPC 往返），原生要把 `code` 翻成平台键码。两处漂移的表现是**按了没反应，且没有
 * 任何提示** —— 前端的表多了 ⇒ 原生回 `unsupported-key`，静默回落旧文案并闩死整条通路；
 * 前端的表少了 ⇒ 那个键永远不让，用户根本不知道有这个能力。两种都不会崩、不会报错，
 * 只在真机上"怪怪的"。
 *
 * 【为什么读源码而不是读一份导出的 JSON】导出一份清单就多了一份需要同步的东西 ——
 * 那正是本测试要防的问题本身。源码是唯一事实，正则只认那张表的字面形状。
 */
function rustRelayableCodes(source: string): string[] {
    const block = source.match(/const CODES:\s*&\[&str\]\s*=\s*&\[([\s\S]*?)\];/);
    if (!block) throw new Error("strokes.rs 里找不到 CODES 表 —— 本测试的锚点没了");
    return [...block[1].matchAll(/"([A-Za-z0-9]+)"/g)].map((match) => match[1]);
}

describe("可让路按键表", () => {
    test("前端与原生逐项一致", () => {
        const rust = rustRelayableCodes(rustStrokesSource);
        expect(rust.length).toBeGreaterThan(0);
        expect([...RELAYABLE_CODES].sort()).toEqual([...rust].sort());
    });

    test("两边都没有重复项", () => {
        const rust = rustRelayableCodes(rustStrokesSource);
        expect(new Set(rust).size).toBe(rust.length);
        expect(RELAYABLE_CODES.size).toBe([...RELAYABLE_CODES].length);
    });

    test("表里每一项都是 DOM `code` 的字面形状（大小写必须一致）", () => {
        for (const code of RELAYABLE_CODES) {
            expect(code, code).toMatch(/^[A-Za-z0-9]+$/);
        }
    });

    test("表里不含可打印字符 —— 那是打字的领地", () => {
        for (const code of RELAYABLE_CODES) {
            expect(
                /^(Key[A-Z]|Digit[0-9]|Semicolon|Quote|Comma|Period|Slash|Backslash|Bracket|Minus|Equal|Backquote)$/.test(
                    code,
                ),
                code,
            ).toBe(false);
        }
    });
});
