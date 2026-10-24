import { describe, expect, test } from "vitest";

// 直接读后端源码文本（与 `invoke.wiring.test.ts` 同一手法）：这样后端增删键时
// 本测试自动跟随，不需要手工维护一份副本。
import rustStrokesSource from "../../../backend/hifishifter-plugin/src/editor/host_key_relay/strokes.rs?raw";

import { RELAYABLE_KEYS } from "./hostKeyRelay";

/**
 * 可让路按键表的**跨语言**一致性。
 *
 * 【为什么必须有】这张表同时存在于两处：前端要先判掉"这个键根本不让"（否则每次按都要
 * 多一次 IPC 往返），原生要把 token 翻成平台键码。两处漂移的表现是**按了没反应，且没有
 * 任何提示** —— 前端的表多了 ⇒ 原生回 `unsupported-key`，静默回落旧文案；前端的表少了
 * ⇒ 那个键永远不让，用户根本不知道有这个能力。两种都不会崩、不会报错，只在真机上"怪怪的"。
 *
 * 【为什么读源码而不是读一份导出的 JSON】导出一份清单就多了一份需要同步的东西 ——
 * 那正是本测试要防的问题本身。源码是唯一事实，正则只认那张表的字面形状。
 */
function rustRelayableTokens(source: string): string[] {
    const block = source.match(/const TOKENS:\s*&\[\(&str, KeyToken\)\]\s*=\s*&\[([\s\S]*?)\];/);
    if (!block) throw new Error("strokes.rs 里找不到 TOKENS 表 —— 本测试的锚点没了");
    return [...block[1].matchAll(/\("([a-z0-9]+)",\s*KeyToken::/g)].map((match) => match[1]);
}

describe("可让路按键表", () => {
    test("前端与原生逐项一致", () => {
        const rust = rustRelayableTokens(rustStrokesSource);
        expect(rust.length).toBeGreaterThan(0);
        expect([...RELAYABLE_KEYS].sort()).toEqual([...rust].sort());
    });

    test("两边都没有重复项", () => {
        const rust = rustRelayableTokens(rustStrokesSource);
        expect(new Set(rust).size).toBe(rust.length);
        expect(RELAYABLE_KEYS.size).toBe([...RELAYABLE_KEYS].length);
    });

    test("表里每一项都是小写裸名（原生按字面量匹配，大小写必须一致）", () => {
        for (const token of RELAYABLE_KEYS) {
            expect(token, token).toMatch(/^[a-z0-9]+$/);
        }
    });
});
