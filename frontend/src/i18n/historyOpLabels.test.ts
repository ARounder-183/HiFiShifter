import { describe, expect, it } from "vitest";

// 直接读后端源码文本（`vite/client` 为 `?raw` 提供类型声明，无需 node 类型，
// 与 invoke.wiring.test.ts 的源码扫描思路一致）。这样后端新增 HistoryOp 时
// 本测试会自动覆盖到，不需要手工维护一份副本。
import backendStateSource from "../../../backend/src-tauri/src/state.rs?raw";

import { messages, type Locale } from "./messages";

/**
 * 撤销记录标签的本地化穷举防回归测试。
 *
 * 【为什么需要】撤销列表的标签是后端 `HistoryOp::key()` 产出的裸标识符
 * （如 `"take_channel_mode"`），前端在 `UndoHistoryPanel.labelOf` 里拼成
 * `history_op_<key>` 查表。查不到时 `t()` 返回 undefined，labelOf 会**静默
 * 回落到裸标识符** —— 用户在列表里看到的是 `take_channel_mode` 而不是
 * 「修改 Take 声道模式」。没有任何断言会失败，只有人眼能发现。
 *
 * 该缺陷已实际发生过一次：`HistoryOp::TakeChannelMode` 在声道模式功能上线时
 * 只加了后端变体、漏了五个语系的 i18n 键。
 *
 * 本测试做两件事：
 * 1. 从后端源码提取全部 `HistoryOp::X => "key"` 的 key；
 * 2. 断言每个 key 在**全部语系**里都有非空的 `history_op_<key>`。
 */

/** 后端 `HistoryOp::key()` 里出现的所有 `=> "..."` 字面量。 */
function extractBackendHistoryOpKeys(source: string): string[] {
    const keys = new Set<string>();
    const pattern = /HistoryOp::[A-Za-z0-9_]+\s*=>\s*"([a-z_0-9]+)"/g;
    for (const match of source.matchAll(pattern)) {
        keys.add(match[1]);
    }
    return [...keys].sort();
}

const LOCALES = Object.keys(messages) as Locale[];

describe("undo history op labels", () => {
    const backendKeys = extractBackendHistoryOpKeys(backendStateSource);

    it("extracts a plausible set of history op keys from the backend", () => {
        // 防"正则失效导致本测试静默通过"：后端确实有几十个操作类型。
        expect(backendKeys.length).toBeGreaterThan(30);
        expect(backendKeys).toContain("take_channel_mode");
        expect(backendKeys).toContain("import_media");
    });

    it("every backend history op has a label key in every locale", () => {
        const missing: string[] = [];
        for (const locale of LOCALES) {
            const table = messages[locale] as Record<string, string | undefined>;
            for (const op of backendKeys) {
                const key = `history_op_${op}`;
                const text = table[key];
                if (typeof text !== "string" || text.length === 0) {
                    missing.push(`${locale}: ${key}`);
                }
            }
        }
        // 报告全部缺失项而不是第一个，便于一次补齐。
        expect(missing).toEqual([]);
    });

    it("does not carry history op labels that the backend never emits", () => {
        // 反向检查：i18n 里的 history_op_* 键必须都能被后端产出（`initial`
        // 例外 —— 它对应"无标签"的空撤销位，由 labelOf 主动使用）。
        const enUS = messages["en-US"] as Record<string, string>;
        const backendSet = new Set(backendKeys);
        const orphans = Object.keys(enUS)
            .filter((key) => key.startsWith("history_op_"))
            .map((key) => key.slice("history_op_".length))
            .filter((op) => op !== "initial" && !backendSet.has(op))
            .sort();
        expect(orphans).toEqual([]);
    });

    it("keeps the locale tables in step with each other for these keys", () => {
        // 任一语系缺键都会在切语言后回落到裸标识符；这里守住"语系之间一致"。
        const keysByLocale = LOCALES.map((locale) => {
            const table = messages[locale] as Record<string, string>;
            return [
                locale,
                Object.keys(table)
                    .filter((key) => key.startsWith("history_op_"))
                    .sort()
                    .join(","),
            ] as const;
        });
        const reference = keysByLocale[0];
        for (const [locale, keys] of keysByLocale.slice(1)) {
            expect(keys, `${locale} 与 ${reference[0]} 的 history_op_* 键集合不一致`).toBe(
                reference[1],
            );
        }
    });
});
