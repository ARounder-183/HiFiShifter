/*
 * 形态能力表的前后端契约。
 *
 * 【为什么这条必须存在】能力 id 是一串**跨越 IPC 边界的字符串**：后端
 * `platform/mod.rs` 里是 `CAP_*` 常量，前端 `surfaceCapabilities.ts` 里是联合类型。
 * 两边没有编译期联系 —— 改一边而不改另一边时，前端会静默读到 `undefined`，而
 * `surfaceCapability()` 对未知 id 的保守处理是"不可用"，于是**某个功能在移动端
 * 整体消失**，没有任何报错、没有任何日志。
 *
 * 这类"字符串契约"在本仓已有先例（`invoke.ts` 的命令名由
 * `check-command-gating.py` 守住）。这条测试是同一类守卫的前端版本。
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, test } from "vitest";

const BACKEND = join("..", "backend", "src-tauri", "src", "platform", "mod.rs");
const FRONTEND = join("src", "services", "surfaceCapabilities.ts");

/** 后端：从 `const CAP_X: &str = "…";` 里取出全部能力 id。 */
function backendCapabilityIds(): string[] {
    const source = readFileSync(BACKEND, "utf8");
    const ids: string[] = [];
    for (const match of source.matchAll(/const CAP_[A-Z_]+:\s*&str\s*=\s*"([^"]+)";/g)) {
        ids.push(match[1]);
    }
    return ids.sort();
}

/** 前端：从 `export type SurfaceCapabilityId = … | …;` 里取出全部成员。 */
function frontendCapabilityIds(): string[] {
    const source = readFileSync(FRONTEND, "utf8");
    const block = source.match(/export type SurfaceCapabilityId =([\s\S]*?);/);
    if (!block) throw new Error("surfaceCapabilities.ts 里找不到 SurfaceCapabilityId");
    const ids: string[] = [];
    for (const match of block[1].matchAll(/"([^"]+)"/g)) ids.push(match[1]);
    return ids.sort();
}

/** 后端：从 `const REASON_X: &str = "…";` 里取出全部理由键。 */
function backendReasonKeys(): string[] {
    const source = readFileSync(BACKEND, "utf8");
    const keys: string[] = [];
    for (const match of source.matchAll(/const REASON_[A-Z_]+:\s*&str\s*=\s*"([^"]+)";/g)) {
        keys.push(match[1]);
    }
    return keys.sort();
}

/** 前端：`REASON_KEYS` 数组里的全部理由键。 */
function frontendReasonKeys(): string[] {
    const source = readFileSync(FRONTEND, "utf8");
    const block = source.match(/const REASON_KEYS = \[([\s\S]*?)\] as const/);
    if (!block) throw new Error("surfaceCapabilities.ts 里找不到 REASON_KEYS");
    const keys: string[] = [];
    for (const match of block[1].matchAll(/"([^"]+)"/g)) keys.push(match[1]);
    return keys.sort();
}

describe("形态能力表的字符串契约", () => {
    test("能力 id 两侧逐字一致", () => {
        const backend = backendCapabilityIds();
        // 先证明正则没失效：一条都抓不到时，"两边都为空"会伪装成通过。
        expect(backend.length).toBeGreaterThan(10);
        expect(frontendCapabilityIds()).toEqual(backend);
    });

    test("理由键两侧逐字一致", () => {
        const backend = backendReasonKeys();
        expect(backend.length).toBeGreaterThan(2);
        expect(frontendReasonKeys()).toEqual(backend);
    });

    test("每个理由键在五语系里都有文案", () => {
        const locales = ["en-US", "zh-CN", "zh-TW", "ja-JP", "ko-KR"];
        for (const key of backendReasonKeys()) {
            for (const locale of locales) {
                const source = readFileSync(join("src", "i18n", `${locale}.ts`), "utf8");
                expect(source.includes(`${key}:`), `${locale} 缺少 ${key}`).toBe(true);
            }
        }
    });
});
