/**
 * webApi 是历史兼容门面。新代码必须走 services/api/* 分组模块；
 * 这里把现有导入方数量钉成只降不升的棘轮。
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { expect, test } from "vitest";
import { spawnSync } from "node:child_process";

test("webApi 门面的导入方不再增加", () => {
    const result = spawnSync("git", ["ls-files", "frontend/src"], {
        encoding: "utf8",
    });
    const files = (result.stdout || "").split("\n").filter(Boolean);
    let importers = 0;
    for (const file of files) {
        if (!/\.(ts|tsx)$/.test(file)) continue;
        const text = readFileSync(join(process.cwd(), "..", file), "utf8");
        if (/from\s+["'][^"']*webviewApi["']/.test(text)) importers += 1;
    }
    // 当前基线：迁移一个就下调一个，禁止回升。
    expect(importers, "新代码请改用 services/api/*，并同步下调本基线").toBeLessThanOrEqual(27);
});
