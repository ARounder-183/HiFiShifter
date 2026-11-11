/**
 * 仓库级配置一致性：重复存在的配置必须有门禁，失效的忽略项必须清理。
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { expect, test } from "vitest";

const root = process.cwd();

test("frontend 与仓库根的 Prettier 配置完全一致", () => {
    const local = readFileSync(join(root, ".prettierrc.json"), "utf8");
    const shared = readFileSync(join(root, "..", ".prettierrc.json"), "utf8");
    expect(JSON.parse(local)).toEqual(JSON.parse(shared));
});

test("frontend 不再保留不存在的 wasm 忽略项", () => {
    const ignore = readFileSync(join(root, ".prettierignore"), "utf8");
    expect(ignore).not.toContain("src/wasm/timeline-kernel");
    expect(ignore).not.toContain("wasm/*/target");
});

test("package.json 明确声明受支持的 Node 版本", () => {
    const pkg = JSON.parse(readFileSync(join(root, "package.json"), "utf8"));
    expect(pkg.engines.node).toMatch(/^>=22/);
});
