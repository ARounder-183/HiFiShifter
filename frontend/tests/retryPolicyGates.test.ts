/**
 * CI 重试只用于吸收 runner 抖动，不能无限放宽，也不能被悄悄改成常态。
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { expect, test } from "vitest";

test("Vitest 重试策略保持在 0/1 两档且有说明", () => {
    const config = readFileSync(join(process.cwd(), "vitest.config.ts"), "utf8");
    expect(config).toMatch(/retry:\s*process\.env\.CI\s*\?\s*1\s*:\s*0/);
    expect(config).toContain("真回归两次都会红");
});
