/**
 * 类型放宽必须留在可审计的边界，而不是铺到整个模块。
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { expect, test } from "vitest";

const root = process.cwd();

test("invoke.ts 不保留文件级 any 豁免", () => {
    const invoke = readFileSync(join(root, "src/services/invoke.ts"), "utf8");
    expect(invoke).not.toContain("eslint-disable @typescript-eslint/no-explicit-any");
    expect(invoke).not.toContain("...args: any[]");
    expect(invoke).toContain("...args: unknown[]");
});

test("无声明的第三方模块使用局部 d.ts 而不是 @ts-expect-error", () => {
    const htmlToMarkdown = readFileSync(
        join(root, "src/components/layout/notebook/htmlToMarkdown.ts"),
        "utf8",
    );
    expect(htmlToMarkdown).not.toContain("@ts-expect-error");
    expect(
        readFileSync(join(root, "src/components/layout/notebook/turndown-plugin-gfm.d.ts"), "utf8"),
    ).toContain('declare module "turndown-plugin-gfm"');
});
