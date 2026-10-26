/**
 * 超大文件预算：只降不升。
 *
 * 【为什么需要】仓库约定“避免单文件过于庞大”，但此前没有任何门禁；
 * TimelinePanel / sessionSlice 已经长到 7000+ 行，继续增长会让 React Compiler
 * 与代码审查都失去可操作性。这里钉住当前基线，拆分后逐步下调。
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { expect, test } from "vitest";

const LIMITS: Record<string, number> = {
    "src/components/layout/TimelinePanel.tsx": 7425,
    "src/features/session/sessionSlice.ts": 7460,
    "src/components/layout/VibratoDialog.tsx": 2650,
    "src/components/layout/FileBrowserPanel.tsx": 2330,
};

test("超大文件只降不升", () => {
    for (const [file, max] of Object.entries(LIMITS)) {
        const lines = readFileSync(join(process.cwd(), file), "utf8").split("\n").length;
        expect(lines, file + " 行数超过预算；请拆分而不是继续增长").toBeLessThanOrEqual(max);
    }
});
