/**
 * 超大文件预算：只降不升。
 *
 * 【为什么需要】仓库约定“避免单文件过于庞大”，但此前没有任何门禁；
 * 2026-10 的拆分把 TimelinePanel / sessionSlice 从 7000+ 行降到 2000 / 1400
 * 行以内。这里钉住拆分后的基线，防止再次长回去；新增巨型模块也必须登记。
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { expect, test } from "vitest";

const LIMITS: Record<string, number> = {
    "src/components/layout/TimelinePanel.tsx": 2050,
    "src/features/session/sessionSlice.ts": 1400,
    "src/components/layout/VibratoDialog.tsx": 1100,
    "src/components/layout/FileBrowserPanel.tsx": 800,
    "src/components/layout/vibratoDialog/useVibratoDialogModel.tsx": 1500,
    "src/components/layout/timeline/hooks/useTimelineKernelTrim.ts": 1350,
    "src/components/layout/timeline/hooks/useTimelineKernelGainCrossfade.ts": 1150,
    "src/components/layout/timeline/hooks/useTimelineKernelSeekDrag.ts": 1150,
};

test("超大文件只降不升", () => {
    for (const [file, max] of Object.entries(LIMITS)) {
        const lines = readFileSync(join(process.cwd(), file), "utf8").split("\n").length;
        expect(lines, file + " 行数超过预算；请拆分而不是继续增长").toBeLessThanOrEqual(max);
    }
});
