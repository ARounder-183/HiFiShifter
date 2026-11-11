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
    "src/components/layout/timeline/hooks/useTimelineKernelTrim.ts": 1351,
    "src/components/layout/timeline/hooks/useTimelineKernelGainCrossfade.ts": 1150,
    "src/components/layout/timeline/hooks/useTimelineKernelSeekDrag.ts": 1150,
    // 全仓审查补录：这些大文件此前完全不受预算约束，先钉住当前行数（只降不升）。
    "../backend/hifishifter-kernel/src/state/model.rs": 12868,
    "../backend/hifishifter-plugin/src/editor/session.rs": 9355,
    "../backend/hifishifter-plugin/src/render/document.rs": 5129,
    "../backend/src-tauri/src/commands.rs": 3898,
    "../backend/hifishifter-kernel/src/mixdown.rs": 3943,
    "../backend/hifishifter-capture/src/recorder.rs": 4705,
    // 2026-11 拆分：块 hover / 命令通道 / 渲染规格三块搬出后，预算同步下调。
    // 2026-11 块写入收口 / 预览释放判据重写之后同步下调（原 7250）。
    "src/components/layout/pianoRoll/panel/usePianoRollPanelModel.tsx": 7129,
    "src/components/layout/timeline/kernel/host/timelineKernelHost.ts": 6453,
    "src/App.tsx": 4718,
};

test("超大文件只降不升", () => {
    for (const [file, max] of Object.entries(LIMITS)) {
        // 末尾换行不是一行；按实际内容行计数，避免"恰好等于预算"被误判超一行。
        const text = readFileSync(join(process.cwd(), file), "utf8").replace(/\r?\n$/, "");
        const lines = text.length === 0 ? 0 : text.split(/\r?\n/).length;
        expect(lines, file + " 行数超过预算；请拆分而不是继续增长").toBeLessThanOrEqual(max);
        expect(lines, file + " 预算已零余量；先拆分再继续增长").toBeLessThan(max);
    }
});
