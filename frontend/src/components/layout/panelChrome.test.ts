/**
 * 面板 chrome 门禁：主编辑面板根节点必须有**同一条**顶部 hairline。
 *
 * 【为什么这条是门禁而不是人工检查】参数编辑器（PianoRollPanel）根节点一直有
 * `border-t border-qt-border`，而时间轴（TimelinePanel）没有 —— 两个面板并排时，
 * 上半部分的窗口拖动条与标尺之间就少一条分隔线。这不是"某个文件忘了写"，而是
 * **契约只存在于某个文件里**：谁新增一个主编辑面板，谁就得记得写同一句 className。
 *
 * 把"根节点必须有这条线"变成一条可执行的断言之后，新增面板会当场红，而不是等用户
 * 在两个面板之间看出不对齐。
 */
import { describe, expect, test } from "vitest";

const ROOT_PANELS = [
    { name: "TimelinePanel", file: "src/components/layout/TimelinePanel.tsx", tag: "timeline" },
    { name: "PianoRollPanel", file: "src/components/layout/PianoRollPanel.tsx", tag: "pianoRoll" },
] as const;

async function readSource(file: string): Promise<string> {
    return import("node:fs").then((fs) => fs.readFileSync(file, "utf8"));
}

describe("主编辑面板根节点 chrome", () => {
    for (const panel of ROOT_PANELS) {
        test(`${panel.name} 根节点带 border-t border-qt-border`, async () => {
            const source = await readSource(panel.file);
            // 只找**根节点**那一段 className：面板内部子元素也可能有自己的边框，
            // 那不是本门禁要管的事。
            const className = source.match(/className="([^"]*bg-qt-graph-bg[^"]*)"/)?.[1] ?? "";
            expect(className, `${panel.name} 根节点应使用 bg-qt-graph-bg`).not.toBe("");
            expect(className).toContain("border-t");
            expect(className).toContain("border-qt-border");
        });

        test(`${panel.name} 根节点标记 data-panel-hairline`, async () => {
            const source = await readSource(panel.file);
            expect(source).toContain(`data-panel-hairline="${panel.tag}"`);
        });
    }
});
