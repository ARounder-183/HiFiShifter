/*
 * ★ 门禁：**发布过吸附高亮的手势，收尾必须清掉它**。
 *
 * ## 防的是什么
 *
 * 吸附高亮（吸附竖线）由 `snapTimelineDetailed(..., { highlight })` 发布，而它
 * **只在再次带 `highlight` 调用时**才会被清除（见 `useTimelineState` 的实现）。
 * 手势松手后不再有预览帧，因此"清"这件事必须由收尾回调显式负责，有两条既有写法：
 *
 * - `clearSnapHighlights(SNAP_HIGHLIGHT_GROUP)` —— clip 拖拽 / 裁切 / 交叉点抓手；
 * - `endSnapGesture()` —— 吸附偏移（手势深度归零时兜底清空）。
 *
 * 交叉点抓手此前两者都没有，于是拖拽中亮起的吸附竖线**一直留在画面上**（用户报告：
 * "按住交叉淡化反向模式拖抓手，松手后高亮线不消失"）。
 *
 * 这类缺陷靠"调用方记得接"就一定会再漏一次 —— 与 `stretchCommitContract` 同一取舍：
 * 用可判定的**文本事实**做门禁，而不是靠人肉 review。
 *
 * ## 判据
 *
 * `handleKernel*Preview` 里出现过 `highlight:` 的，其同名 `*Commit` 里必须出现
 * `clearSnapHighlights(` 或 `endSnapGesture()`。
 *
 * 【为什么只覆盖 Preview/Commit 对】`resolveKernelSeekSec` 也发布高亮，但它不是
 * 手势回调对 —— 它的调用方（`handleKernelSeekTo` / `handleKernelSeek` 的提交分支）
 * 各自显式清了，且那两条路径的语义（单击落点 / rAF 节流拖拽）与本节约束无关。
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

const SOURCE = readFileSync(join("src", "components", "layout", "TimelinePanel.tsx"), "utf8");

/** 去掉块注释与行注释：注释里引用的符号名不得计入文本事实。 */
function stripComments(text: string): string {
    return text.replace(/\/\*[\s\S]*?\*\//g, "").replace(/\/\/[^\n]*/g, "");
}

const CODE = stripComments(SOURCE);

/** 各内核手势回调的名字与其源码区间。 */
function handlerBlocks(): Map<string, string> {
    const matches = [...CODE.matchAll(/const (handleKernel\w+) = React\.useCallback/g)];
    const blocks = new Map<string, string>();
    for (let i = 0; i < matches.length; i += 1) {
        const start = matches[i].index as number;
        const end = i + 1 < matches.length ? (matches[i + 1].index as number) : CODE.length;
        blocks.set(matches[i][1], CODE.slice(start, end));
    }
    return blocks;
}

const BLOCKS = handlerBlocks();

/** 发布吸附高亮的 Preview 回调 → 其收尾回调名。 */
function publishingPreviews(): string[] {
    return [...BLOCKS.entries()]
        .filter(([name, body]) => name.endsWith("Preview") && body.includes("highlight:"))
        .map(([name]) => name);
}

describe("吸附高亮的收尾门禁", () => {
    it("★ 自检：确实扫到了发布高亮的 Preview 回调（否则下面的断言是空转）", () => {
        expect(publishingPreviews().length).toBeGreaterThanOrEqual(3);
    });

    it("★ 每个发布高亮的 Preview，其 Commit 必须清掉高亮", () => {
        const offenders: string[] = [];
        for (const preview of publishingPreviews()) {
            const commit = preview.replace(/Preview$/, "Commit");
            const body = BLOCKS.get(commit);
            if (body === undefined) {
                offenders.push(`${preview} 没有同名的 ${commit}`);
                continue;
            }
            if (!body.includes("clearSnapHighlights(") && !body.includes("endSnapGesture()")) {
                offenders.push(`${commit} 既没有 clearSnapHighlights 也没有 endSnapGesture`);
            }
        }
        expect(
            offenders.length === 0
                ? []
                : [
                      "以下手势会亮起吸附高亮，但收尾没有清除它（高亮线会留在画面上）：",
                      ...offenders.map((line) => `  ${line}`),
                  ].join("\n"),
        ).toEqual([]);
    });

    it("★ 清除必须早于提交派发（否则松手后仍有一帧残留）", () => {
        // 交叉点抓手的收尾在落库前清；裁切同理。只校验"清除出现在落库调用之前"。
        for (const preview of publishingPreviews()) {
            const commit = preview.replace(/Preview$/, "Commit");
            const body = BLOCKS.get(commit) ?? "";
            const clearAt = Math.max(
                body.indexOf("clearSnapHighlights("),
                body.indexOf("endSnapGesture()"),
            );
            const persistAt = body.indexOf("setClipsStateBulkRemote");
            if (clearAt < 0 || persistAt < 0) continue;
            expect(clearAt, `${commit} 的清除应早于落库`).toBeLessThan(persistAt);
        }
    });
});
