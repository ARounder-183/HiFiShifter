/*
 * ★ 门禁：**网格重绘只在输入变化（或 svg 节点被重建）时全量拼接 path**。
 *
 * ## 防的是什么
 *
 * 此前有个无依赖的 `useLayoutEffect` 每次渲染都把 `lastDrawKeyRef` 置 null，
 * 于是 drawKey 完全相同也会走完整重绘：上百个刻度的 `xs.join(",")` 每帧两次
 * 全量字符串拼接（参数编辑器 clip 拖拽逐帧重渲染时尤甚）。drawKey 本身已覆盖
 * 全部影响绘制的输入，键变自然重绘；唯一需要强制失key 的是 svg 节点被 React
 * 重建（新 DOM 上的 path 是空的）。
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

const SOURCE = readFileSync(
    join("src", "components", "layout", "timeline", "BackgroundGrid.tsx"),
    "utf8",
);

function stripComments(text: string): string {
    return text.replace(/\/\*[\s\S]*?\*\//g, "").replace(/\/\/[^\n]*/g, "");
}

const CODE = stripComments(SOURCE);

describe("BackgroundGrid 重绘去重", () => {
    it("★ 不再每次渲染都清空 lastDrawKeyRef", () => {
        expect(CODE).not.toMatch(
            /useLayoutEffect\(\(\)\s*=>\s*\{\s*lastDrawKeyRef\.current\s*=\s*null;?\s*\}\)/,
        );
    });

    it("★ svg 节点被重建时才强制失key", () => {
        expect(CODE).toContain("lastDrawSvgRef.current !== svg");
        expect(CODE).toContain("lastDrawKeyRef.current = null");
    });

    it("★ drawKey 仍覆盖刻度线键（全量位置变化必须重绘）", () => {
        expect(CODE).toContain("latest.weakLineXsKey");
        expect(CODE).toContain("latest.strongLineXsKey");
        expect(CODE).toContain("lastDrawKeyRef.current === drawKey");
    });
});
