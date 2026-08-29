import { describe, expect, it } from "vitest";

import { expandLineSegmentsToQuads, WAVEFORM_STROKE_WIDTH_PX } from "./surfaceRenderer.ts";

/** 读取展开输出中某个角点的位置与颜色。 */
function corner(quads: Float32Array, vertex: number): {
    x: number;
    y: number;
    r: number;
    g: number;
    b: number;
    a: number;
} {
    const base = vertex * 6;
    return {
        x: quads[base],
        y: quads[base + 1],
        r: quads[base + 2],
        g: quads[base + 3],
        b: quads[base + 4],
        a: quads[base + 5],
    };
}

describe("expandLineSegmentsToQuads", () => {
    it("expands a vertical envelope column into a 1-CSS-px-wide quad", () => {
        // 一条竖直包络列：x = 10，y 从 5 到 15，纯黑不透明。
        const vertices = new Float32Array([10, 5, 0, 0, 0, 1, 10, 15, 0, 0, 0, 1]);
        const quads = expandLineSegmentsToQuads(vertices);
        expect(quads.length).toBe(36);

        const half = WAVEFORM_STROKE_WIDTH_PX / 2;
        // 竖直段法线为水平：四角 x 覆盖 [10 - half, 10 + half]，
        // y 覆盖 [5, 15] —— 与 Canvas2D 的 1 CSS px 描边宽度一致。
        for (const index of [0, 1, 2, 3, 4, 5]) {
            const c = corner(quads, index);
            expect(Math.min(c.x, 10 + half)).toBeGreaterThanOrEqual(10 - half - 1e-6);
            expect(Math.max(c.x, 10 - half)).toBeLessThanOrEqual(10 + half + 1e-6);
            expect(c.y >= 5 - 1e-6 && c.y <= 15 + 1e-6).toBe(true);
            expect(c.r).toBe(0);
            expect(c.a).toBe(1);
        }
        // 左右边缘各出现一次（±half）。
        const xs = new Set([0, 1, 2, 3, 4, 5].map((i) => corner(quads, i).x));
        expect([...xs].sort((a, b) => a - b)).toEqual([10 - half, 10 + half]);
    });

    it("expands a horizontal segment vertically (constant 1-px visual thickness)", () => {
        const vertices = new Float32Array([0, 8, 1, 0, 0, 1, 6, 8, 0, 1, 0, 1]);
        const quads = expandLineSegmentsToQuads(vertices);
        const ys = new Set([0, 1, 2, 3, 4, 5].map((i) => corner(quads, i).y));
        expect(ys.size).toBe(2);
        const half = WAVEFORM_STROKE_WIDTH_PX / 2;
        expect([...ys].sort((a, b) => a - b)).toEqual([8 - half, 8 + half]);
    });

    it("handles zero-length segments as a 1-px-tall strip (digital silence)", () => {
        const vertices = new Float32Array([3, 7, 0, 0, 0, 1, 3, 7, 0, 0, 0, 1]);
        const quads = expandLineSegmentsToQuads(vertices);
        const ys = [...new Set([0, 1, 2, 3, 4, 5].map((i) => corner(quads, i).y))].sort(
            (a, b) => a - b,
        );
        expect(ys).toEqual([7 - WAVEFORM_STROKE_WIDTH_PX / 2, 7 + WAVEFORM_STROKE_WIDTH_PX / 2]);
    });

    it("carries start color to corners A/D and end color to B/C", () => {
        // 起点红、终点绿，验证颜色插值端点正确。
        const vertices = new Float32Array([0, 0, 1, 0, 0, 1, 0, 4, 0, 1, 0, 1]);
        const quads = expandLineSegmentsToQuads(vertices);
        // 顶点序：A B C A C D（A/D 起点色，B/C 终点色）。
        const starts = [0, 3, 5];
        const ends = [1, 2, 4];
        for (const index of starts) {
            expect(corner(quads, index).r).toBe(1);
            expect(corner(quads, index).g).toBe(0);
        }
        for (const index of ends) {
            expect(corner(quads, index).g).toBe(1);
            expect(corner(quads, index).r).toBe(0);
        }
    });

    it("reuses the scratch buffer across calls (growth only)", () => {
        const small = new Float32Array(12);
        const first = expandLineSegmentsToQuads(small);
        const second = expandLineSegmentsToQuads(small);
        expect(second.buffer).toBe(first.buffer);
    });
});
