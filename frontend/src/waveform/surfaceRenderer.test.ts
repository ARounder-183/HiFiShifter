import { describe, expect, it } from "vitest";

import { waveformColumnWidthDevicePx } from "./geometry.ts";
import { expandLineSegmentsToQuads } from "./surfaceRenderer.ts";

/**
 * 栅格对齐契约（与 geometry.ts 的 `waveformColumnWidthDevicePx` 同源）：
 * 包络列以**中心线**语义写出，展开后必须恰好覆盖以设备像素边界为边的
 * `W` 物理像素列（W = waveformColumnWidthDevicePx(dpr)）。
 */
function expectedHalfCss(dpr: number): number {
    return waveformColumnWidthDevicePx(dpr) / dpr / 2;
}

/** 读取展开输出中某个角点的位置与颜色。 */
function corner(
    quads: Float32Array,
    vertex: number,
): {
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
    it("expands a vertical envelope column into a W-device-px-wide quad (dpr=1 → 1 CSS px)", () => {
        // 一条竖直包络列：x = 10，y 从 5 到 15，纯黑不透明。
        const vertices = new Float32Array([10, 5, 0, 0, 0, 1, 10, 15, 0, 0, 0, 1]);
        const quads = expandLineSegmentsToQuads(vertices);
        expect(quads.length).toBe(36);

        const half = expectedHalfCss(1);
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

    it("expands a horizontal segment vertically (constant W-px visual thickness)", () => {
        const vertices = new Float32Array([0, 8, 1, 0, 0, 1, 6, 8, 0, 1, 0, 1]);
        const quads = expandLineSegmentsToQuads(vertices);
        const ys = new Set([0, 1, 2, 3, 4, 5].map((i) => corner(quads, i).y));
        expect(ys.size).toBe(2);
        const half = expectedHalfCss(1);
        expect([...ys].sort((a, b) => a - b)).toEqual([8 - half, 8 + half]);
    });

    it("handles zero-length segments as a W-px-tall strip (digital silence)", () => {
        const vertices = new Float32Array([3, 7, 0, 0, 0, 1, 3, 7, 0, 0, 0, 1]);
        const quads = expandLineSegmentsToQuads(vertices);
        const ys = [...new Set([0, 1, 2, 3, 4, 5].map((i) => corner(quads, i).y))].sort(
            (a, b) => a - b,
        );
        expect(ys).toEqual([7 - expectedHalfCss(1), 7 + expectedHalfCss(1)]);
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

    /**
     * 栅格对齐回归（用户报告的"线宽 1~2px 抖动"）：
     * dpr=2 时 W=2，几何端写出的列中心是 `(k + 0.5)·W/dpr = k + 0.5` CSS px，
     * 展开后四角必须精确落在 `[k, k+1]` CSS px = `[2k, 2k+2]` **设备**像素上。
     */
    it("dpr=2: envelope column covers exactly 2 device px with device-grid-aligned edges", () => {
        // dpr=2、W=2 的第 0 列：几何端中心 x = (0+0.5)·2/2 = 0.5 CSS px。
        const vertices = new Float32Array([0.5, 4, 0, 0, 0, 1, 0.5, 9, 0, 0, 0, 1]);
        const quads = expandLineSegmentsToQuads(vertices, 2);
        const xs = [...new Set([0, 1, 2, 3, 4, 5].map((i) => corner(quads, i).x))].sort(
            (a, b) => a - b,
        );
        // 半宽 = W/(2·dpr) = 0.5 CSS px → 覆盖 [0, 1] CSS = [0, 2] 设备像素。
        expect(xs).toEqual([0, 1]);
    });

    /**
     * dpr=1.25（Windows 125% 缩放）：W=1，列宽恒 1 物理像素且边落在设备网格上
     * —— 旧行为（1 CSS px 列宽）下这一列会覆盖 1~2 个物理像素中心，随位置
     * 跳变，即"线宽抖动"。
     */
    it("dpr=1.25: envelope column is exactly 1 device px wide", () => {
        // dpr=1.25、W=1 的第 1 列：中心 x = (1+0.5)·1/1.25 = 1.2 CSS px。
        const vertices = new Float32Array([1.2, 4, 0, 0, 0, 1, 1.2, 9, 0, 0, 0, 1]);
        const quads = expandLineSegmentsToQuads(vertices, 1.25);
        const xs = [...new Set([0, 1, 2, 3, 4, 5].map((i) => corner(quads, i).x))].sort(
            (a, b) => a - b,
        );
        // 半宽 = 1/(2·1.25) = 0.4 CSS px → 覆盖 [0.8, 1.6] CSS = [1, 2] 设备像素。
        // 顶点入参经 Float32 存储舍入，保留 6 位有效精度足够钉住"落在设备网格上"。
        expect(xs[0] * 1.25).toBeCloseTo(1, 6);
        expect(xs[1] * 1.25).toBeCloseTo(2, 6);
    });

    it("non-finite / non-positive dpr falls back to 1 device px (dpr=1 semantics)", () => {
        const vertices = new Float32Array([10, 5, 0, 0, 0, 1, 10, 15, 0, 0, 0, 1]);
        const baseline = expandLineSegmentsToQuads(vertices, 1);
        for (const dpr of [0, -2, Number.NaN]) {
            const quads = expandLineSegmentsToQuads(vertices, dpr);
            for (let i = 0; i < baseline.length; i += 1) {
                expect(Object.is(quads[i], baseline[i])).toBe(true);
            }
        }
    });
});

/**
 * 内联展开的逐字节等价性回归。
 *
 * 背景：热路径上原本有一个 `corners` 数组字面量（每段 7 个数组，15,200 段
 * ≈ 10.6 万次分配/帧），已改为六组直接写入。这两者必须在数值上**完全一致**，
 * 否则波形的四角位置或端点颜色插值会出现肉眼可见的偏差。
 *
 * 参照实现就写在测试里（保持改前的字面形态），对随机段集合做 Float32
 * 位级比对——浮点写入顺序未变，因此期望严格逐位相等而非近似。
 */
describe("expandLineSegmentsToQuads 内联展开 ≡ 改前实现", () => {
    /** 改前的实现：数组字面量 + 循环。仅作正确性参照，不参与生产。 */
    function referenceExpand(vertices: Float32Array, dpr = 1): Float32Array {
        const half = waveformColumnWidthDevicePx(dpr) / dpr / 2;
        const segmentCount = Math.floor(vertices.length / 12);
        const out = new Float32Array(segmentCount * 36);
        for (let segment = 0; segment < segmentCount; segment += 1) {
            const base = segment * 12;
            const x1 = vertices[base];
            const y1 = vertices[base + 1];
            const x2 = vertices[base + 6];
            const y2 = vertices[base + 7];
            const dx = x2 - x1;
            const dy = y2 - y1;
            const length = Math.hypot(dx, dy);
            const nx = length > 1e-9 ? (-dy / length) * half : 0;
            const ny = length > 1e-9 ? (dx / length) * half : half;
            const outBase = segment * 36;
            const corners = [
                [x1 + nx, y1 + ny, base + 2],
                [x2 + nx, y2 + ny, base + 8],
                [x2 - nx, y2 - ny, base + 8],
                [x1 + nx, y1 + ny, base + 2],
                [x2 - nx, y2 - ny, base + 8],
                [x1 - nx, y1 - ny, base + 2],
            ] as const;
            for (let corner = 0; corner < 6; corner += 1) {
                const [cx, cy, colorBase] = corners[corner];
                const o = outBase + corner * 6;
                out[o] = cx;
                out[o + 1] = cy;
                out[o + 2] = vertices[colorBase];
                out[o + 3] = vertices[colorBase + 1];
                out[o + 4] = vertices[colorBase + 2];
                out[o + 5] = vertices[colorBase + 3];
            }
        }
        return out;
    }

    /** 确定性伪随机（回归必须可复现，禁用 Math.random）。 */
    function createRng(seed: number): () => number {
        let state = seed >>> 0;
        return () => {
            state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
            return state / 0x1_0000_0000;
        };
    }

    /** 形态集合：竖列（包络）/ 水平（标记扫描线）/ 斜向 / 零长度（数字静音）。 */
    function buildShapeVertices(shapeIndex: number, segmentCount: number): Float32Array {
        const shapes: Array<
            (rng: () => number, index: number) => [number, number, number, number]
        > = [
            (_rng, index) => [index * 1.5 + 0.5, 4, index * 1.5 + 0.5, 20],
            (_rng, index) => [index * 2, 8, index * 2 + 6, 8],
            (rng, index) => [index * 3, rng() * 40, index * 3 + rng() * 20, rng() * 40],
            (_rng, index) => [index * 2, 12, index * 2, 12],
        ];
        const rng = createRng(shapeIndex + 1);
        const build = shapes[shapeIndex] as (
            rng: () => number,
            index: number,
        ) => [number, number, number, number];
        const vertices = new Float32Array(segmentCount * 12);
        for (let index = 0; index < segmentCount; index += 1) {
            const [x1, y1, x2, y2] = build(rng, index);
            const base = index * 12;
            vertices[base] = x1;
            vertices[base + 1] = y1;
            vertices[base + 2] = rng();
            vertices[base + 3] = rng();
            vertices[base + 4] = rng();
            vertices[base + 5] = rng();
            vertices[base + 6] = x2;
            vertices[base + 7] = y2;
            vertices[base + 8] = rng();
            vertices[base + 9] = rng();
            vertices[base + 10] = rng();
            vertices[base + 11] = rng();
        }
        return vertices;
    }

    it("随机段集合逐位相等（含竖直 / 水平 / 斜向 / 零长度四类）", () => {
        const segmentCount = 32;
        for (let shapeIndex = 0; shapeIndex < 4; shapeIndex += 1) {
            const vertices = buildShapeVertices(shapeIndex, segmentCount);
            const actual = expandLineSegmentsToQuads(vertices);
            const expected = referenceExpand(vertices);
            expect(actual.length).toBe(expected.length);
            for (let i = 0; i < expected.length; i += 1) {
                // 逐位比较：浮点写入顺序未变，必须严格相等。
                expect(Object.is(actual[i], expected[i])).toBe(true);
            }
        }
    });

    it("dpr=1.5 / 2 时与参照实现逐位相等（宽度走设备像素契约）", () => {
        const segmentCount = 24;
        for (let shapeIndex = 0; shapeIndex < 4; shapeIndex += 1) {
            const vertices = buildShapeVertices(shapeIndex, segmentCount);
            for (const dpr of [1.5, 2]) {
                const actual = expandLineSegmentsToQuads(vertices, dpr);
                const expected = referenceExpand(vertices, dpr);
                expect(actual.length).toBe(expected.length);
                for (let i = 0; i < expected.length; i += 1) {
                    expect(Object.is(actual[i], expected[i])).toBe(true);
                }
            }
        }
    });
});
