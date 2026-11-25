import { describe, expect, it } from "vitest";

import { createStrokeBuffer } from "./strokeBuffer";
import type { StrokeBuffer } from "./strokeBuffer";

/**
 * 投影：把**绝对帧**缓冲映射到任意取数窗口。
 *
 * 【为什么单独一份】"笔画的坐标是绝对帧，窗口只是投影"这条不变式唯一可观测的后果就是
 * 这个函数：同一份缓冲必须能投影到步长不同、起点不同、只有部分相交的任意窗口，且结果
 * 与"逐帧查表"完全一致。它是"边缘滚屏时参数线不再更新"那个缺陷的修复核心 ——
 * 旧实现把绝对帧反解成窗口下标并夹住，越界帧被静默丢弃。
 *
 * 参照实现刻意写成最笨的逐帧查表（`valueAt`），与优化实现（只遍历相交段）对拍。
 */
function referenceProject(
    buffer: StrokeBuffer,
    base: readonly number[],
    startFrame: number,
    stride: number,
): number[] {
    return base.map((value, index) => buffer.valueAt(startFrame + index * stride) ?? value);
}

/** 造一份"两段、段间有空洞"的缓冲：帧 10..19 与 40..49。 */
function makeBuffer(): StrokeBuffer {
    const buffer = createStrokeBuffer("v2|t|volume");
    buffer.write(
        10,
        new Array<number>(10).fill(0).map((_, index) => 100 + index),
    );
    buffer.write(
        40,
        new Array<number>(10).fill(0).map((_, index) => 400 + index),
    );
    return buffer;
}

describe("StrokeBuffer.project", () => {
    it("replaces covered frames and keeps the base elsewhere", () => {
        const buffer = makeBuffer();
        const base = new Array<number>(30).fill(-1);

        const projected = buffer.project(base, 0, 1);

        expect(projected).toHaveLength(30);
        expect(projected.slice(0, 10)).toEqual(new Array<number>(10).fill(-1));
        expect(projected.slice(10, 20)).toEqual(
            new Array<number>(10).fill(0).map((_, i) => 100 + i),
        );
        expect(projected.slice(20)).toEqual(new Array<number>(10).fill(-1));
        expect(base[10]).toBe(-1); // 基准数组本身不被改写
    });

    it("samples every stride from the same buffer", () => {
        const buffer = makeBuffer();
        for (const stride of [1, 2, 3, 5, 8]) {
            const base = new Array<number>(20).fill(-1);
            const startFrame = 5;
            expect(buffer.project(base, startFrame, stride)).toEqual(
                referenceProject(buffer, base, startFrame, stride),
            );
        }
    });

    it("handles windows that only partially overlap a run", () => {
        const buffer = makeBuffer();
        // 窗口 [5, 5+8*1) 与 10..19 相交于 10..12。
        expect(buffer.project(new Array<number>(8).fill(-1), 5, 1)).toEqual([
            -1, -1, -1, -1, -1, 100, 101, 102,
        ]);
        // 窗口 [17, 17+6*1) 与 10..19 相交于 17..19。
        expect(buffer.project(new Array<number>(6).fill(-1), 17, 1)).toEqual([
            107, 108, 109, -1, -1, -1,
        ]);
    });

    it("returns the base untouched when the window misses the buffer", () => {
        const buffer = makeBuffer();
        const base = [1, 2, 3, 4];
        expect(buffer.project(base, 1000, 1)).toEqual(base);
        expect(buffer.project(base, 0, 1)).toEqual(base);
    });

    it("reuses the caller's array when the length matches", () => {
        const buffer = makeBuffer();
        const out = new Array<number>(30).fill(0);
        expect(buffer.project(new Array<number>(30).fill(-1), 0, 1, out)).toBe(out);
        // 长度不一致时必须换一份（否则会把上一个窗口的残留写进画布）。
        expect(buffer.project(new Array<number>(31).fill(-1), 0, 1, out)).not.toBe(out);
    });

    it("handles an empty window and a degenerate stride", () => {
        const buffer = makeBuffer();
        expect(buffer.project([], 0, 1)).toEqual([]);
        expect(buffer.project([7, 8], 0, 0)).toEqual([7, 8]);
    });

    it("agrees with per-frame lookup for random windows", () => {
        const buffer = makeBuffer();
        let seed = 12345;
        const random = () => {
            seed = (seed * 1664525 + 1013904223) >>> 0;
            return seed / 0x100000000;
        };
        for (let trial = 0; trial < 200; trial += 1) {
            const startFrame = Math.floor(random() * 120) - 30;
            const stride = 1 + Math.floor(random() * 6);
            const length = 1 + Math.floor(random() * 40);
            const base = new Array<number>(length).fill(-1);
            expect(buffer.project(base, startFrame, stride)).toEqual(
                referenceProject(buffer, base, startFrame, stride),
            );
        }
    });
});
