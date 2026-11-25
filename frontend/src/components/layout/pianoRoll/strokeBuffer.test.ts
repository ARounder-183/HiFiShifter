import { describe, expect, it } from "vitest";

import { createStrokeBuffer, MAX_STROKE_FRAMES } from "./strokeBuffer";
import type { StrokeBuffer } from "./strokeBuffer";

/**
 * 笔画缓冲（绝对帧）的纯逻辑测试。
 *
 * 【为什么用"参照实现对拍"而不是逐条断言】缓冲内部按**有序、互不重叠、已合并**的段
 * 存储，并允许段内数组左右生长；这套结构一旦有边界错误（相邻段没合并、擦除只修了一半、
 * 左延伸把尾巴截掉），症状都是"某个范围的值静默不对" —— 只针对单个场景写断言会漏。
 * 因此这里用最朴素的 `Map<帧, 值>` 作为参照模型，随机序列逐次对拍。
 */

/** 把缓冲展开成"帧 → 值"（测试用的参照读取）。 */
function toMap(buffer: StrokeBuffer): Map<number, number> {
    const map = new Map<number, number>();
    for (const segment of buffer.segments(Number.MAX_SAFE_INTEGER)) {
        for (let i = 0; i < segment.values.length; i += 1) {
            map.set(segment.startFrame + i, segment.values[i]);
        }
    }
    return map;
}

/** 可复现的伪随机数（线性同余）。 */
function makeRandom(seed: number): () => number {
    let state = seed >>> 0;
    return () => {
        state = (state * 1664525 + 1013904223) >>> 0;
        return state / 0x100000000;
    };
}

describe("createStrokeBuffer：写入与合并", () => {
    it("merges a continuous stroke into one run", () => {
        const buffer = createStrokeBuffer("v2|t|volume");
        for (let frame = 0; frame < 10; frame += 1) buffer.write(frame, [frame * 10]);

        const segments = buffer.segments(1024);
        expect(segments).toHaveLength(1);
        expect(segments[0].startFrame).toBe(0);
        expect(segments[0].values).toEqual([0, 10, 20, 30, 40, 50, 60, 70, 80, 90]);
        expect(buffer.frameCount).toBe(10);
    });

    it("overwrites frames that are written twice", () => {
        const buffer = createStrokeBuffer("v2|t|volume");
        buffer.write(0, [1, 1, 1, 1, 1]);
        buffer.write(2, [9, 9, 9, 9, 9]);

        expect(Array.from(toMap(buffer).entries())).toEqual([
            [0, 1],
            [1, 1],
            [2, 9],
            [3, 9],
            [4, 9],
            [5, 9],
            [6, 9],
        ]);
        // 仍然是一段：2..6 与 0..4 重叠且向右延伸。
        expect(buffer.segments(1024)).toHaveLength(1);
    });

    it("extends to the left without losing the tail", () => {
        // 用户反向拖拽（从右往左画）：每次写入都与最后一段的头部重叠并向前延伸。
        const buffer = createStrokeBuffer("v2|t|volume");
        buffer.write(10, [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]);
        for (let frame = 9; frame >= 0; frame -= 1) buffer.write(frame, [frame]);

        const segments = buffer.segments(1024);
        expect(segments).toHaveLength(1);
        expect(segments[0].startFrame).toBe(0);
        expect(segments[0].values).toEqual([
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        ]);
        expect(buffer.frameCount).toBe(20);
    });

    it("keeps disjoint ranges as separate runs", () => {
        const buffer = createStrokeBuffer("v2|t|volume");
        buffer.write(0, [1, 2, 3, 4, 5]);
        buffer.write(10, [6, 7, 8, 9, 10]);

        expect(buffer.segments(1024)).toHaveLength(2);
        expect(buffer.frameCount).toBe(10);
        expect(buffer.range()).toEqual({ start: 0, end: 15 });
    });

    it("bridges two runs when the gap is filled", () => {
        const buffer = createStrokeBuffer("v2|t|volume");
        buffer.write(0, [1, 2]);
        buffer.write(4, [5, 6]);
        buffer.write(2, [3, 4]);

        const segments = buffer.segments(1024);
        expect(segments).toHaveLength(1);
        expect(segments[0].values).toEqual([1, 2, 3, 4, 5, 6]);
    });

    it("applies the clamp to every written value", () => {
        const buffer = createStrokeBuffer("v2|t|volume");
        buffer.write(0, [0, 5, 50], { clamp: (value) => Math.min(10, value) });

        expect(buffer.segments(1024)[0].values).toEqual([0, 5, 10]);
    });

    it("skips holes instead of inventing values", () => {
        // 步长 > 1 的窗口上，拖拽预览只落在栅格帧上（dense 里是空洞）。
        const buffer = createStrokeBuffer("v2|t|volume");
        buffer.write(0, [1, undefined, 3, undefined, 5]);

        expect(Array.from(toMap(buffer).entries())).toEqual([
            [0, 1],
            [2, 3],
            [4, 5],
        ]);
        // 空洞把一次写入切成三段，但绝不能凭空造出帧 1 / 3 的值。
        expect(buffer.valueAt(1)).toBeUndefined();
        expect(buffer.valueAt(3)).toBeUndefined();
    });

    it("honours the from/to bounds", () => {
        const buffer = createStrokeBuffer("v2|t|volume");
        buffer.write(0, [1, 2, 3, 4, 5], { from: 1, to: 3 });

        expect(Array.from(toMap(buffer).entries())).toEqual([
            [1, 2],
            [2, 3],
            [3, 4],
        ]);
    });
});

describe("createStrokeBuffer：擦除", () => {
    it("splits a run in two", () => {
        const buffer = createStrokeBuffer("v2|t|volume");
        buffer.write(0, [1, 2, 3, 4, 5]);
        expect(buffer.erase(2, 3)).toBe(true);

        expect(buffer.segments(1024).map((s) => [s.startFrame, s.values])).toEqual([
            [0, [1, 2]],
            [3, [4, 5]],
        ]);
        expect(buffer.frameCount).toBe(4);
    });

    it("drops whole runs and trims partial ones", () => {
        const buffer = createStrokeBuffer("v2|t|volume");
        buffer.write(0, [1, 2]);
        buffer.write(10, [3, 4]);
        buffer.write(20, [5, 6]);
        buffer.erase(1, 21);

        expect(buffer.segments(1024).map((s) => [s.startFrame, s.values])).toEqual([
            [0, [1]],
            [21, [6]],
        ]);
    });

    it("reports a no-op erase and does not bump the version", () => {
        const buffer = createStrokeBuffer("v2|t|volume");
        buffer.write(10, [1, 2]);
        const version = buffer.version;

        expect(buffer.erase(0, 5)).toBe(false);
        expect(buffer.version).toBe(version);
    });

    it("empties the buffer when everything is erased", () => {
        const buffer = createStrokeBuffer("v2|t|volume");
        buffer.write(0, [1, 2, 3]);
        buffer.erase(0, 3);

        expect(buffer.range()).toBeNull();
        expect(buffer.frameCount).toBe(0);
        expect(buffer.segments(1024)).toEqual([]);
    });
});

describe("createStrokeBuffer：查询与版本", () => {
    it("answers valueAt only for covered frames", () => {
        const buffer = createStrokeBuffer("v2|t|volume");
        buffer.write(5, [10, 11, 12]);

        expect(buffer.valueAt(4)).toBeUndefined();
        expect(buffer.valueAt(5)).toBe(10);
        expect(buffer.valueAt(7)).toBe(12);
        expect(buffer.valueAt(8)).toBeUndefined();
    });

    it("advances the version on every content change", () => {
        const buffer = createStrokeBuffer("v2|t|volume");
        const start = buffer.version;
        buffer.write(0, [1]);
        const afterWrite = buffer.version;
        buffer.erase(0, 1);
        const afterErase = buffer.version;

        expect(afterWrite).toBeGreaterThan(start);
        expect(afterErase).toBeGreaterThan(afterWrite);
    });

    it("splits long runs into bounded segments", () => {
        const buffer = createStrokeBuffer("v2|t|volume");
        const dense = new Array<number>(10).fill(1).map((_, index) => index);
        buffer.write(0, dense);

        const segments = buffer.segments(4);
        expect(segments.map((s) => [s.startFrame, s.values.length])).toEqual([
            [0, 4],
            [4, 4],
            [8, 2],
        ]);
        expect(segments.flatMap((s) => s.values)).toEqual(dense);
    });

    it("stops accepting frames past MAX_STROKE_FRAMES and says so", () => {
        const buffer = createStrokeBuffer("v2|t|volume");
        const dense = new Array<number>(MAX_STROKE_FRAMES + 1).fill(1);
        buffer.write(0, dense);

        expect(buffer.truncated).toBe(true);
        expect(buffer.frameCount).toBe(0);
    });

    it("clears everything and bumps the version", () => {
        const buffer = createStrokeBuffer("v2|t|volume");
        buffer.write(0, [1, 2, 3]);
        const version = buffer.version;
        buffer.clear();

        expect(buffer.frameCount).toBe(0);
        expect(buffer.version).toBeGreaterThan(version);
    });
});

describe("createStrokeBuffer：与朴素模型对拍", () => {
    it("matches a Map<frame, value> model over a random operation sequence", () => {
        const random = makeRandom(0x5eed);
        const buffer = createStrokeBuffer("v2|t|volume");
        const model = new Map<number, number>();

        for (let step = 0; step < 400; step += 1) {
            const startFrame = Math.floor(random() * 200) - 20;
            if (random() < 0.75) {
                const length = 1 + Math.floor(random() * 12);
                const dense: (number | undefined)[] = [];
                for (let i = 0; i < length; i += 1) {
                    // 20% 的帧是空洞（步长 > 1 的窗口留下的）。
                    dense.push(random() < 0.2 ? undefined : Math.round(random() * 100));
                }
                const from = startFrame;
                const to = startFrame + length - 1;
                buffer.write(startFrame, dense, { from, to });
                for (let i = 0; i < length; i += 1) {
                    const value = dense[i];
                    if (value === undefined) continue;
                    model.set(startFrame + i, value);
                }
            } else {
                const end = startFrame + 1 + Math.floor(random() * 30);
                buffer.erase(startFrame, end);
                for (const frame of Array.from(model.keys())) {
                    if (frame >= startFrame && frame < end) model.delete(frame);
                }
            }

            const actual = toMap(buffer);
            expect(actual.size).toBe(model.size);
            for (const [frame, value] of model) {
                expect(actual.get(frame)).toBe(value);
            }
        }
    });
});
