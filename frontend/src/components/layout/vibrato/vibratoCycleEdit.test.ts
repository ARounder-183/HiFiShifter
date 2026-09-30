import { describe, expect, test } from "vitest";

import { sampleCycle } from "../../../features/vibrato/vibratoCycle";
import {
    CYCLE_EDIT_BINS,
    cycleEditorPoint,
    paintCycleBin,
    paintCycleSegment,
    smoothCycleTable,
    tableFromCycle,
} from "./vibratoCycleEdit";

describe("tableFromCycle", () => {
    test("把参数形状采样成定长表（正弦起步）", () => {
        const table = tableFromCycle({ kind: "shape", shape: "sine", skew: 0.5 });
        expect(table.length).toBe(CYCLE_EDIT_BINS);
        // 与直接采样一致（同一套 sampleCycle）。
        for (let i = 0; i < table.length; i += 1) {
            expect(table[i]).toBeCloseTo(
                sampleCycle({ kind: "shape", shape: "sine", skew: 0.5 }, i / table.length),
                9,
            );
        }
        expect(table[0]).toBeCloseTo(0, 9);
    });

    test("table 来源原样采样（不引入额外变形）", () => {
        const source = [0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5];
        const table = tableFromCycle({ kind: "table", table: source }, 8);
        expect(table.length).toBe(8);
        for (let i = 0; i < 8; i += 1) expect(table[i]).toBeCloseTo(source[i], 9);
    });

    test("格数被钳在合法范围内", () => {
        expect(tableFromCycle({ kind: "shape", shape: "sine", skew: 0.5 }, 1).length).toBe(8);
        expect(tableFromCycle({ kind: "shape", shape: "sine", skew: 0.5 }, 9999).length).toBe(256);
    });
});

describe("paintCycleBin", () => {
    test("写入一格并钳到 [-1,1]，不改动原表", () => {
        const table = [0, 0, 0, 0];
        const next = paintCycleBin(table, 2, 5);
        expect(next[2]).toBe(1);
        expect(table[2]).toBe(0);
        expect(paintCycleBin(table, 2, -9)[2]).toBe(-1);
    });

    test("格号越界被钳到边界", () => {
        expect(paintCycleBin([0, 0, 0], -5, 1)[0]).toBe(1);
        expect(paintCycleBin([0, 0, 0], 99, 1)[2]).toBe(1);
    });

    test("空表返回空表", () => {
        expect(paintCycleBin([], 0, 1)).toEqual([]);
    });
});

describe("paintCycleSegment", () => {
    test("在两点之间线性补间，填满每一格（快笔不留断线）", () => {
        const next = paintCycleSegment(
            [0, 0, 0, 0, 0],
            { bin: 0, value: -1 },
            { bin: 4, value: 1 },
        );
        expect(next[0]).toBeCloseTo(-1, 9);
        expect(next[2]).toBeCloseTo(0, 9);
        expect(next[4]).toBeCloseTo(1, 9);
    });

    test("方向无关（从右往左画得到同一结果）", () => {
        const forward = paintCycleSegment(
            [0, 0, 0, 0, 0],
            { bin: 0, value: -1 },
            { bin: 4, value: 1 },
        );
        const backward = paintCycleSegment(
            [0, 0, 0, 0, 0],
            { bin: 4, value: 1 },
            { bin: 0, value: -1 },
        );
        expect(backward).toEqual(forward);
    });

    test("同一格（span 为 0）只写该格", () => {
        const next = paintCycleSegment([0, 0, 0], { bin: 1, value: 0.5 }, { bin: 1, value: 0.5 });
        expect(next).toEqual([0, 0.5, 0]);
    });
});

describe("smoothCycleTable", () => {
    test("三点平均削平尖峰", () => {
        const table = [0, 0, 1, 0, 0];
        const next = smoothCycleTable(table);
        expect(next[2]).toBeCloseTo(1 / 3, 9);
    });

    test("循环：端点参与邻居平均（接缝不产生折角）", () => {
        // 表首尾相接，索引 0 的邻居是末项与第 2 项。
        const table = [1, 0, 0, 0, 1];
        const next = smoothCycleTable(table);
        expect(next[0]).toBeCloseTo((table[4] + table[0] + table[1]) / 3, 9);
    });

    test("长度不变、可连点（循环平均守恒总量）", () => {
        const table = [0, 0, 1, 0, 0, 0];
        const once = smoothCycleTable(table);
        const twice = smoothCycleTable(once);
        expect(once.length).toBe(table.length);
        expect(twice.length).toBe(table.length);
        // 循环滑动平均是守恒的：总和不变。
        const sum = (values: number[]) => values.reduce((total, value) => total + value, 0);
        expect(sum(once)).toBeCloseTo(sum(table), 9);
        expect(sum(twice)).toBeCloseTo(sum(table), 9);
    });

    test("空表返回空表", () => {
        expect(smoothCycleTable([])).toEqual([]);
    });
});

describe("cycleEditorPoint", () => {
    test("x → 格号，y → 值（顶端为 +1）", () => {
        // 宽 640 / 64 格 → 每格 10px。
        expect(cycleEditorPoint(5, 0, 640, 120, 64)).toEqual({ bin: 0, value: 1 });
        expect(cycleEditorPoint(15, 120, 640, 120, 64)).toEqual({ bin: 1, value: -1 });
        const mid = cycleEditorPoint(320, 60, 640, 120, 64);
        expect(mid.bin).toBe(32);
        expect(mid.value).toBeCloseTo(0, 9);
    });

    test("退化尺寸不产生 NaN", () => {
        const point = cycleEditorPoint(0, 0, 0, 0, 64);
        expect(Number.isFinite(point.bin)).toBe(true);
        expect(Number.isFinite(point.value)).toBe(true);
    });
});
