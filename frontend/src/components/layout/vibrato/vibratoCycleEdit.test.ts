import { describe, expect, test } from "vitest";

import { sampleCycle } from "../../../features/vibrato/vibratoCycle";
import {
    CYCLE_EDIT_BINS,
    CYCLE_SCALE_MAX,
    CYCLE_SCALE_MIN,
    EDITOR_VALUE_REACH_INSET,
    cycleEditorPoint,
    cycleEditorY,
    cycleRightDragTransform,
    editorValueReach,
    paintCycleBin,
    paintCycleSegment,
    rotateCycleTable,
    scaleCycleTable,
    smoothCycleTable,
    tableFromCycle,
    transformCycleTable,
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

    test("纵轴量程与绘制内缩同源：值 ±1 恰在内缩处", () => {
        const height = 120;
        // 峰顶画在离边缘 INSET 处；那里恰好是 1.0。
        expect(cycleEditorY(1, height)).toBeCloseTo(EDITOR_VALUE_REACH_INSET, 9);
        expect(cycleEditorY(-1, height)).toBeCloseTo(height - EDITOR_VALUE_REACH_INSET, 9);
        // 反过来：落在峰顶那一行读到 1.0（旧实现只有 0.93）。
        expect(cycleEditorPoint(0, EDITOR_VALUE_REACH_INSET, 640, height, 64).value).toBe(1);
        // 边带内（越过峰顶）钳在 1，不随像素继续上飘。
        expect(cycleEditorPoint(0, 0, 640, height, 64).value).toBe(1);
    });

    test("点与 y 互为逆运算", () => {
        const height = 120;
        for (const value of [-1, -0.5, 0, 0.5, 1]) {
            const y = cycleEditorY(value, height);
            expect(cycleEditorPoint(0, y, 640, height, 64).value).toBeCloseTo(value, 9);
        }
    });

    test("退化的纵轴量程保持为正（顶端不会翻转成 -1）", () => {
        expect(editorValueReach(0)).toBeGreaterThan(0);
        expect(editorValueReach(1)).toBeGreaterThan(0);
    });
});

describe("rotateCycleTable", () => {
    const sum = (values: readonly number[]) => values.reduce((total, value) => total + value, 0);

    test("零位移与整圈位移都是恒等", () => {
        const table = [0, 1, 0, -1, 0.5, 0];
        expect(rotateCycleTable(table, 0)).toEqual(table);
        expect(rotateCycleTable(table, table.length)).toEqual(table);
        expect(rotateCycleTable(table, -table.length)).toEqual(table);
    });

    test("不修改原表", () => {
        const table = [0, 1, 0];
        rotateCycleTable(table, 1);
        expect(table).toEqual([0, 1, 0]);
    });

    test("正位移把图像向右挪（峰值从 index 2 到 index 3）", () => {
        const next = rotateCycleTable([0, 0, 1, 0, 0, 0], 1);
        expect(next.indexOf(1)).toBe(3);
    });

    test("循环：越过右缘的峰值绕回左缘", () => {
        const next = rotateCycleTable([0, 0, 0, 0, 0, 1], 1);
        expect(next[0]).toBe(1);
    });

    test("位移取模：n + 1 与 1 等价", () => {
        const table = [0, 1, 0.2, -0.4, 0.8, 0];
        expect(rotateCycleTable(table, table.length + 1)).toEqual(rotateCycleTable(table, 1));
    });

    test("分数位移做线性插值", () => {
        // n = 4，位移 0.5 格：新 index 1 取旧 0 与 1 的中点。
        const next = rotateCycleTable([0, 1, 0, 0], 0.5);
        expect(next[1]).toBeCloseTo(0.5, 9);
    });

    test("循环重采样守恒总和（凸组合按周期配满权重）", () => {
        const table = [0, 1, 0.5, -1, 0.25, 0];
        expect(sum(rotateCycleTable(table, 1.3))).toBeCloseTo(sum(table), 9);
    });

    test("空表返回空表", () => {
        expect(rotateCycleTable([], 3)).toEqual([]);
    });
});

describe("scaleCycleTable", () => {
    test("1 倍是恒等，0.5 / 2 倍逐格缩放", () => {
        const table = [0, 0.5, 1, -0.5];
        expect(scaleCycleTable(table, 1)).toEqual(table);
        expect(scaleCycleTable(table, 0.5)).toEqual([0, 0.25, 0.5, -0.25]);
        // 放大到超出 ±1 的部分被钳住（增益顶到天花板）。
        expect(scaleCycleTable(table, 2)).toEqual([0, 1, 1, -1]);
    });

    test("结果始终落在 [-1,1]", () => {
        const table = [-1, 0.7, 0.3, 1];
        for (const factor of [0.05, 0.2, 1, 3, 4, 100]) {
            for (const value of scaleCycleTable(table, factor)) {
                expect(value).toBeGreaterThanOrEqual(-1);
                expect(value).toBeLessThanOrEqual(1);
            }
        }
    });

    test("缩到 0 附近被钳到下限：形状不被抹平，仍可恢复", () => {
        const next = scaleCycleTable([0, 1, 0, -1], 0);
        expect(Math.max(...next.map(Math.abs))).toBeGreaterThan(0);
        expect(next[1]).toBe(0.05);
    });

    test("非有限倍数按 1 处理，不产生 NaN", () => {
        expect(scaleCycleTable([0.5, 1], Number.NaN)).toEqual([0.5, 1]);
    });

    test("空表返回空表", () => {
        expect(scaleCycleTable([], 2)).toEqual([]);
    });
});

describe("transformCycleTable", () => {
    const table = [0, 1, 0.5, -1, 0.25, 0, -0.5, 0.75];

    test("等于「先旋转再缩放」两步的结果", () => {
        const combined = transformCycleTable(table, 1.5, 0.6);
        const stepwise = scaleCycleTable(rotateCycleTable(table, 1.5), 0.6);
        expect(combined.length).toBe(stepwise.length);
        for (let i = 0; i < combined.length; i += 1) {
            expect(combined[i]).toBeCloseTo(stepwise[i], 9);
        }
    });

    test("恒等变换逐位复原", () => {
        expect(transformCycleTable(table, 0, 1)).toEqual(table);
    });

    test("输出合法：定长、有限、钳在 [-1,1]", () => {
        const next = transformCycleTable(table, 2.7, 3);
        expect(next.length).toBe(table.length);
        for (const value of next) {
            expect(Number.isFinite(value)).toBe(true);
            expect(Math.abs(value)).toBeLessThanOrEqual(1);
        }
    });

    test("空表返回空表", () => {
        expect(transformCycleTable([], 1, 2)).toEqual([]);
    });
});

describe("cycleRightDragTransform", () => {
    test("水平拖满一个画布宽 = 一个整周期（恒等旋转量）", () => {
        expect(cycleRightDragTransform(640, 0, 640, 120, 64).rotateBins).toBe(64);
        expect(cycleRightDragTransform(320, 0, 640, 120, 64).rotateBins).toBe(32);
        // 非默认表长同样按"一个画布宽 = 一个周期"换算。
        expect(cycleRightDragTransform(640, 0, 640, 120, 100).rotateBins).toBe(100);
    });

    test("垂直向上拖满画布高 = 放大一倍，向下 = 减半", () => {
        expect(cycleRightDragTransform(0, -120, 640, 120, 64).scale).toBeCloseTo(2, 9);
        expect(cycleRightDragTransform(0, 120, 640, 120, 64).scale).toBeCloseTo(0.5, 9);
        expect(cycleRightDragTransform(0, 0, 640, 120, 64).scale).toBe(1);
    });

    test("放大 / 缩小的钳制生效", () => {
        expect(cycleRightDragTransform(0, -100000, 640, 120, 64).scale).toBe(CYCLE_SCALE_MAX);
        expect(cycleRightDragTransform(0, 100000, 640, 120, 64).scale).toBe(CYCLE_SCALE_MIN);
    });

    test("退化尺寸（宽 / 高为 0）退化为不变换，不产生 NaN", () => {
        const spec = cycleRightDragTransform(50, 50, 0, 0, 64);
        expect(spec.rotateBins).toBe(0);
        expect(spec.scale).toBe(1);
    });

    test("非有限位移按 0 处理", () => {
        const spec = cycleRightDragTransform(Number.NaN, Number.NaN, 640, 120, 64);
        expect(spec.rotateBins).toBe(0);
        expect(spec.scale).toBe(1);
    });
});
