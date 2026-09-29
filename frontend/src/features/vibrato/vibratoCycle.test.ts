import { describe, expect, test } from "vitest";

import {
    CYCLE_TABLE_DEFAULT_LEN,
    CYCLE_TABLE_MAX_LEN,
    CYCLE_TABLE_MIN_LEN,
    makeSineTable,
    normalizeCycleTable,
    sampleCycle,
    shapeUsesSkew,
    wrap01,
} from "./vibratoCycle";
import type { CycleSource, WaveShape } from "./vibratoTypes";

describe("wrap01", () => {
    test("把任意实数折进 [0,1)", () => {
        expect(wrap01(0)).toBe(0);
        expect(wrap01(0.25)).toBeCloseTo(0.25, 12);
        expect(wrap01(1)).toBeCloseTo(0, 12);
        expect(wrap01(1.5)).toBeCloseTo(0.5, 12);
        // 负相位必须折到正向，否则 `u = -0.1` 会得到波形的"倒退"段。
        expect(wrap01(-0.25)).toBeCloseTo(0.75, 12);
        expect(wrap01(-1.25)).toBeCloseTo(0.75, 12);
    });

    test("非有限值折成 0 而不是 NaN", () => {
        expect(wrap01(Number.NaN)).toBe(0);
        expect(wrap01(Number.POSITIVE_INFINITY)).toBe(0);
    });
});

describe("形状采样", () => {
    const shape = (s: WaveShape, skew = 0.5): CycleSource => ({ kind: "shape", shape: s, skew });

    test("正弦：一个周期内四个象限的取值", () => {
        const sine = shape("sine");
        expect(sampleCycle(sine, 0)).toBeCloseTo(0, 12);
        expect(sampleCycle(sine, 0.25)).toBeCloseTo(1, 12);
        expect(sampleCycle(sine, 0.5)).toBeCloseTo(0, 12);
        expect(sampleCycle(sine, 0.75)).toBeCloseTo(-1, 12);
        expect(sampleCycle(sine, 1)).toBeCloseTo(0, 12);
    });

    test("三角：与正弦同一起点（u=0 过零上行），偏斜 0.5 时为标准三角", () => {
        const tri = shape("triangle");
        expect(sampleCycle(tri, 0)).toBeCloseTo(0, 12);
        expect(sampleCycle(tri, 0.25)).toBeCloseTo(1, 12);
        expect(sampleCycle(tri, 0.5)).toBeCloseTo(0, 12);
        expect(sampleCycle(tri, 0.75)).toBeCloseTo(-1, 12);
    });

    test("三角：偏斜改变上升段占比", () => {
        // 上升段占比 0.25 时，上升边横跨 u ∈ [-0.125, 0.125]，峰值在 u = 0.125。
        const tri = shape("triangle", 0.25);
        expect(sampleCycle(tri, 0.125)).toBeCloseTo(1, 12);
        // u = 0 仍在上升边的中点 —— 即过零点。
        expect(sampleCycle(tri, 0)).toBeCloseTo(0, 12);
        // 下降边从 0.125 走到 0.875，其中点 0.5 是第二个过零点。
        expect(sampleCycle(tri, 0.5)).toBeCloseTo(0, 12);
        expect(sampleCycle(tri, 0.6)).toBeLessThan(0);
        expect(sampleCycle(tri, 0.05)).toBeGreaterThan(0);
    });

    test("方波：u=0 在高电平，占空比由偏斜决定", () => {
        expect(sampleCycle(shape("square", 0.5), 0)).toBe(1);
        expect(sampleCycle(shape("square", 0.5), 0.49)).toBe(1);
        expect(sampleCycle(shape("square", 0.5), 0.5)).toBe(-1);
        expect(sampleCycle(shape("square", 0.25), 0.3)).toBe(-1);
        expect(sampleCycle(shape("square", 0.75), 0.7)).toBe(1);
    });

    test("锯齿：u=0 过零上行，半周期处发生跳变", () => {
        const up = shape("sawUp");
        expect(sampleCycle(up, 0)).toBeCloseTo(0, 12);
        expect(sampleCycle(up, 0.25)).toBeCloseTo(0.5, 12);
        expect(sampleCycle(up, 0.5)).toBeCloseTo(-1, 12);
        expect(sampleCycle(up, 0.75)).toBeCloseTo(-0.5, 12);

        const down = shape("sawDown");
        expect(sampleCycle(down, 0)).toBeCloseTo(0, 12);
        expect(sampleCycle(down, 0.25)).toBeCloseTo(-0.5, 12);
        expect(sampleCycle(down, 0.5)).toBeCloseTo(1, 12);
    });

    test("梯形：u=0 处于过渡边上（值 0），平台段取 ±1", () => {
        const trap = shape("trapezoid", 0.5);
        expect(sampleCycle(trap, 0)).toBeCloseTo(0, 12);
        // 占空比 0.5、边宽 0.12：平台在 [0.12, 0.38]。
        expect(sampleCycle(trap, 0.25)).toBeCloseTo(1, 12);
        expect(sampleCycle(trap, 0.75)).toBeCloseTo(-1, 12);
    });

    test("颤指比梯形更陡（边宽更窄）", () => {
        const trap = shape("trapezoid", 0.5);
        const trill = shape("trill", 0.5);
        // 同样在过渡区内，颤指更接近满幅。
        expect(Math.abs(sampleCycle(trill, 0.04))).toBeGreaterThan(
            Math.abs(sampleCycle(trap, 0.04)),
        );
    });

    test("所有形状值域都在 [-1,1] 内且无 NaN", () => {
        const sources: CycleSource[] = [
            shape("sine"),
            shape("triangle"),
            shape("sawUp"),
            shape("sawDown"),
            shape("square"),
            shape("trapezoid"),
            shape("trill"),
        ];
        for (const source of sources) {
            for (let i = 0; i <= 200; i += 1) {
                const value = sampleCycle(source, i / 200);
                expect(Number.isFinite(value)).toBe(true);
                expect(Math.abs(value)).toBeLessThanOrEqual(1);
            }
        }
    });

    test("周期连续：u→1 与 u=0 的值收敛到同一点（锯齿与方波除外）", () => {
        // 锯齿 / 方波在一个周期内本来就含跳变，端点差不为 0 是正确的。
        for (const name of ["sine", "triangle", "trapezoid", "trill"] as const) {
            const source = shape(name);
            expect(Math.abs(sampleCycle(source, 1 - 1e-9) - sampleCycle(source, 0))).toBeLessThan(
                1e-6,
            );
        }
    });
});

describe("shapeUsesSkew", () => {
    test("只有三角与三种方波类形状使用偏斜", () => {
        expect(shapeUsesSkew("triangle")).toBe(true);
        expect(shapeUsesSkew("square")).toBe(true);
        expect(shapeUsesSkew("trapezoid")).toBe(true);
        expect(shapeUsesSkew("trill")).toBe(true);
        expect(shapeUsesSkew("sine")).toBe(false);
        expect(shapeUsesSkew("sawUp")).toBe(false);
        expect(shapeUsesSkew("sawDown")).toBe(false);
    });
});

describe("采样表", () => {
    test("线性插值，并按周期首尾相接", () => {
        const table: CycleSource = { kind: "table", table: [-1, 0, 1, 0] };
        expect(sampleCycle(table, 0)).toBeCloseTo(-1, 12);
        expect(sampleCycle(table, 0.125)).toBeCloseTo(-0.5, 12);
        expect(sampleCycle(table, 0.25)).toBeCloseTo(0, 12);
        expect(sampleCycle(table, 0.375)).toBeCloseTo(0.5, 12);
        expect(sampleCycle(table, 0.5)).toBeCloseTo(1, 12);
        // 末项之后回绕到首项（首尾相接），而不是钳制。
        expect(sampleCycle(table, 0.875)).toBeCloseTo(-0.5, 12);
    });

    test("越界相位与负相位都按周期回绕", () => {
        const table: CycleSource = { kind: "table", table: [-1, 0, 1, 0] };
        expect(sampleCycle(table, 1.25)).toBeCloseTo(sampleCycle(table, 0.25), 12);
        expect(sampleCycle(table, -0.25)).toBeCloseTo(sampleCycle(table, 0.75), 12);
    });

    test("空表 / 单值表 / 非有限值都不产生 NaN", () => {
        expect(sampleCycle({ kind: "table", table: [] }, 0.3)).toBe(0);
        expect(sampleCycle({ kind: "table", table: [0.5] }, 0.3)).toBe(0.5);
        // 非有限项按 0 参与插值（而不是把整条曲线污染成 NaN）。
        expect(sampleCycle({ kind: "table", table: [Number.NaN, 1] }, 0.1)).toBeCloseTo(0.2, 12);
    });

    test("makeSineTable 生成指定长度、首值为 0 的正弦", () => {
        const table = makeSineTable(8);
        expect(table).toHaveLength(8);
        expect(table[0]).toBeCloseTo(0, 12);
        expect(table[2]).toBeCloseTo(1, 12);
        expect(table[6]).toBeCloseTo(-1, 12);
    });

    test("makeSineTable 的长度被钳进合法区间", () => {
        expect(makeSineTable(1)).toHaveLength(CYCLE_TABLE_MIN_LEN);
        expect(makeSineTable(10_000)).toHaveLength(CYCLE_TABLE_MAX_LEN);
    });

    test("normalizeCycleTable 对非法输入回退成正弦表", () => {
        expect(normalizeCycleTable(null)).toHaveLength(CYCLE_TABLE_DEFAULT_LEN);
        expect(normalizeCycleTable([])).toHaveLength(CYCLE_TABLE_DEFAULT_LEN);
        // 长度不足下限 → 回退。
        expect(normalizeCycleTable([0, 1, 0])).toHaveLength(CYCLE_TABLE_DEFAULT_LEN);
    });

    test("normalizeCycleTable 钳制越界值并过滤非有限值", () => {
        const table = normalizeCycleTable([0, 5, -5, Number.NaN, 0, 0, 0, 0]);
        expect(table[1]).toBe(1);
        expect(table[2]).toBe(-1);
        expect(table[3]).toBe(0);
    });
});
