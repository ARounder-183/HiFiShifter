/**
 * live 覆盖窗口写入 / 回滚的**语义等价性**回归。
 *
 * ## 为什么必须有这份测试
 *
 * `writeDenseIntoLiveWindow` 把"遍历整份参数窗口"换成了"反解受影响的索引区间"
 * （见 `liveEditWindow.ts` 的文件头说明）。这是纯性能改写，**语义必须逐值不变**
 * —— 一旦区间反解差一帧，画出来的曲线就会与手指位置错位，而错位只在这种
 * 边界条件下出现，人工回归极难发现。因此这里用**旧的全扫实现**作为参照实现，
 * 在一组刻意覆盖边界的输入上逐值对拍。
 *
 * ## 覆盖的边界
 *
 * - `stride > 1`（快照降采样后窗口步长非 1，帧号与下标不再一一对应）；
 * - `stride <= 0`（退化；旧实现仍会跑，新区间反解必须退回全扫才能等价）；
 * - 受影响帧区间部分 / 完全落在窗口之外（两侧各有越界）；
 * - `dense` 起点不在窗口内（`j` 越界分支）；
 * - `dense` 短于受影响帧数（`dense[j] ?? edit[i]` 的兜底分支）；
 * - `mode === "restore"`（还原源 `orig` 可能短于窗口 → `?? edit[i]` 兜底）；
 * - 受影响区间与窗口不相交（应完全不改动）。
 */
import { describe, expect, test } from "vitest";

import { restoreLiveEditRange, writeDenseIntoLiveWindow } from "./liveEditWindow";
import { clampParamWriteValue, DYN_FOLLOW_ORIG } from "./paramRanges";
import type { StrokeMode } from "./types";

/**
 * 参照实现：**旧的全窗口线性扫描**（改写前 `applyDenseToLiveEdit` 的循环体）。
 *
 * 逐字保留旧语义，包括"遍历全部下标 + `f` 边界判定跳过"，不接受任何优化 ——
 * 它的价值就是"与现在不同的一把尺子"。
 */
function referenceWrite(args: {
    edit: number[];
    orig: readonly number[];
    startFrame: number;
    stride: number;
    dense: number[] | null;
    denseStartFrame: number;
    minF: number;
    maxF: number;
    mode: StrokeMode;
}): number[] {
    const next = args.edit.slice();
    const start = args.startFrame;
    const step = args.stride;
    for (let i = 0; i < next.length; i += 1) {
        const f = start + i * step;
        if (f < args.minF || f > args.maxF) continue;
        if (args.mode === "restore") {
            next[i] = args.orig[i] ?? next[i];
        } else if (args.dense) {
            const j = f - args.denseStartFrame;
            if (j >= 0 && j < args.dense.length) next[i] = args.dense[j] ?? next[i];
        }
    }
    return next;
}

function numericArray(length: number, seed: number): number[] {
    const out: number[] = new Array(length);
    let s = seed;
    for (let i = 0; i < length; i += 1) {
        s = (s * 1103515245 + 12345) & 0x7fffffff;
        out[i] = (s / 0x7fffffff) * 2 - 1;
    }
    return out;
}

describe("writeDenseIntoLiveWindow", () => {
    test("与旧的全窗口扫描逐值等价（覆盖各种步长与越界组合）", () => {
        const cases: Array<{
            startFrame: number;
            stride: number;
            length: number;
            denseStartFrame: number;
            minF: number;
            maxF: number;
            mode: StrokeMode;
        }> = [
            // 常规：stride 1，区间落在窗口内。
            {
                startFrame: 100,
                stride: 1,
                length: 200,
                denseStartFrame: 150,
                minF: 150,
                maxF: 180,
                mode: "draw",
            },
            // 区间左越界（minF < 窗口首帧）。
            {
                startFrame: 100,
                stride: 1,
                length: 200,
                denseStartFrame: 40,
                minF: 40,
                maxF: 130,
                mode: "draw",
            },
            // 区间右越界。
            {
                startFrame: 100,
                stride: 1,
                length: 200,
                denseStartFrame: 250,
                minF: 250,
                maxF: 400,
                mode: "draw",
            },
            // 区间完全在窗口之前 / 之后（应无改动）。
            {
                startFrame: 100,
                stride: 1,
                length: 200,
                denseStartFrame: 0,
                minF: 0,
                maxF: 50,
                mode: "draw",
            },
            {
                startFrame: 100,
                stride: 1,
                length: 200,
                denseStartFrame: 400,
                minF: 400,
                maxF: 500,
                mode: "draw",
            },
            // stride > 1：帧号与下标不再一一对应。
            {
                startFrame: 0,
                stride: 4,
                length: 120,
                denseStartFrame: 60,
                minF: 61,
                maxF: 200,
                mode: "draw",
            },
            {
                startFrame: 7,
                stride: 16,
                length: 64,
                denseStartFrame: 63,
                minF: 63,
                maxF: 500,
                mode: "draw",
            },
            // dense 起点远在受影响区间之外（j 越界分支）。
            {
                startFrame: 0,
                stride: 1,
                length: 80,
                denseStartFrame: 500,
                minF: 10,
                maxF: 20,
                mode: "draw",
            },
            // dense 比受影响帧数短（dense[j] 兜底分支）。
            {
                startFrame: 0,
                stride: 1,
                length: 80,
                denseStartFrame: 10,
                minF: 10,
                maxF: 60,
                mode: "draw",
            },
            // restore：orig 短于窗口（?? 兜底分支）。
            {
                startFrame: 0,
                stride: 1,
                length: 80,
                denseStartFrame: 0,
                minF: 0,
                maxF: 79,
                mode: "restore",
            },
            // 退化步长：区间反解必须退回全扫才等价。
            {
                startFrame: 30,
                stride: 0,
                length: 50,
                denseStartFrame: 30,
                minF: 30,
                maxF: 30,
                mode: "draw",
            },
            {
                startFrame: 30,
                stride: -1,
                length: 50,
                denseStartFrame: 0,
                minF: 0,
                maxF: 60,
                mode: "draw",
            },
            // 单帧区间（边界两侧各差一帧）。
            {
                startFrame: 100,
                stride: 1,
                length: 200,
                denseStartFrame: 199,
                minF: 199,
                maxF: 199,
                mode: "draw",
            },
            {
                startFrame: 100,
                stride: 1,
                length: 200,
                denseStartFrame: 100,
                minF: 100,
                maxF: 100,
                mode: "draw",
            },
        ];

        for (const c of cases) {
            const edit = numericArray(c.length, 7 + c.length);
            const orig = numericArray(Math.max(1, c.length - 13), 991);
            const dense = numericArray(Math.max(1, c.maxF - c.denseStartFrame + 1), 12345);
            // restore 时 dense 传 null（与调用方一致）。
            const denseArg = c.mode === "restore" ? null : dense;

            const expected = referenceWrite({
                edit,
                orig,
                startFrame: c.startFrame,
                stride: c.stride,
                dense: denseArg,
                denseStartFrame: c.denseStartFrame,
                minF: c.minF,
                maxF: c.maxF,
                mode: c.mode,
            });

            const got = edit.slice();
            writeDenseIntoLiveWindow({
                edit: got,
                orig,
                startFrame: c.startFrame,
                stride: c.stride,
                dense: denseArg,
                denseStartFrame: c.denseStartFrame,
                minF: c.minF,
                maxF: c.maxF,
                mode: c.mode,
            });

            const label = JSON.stringify(c);
            expect(got.length).toBe(expected.length);
            for (let i = 0; i < expected.length; i += 1) {
                // 逐值严格相等（不是近似）：这是纯下标算术的改写。
                expect(`${label} idx=${i} got=${got[i]} want=${expected[i]}`).toBe(
                    `${label} idx=${i} got=${expected[i]} want=${expected[i]}`,
                );
            }
        }
    });

    test("返回的区间覆盖全部被改动的下标（不多不少，供回滚使用）", () => {
        const edit = numericArray(200, 42);
        const orig = numericArray(200, 43);
        const dense: number[] = new Array(30).fill(0.75);
        const range = writeDenseIntoLiveWindow({
            edit,
            orig,
            startFrame: 100,
            stride: 1,
            dense,
            denseStartFrame: 150,
            minF: 150,
            maxF: 179,
            mode: "draw",
        });
        expect(range).toEqual({ lo: 50, hi: 79 });
    });

    test("受影响的帧区间与窗口不相交时不改动任何值", () => {
        const edit = numericArray(50, 5);
        const before = edit.slice();
        const dense: number[] = new Array(10).fill(9);
        writeDenseIntoLiveWindow({
            edit,
            orig: edit,
            startFrame: 1000,
            stride: 1,
            dense,
            denseStartFrame: 0,
            minF: 0,
            maxF: 9,
            mode: "draw",
        });
        expect(edit).toEqual(before);
    });

    test("空窗口返回 null", () => {
        expect(
            writeDenseIntoLiveWindow({
                edit: [],
                orig: [],
                startFrame: 0,
                stride: 1,
                dense: [1],
                denseStartFrame: 0,
                minF: 0,
                maxF: 10,
                mode: "draw",
            }),
        ).toBeNull();
    });

    test("写入成本与窗口长度无关（只遍历受影响区间）", () => {
        // 直接断言"不改动的下标不被访问"：用一个越界写入即抛的数组代理，
        // 任何对区间外下标的访问都会失败。
        const touched: number[] = [];
        const edit = new Proxy([...new Array(20000).fill(0.5)], {
            set(target, prop, value) {
                const i = Number(prop);
                if (Number.isInteger(i)) touched.push(i);
                target[i] = value as number;
                return true;
            },
        });
        const dense: number[] = new Array(20).fill(0.9);
        writeDenseIntoLiveWindow({
            edit,
            orig: edit,
            startFrame: 0,
            stride: 1,
            dense,
            denseStartFrame: 10000,
            minF: 10000,
            maxF: 10019,
            mode: "draw",
        });
        expect(touched.length).toBe(20);
        expect(Math.min(...touched)).toBe(10000);
        expect(Math.max(...touched)).toBe(10019);
    });
});

describe("restoreLiveEditRange", () => {
    test("只还原给定区间，区间外保持不变", () => {
        const edit = [...new Array(100).fill(1)];
        const committed = numericArray(100, 77);
        const changed = restoreLiveEditRange({
            edit,
            committed,
            range: { lo: 10, hi: 19 },
        });
        expect(changed).toBe(true);
        for (let i = 0; i < 100; i += 1) {
            if (i >= 10 && i <= 19) expect(edit[i]).toBe(committed[i]);
            else expect(edit[i]).toBe(1);
        }
    });

    test("区间越界时按数组边界裁剪，不抛错", () => {
        const edit = [...new Array(10).fill(1)];
        const committed = numericArray(10, 3);
        restoreLiveEditRange({ edit, committed, range: { lo: -5, hi: 100 } });
        expect(edit).toEqual(committed);
    });

    test("空区间（lo > hi）不作改动并返回 false", () => {
        const edit = [...new Array(10).fill(1)];
        expect(restoreLiveEditRange({ edit, committed: edit, range: { lo: 5, hi: 4 } })).toBe(
            false,
        );
    });
});

describe("写入路径上的值域钳制（拖拽预览不得超出后端值域）", () => {
    /** 模拟选区上拖：`orig × 2^(Δ/0.5)`（动态）跑出值域后的预览写入。 */
    test("动态：乘性拖拽越界时预览值被钳到 1（与后端存下的值一致）", () => {
        const edit = [0.25, 0.5, 0.9];
        const dense = [0.25 * 4, 0.5 * 4, 0.9 * 4]; // ×4（Δ = +1 值单位）
        writeDenseIntoLiveWindow({
            edit,
            orig: edit.slice(),
            startFrame: 0,
            stride: 1,
            dense,
            denseStartFrame: 0,
            minF: 0,
            maxF: 2,
            mode: "draw",
            clampValue: (v) => clampParamWriteValue("dyn", v),
        });
        expect(edit).toEqual([1, 1, 1]);
    });

    test("音量：线性拖拽越界时预览值被钳到 2", () => {
        const edit = [1.0, 1.5, 2.0];
        const dense = [3.0, 3.5, 4.0];
        writeDenseIntoLiveWindow({
            edit,
            orig: edit.slice(),
            startFrame: 0,
            stride: 1,
            dense,
            denseStartFrame: 0,
            minF: 0,
            maxF: 2,
            mode: "draw",
            clampValue: (v) => clampParamWriteValue("volume", v),
        });
        expect(edit).toEqual([2, 2, 2]);
    });

    test("动态：拖成负值时收敛到「沿用原声」哨兵（而非 0 = 静音）", () => {
        const edit = [0.5];
        writeDenseIntoLiveWindow({
            edit,
            orig: edit.slice(),
            startFrame: 0,
            stride: 1,
            dense: [-0.4],
            denseStartFrame: 0,
            minF: 0,
            maxF: 0,
            mode: "draw",
            clampValue: (v) => clampParamWriteValue("dyn", v),
        });
        expect(edit).toEqual([DYN_FOLLOW_ORIG]);
    });

    test("未提供钳制函数时行为与既有实现一致（恒等）", () => {
        const edit = [0.25];
        writeDenseIntoLiveWindow({
            edit,
            orig: edit.slice(),
            startFrame: 0,
            stride: 1,
            dense: [9],
            denseStartFrame: 0,
            minF: 0,
            maxF: 0,
            mode: "draw",
        });
        expect(edit).toEqual([9]);
    });

    test("钳制只作用于受影响区间之外的下标（不越界改写）", () => {
        const edit = [1, 1, 1, 1, 1];
        writeDenseIntoLiveWindow({
            edit,
            orig: edit.slice(),
            startFrame: 0,
            stride: 1,
            dense: [99],
            denseStartFrame: 2,
            minF: 2,
            maxF: 2,
            mode: "draw",
            clampValue: (v) => clampParamWriteValue("dyn", v),
        });
        expect(edit).toEqual([1, 1, 1, 1, 1]);
    });
});
