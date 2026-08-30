import { test } from "vitest";

import {
    buildSelectionDragDense,
    pvCoversFullRes,
    selectionDragRange,
} from "./selectionEditData.js";
import type { ParamViewSegment } from "./types";

/**
 * 这里的断言锁住的是「选区拖动」的两条不变量：
 *   1. dense 数组是逐帧索引的（values[k] 对应第 startFrame + k 帧），
 *      否则与 setParamFrames 的逐帧写回语义对不上；
 *   2. 降采样（stride > 1）的 pv 绝不能被当成可编辑数据，
 *      否则会把低分辨率的值写回后端、覆盖掉全分辨率曲线。
 */
test("components/layout/pianoRoll/selectionEditData.test.ts scripted checks", async () => {
    function assertEqual(actual: unknown, expected: unknown, label: string): void {
        if (actual !== expected) {
            throw new Error(`${label}: expected ${String(expected)}, received ${String(actual)}`);
        }
    }

    // ── pvCoversFullRes ──────────────────────────────────────────────────
    const pv = (startFrame: number, stride: number, edit: number[]): ParamViewSegment => ({
        key: "k",
        framePeriodMs: 5.8,
        startFrame,
        stride,
        referenceKind: "source_curve",
        orig: edit.slice(),
        edit,
    });

    // stride=1 且完整覆盖 → 可直接切片（零 IPC 快路径）
    assertEqual(pvCoversFullRes(pv(100, 1, [1, 2, 3, 4, 5]), 101, 103), true, "covers stride1");
    // 覆盖边界
    assertEqual(pvCoversFullRes(pv(100, 1, [1, 2, 3, 4, 5]), 100, 104), true, "covers full");
    // 超出右边界 → 不覆盖
    assertEqual(pvCoversFullRes(pv(100, 1, [1, 2, 3, 4, 5]), 100, 105), false, "beyond right");
    // 超出左边界 → 不覆盖
    assertEqual(pvCoversFullRes(pv(100, 1, [1, 2, 3, 4, 5]), 99, 104), false, "beyond left");
    // 关键：stride>1 属于降采样数据，必须重新取全分辨率，不能拿来编辑
    assertEqual(pvCoversFullRes(pv(100, 4, [1, 2, 3, 4, 5]), 100, 116), false, "decimated pv");
    // 空 pv / null
    assertEqual(pvCoversFullRes(null, 0, 1), false, "null pv");
    assertEqual(pvCoversFullRes(pv(0, 1, []), 0, 1), false, "empty pv");

    // ── selectionDragRange ───────────────────────────────────────────────
    // 纯上下拖动（frameDelta = 0）：范围 = 选区 ± 边缘扩展
    {
        const r = selectionDragRange({
            origStartFrame: 1000,
            origValuesLength: 100,
            frameDelta: 0,
            extraEdgeFrames: 10,
        });
        assertEqual(r.startFrame, 990, "y-drag start");
        assertEqual(r.endFrame, 1109, "y-drag end");
    }
    // 带 X 位移：范围要覆盖「原位置 ∪ 新位置」
    {
        const r = selectionDragRange({
            origStartFrame: 1000,
            origValuesLength: 100,
            frameDelta: 50,
            extraEdgeFrames: 0,
        });
        assertEqual(r.startFrame, 1000, "x-drag start");
        assertEqual(r.endFrame, 1149, "x-drag end");
    }
    // 负向位移
    {
        const r = selectionDragRange({
            origStartFrame: 1000,
            origValuesLength: 100,
            frameDelta: -40,
            extraEdgeFrames: 0,
        });
        assertEqual(r.startFrame, 960, "x-drag negative start");
        assertEqual(r.endFrame, 1099, "x-drag negative end");
    }
    // 起点被夹到 0，不能出现负帧
    {
        const r = selectionDragRange({
            origStartFrame: 5,
            origValuesLength: 10,
            frameDelta: -20,
            extraEdgeFrames: 8,
        });
        assertEqual(r.startFrame, 0, "clamped to 0");
    }
    // 空选区
    {
        const r = selectionDragRange({
            origStartFrame: 500,
            origValuesLength: 0,
            frameDelta: 0,
            extraEdgeFrames: 4,
        });
        assertEqual(r.startFrame, 500, "empty selection start");
        assertEqual(r.endFrame, 500, "empty selection end");
    }

    // ── buildSelectionDragDense：逐帧索引 + 只改选区内 ────────────────────
    {
        // sourceAt 返回帧号本身，方便直接验证索引语义
        const built = buildSelectionDragDense({
            sourceAt: (frame) => frame,
            origValues: [10, 20, 30],
            origStartFrame: 100,
            frameDelta: 0,
            extraEdgeFrames: 2,
            transform: (orig) => orig + 5,
        });
        // 选区 [100,102]，向两侧各扩 2 帧 → [98,104]，共 7 帧
        assertEqual(built.startFrame, 98, "dense start");
        assertEqual(built.endFrame, 104, "dense end");
        assertEqual(built.values.length, 7, "dense length");
        // 选区外保持原值（98,99 与 103,104）
        assertEqual(built.values[0], 98, "context before");
        assertEqual(built.values[1], 99, "context before 2");
        assertEqual(built.values[5], 103, "context after");
        assertEqual(built.values[6], 104, "context after 2");
        // 选区内被变换：10→15, 20→25, 30→35，且落在正确的帧上
        assertEqual(built.values[2], 15, "frame 100");
        assertEqual(built.values[3], 25, "frame 101");
        assertEqual(built.values[4], 35, "frame 102");
    }

    // ── X 位移：值落到新位置，原位置回填当前值 ────────────────────────────
    {
        const built = buildSelectionDragDense({
            sourceAt: (frame) => frame,
            origValues: [10, 20],
            origStartFrame: 100,
            frameDelta: 3,
            extraEdgeFrames: 0,
            transform: (orig) => orig + 1,
        });
        // 原位置 [100,101] ∪ 新位置 [103,104] → [100,104]
        assertEqual(built.startFrame, 100, "moved start");
        assertEqual(built.endFrame, 104, "moved end");
        assertEqual(built.values.join(","), "100,101,102,11,21", "moved values");
    }

    // ── 预览 / 提交一致性：数据源同为全分辨率时结果必须完全一致 ────────────
    // 这条断言保证「用户拖出来的线」与「写回后端的数据」是同一条。
    {
        const full = [1, 2, 3, 4, 5, 6, 7, 8];
        const args = {
            origValues: [3, 4, 5],
            origStartFrame: 2,
            frameDelta: 0,
            extraEdgeFrames: 1,
            transform: (orig: number) => orig * 2,
        };
        const fromFull = buildSelectionDragDense({
            ...args,
            sourceAt: (frame) => full[frame] ?? 0,
        });
        // 模拟 pv 恰好以 stride=1 覆盖同一区间时的切片
        const pvSlice = full.slice(1, 7);
        const fromPv = buildSelectionDragDense({
            ...args,
            sourceAt: (frame) => pvSlice[frame - 1] ?? 0,
        });
        assertEqual(fromFull.values.join(","), fromPv.values.join(","), "preview == commit");
    }
});
