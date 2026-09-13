import { test, vi } from "vitest";

vi.mock("../../../services/api", () => ({
    paramsApi: {
        getParamFrames: vi.fn(),
        setParamFrames: vi.fn(),
    },
}));

import { paramsApi } from "../../../services/api";
import {
    buildMultiRangeEditPlan,
    buildSelectionDragDense,
    expandStrideSampledDense,
    pvCoversFullRes,
    selectionDragRange,
    uploadFullResCurveSegments,
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

    // ── buildMultiRangeEditPlan：多段独立 + 断层保持原值 ──────────────────
    {
        // 帧号即基准值：未写入的帧应当仍是基准值（等价于未改动）
        const plan = buildMultiRangeEditPlan({
            ranges: [
                { startFrame: 100, endFrame: 104 },
                { startFrame: 200, endFrame: 204 },
            ],
            valuesAt: (i) => Array.from({ length: 5 }, () => (i === 0 ? 10 : 20)),
            sourceAt: (frame) => frame,
        });
        // 段互不相连且无边缘淡化 → 两个独立片段，缝隙 [105,199] 完全不写
        assertEqual(plan.length, 2, "two pieces");
        assertEqual(plan[0].startFrame, 100, "piece0 start");
        assertEqual(plan[0].endFrame, 104, "piece0 end");
        assertEqual(plan[1].startFrame, 200, "piece1 start");
        assertEqual(plan[1].values.join(","), "20,20,20,20,20", "piece1 values");
        assertEqual(plan[0].values.join(","), "10,10,10,10,10", "piece0 values");
    }
    // 边缘淡化：窗口向两侧扩展，扩展帧回到基准值
    {
        const plan = buildMultiRangeEditPlan({
            ranges: [{ startFrame: 100, endFrame: 104 }],
            valuesAt: () => [10, 10, 10, 10, 10],
            sourceAt: (frame) => frame,
            edgeHalfSpanAt: () => 2,
            transformAt: (_i, _index, sourceValue) => sourceValue,
        });
        assertEqual(plan[0].startFrame, 98, "edge-extended start");
        assertEqual(plan[0].endFrame, 106, "edge-extended end");
        // 扩展区的基准值仍在（淡化权重为 0 时保持不变）
        assertEqual(plan[0].values[0], 98, "pre-edge base value preserved");
        assertEqual(plan[0].values[plan[0].values.length - 1], 106, "post-edge base value");
    }
    // X 位移：写入窗口覆盖「原位 ∪ 落地位」，原位回填基准值
    {
        const plan = buildMultiRangeEditPlan({
            ranges: [{ startFrame: 10, endFrame: 12 }],
            frameDelta: 5,
            valuesAt: () => [1, 2, 3],
            sourceAt: (frame) => frame,
        });
        assertEqual(plan.length, 1, "merged into one piece");
        assertEqual(plan[0].startFrame, 10, "moved window start");
        assertEqual(plan[0].endFrame, 17, "moved window end");
        // 12,13,14 是原值（未改动），15..17 是移动后的选区值
        assertEqual(plan[0].values.join(","), "10,11,12,13,14,1,2,3", "moved values");
    }
    // 位移后两段落地位重叠 → 合并为一个写入片段（后写覆盖先写，确定性）
    {
        const plan = buildMultiRangeEditPlan({
            ranges: [
                { startFrame: 0, endFrame: 1 },
                { startFrame: 4, endFrame: 5 },
            ],
            frameDelta: 3,
            valuesAt: (i) => (i === 0 ? [1, 1] : [2, 2]),
            sourceAt: (frame) => frame,
        });
        assertEqual(plan.length, 1, "overlapping landings merge");
        assertEqual(plan[0].startFrame, 0, "merged start");
        assertEqual(plan[0].endFrame, 8, "merged end");
        // 段 1 落在 [3,4]（值 1），段 2 落在 [7,8]（值 2），其余为基准值
        assertEqual(plan[0].values.join(","), "0,1,2,1,1,5,6,2,2", "overlap last-writer-wins");
    }
    // 无段可用值（例如尚未取到全分辨率数据）→ 不产生任何实际变更（窗口回填基准值）
    {
        const plan = buildMultiRangeEditPlan({
            ranges: [{ startFrame: 0, endFrame: 3 }],
            valuesAt: () => null,
            sourceAt: (frame) => frame,
        });
        assertEqual(plan.length, 1, "window still materialised");
        assertEqual(plan[0].values.join(","), "0,1,2,3", "base values untouched");
    }
    // 空选区
    assertEqual(
        buildMultiRangeEditPlan({ ranges: [], valuesAt: () => [1], sourceAt: () => 0 }).length,
        0,
        "empty ranges no plan",
    );
});

/**
 * 拉伸边缘拖拽提交路径（stride → 逐帧展开）的不变量：
 * 采样点精确保留、点间按渲染同款线性插值展开、stride=1 原样返回。
 */
test("components/layout/pianoRoll/selectionEditData.test.ts expandStrideSampledDense checks", () => {
    function assertEqual(actual: unknown, expected: unknown, label: string): void {
        if (actual !== expected) {
            throw new Error(`${label}: expected ${String(expected)}, received ${String(actual)}`);
        }
    }

    // stride=1 → 原样返回（同一引用，零开销）
    const dense = [1, 2, 3];
    assertEqual(expandStrideSampledDense(dense, 1), dense, "stride1 same ref");
    // 空输入
    assertEqual(expandStrideSampledDense([], 4).length, 0, "empty stays empty");
    // 单样本：无插值可言，长度 1
    {
        const out = expandStrideSampledDense([7], 5);
        assertEqual(out.length, 1, "single sample length");
        assertEqual(out[0], 7, "single sample value");
    }
    // 采样点精确保留 + 点间线性插值：dense[k] ↔ startFrame + k×stride
    {
        const out = expandStrideSampledDense([0, 10, 30], 4);
        assertEqual(out.length, 9, "expanded length");
        const expected = [0, 2.5, 5, 7.5, 10, 15, 20, 25, 30];
        for (let i = 0; i < expected.length; i += 1) {
            if (Math.abs(out[i] - expected[i]) > 1e-9) {
                throw new Error(`out[${i}]: expected ${expected[i]}, received ${out[i]}`);
            }
        }
    }
    // 小数 stride 按 Math.floor 收敛（与 pv 读取一致）
    {
        const out = expandStrideSampledDense([1, 3], 2.9);
        assertEqual(out.length, 3, "fractional stride length");
        assertEqual(out[1], 2, "fractional stride midpoint");
    }
});

/**
 * 多段回写的「整批一个撤销点」纪律：
 * 后端 set_param_frames 的 checkpoint 是「写入前先快照时间线」，因此一次
 * 用户操作（可能跨多段、多块）只能有**一个** checkpoint=true —— 否则撤销
 * 需要按多次才能回到操作前。
 */
test("components/layout/pianoRoll/selectionEditData.test.ts uploadFullResCurveSegments checks", async () => {
    const mockedSet = vi.mocked(paramsApi.setParamFrames);
    mockedSet.mockReset();
    mockedSet.mockResolvedValue({ ok: true } as never);

    // 两段：各一段，首段打撤销点、次段不打
    mockedSet.mockClear();
    await uploadFullResCurveSegments({
        trackId: "t1",
        param: "pitch",
        segments: [
            { startFrame: 100, values: [1, 2, 3] },
            { startFrame: 500, values: [4, 5] },
        ],
    });
    const calls = mockedSet.mock.calls;
    if (calls.length !== 2) throw new Error(`expected 2 writes, got ${calls.length}`);
    if (calls[0][2] !== 100 || calls[0][4] !== true) {
        throw new Error(`first write mismatch: ${JSON.stringify(calls[0])}`);
    }
    if (calls[1][2] !== 500 || calls[1][4] !== false) {
        throw new Error(`second write must not checkpoint: ${JSON.stringify(calls[1])}`);
    }

    // 单段跨块（> CHUNK_FRAMES）：仅首块 checkpoint=true
    mockedSet.mockClear();
    await uploadFullResCurveSegments({
        trackId: "t1",
        param: "pitch",
        segments: [{ startFrame: 0, values: new Array<number>(40_000).fill(1) }],
    });
    const chunkCalls = mockedSet.mock.calls;
    if (chunkCalls.length !== 2) throw new Error(`expected 2 chunks, got ${chunkCalls.length}`);
    if (chunkCalls[0][4] !== true || chunkCalls[1][4] !== false) {
        throw new Error("only the first chunk may checkpoint");
    }
    if (chunkCalls[0][3].length !== 32_768) {
        throw new Error(`unexpected chunk size ${chunkCalls[0][3].length}`);
    }

    // 全空片段：不产生任何写入（不会白打撤销点）
    mockedSet.mockClear();
    await uploadFullResCurveSegments({
        trackId: "t1",
        param: "pitch",
        segments: [{ startFrame: 0, values: [] }],
    });
    if (mockedSet.mock.calls.length !== 0) throw new Error("empty segments must not write");
});
