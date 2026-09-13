// applySelectionEditWithEdgeSmoothing 编排的集成测试（mock paramsApi）。
//
// 锁定「取数范围 → 选区编辑 → 边缘淡化 → 回写」协议：
//   - 取数范围覆盖 ±⌈halfSpan⌉（毫秒定标：100% @5ms = 每侧 12 帧，选区 20 帧
//     时受 floor(20/4)=5 上限 → 每侧 5 帧）；
//   - 回写从扩展起点开始、首块 checkpoint=true（撤销语义）；
//   - 未浊哨兵帧（pitch=0）不被写非零；
//   - strength=0 时只写选区本身。

import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../../../services/api", () => ({
    paramsApi: {
        getParamFrames: vi.fn(),
        setParamFrames: vi.fn(),
    },
}));

import { paramsApi } from "../../../services/api";
import type { ParamFramesPayload } from "../../../types/api";
import { applySelectionEditOverRanges, applySelectionEditWithEdgeSmoothing } from "./selectionEditApply";

const mockedGet = vi.mocked(paramsApi.getParamFrames);
const mockedSet = vi.mocked(paramsApi.setParamFrames);

// 生产代码同样以 `res as ParamFramesPayload` 消费部分字段（ok/edit/frame_period_ms），
// 测试按同一契约只提供被读取的字段。
const asPayload = (edit: number[], framePeriodMs = 5): ParamFramesPayload =>
    ({ ok: true, edit, frame_period_ms: framePeriodMs }) as unknown as ParamFramesPayload;
const failedPayload = { ok: false } as unknown as ParamFramesPayload;

beforeEach(() => {
    mockedGet.mockReset();
    mockedSet.mockReset();
    mockedSet.mockResolvedValue({ ok: true });
});

const pitchEditable = (v: number) => Number.isFinite(v) && v !== 0;

describe("applySelectionEditWithEdgeSmoothing", () => {
    it("常数移调 + 平滑：取数扩展、折半边界、扩展回写", async () => {
        mockedGet.mockResolvedValue(asPayload(new Array<number>(30).fill(60)));
        const ok = await applySelectionEditWithEdgeSmoothing({
            trackId: "t1",
            param: "pitch",
            startFrame: 10,
            frameCount: 20,
            framePeriodMs: 5,
            smoothnessPercent: 100,
            editSelection: (vals) => vals.map((v) => (v === 0 ? 0 : v + 2)),
            isEditable: pitchEditable,
        });
        expect(ok).toBe(true);
        // 取数范围 [10−5, 10+20+5) = [5, 35)，30 帧
        expect(mockedGet).toHaveBeenCalledWith("t1", "pitch", 5, 30, 1);
        expect(mockedSet).toHaveBeenCalledTimes(1);
        const [, param, writeStart, written, checkpoint] = mockedSet.mock.calls[0];
        expect(param).toBe("pitch");
        expect(writeStart).toBe(5);
        expect(checkpoint).toBe(true);
        expect(written.length).toBe(30);
        // 选区 = dense idx [5,24]；左带 [0,10]，右带 [19,29]
        expect(written[0]).toBe(60); // 左带 u=0 → 原值
        expect(written[5]).toBeCloseTo(61, 9); // 左边界帧 w=0.5
        expect(written[10]).toBe(62); // 左带 u=1 → 完整编辑值
        expect(written[19]).toBe(62); // 右带 u=1
        expect(written[24]).toBeCloseTo(61, 9); // 右边界帧 w=0.5
        expect(written[29]).toBe(60); // 右带 u=0 → 原值
    });

    it("未浊哨兵：选区边外的 0 帧不被写非零", async () => {
        const edit = new Array<number>(30).fill(60);
        edit[4] = 0; // 绝对帧 9，选区左边界外一帧
        edit[25] = 0; // 绝对帧 30，选区右边界外一帧
        mockedGet.mockResolvedValue(asPayload(edit));
        await applySelectionEditWithEdgeSmoothing({
            trackId: "t1",
            param: "pitch",
            startFrame: 10,
            frameCount: 20,
            framePeriodMs: 5,
            smoothnessPercent: 100,
            editSelection: (vals) => vals.map((v) => (v === 0 ? 0 : v + 2)),
            isEditable: pitchEditable,
        });
        const written = mockedSet.mock.calls[0][3] as number[];
        expect(written[4]).toBe(0); // 绝对帧 9 保持未浊
        expect(written[25]).toBe(0); // 绝对帧 30 保持未浊
        expect(written[5]).toBeCloseTo(61, 9); // 有声帧淡化照常
    });

    it("setPitch（editedAt 延拓）：选区外向目标滑移", async () => {
        mockedGet.mockResolvedValue(asPayload(new Array<number>(30).fill(60)));
        await applySelectionEditWithEdgeSmoothing({
            trackId: "t1",
            param: "pitch",
            startFrame: 10,
            frameCount: 20,
            framePeriodMs: 5,
            smoothnessPercent: 100,
            editSelection: (vals) => vals.map((v) => (v === 0 ? 0 : 65)),
            extension: { kind: "editedAt", editedAt: () => 65 },
            isEditable: pitchEditable,
        });
        const written = mockedSet.mock.calls[0][3] as number[];
        // 左带外缘 → 60；边界帧 → 半途（62.5）；选区中部 → 65
        expect(written[0]).toBe(60);
        expect(written[5]).toBeCloseTo(62.5, 9);
        expect(written[10]).toBe(65);
        // 右带外缘回到 60
        expect(written[29]).toBe(60);
    });

    it("strength=0：只写选区本身，无淡化", async () => {
        mockedGet.mockResolvedValue(asPayload(new Array<number>(20).fill(60)));
        await applySelectionEditWithEdgeSmoothing({
            trackId: "t1",
            param: "pitch",
            startFrame: 10,
            frameCount: 20,
            framePeriodMs: 5,
            smoothnessPercent: 0,
            editSelection: (vals) => vals.map((v) => (v === 0 ? 0 : v + 2)),
            isEditable: pitchEditable,
        });
        expect(mockedGet).toHaveBeenCalledWith("t1", "pitch", 10, 20, 1);
        const [, , writeStart, written] = mockedSet.mock.calls[0];
        expect(writeStart).toBe(10);
        expect(written).toEqual(new Array<number>(20).fill(62));
    });

    it("缺省延拓（量化类）：选区外按边界帧实际 delta 常数延拓", async () => {
        mockedGet.mockResolvedValue(asPayload(new Array<number>(30).fill(60)));
        // 模拟量化：选区前半 +2、后半不动 → 左边界 delta=+2，右边界 delta=0
        const ok = await applySelectionEditWithEdgeSmoothing({
            trackId: "t1",
            param: "pitch",
            startFrame: 10,
            frameCount: 20,
            framePeriodMs: 5,
            smoothnessPercent: 100,
            editSelection: (vals) => vals.map((v, i) => (i < 10 ? v + 2 : v)),
            isEditable: pitchEditable,
        });
        expect(ok).toBe(true);
        const written = mockedSet.mock.calls[0][3] as number[];
        // 左带 [0,10]：按 +2 delta 淡出；右边界 delta=0 → 右带完全不动
        expect(written[0]).toBe(60); // 左带 u=0 → 原值
        expect(written[5]).toBeCloseTo(61, 9); // 左边界帧折半（60+0.5·2）
        expect(written[10]).toBe(62); // 左带 u=1 → 完整 +2
        expect(written[19]).toBe(60); // 选区内未编辑的半段保持原值
        expect(written[24]).toBe(60); // 右边界 delta=0 → 无淡化
        expect(written[29]).toBe(60); // 右带外原值
    });

    it("取数失败 / 空数据 → 返回 false 且不回写", async () => {
        mockedGet.mockResolvedValue(failedPayload);
        const ok = await applySelectionEditWithEdgeSmoothing({
            trackId: "t1",
            param: "pitch",
            startFrame: 10,
            frameCount: 20,
            framePeriodMs: 5,
            smoothnessPercent: 50,
            editSelection: (vals) => vals,
        });
        expect(ok).toBe(false);
        expect(mockedSet).not.toHaveBeenCalled();
    });
});

describe("applySelectionEditOverRanges（多选区：每段独立 + 单撤销点）", () => {
    it("逐段独立取数写回，仅首段带撤销点", async () => {
        mockedGet.mockImplementation(async (_track, _param, startFrame, frameCount) =>
            asPayload(new Array<number>(frameCount).fill(startFrame)),
        );
        const ok = await applySelectionEditOverRanges({
            trackId: "t1",
            param: "pitch",
            framePeriodMs: 5,
            smoothnessPercent: 0,
            // 断层两侧各写一次，互不影响
            ranges: [
                { startFrame: 10, frameCount: 5 },
                { startFrame: 100, frameCount: 5 },
            ],
            editSelection: (vals) => vals.map((v) => v + 1),
        });
        expect(ok).toBe(true);
        expect(mockedGet).toHaveBeenCalledWith("t1", "pitch", 10, 5, 1);
        expect(mockedGet).toHaveBeenCalledWith("t1", "pitch", 100, 5, 1);
        expect(mockedSet).toHaveBeenCalledTimes(2);
        const [, , firstStart, firstWritten, firstCheckpoint] = mockedSet.mock.calls[0];
        expect(firstStart).toBe(10);
        expect(firstWritten).toEqual(new Array<number>(5).fill(11));
        expect(firstCheckpoint).toBe(true);
        const [, , secondStart, secondWritten, secondCheckpoint] = mockedSet.mock.calls[1];
        expect(secondStart).toBe(100);
        expect(secondWritten).toEqual(new Array<number>(5).fill(101));
        expect(secondCheckpoint).toBe(false);
    });

    it("首段取数失败时，撤销点顺延到第一个真正写入的段", async () => {
        let call = 0;
        mockedGet.mockImplementation(async (_track, _param, _startFrame, frameCount) => {
            call += 1;
            return call === 1 ? failedPayload : asPayload(new Array<number>(frameCount).fill(60));
        });
        const ok = await applySelectionEditOverRanges({
            trackId: "t1",
            param: "pitch",
            framePeriodMs: 5,
            smoothnessPercent: 0,
            ranges: [
                { startFrame: 10, frameCount: 5 },
                { startFrame: 100, frameCount: 5 },
            ],
            editSelection: (vals) => vals.map((v) => v + 1),
        });
        expect(ok).toBe(true);
        expect(mockedSet).toHaveBeenCalledTimes(1);
        expect(mockedSet.mock.calls[0][2]).toBe(100);
        expect(mockedSet.mock.calls[0][4]).toBe(true);
    });

    it("空选区不产生任何调用", async () => {
        const ok = await applySelectionEditOverRanges({
            trackId: "t1",
            param: "pitch",
            framePeriodMs: 5,
            smoothnessPercent: 0,
            ranges: [],
            editSelection: (vals) => vals,
        });
        expect(ok).toBe(false);
        expect(mockedGet).not.toHaveBeenCalled();
        expect(mockedSet).not.toHaveBeenCalled();
    });
});
