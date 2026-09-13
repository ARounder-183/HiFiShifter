/**
 * 内核编辑手势参与集合（./kernelEditSet）行为自检。
 *
 * 【主要内容】
 * 1. 参与集合：单点 / 多选 / 编组展开 / 忽略编组 / 禁用组 / 淡变不展开；
 * 2. 位移换算：起点钳到 0、逐参与者独立钳制、跨轨按各自初始序号 + 同一偏移量、
 *    轨道越界钳制；
 * 3. 边界：锚点不存在返回空、轨道不存在不移动轨道。
 *
 * 【作用】这些断言是「选中多个 clip 只动一个」「同组不联动」这类数据语义缺陷的
 * 回归护栏：一旦规则退化成单 clip，下游提交会静默只改一个 clip。
 *
 * 【与其他模块的关系】覆盖 `kernelEditSet.ts`；不依赖 React / Redux。
 */

import { describe, expect, it } from "vitest";

import { applyKernelEditDelta, resolveKernelEditParticipants } from "./kernelEditSet";

const TRACKS = ["t1", "t2", "t3"];

function clip(
    id: string,
    trackId: string,
    startSec: number,
    extra: { groupId?: string; snapOffsetSec?: number } = {},
) {
    return { id, trackId, startSec, lengthSec: 2, ...extra };
}

function resolve(
    anchorClipId: string,
    clips: ReturnType<typeof clip>[],
    overrides: Partial<Parameters<typeof resolveKernelEditParticipants>[0]> = {},
) {
    return resolveKernelEditParticipants({
        anchorClipId,
        multiSelectedClipIds: [],
        clips,
        trackIds: TRACKS,
        ignoreGrouping: false,
        disabledGroupIds: [],
        expandGroups: true,
        ...overrides,
    });
}

describe("resolveKernelEditParticipants", () => {
    it("未在多选集合内时只返回锚点自己", () => {
        const clips = [clip("a", "t1", 1), clip("b", "t1", 5)];
        const result = resolve("a", clips, { multiSelectedClipIds: ["b"] });
        expect(result.map((p) => p.clipId)).toEqual(["a"]);
    });

    it("锚点在多选集合内时整组参与", () => {
        const clips = [clip("a", "t1", 1), clip("b", "t1", 5), clip("c", "t2", 9)];
        const result = resolve("a", clips, { multiSelectedClipIds: ["a", "c"] });
        expect(result.map((p) => p.clipId)).toEqual(["a", "c"]);
    });

    it("同组 clip 联动参与（编组展开）", () => {
        const clips = [
            clip("a", "t1", 1, { groupId: "g1" }),
            clip("b", "t2", 5, { groupId: "g1" }),
            clip("c", "t3", 9),
        ];
        const result = resolve("a", clips);
        expect(result.map((p) => p.clipId).sort()).toEqual(["a", "b"]);
    });

    it("全局「忽略编组」时不展开", () => {
        const clips = [
            clip("a", "t1", 1, { groupId: "g1" }),
            clip("b", "t2", 5, { groupId: "g1" }),
        ];
        const result = resolve("a", clips, { ignoreGrouping: true });
        expect(result.map((p) => p.clipId)).toEqual(["a"]);
    });

    it("组被临时禁用联动时不展开", () => {
        const clips = [
            clip("a", "t1", 1, { groupId: "g1" }),
            clip("b", "t2", 5, { groupId: "g1" }),
        ];
        const result = resolve("a", clips, { disabledGroupIds: ["g1"] });
        expect(result.map((p) => p.clipId)).toEqual(["a"]);
    });

    it("淡变 / 增益手势（expandGroups=false）不展开编组，但多选仍生效", () => {
        const clips = [
            clip("a", "t1", 1, { groupId: "g1" }),
            clip("b", "t2", 5, { groupId: "g1" }),
        ];
        const grouped = resolve("a", clips, { expandGroups: false });
        expect(grouped.map((p) => p.clipId)).toEqual(["a"]);
        const selected = resolve("a", clips, {
            expandGroups: false,
            multiSelectedClipIds: ["a", "b"],
        });
        expect(selected.map((p) => p.clipId)).toEqual(["a", "b"]);
    });

    it("锚点不存在时返回空数组", () => {
        expect(resolve("missing", [clip("a", "t1", 1)])).toEqual([]);
    });

    it("带出初始几何与轨道序号（含吸附偏移点）", () => {
        const result = resolve("a", [{ ...clip("a", "t2", 3), snapOffsetSec: 1.5 }]);
        expect(result[0]).toEqual({
            clipId: "a",
            startSec: 3,
            lengthSec: 2,
            trackId: "t2",
            trackIndex: 1,
            snapOffsetSec: 1.5,
        });
    });
});

describe("applyKernelEditDelta", () => {
    const participants = resolve(
        "a",
        [clip("a", "t1", 2), clip("b", "t2", 0.5), clip("c", "t3", 8)],
        { multiSelectedClipIds: ["a", "b", "c"] },
    );

    it("按同一位移平移全部参与者（未触及边界时不改变位移）", () => {
        const { moves, deltaStartSec } = applyKernelEditDelta({
            participants,
            deltaStartSec: -0.3,
            deltaTrack: 0,
            trackIds: TRACKS,
        });
        expect(deltaStartSec).toBe(-0.3);
        expect(moves).toEqual([
            { clipId: "a", startSec: 1.7, trackId: "t1" },
            { clipId: "b", startSec: 0.2, trackId: "t2" },
            { clipId: "c", startSec: 7.7, trackId: "t3" },
        ]);
    });

    it("位移越过左边界时钳制共享位移：整组间距保持不变", () => {
        // 最靠左的参与者 start = 0.5 → 位移下界 = -0.5。
        const { moves, deltaStartSec } = applyKernelEditDelta({
            participants,
            deltaStartSec: -10,
            deltaTrack: 0,
            trackIds: TRACKS,
        });
        expect(deltaStartSec).toBe(-0.5);
        expect(moves.map((m) => m.startSec)).toEqual([1.5, 0, 7.5]);
    });

    it("跨轨按各自初始序号 + 同一偏移量移动", () => {
        const { moves } = applyKernelEditDelta({
            participants,
            deltaStartSec: 0,
            deltaTrack: 1,
            trackIds: TRACKS,
        });
        expect(moves.map((m) => m.trackId)).toEqual(["t2", "t3", "t3"]);
    });

    it("轨道越界被钳制到合法范围", () => {
        const up = applyKernelEditDelta({
            participants,
            deltaStartSec: 0,
            deltaTrack: -5,
            trackIds: TRACKS,
        });
        expect(up.moves.map((m) => m.trackId)).toEqual(["t1", "t1", "t1"]);
        const down = applyKernelEditDelta({
            participants,
            deltaStartSec: 0,
            deltaTrack: 5,
            trackIds: TRACKS,
        });
        expect(down.moves.map((m) => m.trackId)).toEqual(["t3", "t3", "t3"]);
    });

    it("非法位移退化为不位移", () => {
        const { moves, deltaStartSec } = applyKernelEditDelta({
            participants,
            deltaStartSec: Number.NaN,
            deltaTrack: Number.NaN,
            trackIds: TRACKS,
        });
        expect(deltaStartSec).toBe(0);
        expect(moves.map((m) => m.startSec)).toEqual([2, 0.5, 8]);
        expect(moves.map((m) => m.trackId)).toEqual(["t1", "t2", "t3"]);
    });

    it("轨道不存在（trackIndex = -1）时不改轨道", () => {
        const orphan = resolve("x", [clip("x", "gone", 1)]);
        expect(orphan[0]?.trackIndex).toBe(-1);
        const { moves } = applyKernelEditDelta({
            participants: orphan,
            deltaStartSec: 0,
            deltaTrack: 1,
            trackIds: TRACKS,
        });
        expect(moves[0]?.trackId).toBe("gone");
    });

    it("空参与集合返回空位移且不抛错", () => {
        const { moves, deltaStartSec } = applyKernelEditDelta({
            participants: [],
            deltaStartSec: 5,
            deltaTrack: 1,
            trackIds: TRACKS,
        });
        expect(moves).toEqual([]);
        expect(deltaStartSec).toBe(5);
    });
});
