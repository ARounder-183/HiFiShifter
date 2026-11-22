/**
 * 分批与汇总：分批不能改变"一共转了几条"，也不能让同一条片段被转两次。
 */
import { describe, expect, test } from "vitest";

import {
    PITCH_BRIDGE_CHUNK,
    emptyOutcome,
    mergeOutcome,
    planPitchBridgeBatches,
    skippedClips,
    uniqueClipIds,
} from "./run";

const ids = (count: number) => Array.from({ length: count }, (_, index) => `clip-${index}`);

describe("分批", () => {
    test("不足一批时只有一批", () => {
        expect(planPitchBridgeBatches(ids(3))).toEqual([["clip-0", "clip-1", "clip-2"]]);
    });

    test("正好一批不多出空批", () => {
        expect(planPitchBridgeBatches(ids(PITCH_BRIDGE_CHUNK))).toHaveLength(1);
    });

    test("多出 1 条就分两批，且顺序不变", () => {
        const batches = planPitchBridgeBatches(ids(PITCH_BRIDGE_CHUNK + 1));
        expect(batches.map((batch) => batch.length)).toEqual([PITCH_BRIDGE_CHUNK, 1]);
        expect(batches.flat()).toEqual(ids(PITCH_BRIDGE_CHUNK + 1));
    });

    test("空输入得到空数组（调用方据此不发明细）", () => {
        expect(planPitchBridgeBatches([])).toEqual([]);
    });
});

describe("去重", () => {
    test("保留首次出现的位置", () => {
        expect(uniqueClipIds(["b", "a", "b", "c", "a"])).toEqual(["b", "a", "c"]);
    });
});

describe("汇总", () => {
    test("两批的数量相加，unresolved 合并", () => {
        let outcome = emptyOutcome(4);
        outcome = mergeOutcome(outcome, {
            clipsConverted: 2,
            framesWritten: 10,
            unresolved: ["x"],
        });
        outcome = mergeOutcome(outcome, {
            clipsConverted: 1,
            framesWritten: 5,
            unresolved: ["y"],
        });
        expect(outcome.converted).toBe(3);
        expect(outcome.framesWritten).toBe(15);
        expect(outcome.unresolved).toEqual(["x", "y"]);
        expect(skippedClips(outcome)).toBe(1);
    });

    test("缺字段按 0 处理（老后端只回 ok/clips）", () => {
        const outcome = mergeOutcome(emptyOutcome(2), { ok: true, clips: 2 });
        expect(outcome.converted).toBe(0);
        expect(outcome.framesWritten).toBe(0);
        expect(outcome.unresolved).toEqual([]);
    });

    test("负数与 NaN 不把总账拉低", () => {
        const outcome = mergeOutcome(emptyOutcome(1), {
            clipsConverted: Number.NaN,
            framesWritten: -5,
        });
        expect(outcome.converted).toBe(0);
        expect(outcome.framesWritten).toBe(0);
    });

    test("转换数不会超过请求数（后端若多报，跳过数不为负）", () => {
        const outcome = mergeOutcome(emptyOutcome(1), { clipsConverted: 3 });
        expect(skippedClips(outcome)).toBe(0);
    });
});
