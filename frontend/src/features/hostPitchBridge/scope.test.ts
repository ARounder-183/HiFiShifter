/**
 * 作用域解析的边界：轨道组、选区重叠、非有限几何。
 *
 * 【为什么要钉"端点相接不算重叠"】这条判据决定了"选区紧贴下一个片段"时会不会
 * 顺手把那个片段也转掉。差一帧的越界写入在界面上完全看不出来，只有断言能拦住。
 */
import { describe, expect, test } from "vitest";

import { resolvePitchBridgeScopes, scopeClipIds } from "./scope";

const clip = (id: string, trackId: string, startSec: number, lengthSec: number) => ({
    id,
    trackId,
    startSec,
    lengthSec,
});

const trackIds = new Set(["root", "child"]);

describe("作用域解析", () => {
    test("当前轨道只取轨道组内的片段（根轨道 + 子轨道）", () => {
        const clips = [clip("a", "root", 0, 1), clip("b", "child", 1, 1), clip("c", "other", 2, 1)];
        const scopes = resolvePitchBridgeScopes({ clips, trackIds, rangesSec: [] });
        expect(scopes.track).toEqual(["a", "b"]);
        expect(scopes.all).toEqual(["a", "b", "c"]);
    });

    test("选区内按重叠取，且不把端点相接的片段算进来", () => {
        const clips = [
            clip("before", "root", 0, 1), // [0,1) 与 [1,2) 相接
            clip("inside", "root", 1.5, 0.4),
            clip("straddle", "root", 1.9, 3),
            clip("after", "root", 5, 1),
        ];
        const scopes = resolvePitchBridgeScopes({
            clips,
            trackIds,
            rangesSec: [[1, 2]],
        });
        expect(scopes.selection).toEqual(["inside", "straddle"]);
    });

    test("多段选区取并集", () => {
        const clips = [
            clip("a", "root", 0, 0.5),
            clip("b", "root", 10, 0.5),
            clip("c", "root", 5, 0.5),
        ];
        const scopes = resolvePitchBridgeScopes({
            clips,
            trackIds,
            rangesSec: [
                [0, 1],
                [9.9, 11],
            ],
        });
        expect(scopes.selection).toEqual(["a", "b"]);
    });

    test("没有选区时选区内为空（菜单据此不显示那一项）", () => {
        const clips = [clip("a", "root", 0, 1)];
        const scopes = resolvePitchBridgeScopes({ clips, trackIds, rangesSec: [] });
        expect(scopes.selection).toEqual([]);
    });

    test("倒着给的区间（终点小于起点）按端点夹正，不会一个都不选", () => {
        const clips = [clip("a", "root", 0.2, 0.2)];
        const scopes = resolvePitchBridgeScopes({ clips, trackIds, rangesSec: [[1, 0]] });
        expect(scopes.selection).toEqual(["a"]);
    });

    test("非有限几何的片段不进选区内（但仍在全部里）", () => {
        const clips = [clip("nan", "root", Number.NaN, 1), clip("ok", "root", 0, 1)];
        const scopes = resolvePitchBridgeScopes({ clips, trackIds, rangesSec: [[0, 1]] });
        expect(scopes.selection).toEqual(["ok"]);
        expect(scopes.all).toEqual(["nan", "ok"]);
    });

    test("scopeClipIds 只是取列表，不复制语义", () => {
        const scopes = { track: ["a"], selection: [], all: ["a", "b"] };
        expect(scopeClipIds(scopes, "track")).toEqual(["a"]);
        expect(scopeClipIds(scopes, "all")).toEqual(["a", "b"]);
    });
});
