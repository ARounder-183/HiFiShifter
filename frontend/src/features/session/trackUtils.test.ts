import { describe, expect, it } from "vitest";
import { computeInsertBelowPlacement, type TrackParentRef } from "./trackUtils";

/**
 * 场景树（DFS 显示顺序，与后端 build_track_payload 一致）：
 *   A (root)
 *   ├─ A1
 *   │  └─ A1a
 *   └─ A2
 *   B (root)
 *   └─ B1
 *   C (root)
 */
const TRACKS: TrackParentRef[] = [
    { id: "A", parentId: null },
    { id: "A1", parentId: "A" },
    { id: "A1a", parentId: "A1" },
    { id: "A2", parentId: "A" },
    { id: "B", parentId: null },
    { id: "B1", parentId: "B" },
    { id: "C", parentId: null },
];

describe("computeInsertBelowPlacement — 添加轨道紧贴选中轨道下方", () => {
    it("无选中轨道：追加为根轨道列表末尾", () => {
        expect(computeInsertBelowPlacement(TRACKS, null)).toEqual({
            parentTrackId: null,
            index: 3,
        });
    });

    it("选中不存在轨道：按无选中处理", () => {
        expect(computeInsertBelowPlacement(TRACKS, "missing")).toEqual({
            parentTrackId: null,
            index: 3,
        });
    });

    it("选中根轨道 A：新根轨道插入到 A 之后（其子树之后）", () => {
        expect(computeInsertBelowPlacement(TRACKS, "A")).toEqual({
            parentTrackId: null,
            index: 1,
        });
    });

    it("选中根轨道 B：新根轨道紧跟 B（而非插入到列表末尾）", () => {
        // B 的扁平下标是 4，但根列表位置是 1 → 目标 index 应为 2（B、C 之间）
        expect(computeInsertBelowPlacement(TRACKS, "B")).toEqual({
            parentTrackId: null,
            index: 2,
        });
    });

    it("选中最后一条根轨道 C：插入到根列表末尾", () => {
        expect(computeInsertBelowPlacement(TRACKS, "C")).toEqual({
            parentTrackId: null,
            index: 3,
        });
    });

    it("选中子轨道 A1：继承父级 A，插入到 A1 之后", () => {
        expect(computeInsertBelowPlacement(TRACKS, "A1")).toEqual({
            parentTrackId: "A",
            index: 1,
        });
    });

    it("选中子轨道 A2（A 的第二个孩子）：插入到 A2 之后", () => {
        expect(computeInsertBelowPlacement(TRACKS, "A2")).toEqual({
            parentTrackId: "A",
            index: 2,
        });
    });

    it("选中孙轨道 A1a：继承 A1，紧贴 A1a 下方", () => {
        expect(computeInsertBelowPlacement(TRACKS, "A1a")).toEqual({
            parentTrackId: "A1",
            index: 1,
        });
    });

    it("单轨道工程：选中唯一根轨道 → 紧跟其后", () => {
        expect(computeInsertBelowPlacement([{ id: "M", parentId: null }], "M")).toEqual({
            parentTrackId: null,
            index: 1,
        });
    });
});
