import { describe, expect, it } from "vitest";
import type { ParamBlock } from "./paramBlocks";
import { resolveParamBlockDragCenters } from "./paramBlockDragCenters";

function block(args: {
    id: string;
    userBlockId?: string;
    startSec?: number;
    endSec?: number;
    averageValue?: number | null;
    manualValue?: number | null;
}): ParamBlock {
    return {
        id: args.id,
        ...(args.userBlockId === undefined ? {} : { userBlockId: args.userBlockId }),
        rootTrackId: "root",
        kind: args.userBlockId === undefined ? "auto" : "user",
        members: [],
        startSec: args.startSec ?? 0,
        endSec: args.endSec ?? 1,
        averageValue: args.averageValue === undefined ? 60 : args.averageValue,
        manualValue: args.manualValue ?? null,
    };
}

describe("resolveParamBlockDragCenters", () => {
    it("resolves every selected block, not only the anchor", () => {
        const a = block({ id: "a", averageValue: 60 });
        const b = block({ id: "b", startSec: 2, endSec: 3, averageValue: 62 });
        const c = block({ id: "c", startSec: 4, endSec: 5, averageValue: 64 });
        const centers = resolveParamBlockDragCenters({
            selectedBlocks: [a, b, c],
            anchorBlockId: "a",
            anchorValue: 60.25,
            centerValues: new Map([
                ["a", 60],
                ["b", 62],
                ["c", 64],
            ]),
            param: "pitch",
        });
        expect(centers.byLogical.get("a")).toBe(60.25);
        expect(centers.byLogical.get("b")).toBe(62);
        expect(centers.byLogical.get("c")).toBe(64);
        expect(centers.byRun.get("b")).toBe(62);
        expect(centers.byRun.get("c")).toBe(64);
    });

    it("gives every run of a logical block the same center", () => {
        const first = block({
            id: "parent:0",
            userBlockId: "parent",
            averageValue: 48,
        });
        const second = block({
            id: "parent:1",
            userBlockId: "parent",
            startSec: 2,
            endSec: 3,
            averageValue: 50,
        });
        const centers = resolveParamBlockDragCenters({
            selectedBlocks: [first, second],
            anchorBlockId: "parent:0",
            anchorValue: 48,
            param: "pitch",
        });
        expect(centers.byLogical.get("parent")).toBe(48);
        expect(centers.byRun.get("parent:0")).toBe(48);
        expect(centers.byRun.get("parent:1")).toBe(48);
    });

    it("uses rendered geometry before manual/average fallbacks", () => {
        const a = block({ id: "a", averageValue: 60, manualValue: 70 });
        const b = block({ id: "b", averageValue: 50, manualValue: 40 });
        const centers = resolveParamBlockDragCenters({
            selectedBlocks: [a, b],
            anchorBlockId: "a",
            centerValues: new Map([
                ["a", 61],
                ["b", 51],
            ]),
            param: "pitch",
        });
        expect(centers.byLogical.get("a")).toBe(61);
        expect(centers.byLogical.get("b")).toBe(51);
    });

    it("ignores manualValue for non-pitch parameters", () => {
        const a = block({ id: "a", averageValue: 0.5, manualValue: 70 });
        const centers = resolveParamBlockDragCenters({
            selectedBlocks: [a],
            anchorBlockId: "a",
            param: "volume",
        });
        expect(centers.byLogical.get("a")).toBe(0.5);
    });

    it("does not invent a center when the block has no rendered value", () => {
        const a = block({ id: "a", averageValue: null, manualValue: null });
        const centers = resolveParamBlockDragCenters({
            selectedBlocks: [a],
            anchorBlockId: "a",
            param: "pitch",
        });
        expect(centers.byLogical.has("a")).toBe(false);
        expect(centers.byRun.has("a")).toBe(false);
    });
});
