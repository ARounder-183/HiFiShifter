import { describe, expect, it } from "vitest";
import {
    resolveBlockAnchor,
    resolveBlockClickSelection,
    resolveBlockPressPlan,
    resolveBlockRangeSelection,
} from "./paramBlockSelectionGesture";

const blocks = [
    { id: "a:0", userBlockId: "a", startFrame: 0, endFrameExclusive: 20 },
    { id: "a:1", userBlockId: "a", startFrame: 100, endFrameExclusive: 120 },
    { id: "b", startFrame: 40, endFrameExclusive: 60 },
    { id: "c", startFrame: 60, endFrameExclusive: 80 },
];

describe("paramBlockSelectionGesture", () => {
    it("replace selects every run of the logical block and sets the anchor", () => {
        const outcome = resolveBlockClickSelection({
            blocks,
            selection: null,
            anchor: null,
            targetBlockId: "a:0",
            targetFrame: 5,
            mode: "replace",
        });
        expect(outcome.selection).toEqual([
            { startFrame: 0, frameCount: 20 },
            { startFrame: 100, frameCount: 20 },
        ]);
        expect(outcome.anchor).toEqual({ blockId: "a", frame: 5 });
        expect(outcome.targetSelected).toBe(true);
    });

    it("ctrl toggle adds and removes the whole logical block", () => {
        const add = resolveBlockClickSelection({
            blocks,
            selection: null,
            anchor: null,
            targetBlockId: "b",
            targetFrame: 45,
            mode: "toggle",
        });
        expect(add.selection).toEqual([{ startFrame: 40, frameCount: 20 }]);
        expect(add.targetSelected).toBe(true);
        const remove = resolveBlockClickSelection({
            blocks,
            selection: add.selection,
            anchor: add.anchor,
            targetBlockId: "b",
            targetFrame: 45,
            mode: "toggle",
        });
        expect(remove.selection).toBeNull();
        expect(remove.targetSelected).toBe(false);
    });

    it("shift range selects whole blocks between anchor and target, including touching blocks", () => {
        const selection = resolveBlockRangeSelection({
            blocks,
            anchorFrame: 5,
            targetFrame: 65,
            anchorBlockId: "a",
        });
        expect(selection).toEqual([
            { startFrame: 0, frameCount: 20 },
            { startFrame: 40, frameCount: 40 },
            { startFrame: 100, frameCount: 20 },
        ]);
    });

    it("replace press on a selected block preserves the multi-selection until drag starts", () => {
        const selection = [
            { startFrame: 0, frameCount: 20 },
            { startFrame: 40, frameCount: 20 },
            { startFrame: 100, frameCount: 20 },
        ];
        const plan = resolveBlockPressPlan({
            blocks,
            selection,
            anchor: { blockId: "a", frame: 5 },
            targetBlockId: "b",
            targetFrame: 45,
            mode: "replace",
        });
        expect(plan.downSelection).toBe(selection);
        expect(plan.dragIds).toEqual(["a", "b"]);
        expect(plan.clickSelection).toEqual([{ startFrame: 40, frameCount: 20 }]);
        expect(plan.startsDrag).toBe(true);
    });

    it("replace press on an unselected block collapses selection immediately", () => {
        const plan = resolveBlockPressPlan({
            blocks,
            selection: [{ startFrame: 40, frameCount: 20 }],
            anchor: { blockId: "b", frame: 40 },
            targetBlockId: "c",
            targetFrame: 65,
            mode: "replace",
        });
        expect(plan.downSelection).toEqual([{ startFrame: 60, frameCount: 20 }]);
        expect(plan.dragIds).toEqual(["c"]);
        expect(plan.clickSelection).toEqual([{ startFrame: 60, frameCount: 20 }]);
    });

    it("ctrl removal never starts a drag and moves the anchor off the removed block", () => {
        const selection = [
            { startFrame: 40, frameCount: 20 },
            { startFrame: 60, frameCount: 20 },
        ];
        const plan = resolveBlockPressPlan({
            blocks,
            selection,
            anchor: { blockId: "b", frame: 45 },
            targetBlockId: "b",
            targetFrame: 45,
            mode: "toggle",
        });
        expect(plan.downSelection).toEqual([{ startFrame: 60, frameCount: 20 }]);
        expect(plan.dragIds).toEqual([]);
        expect(plan.startsDrag).toBe(false);
        expect(plan.downAnchor?.blockId).toBe("c");
    });

    it("ctrl addition drags the whole new selection", () => {
        const plan = resolveBlockPressPlan({
            blocks,
            selection: [{ startFrame: 40, frameCount: 20 }],
            anchor: { blockId: "b", frame: 40 },
            targetBlockId: "c",
            targetFrame: 65,
            mode: "toggle",
        });
        expect(plan.dragIds).toEqual(["b", "c"]);
        expect(plan.startsDrag).toBe(true);
        expect(plan.clickSelection).toBeNull();
    });

    it("shift press drags the range result instead of the stale selection", () => {
        const plan = resolveBlockPressPlan({
            blocks,
            selection: [{ startFrame: 40, frameCount: 20 }],
            anchor: { blockId: "b", frame: 45 },
            targetBlockId: "c",
            targetFrame: 65,
            mode: "range",
        });
        expect(plan.dragIds).toEqual(["b", "c"]);
        expect(plan.dragSelection).toEqual([{ startFrame: 40, frameCount: 40 }]);
        expect(plan.clickSelection).toEqual([{ startFrame: 40, frameCount: 40 }]);
        expect(plan.startsDrag).toBe(true);
    });

    it("anchor fallback uses the first selected block then the playhead block", () => {
        expect(
            resolveBlockAnchor({
                blocks,
                anchor: null,
                selection: [{ startFrame: 40, frameCount: 20 }],
            }),
        ).toEqual({ blockId: "b", frame: 40 });
        expect(
            resolveBlockAnchor({ blocks, anchor: null, selection: null, playheadFrame: 70 }),
        ).toEqual({ blockId: "c", frame: 70 });
    });
});
