import { describe, expect, it } from "vitest";
import {
    resolveBlockAnchor,
    resolveBlockClickSelection,
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
