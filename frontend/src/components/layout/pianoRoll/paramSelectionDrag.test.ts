import { describe, expect, it } from "vitest";
import { selectionForRangeDrag } from "./paramSelectionDrag";

describe("selectionForRangeDrag", () => {
    it("left drag replaces the previous selection", () => {
        expect(
            selectionForRangeDrag({
                base: [{ startFrame: 0, frameCount: 10 }],
                anchorFrame: 20,
                pointerFrame: 30,
                mode: "replace",
            }),
        ).toEqual([{ startFrame: 20, frameCount: 11 }]);
    });

    it("right / ctrl drag appends to the previous selection", () => {
        expect(
            selectionForRangeDrag({
                base: [{ startFrame: 0, frameCount: 10 }],
                anchorFrame: 20,
                pointerFrame: 30,
                mode: "add",
            }),
        ).toEqual([
            { startFrame: 0, frameCount: 10 },
            { startFrame: 20, frameCount: 11 },
        ]);
    });

    it("touching ranges merge exactly like the selection tool", () => {
        expect(
            selectionForRangeDrag({
                base: [{ startFrame: 0, frameCount: 10 }],
                anchorFrame: 10,
                pointerFrame: 15,
                mode: "add",
            }),
        ).toEqual([{ startFrame: 0, frameCount: 16 }]);
    });
});
