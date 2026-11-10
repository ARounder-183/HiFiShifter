import { describe, expect, it } from "vitest";
import { canRollbackParamBlockDragCommit } from "./paramBlockDrag";

describe("canRollbackParamBlockDragCommit", () => {
    it("allows rolling back to the pre-commit revision", () => {
        expect(
            canRollbackParamBlockDragCommit({
                currentRevision: 7,
                beforeRevision: 7,
                appliedRevision: null,
            }),
        ).toBe(true);
    });

    it("allows rolling back the gesture's own applied revision", () => {
        expect(
            canRollbackParamBlockDragCommit({
                currentRevision: 8,
                beforeRevision: 7,
                appliedRevision: 8,
            }),
        ).toBe(true);
    });

    it("refuses to overwrite a third-party structural edit", () => {
        expect(
            canRollbackParamBlockDragCommit({
                currentRevision: 9,
                beforeRevision: 7,
                appliedRevision: 8,
            }),
        ).toBe(false);
    });

    it("refuses when no applied revision is known and current moved on", () => {
        expect(
            canRollbackParamBlockDragCommit({
                currentRevision: 9,
                beforeRevision: 7,
                appliedRevision: null,
            }),
        ).toBe(false);
    });
});
