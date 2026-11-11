import { describe, expect, it } from "vitest";

import { markParamBlockPreviewOwnedRevision } from "./paramBlockPreviewOwnership";

describe("markParamBlockPreviewOwnedRevision", () => {
    it("marks every live preview ref with the local commit revision", () => {
        const visualTransaction = { current: { ownedEditRevision: null } };
        const dragSnapshot = { current: { ownedEditRevision: null } };
        const resize = { current: { ownedEditRevision: null } };
        markParamBlockPreviewOwnedRevision(
            { visualTransaction, dragSnapshot, resize },
            42,
        );
        expect(visualTransaction.current.ownedEditRevision).toBe(42);
        expect(dragSnapshot.current.ownedEditRevision).toBe(42);
        expect(resize.current.ownedEditRevision).toBe(42);
    });

    it("skips absent previews and ignores non-finite revisions", () => {
        const visualTransaction = { current: null };
        const dragSnapshot = { current: null };
        const resize = { current: { ownedEditRevision: 7 } };
        markParamBlockPreviewOwnedRevision(
            { visualTransaction, dragSnapshot, resize },
            9,
        );
        expect(resize.current.ownedEditRevision).toBe(9);
        markParamBlockPreviewOwnedRevision(
            { visualTransaction, dragSnapshot, resize },
            Number.NaN,
        );
        expect(resize.current.ownedEditRevision).toBe(9);
    });
});
