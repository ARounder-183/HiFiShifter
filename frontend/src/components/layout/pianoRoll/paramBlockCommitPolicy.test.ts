import { describe, expect, it } from "vitest";

import { assertCurveCommitAuthority } from "./paramBlockCommitPolicy";

describe("assertCurveCommitAuthority", () => {
    it("allows patch-free commits without authority", () => {
        expect(() =>
            assertCurveCommitAuthority({ hasPatches: false, authority: undefined }),
        ).not.toThrow();
    });

    it("requires a finite expectedCurveRevision for curve patches", () => {
        expect(() =>
            assertCurveCommitAuthority({ hasPatches: true, authority: undefined }),
        ).toThrow("expectedCurveRevision");
        expect(() =>
            assertCurveCommitAuthority({
                hasPatches: true,
                authority: { expectedCurveRevision: Number.NaN },
            }),
        ).toThrow("expectedCurveRevision");
    });

    it("accepts a valid authority object", () => {
        expect(() =>
            assertCurveCommitAuthority({
                hasPatches: true,
                authority: {
                    expectedCurveRevision: 7,
                    expectedGeometryRevision: 3,
                    expectedGeometrySignature: "sig",
                },
            }),
        ).not.toThrow();
    });
});
