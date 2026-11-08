import { describe, expect, it } from "vitest";
import {
    isParamAuthorityConfirmedAfterCommit,
    ownsParamBlockGesture,
} from "./paramBlockTransaction";

describe("param block transaction authority", () => {
    it("rejects an in-flight request that was issued before the commit", () => {
        expect(
            isParamAuthorityConfirmedAfterCommit({
                watermark: { issuedRequestSeq: 4, appliedRequestSeq: 4 },
                issuedRequestSeqAtCommit: 4,
            }),
        ).toBe(false);
    });

    it("accepts only a response from a request issued after the commit", () => {
        expect(
            isParamAuthorityConfirmedAfterCommit({
                watermark: { issuedRequestSeq: 5, appliedRequestSeq: 5 },
                issuedRequestSeqAtCommit: 4,
            }),
        ).toBe(true);
    });

    it("does not let the A-to-B completion clear a newer B-to-C gesture", () => {
        expect(
            ownsParamBlockGesture({
                activeGestureToken: 2,
                gestureToken: 1,
            }),
        ).toBe(false);
        expect(
            ownsParamBlockGesture({
                activeGestureToken: 2,
                gestureToken: 2,
            }),
        ).toBe(true);
    });
});
