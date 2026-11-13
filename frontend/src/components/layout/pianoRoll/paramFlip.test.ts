import { describe, expect, it } from "vitest";

import { flipParamValue, supportsParamFlip } from "./paramFlip";

describe("supportsParamFlip", () => {
    it("accepts zero-symmetric params", () => {
        expect(supportsParamFlip({ min: -1, max: 1 }, 0)).toBe(true);
        expect(supportsParamFlip({ min: -100, max: 100 }, 0)).toBe(true);
    });

    it("rejects non-negative and non-zero-centered params", () => {
        expect(supportsParamFlip({ min: 0, max: 2 }, 1)).toBe(false);
        expect(supportsParamFlip({ min: 0, max: 1 }, 0)).toBe(false);
        expect(supportsParamFlip({ min: 0, max: 127 }, 60)).toBe(false);
        expect(supportsParamFlip(undefined, 0)).toBe(false);
    });
});

describe("flipParamValue", () => {
    it("mirrors around zero and preserves zero", () => {
        expect(flipParamValue(-0.3)).toBe(0.3);
        expect(flipParamValue(0.3)).toBe(-0.3);
        expect(flipParamValue(0)).toBe(0);
        expect(flipParamValue(-0)).toBe(0);
        expect(flipParamValue(Number.NaN)).toBe(0);
    });
});
