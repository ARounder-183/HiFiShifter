import { describe, expect, it } from "vitest";
import { resolveTrackSnap } from "./trackSnap";

describe("resolveTrackSnap", () => {
    it("accepts the raw row when the pointer is already well inside it", () => {
        expect(
            resolveTrackSnap({
                contentY: 90,
                rowHeight: 80,
                trackCount: 3,
                previousIndex: 0,
            }),
        ).toEqual({ index: 1, snapped: true, boundaryIndex: 1 });
    });

    it("keeps the previous row until the pointer is inside the new row", () => {
        expect(
            resolveTrackSnap({
                contentY: 83,
                rowHeight: 80,
                trackCount: 3,
                previousIndex: 0,
            }),
        ).toEqual({ index: 0, snapped: false, boundaryIndex: null });
        expect(
            resolveTrackSnap({
                contentY: 91,
                rowHeight: 80,
                trackCount: 3,
                previousIndex: 0,
            }),
        ).toEqual({ index: 1, snapped: true, boundaryIndex: 1 });
    });

    it("requires leaving the old row before snapping upward", () => {
        expect(
            resolveTrackSnap({
                contentY: 78,
                rowHeight: 80,
                trackCount: 3,
                previousIndex: 1,
            }),
        ).toEqual({ index: 1, snapped: false, boundaryIndex: null });
        expect(
            resolveTrackSnap({
                contentY: 69,
                rowHeight: 80,
                trackCount: 3,
                previousIndex: 1,
            }),
        ).toEqual({ index: 0, snapped: true, boundaryIndex: 1 });
    });

    it("uses the sentinel row below all tracks with the same threshold", () => {
        expect(
            resolveTrackSnap({
                contentY: 242,
                rowHeight: 80,
                trackCount: 3,
                previousIndex: 2,
            }),
        ).toEqual({ index: 2, snapped: false, boundaryIndex: null });
        expect(
            resolveTrackSnap({
                contentY: 251,
                rowHeight: 80,
                trackCount: 3,
                previousIndex: 2,
            }),
        ).toEqual({ index: 3, snapped: true, boundaryIndex: 3 });
    });

    it("is safe for invalid geometry and fractional row heights", () => {
        expect(
            resolveTrackSnap({
                contentY: 0,
                rowHeight: 0,
                trackCount: 3,
                previousIndex: 0,
            }),
        ).toEqual({ index: -1, snapped: false, boundaryIndex: null });
        expect(
            resolveTrackSnap({
                contentY: Number.NaN,
                rowHeight: 80.5,
                trackCount: 2,
                previousIndex: null,
            }),
        ).toEqual({ index: 0, snapped: false, boundaryIndex: null });
    });
});
