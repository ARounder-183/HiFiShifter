import { describe, expect, it } from "vitest";
import {
    CLIP_ORIGIN_SNAP_MAX_PX,
    CLIP_ORIGIN_SNAP_MIN_PX,
    resolveClipOriginSnap,
    resolveClipOriginSnapThresholdPx,
} from "./clipOriginSnap";

describe("resolveClipOriginSnapThresholdPx", () => {
    it("clamps the scale-aware threshold to 24..32 px", () => {
        expect(resolveClipOriginSnapThresholdPx(100)).toBe(CLIP_ORIGIN_SNAP_MIN_PX);
        expect(resolveClipOriginSnapThresholdPx(400)).toBe(24);
        expect(resolveClipOriginSnapThresholdPx(1000)).toBe(CLIP_ORIGIN_SNAP_MAX_PX);
        expect(resolveClipOriginSnapThresholdPx(100_000)).toBe(CLIP_ORIGIN_SNAP_MAX_PX);
    });

    it("falls back to the minimum for invalid zoom", () => {
        expect(resolveClipOriginSnapThresholdPx(Number.NaN)).toBe(CLIP_ORIGIN_SNAP_MIN_PX);
        expect(resolveClipOriginSnapThresholdPx(0)).toBe(CLIP_ORIGIN_SNAP_MIN_PX);
        expect(resolveClipOriginSnapThresholdPx(-10)).toBe(CLIP_ORIGIN_SNAP_MIN_PX);
    });
});

describe("resolveClipOriginSnap", () => {
    const base = {
        rawStartSec: 10,
        originStartSec: 10,
        pxPerSec: 400,
        crossTrack: true,
        enabled: true,
    };

    it("does not snap while the clip stays on its origin track", () => {
        expect(resolveClipOriginSnap({ ...base, crossTrack: false })).toMatchObject({
            active: false,
            startSec: 10,
        });
    });

    it("does not snap when snapping is disabled", () => {
        expect(resolveClipOriginSnap({ ...base, enabled: false })).toMatchObject({
            active: false,
            startSec: 10,
        });
    });

    it("snaps inside the screen-space threshold", () => {
        // 24px at 400px/s = 0.06s.
        expect(resolveClipOriginSnap({ ...base, rawStartSec: 10.05 })).toMatchObject({
            active: true,
            startSec: 10,
        });
    });

    it("snaps exactly at the threshold and releases just past it", () => {
        expect(resolveClipOriginSnap({ ...base, rawStartSec: 10.06 })).toMatchObject({
            active: true,
            startSec: 10,
        });
        expect(resolveClipOriginSnap({ ...base, rawStartSec: 10.07 })).toMatchObject({
            active: false,
            startSec: 10.07,
        });
    });

    it("uses the same screen-space threshold at different zooms", () => {
        const slow = resolveClipOriginSnap({
            ...base,
            pxPerSec: 200,
            rawStartSec: 10.1,
        });
        const fast = resolveClipOriginSnap({
            ...base,
            pxPerSec: 800,
            rawStartSec: 10.025,
        });
        expect(slow.active).toBe(true);
        expect(fast.active).toBe(true);
    });

    it("does not snap for invalid input", () => {
        expect(
            resolveClipOriginSnap({ ...base, rawStartSec: Number.NaN }),
        ).toMatchObject({ active: false });
        expect(
            resolveClipOriginSnap({ ...base, originStartSec: Number.NaN }),
        ).toMatchObject({ active: false });
        expect(resolveClipOriginSnap({ ...base, pxPerSec: 0 })).toMatchObject({
            active: false,
        });
        expect(resolveClipOriginSnap({ ...base, pxPerSec: Number.NaN })).toMatchObject({
            active: false,
        });
    });
});
