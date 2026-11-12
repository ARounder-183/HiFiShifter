import { describe, expect, it } from "vitest";
import { buildClipPitchOverlays, resolveClipPitchOverlayVisual } from "./clipPitchOverlay";

const ROOT = new Set(["root"]);

function clip(overrides: Record<string, unknown> = {}) {
    return {
        id: "a",
        trackId: "root",
        startSec: 0,
        lengthSec: 1,
        color: "blue",
        sourcePath: "a.wav",
        ...overrides,
    };
}

function curve(values: number[], framePeriodMs = 100, curveStartSec = 0) {
    return { curveStartSec, midiCurve: values, framePeriodMs };
}

describe("buildClipPitchOverlays", () => {
    it("把 Clip 的起点与长度换算成工程帧", () => {
        const overlays = buildClipPitchOverlays({
            clips: [clip({ id: "a", startSec: 1.25, lengthSec: 2 })],
            curves: { a: curve(new Array(20).fill(60), 100, 1.25) },
            groupTrackIds: ROOT,
            selectedClipIds: new Set(),
        });
        expect(overlays).toHaveLength(1);
        expect(overlays[0]!.startFrame).toBe(13);
        expect(overlays[0]!.frameCount).toBe(20);
        expect(overlays[0]!.framePeriodMs).toBe(100);
    });

    it("亚帧相位不改变工程帧号（曲线仍属于这个 Clip）", () => {
        // 1.0023 s / 5 ms = 200.46 → 工程帧 200；与 1.000 s 的 Clip 是同一帧。
        const overlays = buildClipPitchOverlays({
            clips: [clip({ id: "a", startSec: 1.0023, lengthSec: 0.2 })],
            curves: { a: curve(new Array(40).fill(60), 5, 1.0023) },
            groupTrackIds: ROOT,
            selectedClipIds: new Set(),
        });
        expect(overlays).toHaveLength(1);
        expect(overlays[0]!.startFrame).toBe(200);
        expect(overlays[0]!.frameCount).toBe(40);
    });

    it("帧周期与工程不一致时整条丢弃", () => {
        const overlays = buildClipPitchOverlays({
            clips: [clip({ id: "a" })],
            curves: { a: curve(new Array(10).fill(60), 10) },
            groupTrackIds: ROOT,
            selectedClipIds: new Set(),
            projectFramePeriodMs: 5,
        });
        expect(overlays).toEqual([]);
    });

    it("keeps two overlapping clips as distinct overlays and marks overlap depth", () => {
        const overlays = buildClipPitchOverlays({
            clips: [
                clip({ id: "a", color: "blue" }),
                clip({ id: "b", startSec: 0.5, color: "violet" }),
            ],
            curves: {
                a: curve([60, 61, 62, 63, 64, 65, 66, 67, 68, 69]),
                b: curve([70, 71, 72, 73, 74, 75, 76, 77, 78, 79], 100, 0.5),
            },
            groupTrackIds: ROOT,
            selectedClipIds: new Set(),
        });
        expect(overlays.map((overlay) => overlay.clipId)).toEqual(["a", "b"]);
        expect(overlays.every((overlay) => overlay.overlapDepth === 2)).toBe(true);
        expect(overlays[0]!.strokeColor).not.toBe(overlays[1]!.strokeColor);
    });

    it("keeps the audio source overlay when a pitch reference clip covers it", () => {
        const overlays = buildClipPitchOverlays({
            clips: [
                clip({ id: "audio", sourcePath: "a.wav" }),
                clip({ id: "reference", midiNoteData: [{ startSec: 0, endSec: 1, note: 72 }] }),
            ],
            curves: {
                audio: curve([60, 61, 62, 63, 64, 65, 66, 67, 68, 69]),
                reference: curve([72, 72, 72, 72, 72, 72, 72, 72, 72, 72]),
            },
            groupTrackIds: ROOT,
            selectedClipIds: new Set(),
        });
        expect(overlays.map((overlay) => overlay.kind)).toEqual(["sourceF0", "midiReference"]);
        expect(overlays[0]!.clipId).toBe("audio");
    });

    it("splits voiced runs at silent frames instead of connecting across them", () => {
        const overlays = buildClipPitchOverlays({
            clips: [clip({ id: "a", lengthSec: 0.5 })],
            curves: {
                a: curve([60, 61, 0, 64, 65], 100),
            },
            groupTrackIds: ROOT,
            selectedClipIds: new Set(),
        });
        expect(overlays).toHaveLength(1);
        expect(overlays[0]!.segments).toEqual([
            { startIndex: 0, endIndexExclusive: 2, overlapDepth: 1 },
            { startIndex: 3, endIndexExclusive: 5, overlapDepth: 1 },
        ]);
    });

    it("drops stale curves whose start frame or length does not match the clip", () => {
        // 起点落在另一个工程帧上：旧缓存属于别的几何。
        expect(
            buildClipPitchOverlays({
                clips: [clip({ id: "a" })],
                curves: { a: curve([60, 61, 62], 100, 0.1) },
                groupTrackIds: ROOT,
                selectedClipIds: new Set(),
            }),
        ).toEqual([]);
        // 曲线帧数不覆盖 Clip 帧数：不拉伸、不补零。
        expect(
            buildClipPitchOverlays({
                clips: [clip({ id: "a" })],
                curves: { a: curve([60, 61, 62], 100) },
                groupTrackIds: ROOT,
                selectedClipIds: new Set(),
            }),
        ).toEqual([]);
    });

    it("puts the selected clip last and gives it a stronger visual", () => {
        const overlays = buildClipPitchOverlays({
            clips: [clip({ id: "a" }), clip({ id: "b", startSec: 0.5 })],
            curves: {
                a: curve(new Array(10).fill(60)),
                b: curve(new Array(10).fill(70), 100, 0.5),
            },
            groupTrackIds: ROOT,
            selectedClipIds: new Set(["a"]),
        });
        expect(overlays[overlays.length - 1]!.clipId).toBe("a");
        const selected = overlays.find((overlay) => overlay.selected)!;
        const plain = overlays.find((overlay) => !overlay.selected)!;
        const selectedVisual = resolveClipPitchOverlayVisual(selected, 2);
        const plainVisual = resolveClipPitchOverlayVisual(plain, 2);
        expect(selectedVisual.alpha).toBeGreaterThan(plainVisual.alpha);
        expect(selectedVisual.lineWidthPx).toBeGreaterThan(plainVisual.lineWidthPx);
    });
});
