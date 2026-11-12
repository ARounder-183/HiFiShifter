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

/** 一条 1 s / 100 ms 帧周期的整段曲线（10 帧）。 */
function flatCurve(value = 60, curveStartSec = 0) {
    return curve(new Array(10).fill(value), 100, curveStartSec);
}

describe("buildClipPitchOverlays · 帧基准", () => {
    it("把 Clip 的起点与长度换算成工程帧", () => {
        const overlays = buildClipPitchOverlays({
            clips: [clip({ id: "a", startSec: 1.25, lengthSec: 2 })],
            curves: { a: curve(new Array(20).fill(60), 100, 1.25) },
            groupTrackIds: ROOT,
            focusClipId: null,
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
            focusClipId: null,
        });
        expect(overlays).toHaveLength(1);
        expect(overlays[0]!.startFrame).toBe(200);
        expect(overlays[0]!.frameCount).toBe(40);
    });

    it("帧周期与工程不一致时整条丢弃", () => {
        const overlays = buildClipPitchOverlays({
            clips: [clip({ id: "a" })],
            curves: { a: flatCurve() },
            groupTrackIds: ROOT,
            focusClipId: null,
            projectFramePeriodMs: 5,
        });
        expect(overlays).toEqual([]);
    });

    it("drops stale curves whose start frame or length does not match the clip", () => {
        // 起点落在另一个工程帧上：旧缓存属于别的几何。
        expect(
            buildClipPitchOverlays({
                clips: [clip({ id: "a" })],
                curves: { a: curve([60, 61, 62], 100, 0.1) },
                groupTrackIds: ROOT,
                focusClipId: null,
            }),
        ).toEqual([]);
        // 曲线帧数不覆盖 Clip 帧数：不拉伸、不补零。
        expect(
            buildClipPitchOverlays({
                clips: [clip({ id: "a" })],
                curves: { a: curve([60, 61, 62], 100) },
                groupTrackIds: ROOT,
                focusClipId: null,
            }),
        ).toEqual([]);
    });
});

describe("buildClipPitchOverlays · 身份色", () => {
    it("重合的两个 Clip 必不同色", () => {
        const overlays = buildClipPitchOverlays({
            clips: [clip({ id: "a" }), clip({ id: "b", startSec: 0.5 })],
            curves: { a: flatCurve(60), b: flatCurve(70, 0.5) },
            groupTrackIds: ROOT,
            focusClipId: null,
        });
        expect(overlays).toHaveLength(2);
        expect(overlays.every((overlay) => overlay.overlapDepth === 2)).toBe(true);
        expect(overlays[0]!.colorIndex).not.toBe(overlays[1]!.colorIndex);
    });

    it("哈希撞车时，重合会强制让位、不重合则允许同色", () => {
        // clip-0 与 clip-8 的 FNV-1a 落在同一个色槽（已验证）。
        const apart = buildClipPitchOverlays({
            clips: [clip({ id: "clip-0" }), clip({ id: "clip-8", startSec: 2 })],
            curves: { "clip-0": flatCurve(60), "clip-8": flatCurve(70, 2) },
            groupTrackIds: ROOT,
            focusClipId: null,
        });
        expect(apart[0]!.colorIndex).toBe(apart[1]!.colorIndex);

        const overlapping = buildClipPitchOverlays({
            clips: [clip({ id: "clip-0" }), clip({ id: "clip-8", startSec: 0.5 })],
            curves: { "clip-0": flatCurve(60), "clip-8": flatCurve(70, 0.5) },
            groupTrackIds: ROOT,
            focusClipId: null,
        });
        expect(overlapping[0]!.colorIndex).not.toBe(overlapping[1]!.colorIndex);
    });

    it("同一个 Clip 恒同色，且与输入顺序无关", () => {
        const forward = buildClipPitchOverlays({
            clips: [
                clip({ id: "a" }),
                clip({ id: "b", startSec: 0.5 }),
                clip({ id: "c", startSec: 3 }),
            ],
            curves: { a: flatCurve(60), b: flatCurve(70, 0.5), c: flatCurve(80, 3) },
            groupTrackIds: ROOT,
            focusClipId: null,
        });
        const reversed = buildClipPitchOverlays({
            clips: [
                clip({ id: "c", startSec: 3 }),
                clip({ id: "b", startSec: 0.5 }),
                clip({ id: "a" }),
            ],
            curves: { a: flatCurve(60), b: flatCurve(70, 0.5), c: flatCurve(80, 3) },
            groupTrackIds: ROOT,
            focusClipId: null,
        });
        const byId = (list: typeof forward) =>
            Object.fromEntries(list.map((overlay) => [overlay.clipId, overlay.colorIndex]));
        expect(byId(forward)).toEqual(byId(reversed));
    });
});

describe("buildClipPitchOverlays · 曲线内容", () => {
    it("keeps the audio source overlay when a pitch reference clip covers it", () => {
        const overlays = buildClipPitchOverlays({
            clips: [
                clip({ id: "audio", sourcePath: "a.wav" }),
                clip({ id: "reference", midiNoteData: [{ startSec: 0, endSec: 1, note: 72 }] }),
            ],
            curves: { audio: flatCurve(60), reference: flatCurve(72) },
            groupTrackIds: ROOT,
            focusClipId: null,
        });
        expect(overlays.map((overlay) => overlay.kind)).toEqual(["sourceF0", "midiReference"]);
        expect(overlays[0]!.clipId).toBe("audio");
    });

    it("splits voiced runs at silent frames instead of connecting across them", () => {
        const overlays = buildClipPitchOverlays({
            clips: [clip({ id: "a", lengthSec: 0.5 })],
            curves: { a: curve([60, 61, 0, 64, 65], 100) },
            groupTrackIds: ROOT,
            focusClipId: null,
        });
        expect(overlays).toHaveLength(1);
        expect(overlays[0]!.segments).toEqual([
            { startIndex: 0, endIndexExclusive: 2, overlapDepth: 1 },
            { startIndex: 3, endIndexExclusive: 5, overlapDepth: 1 },
        ]);
    });
});

describe("buildClipPitchOverlays · 焦点", () => {
    it("焦点 Clip 最后绘制且视觉更强", () => {
        const overlays = buildClipPitchOverlays({
            clips: [clip({ id: "a" }), clip({ id: "b", startSec: 0.5 })],
            curves: { a: flatCurve(60), b: flatCurve(70, 0.5) },
            groupTrackIds: ROOT,
            focusClipId: "a",
        });
        expect(overlays[overlays.length - 1]!.clipId).toBe("a");
        const focused = overlays.find((overlay) => overlay.focused)!;
        const plain = overlays.find((overlay) => !overlay.focused)!;
        const focusedVisual = resolveClipPitchOverlayVisual(focused, 2);
        const plainVisual = resolveClipPitchOverlayVisual(plain, 2);
        expect(focusedVisual.alpha).toBeGreaterThan(plainVisual.alpha);
        expect(focusedVisual.lineWidthPx).toBeGreaterThan(plainVisual.lineWidthPx);
    });

    it("多选不会把多条曲线一起点亮（焦点只认单选）", () => {
        // 切分后 multiSelectedClipIds 会累积多个 Clip，但焦点只取 selectedClipId；
        // 这里用 focusClipId 表达同一语义：只有一条线被点亮。
        const overlays = buildClipPitchOverlays({
            clips: [clip({ id: "a" }), clip({ id: "b", startSec: 0.5 })],
            curves: { a: flatCurve(60), b: flatCurve(70, 0.5) },
            groupTrackIds: ROOT,
            focusClipId: null,
        });
        expect(overlays.filter((overlay) => overlay.focused)).toHaveLength(0);
    });
});
