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

describe("buildClipPitchOverlays · 切段只在静音处", () => {
    it("亚帧重合不得把前一条曲线切出空洞（重合处曲线必须连续）", () => {
        // 复现用户报的"两条曲线都在后一条 Clip 的起点断一小段"：
        // A 0–1.5 s、B 1.002–1.502 s，两者只在亚帧级别重合。旧实现按"重合深度
        // 变化"切段：A 在 200→201 处断成两条独立折线（一帧宽的洞），B 的
        // [0,1) 段只有 1 个采样被整段丢掉（曲线晚一帧开始）。
        const overlays = buildClipPitchOverlays({
            clips: [
                clip({ id: "a", startSec: 0, lengthSec: 1.5 }),
                clip({ id: "b", startSec: 1.002, lengthSec: 0.5 }),
            ],
            curves: {
                a: { curveStartSec: 0, midiCurve: new Array(300).fill(60), framePeriodMs: 5 },
                b: { curveStartSec: 1.002, midiCurve: new Array(100).fill(70), framePeriodMs: 5 },
            },
            groupTrackIds: ROOT,
            focusClipId: "a",
        });
        const byId = (id: string) => overlays.find((overlay) => overlay.clipId === id)!;
        expect(byId("a").runs).toEqual([{ startIndex: 0, endIndexExclusive: 300 }]);
        expect(byId("b").runs).toEqual([{ startIndex: 0, endIndexExclusive: 100 }]);
    });

    it("帧对齐重合同样不切段", () => {
        const overlays = buildClipPitchOverlays({
            clips: [
                clip({ id: "a", startSec: 0, lengthSec: 1.5 }),
                clip({ id: "b", startSec: 1, lengthSec: 0.5 }),
            ],
            curves: {
                a: { curveStartSec: 0, midiCurve: new Array(300).fill(60), framePeriodMs: 5 },
                b: { curveStartSec: 1, midiCurve: new Array(100).fill(70), framePeriodMs: 5 },
            },
            groupTrackIds: ROOT,
            focusClipId: null,
        });
        expect(overlays.every((overlay) => overlay.runs.length === 1)).toBe(true);
        expect(overlays.every((overlay) => overlay.runs[0]!.startIndex === 0)).toBe(true);
    });

    it("splits voiced runs at silent frames instead of connecting across them", () => {
        const overlays = buildClipPitchOverlays({
            clips: [clip({ id: "a", lengthSec: 0.5 })],
            curves: { a: curve([60, 61, 0, 64, 65], 100) },
            groupTrackIds: ROOT,
            focusClipId: null,
        });
        expect(overlays).toHaveLength(1);
        expect(overlays[0]!.runs).toEqual([
            { startIndex: 0, endIndexExclusive: 2 },
            { startIndex: 3, endIndexExclusive: 5 },
        ]);
    });

    it("首尾静音不会让 run 越过有声区间", () => {
        const overlays = buildClipPitchOverlays({
            clips: [clip({ id: "a", lengthSec: 0.8 })],
            curves: { a: curve([0, 0, 62, 63, 64, 0, 0, 0], 100) },
            groupTrackIds: ROOT,
            focusClipId: null,
        });
        expect(overlays[0]!.runs).toEqual([{ startIndex: 2, endIndexExclusive: 5 }]);
    });
});

describe("buildClipPitchOverlays · 分组过滤", () => {
    it("只画同一根轨道组内的 Clip", () => {
        const overlays = buildClipPitchOverlays({
            clips: [
                clip({ id: "a" }),
                // 另一条轨道上时间完全重叠的 Clip：不属于当前组，不参与绘制。
                clip({ id: "other", trackId: "elsewhere" }),
            ],
            curves: { a: flatCurve(60), other: flatCurve(70) },
            groupTrackIds: ROOT,
            focusClipId: null,
        });
        expect(overlays.map((overlay) => overlay.clipId)).toEqual(["a"]);
    });

    it("muted 的 Clip 不画", () => {
        const overlays = buildClipPitchOverlays({
            clips: [clip({ id: "a" }), clip({ id: "b", startSec: 0.5, muted: true })],
            curves: { a: flatCurve(60), b: flatCurve(70, 0.5) },
            groupTrackIds: ROOT,
            focusClipId: null,
        });
        expect(overlays).toHaveLength(1);
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
        const focusedVisual = resolveClipPitchOverlayVisual(focused);
        const plainVisual = resolveClipPitchOverlayVisual(plain);
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

describe("resolveClipPitchOverlayVisual · 视觉层级", () => {
    const overlay = (kind: "sourceF0" | "midiReference", focused: boolean) => ({
        clipId: "a",
        startFrame: 0,
        frameCount: 10,
        framePeriodMs: 100,
        midiCurve: [],
        kind,
        colorIndex: 0,
        focused,
        zOrder: 0,
        runs: [],
    });

    /** 深色主题下"色值 × alpha 叠在 #1f1f1f 上"的相对亮度（0-255）。 */
    const lumaOnDark = (alpha: number, rgb: readonly [number, number, number]) =>
        0.2126 * (rgb[0] * alpha + 31 * (1 - alpha)) +
        0.7152 * (rgb[1] * alpha + 31 * (1 - alpha)) +
        0.0722 * (rgb[2] * alpha + 31 * (1 - alpha));

    it("焦点比非焦点强，但永远压不过 pv.orig", () => {
        // pv.orig = rgba(200,200,200,0.55) → 亮度 124。
        const PV_ORIG_LUMA = lumaOnDark(0.55, [200, 200, 200]);
        // 调色板里最亮的一支（黄）与最暗的一支（蓝紫），取包络。
        const brightest = lumaOnDark(
            resolveClipPitchOverlayVisual(overlay("sourceF0", true)).alpha,
            [222, 222, 115],
        );
        expect(brightest).toBeLessThanOrEqual(PV_ORIG_LUMA);
        expect(
            lumaOnDark(
                resolveClipPitchOverlayVisual(overlay("sourceF0", false)).alpha,
                [222, 222, 115],
            ),
        ).toBeLessThan(brightest);
    });

    it("线宽永远细于 pv.orig（1.8）", () => {
        for (const kind of ["sourceF0", "midiReference"] as const) {
            for (const focused of [true, false]) {
                expect(
                    resolveClipPitchOverlayVisual(overlay(kind, focused)).lineWidthPx,
                ).toBeLessThan(1.8);
            }
        }
    });

    it("sourceF0 比 midiReference 重一档（实线 > 虚线）", () => {
        const solid = resolveClipPitchOverlayVisual(overlay("sourceF0", false));
        const dashed = resolveClipPitchOverlayVisual(overlay("midiReference", false));
        expect(solid.alpha).toBeGreaterThan(dashed.alpha);
        expect(solid.lineWidthPx).toBeGreaterThan(dashed.lineWidthPx);
        expect(solid.dash).toBeNull();
        expect(dashed.dash).not.toBeNull();
    });

    it("alpha 上界写死：任何组合都不得回到「压过主线」的 0.95", () => {
        for (const kind of ["sourceF0", "midiReference"] as const) {
            for (const focused of [true, false]) {
                expect(
                    resolveClipPitchOverlayVisual(overlay(kind, focused)).alpha,
                ).toBeLessThanOrEqual(0.5);
            }
        }
    });
});
