import type { ClipInfo, TrackInfo } from "../features/session/sessionTypes";
import { createDefaultTimelineSnapSettings } from "../features/session/sessionSlice";
import {
    alignClipsToSwingGrid,
    snapStepBeats,
    snapTimelinePosition,
    snapToConfiguredGrid,
    type TimelineSnapContext,
} from "./timelineSnapping";

let checks = 0;
function assertNear(actual: number, expected: number, label: string, eps = 1e-9) {
    checks += 1;
    if (Math.abs(actual - expected) > eps) {
        throw new Error(`${label}: expected ${expected}, got ${actual}`);
    }
}
function assertTrue(value: boolean, label: string) {
    checks += 1;
    if (!value) throw new Error(`assertion failed: ${label}`);
}

const track: TrackInfo = {
    id: "t0",
    name: "Track",
    muted: false,
    solo: false,
    volume: 1,
    composeEnabled: false,
    pitchAnalysisAlgo: "nsf_hifigan_onnx",
};
const clips: ClipInfo[] = [
    {
        id: "c0",
        trackId: "t0",
        name: "a",
        startSec: 1,
        lengthSec: 2,
        color: "blue",
        sourceStartSec: 0,
        sourceEndSec: 2,
        playbackRate: 1,
        reversed: false,
        loopEnabled: false,
        fadeInSec: 0,
        fadeOutSec: 0,
        gain: 1,
        muted: false,
        fadeInCurve: "sine",
        fadeOutCurve: "sine",
    },
    {
        id: "c1",
        trackId: "t0",
        name: "b",
        startSec: 3,
        lengthSec: 1,
        color: "cyan",
        sourceStartSec: 0,
        sourceEndSec: 1,
        playbackRate: 1,
        reversed: false,
        loopEnabled: false,
        fadeInSec: 0,
        fadeOutSec: 0,
        gain: 1,
        muted: false,
        fadeInCurve: "sine",
        fadeOutCurve: "sine",
    },
];

const baseSettings = createDefaultTimelineSnapSettings();

function context(
    overrides: Partial<TimelineSnapContext> &
        Partial<ReturnType<typeof createDefaultTimelineSnapSettings>> = {},
): TimelineSnapContext {
    const ctxOverrides = overrides as Partial<TimelineSnapContext>;
    const settingsOverrides = overrides as Partial<
        ReturnType<typeof createDefaultTimelineSnapSettings>
    >;
    return {
        settings: {
            ...baseSettings,
            ...(ctxOverrides.settings ?? {}),
            ...settingsOverrides,
        },
        grid: "1/4" as const,
        bpm: 120,
        beatsPerBar: 4,
        tempoMap: null,
        pxPerSec: 40,
        clips,
        tracks: [track],
        selectedClipIds: [],
        playheadSec: 1.5,
        object: "clip" as const,
        ...ctxOverrides,
    };
}

// 吸附到网格：1/4 = 0.5s @120BPM。
assertNear(snapToConfiguredGrid(0.51, null, 1, 120, baseSettings), 0.5, "grid snap quarter");
// Swing：奇数格延迟 25%（100% → 0.25s）。
assertNear(
    snapToConfiguredGrid(0.75, null, 1, 120, { swingEnabled: true, swingPercent: 100 }),
    0.75,
    "swing grid keeps odd candidate",
);
assertNear(
    snapToConfiguredGrid(0.6, null, 1, 120, { swingEnabled: true, swingPercent: 100 }),
    0.75,
    "swing snap picks shifted odd line when closer",
);
// 独立吸附间距。
assertNear(
    snapStepBeats({ ...baseSettings, useIndependentSnapSpacing: true, snapSpacing: "1/8" }, "1/4"),
    0.5,
    "independent snap spacing",
);
// 像素距离阈值。
{
    const ctx = context({ snapClipsToGrid: true, snapDistancePx: 4 });
    const result = snapTimelinePosition(ctx, 0.53);
    assertNear(result.sec, 0.5, "snap within pixel threshold");
}
{
    const ctx = context({ snapClipsToGrid: true, snapDistancePx: 0 });
    const result = snapTimelinePosition(ctx, 0.53);
    assertNear(result.sec, 0.53, "no snap outside threshold");
}
// 任意距离吸附到网格。
{
    const ctx = context({ snapClipsToGrid: true, snapToGridAnyDistance: true });
    assertNear(snapTimelinePosition(ctx, 0.49).sec, 0.5, "any-distance grid snap");
}
// 媒体项边缘候选（排除自身后仍能吸附另一 clip）。
{
    const ctx = context({
        snapClipsToGrid: false,
        snapClipsToSelectionMarkersCursor: true,
        excludeClipIds: new Set(["c1"]),
        snapDistancePx: 40,
    });
    const result = snapTimelinePosition(ctx, 1.02);
    assertNear(result.sec, 1, "snap to other media item start");
}
// 光标候选。
{
    const ctx = context({
        snapClipsToGrid: false,
        snapClipsToSelectionMarkersCursor: true,
        snapDistancePx: 40,
        excludeClipIds: new Set(["c0", "c1"]),
    });
    const result = snapTimelinePosition(ctx, 1.48);
    assertNear(result.sec, 1.5, "snap to cursor");
}
// 网格显示关闭且联动时不再吸网格。
{
    const ctx = context({
        snapClipsToGrid: true,
        snapClipsToSelectionMarkersCursor: false,
        gridVisible: false,
        snapFollowsGridVisibility: true,
        snapDistancePx: 40,
        playheadSec: 10,
    });
    assertNear(snapTimelinePosition(ctx, 0.51).sec, 0.51, "hidden grid is not a target");
}
// Swing 对齐现有 items。
{
    const swingClips = [
        clips[0],
        { ...clips[1], id: "swing-c", startSec: 0.6 },
    ];
    const updates = alignClipsToSwingGrid({
        clips: swingClips,
        settings: { ...baseSettings, swingEnabled: true, swingPercent: 100 },
        grid: "1/4",
        tempoMap: null,
        bpm: 120,
    });
    assertTrue(Math.abs((updates["c0"] ?? 1) - 1) < 1e-9, "on-grid clip stays");
    assertNear(updates["swing-c"] ?? 0.6, 0.75, "off-grid clip moves to swung grid", 1e-9);
}

console.log(`timelineSnapping checks passed (${checks})`);
