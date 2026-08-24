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
// （边缘/内容起点/源素材首尾是独立目标族，须一并关闭才能得到"无候选"场景。）
{
    const ctx = context({
        snapClipsToGrid: true,
        snapClipsToSelectionMarkersCursor: false,
        snapClipEdges: false,
        snapClipSnapOffset: false,
        snapClipsToSourceMedia: false,
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

// ── 源素材首尾候选：方向感知投影 / Loop 相位族 / 范围过滤 ─────────────
// 倒放非 Loop：消费窗口锚定 se（win=[se−len·r, se]），s=b 的投影按
// (winEnd − b)/r 计算 —— 旧实现用正放公式会给出镜像错位的目标。
{
    const revClip: ClipInfo = {
        ...clips[0],
        id: "rev",
        startSec: 10,
        lengthSec: 4,
        reversed: true,
        sourceStartSec: 0,
        sourceEndSec: 3,
        durationSec: 10,
    };
    const ctx = context({
        snapClipsToGrid: false,
        snapClipsToSelectionMarkersCursor: false,
        snapClipEdges: false,
        snapClipSnapOffset: false,
        snapClipsToSourceMedia: true,
        snapDistancePx: 40,
        clips: [revClip],
    });
    // win=[−1,3]：s=0 投影到 t=10+(3−0)=13（clip 内，真实媒体起点穿越点）；
    // s=10 投影到 t=10+(3−10)=3（clip 之前，不生成幻影）。
    // 旧实现在 11 与 21 处给出镜像幻影目标 —— 均不得存在。
    assertNear(snapTimelinePosition(ctx, 13.05).sec, 13, "reversed media-start projects direction-aware");
    assertNear(snapTimelinePosition(ctx, 11.05).sec, 11.05, "mirrored phantom target is gone");
}
// Loop 正放：媒体边界呈 mod-D 等差回绕族（首回绕点 = mod(−ss,D)/r）。
{
    const loopClip: ClipInfo = {
        ...clips[0],
        id: "loop",
        startSec: 0,
        lengthSec: 30,
        loopEnabled: true,
        sourceStartSec: 2,
        sourceEndSec: 4,
        durationSec: 10,
    };
    const ctx = context({
        snapClipsToGrid: false,
        snapClipsToSelectionMarkersCursor: false,
        snapClipEdges: false,
        snapClipSnapOffset: false,
        snapClipsToSourceMedia: true,
        snapDistancePx: 40,
        clips: [loopClip],
    });
    // ss=2、D=10 → 回绕点 8、18、28…（旧实现给出负值被 clamp 成 0 的幻影）。
    assertNear(snapTimelinePosition(ctx, 8.1).sec, 8, "loop first wrap phase");
    assertNear(snapTimelinePosition(ctx, 17.9).sec, 18, "loop second wrap = +D/r");
}
// Loop 倒放：锚点 clamp 到 min(se,D) 后取 mod(φ,D) 相位。
{
    const revLoop: ClipInfo = {
        ...clips[0],
        id: "rev-loop",
        startSec: 100,
        lengthSec: 30,
        reversed: true,
        loopEnabled: true,
        sourceStartSec: 0,
        sourceEndSec: 13, // > D：引擎锚点 φ=min(13,10)=10 → 首回绕点 mod(10,10)=0
        durationSec: 10,
    };
    const ctx = context({
        snapClipsToGrid: false,
        snapClipsToSelectionMarkersCursor: false,
        snapClipEdges: false,
        snapClipSnapOffset: false,
        snapClipsToSourceMedia: true,
        snapDistancePx: 40,
        clips: [revLoop],
    });
    assertNear(
        snapTimelinePosition(ctx, 110.05).sec,
        110,
        "reversed loop anchor clamped to media end before phasing",
    );
}
// 内容起点（snapOffset）：前导静音之后的首个可听采样。
// 左延伸正放 clip（ss=−3）：静音占 3s → 目标在 start+3。
{
    const extended: ClipInfo = {
        ...clips[0],
        id: "ext",
        startSec: 100,
        lengthSec: 6,
        sourceStartSec: -3,
        sourceEndSec: 3,
        durationSec: 20,
    };
    const ctx = context({
        snapClipsToGrid: false,
        snapClipsToSelectionMarkersCursor: false,
        snapClipEdges: false,
        snapClipsToSourceMedia: false,
        snapClipSnapOffset: true,
        snapDistancePx: 40,
        clips: [extended],
    });
    assertNear(snapTimelinePosition(ctx, 103.05).sec, 103, "snap offset = first audible sample");
}

console.log(`timelineSnapping checks passed (${checks})`);
