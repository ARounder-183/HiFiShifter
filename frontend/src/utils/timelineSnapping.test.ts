import type { ClipInfo, TrackInfo } from "../features/session/sessionTypes";
import { createDefaultTimelineSnapSettings } from "../features/session/sessionSlice";
import {
    alignClipsToSwingGrid,
    snapStepBeats,
    snapTimelineClipMove,
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
        snapOffsetSec: 0,
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
        snapOffsetSec: 0,
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
// 显式 SnapOffset（相对 Clip 起点的偏移）：非 0 时是独立吸附目标；
// 为 0 时不生成候选（与 clipStart 重合）。
{
    const offsetClip: ClipInfo = {
        ...clips[0],
        id: "with-offset",
        startSec: 100,
        lengthSec: 6,
        snapOffsetSec: 2.5,
    };
    const ctxOff = context({
        snapClipsToGrid: false,
        snapClipsToSelectionMarkersCursor: false,
        snapClipEdges: false,
        snapClipsToSourceMedia: false,
        snapClipSnapOffset: true,
        snapDistancePx: 40,
        clips: [offsetClip],
    });
    assertNear(
        snapTimelinePosition(ctxOff, 102.55).sec,
        102.5,
        "explicit snap offset is a target",
    );
    const ctxZero = context({
        snapClipsToGrid: false,
        snapClipsToSelectionMarkersCursor: false,
        snapClipEdges: false,
        snapClipsToSourceMedia: false,
        snapClipSnapOffset: true,
        snapDistancePx: 40,
        clips: [{ ...offsetClip, snapOffsetSec: 0 }],
    });
    assertNear(
        snapTimelinePosition(ctxZero, 100.05).sec,
        100.05,
        "zero snap offset generates no candidate",
    );
}

// 双缘吸附（拖拽移动）：前缘与后缘同时参与匹配，取更近者。
{
    const others: ClipInfo[] = [
        { ...clips[0], id: "grid-a", startSec: 10, lengthSec: 2 },
        { ...clips[0], id: "far", startSec: 50, lengthSec: 2 },
    ];
    const base = {
        settings: {
            ...baseSettings,
            snapClipsToGrid: true,
            snapClipsToSelectionMarkersCursor: false,
            snapClipEdges: true,
            snapClipSnapOffset: false,
            snapClipsToSourceMedia: false,
            snapDistancePx: 40,
        },
        clips: others,
    };
    // 后缘更近：rawStart=12.75，len=1.2 → 后缘 13.95 距网格 14 差 0.05s，
    // 前缘距网格 13 差 0.25s → 后缘胜出，起点吸附到 14−1.2=12.8。
    const ctxEnd: TimelineSnapContext = { ...context(base), pxPerSec: 40 };
    const endWin = snapTimelineClipMove(ctxEnd, 12.75, 1.2);
    assertTrue(endWin.snapped && endWin.edgeSide === "end", "end edge wins when closer");
    assertNear(endWin.sec, 12.8, "end-aligned move start");
    // 前缘更近：rawStart=10.1，len=1.2 → 前缘差 0.1s（4px）；
    // 后缘 11.3 距网格 11.5 差 0.2s（8px）→ 前缘胜出。
    const startWin = snapTimelineClipMove(ctxEnd, 10.1, 1.2);
    assertTrue(startWin.snapped && startWin.edgeSide === "start", "start edge wins when closer");
    assertNear(startWin.sec, 10, "start-aligned move start");
    // 无任何候选族（全部关闭）：不吸附，返回原始起点。
    const noSnapCtx: TimelineSnapContext = {
        ...context({
            settings: {
                ...baseSettings,
                snapClipsToGrid: false,
                snapClipsToSelectionMarkersCursor: false,
                snapClipEdges: false,
                snapClipSnapOffset: false,
                snapClipsToSourceMedia: false,
                snapDistancePx: 40,
            },
            clips: others,
        }),
        pxPerSec: 40,
    };
    const noSnap = snapTimelineClipMove(noSnapCtx, 30.5, 1);
    assertTrue(!noSnap.snapped, "no snap without candidates");
    assertNear(noSnap.sec, 30.5, "unsnapped move keeps raw start");

    // ── 自身 SnapOffset 作为第三吸附源 ──
    // 网格间距 0.5s。len=0.7、offset=0.42、rawStart=20.15：
    //   前缘 20.15 距 20 差 0.15s(6px)；后缘 20.85 距 21 差 0.15s(6px)；
    //   偏移点 20.57 距 20.5 差 0.07s(2.8px) → 偏移点唯一最近 ✓
    // 命中后新起点 = 20.5 − 0.42 = 20.08。
    const offWin = snapTimelineClipMove(ctxEnd, 20.15, 0.7, 0.42);
    assertTrue(
        offWin.snapped && offWin.edgeSide === "snap_offset",
        "snap offset wins as its own source",
    );
    assertNear(offWin.sec, 20.08, "offset-aligned move start");
    // offset=0 时退化为双缘行为（不生成重复偏移候选）。
    const zeroOffset = snapTimelineClipMove(ctxEnd, 10.1, 1.2, 0);
    assertTrue(zeroOffset.edgeSide === "start", "zero offset keeps dual-edge behavior");
}

console.log(`timelineSnapping checks passed (${checks})`);
