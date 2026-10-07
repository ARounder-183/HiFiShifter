/**
 * ★ 回归：吸附候选的**范围过滤**（`candidateRangeSec`）。
 *
 * ## 缺陷形态
 *
 * 被吸附对象可能有**活动范围**：Clip 的吸附偏移手柄必须留在 Clip 内部
 * （`[clipStart, clipStart + clipLen]`）。不过滤候选时，最近的候选可能落在 Clip
 * 之外 —— 吸附把"落点"算到外面，调用方再把偏移钳回边界，于是**高亮线画在 Clip
 * 外、手柄却停在边界**，用户看到的是"范围外的位置也产生了吸附"。
 *
 * ## 本用例钉住什么
 *
 * 同一组候选与同一个查询位置：不传范围时吸附到范围外那个更近的候选；传入范围后
 * 它被排除，落到范围内的次近候选上。两个方向都断言，避免"过滤把一切都滤掉"也能
 * 通过。
 */
import { describe, expect, it } from "vitest";

import type { ClipInfo, TrackInfo } from "../features/session/sessionTypes";
import { createDefaultTimelineSnapSettings } from "../features/session/sessionSlice";
import { snapTimelinePosition, type TimelineSnapContext } from "./timelineSnapping";

const track: TrackInfo = {
    id: "t0",
    name: "Track",
    muted: false,
    solo: false,
    volume: 1,
    composeEnabled: false,
    pitchAnalysisAlgo: "nsf_hifigan_onnx",
};

function clip(id: string, startSec: number, lengthSec: number): ClipInfo {
    return {
        id,
        trackId: "t0",
        name: id,
        startSec,
        lengthSec,
        color: "blue",
        sourceStartSec: 0,
        sourceEndSec: lengthSec,
        playbackRate: 1,
        reversed: false,
        loopEnabled: false,
        channelMode: 0,
        snapOffsetSec: 0,
        fadeInSec: 0,
        fadeOutSec: 0,
        gain: 1,
        muted: false,
        fadeInShape: 0,
        fadeInDir: 0,
        fadeOutShape: 0,
        fadeOutDir: 0,
    };
}

/** 阈值 = snapDistancePx / pxPerSec = 10 / 100 = 0.1s。 */
function context(): TimelineSnapContext {
    return {
        settings: {
            ...createDefaultTimelineSnapSettings(),
            enabled: true,
            snapDistancePx: 10,
            snapClipsToGrid: false,
            snapClipEdges: true,
            snapClipsToSelectionMarkersCursor: true,
            snapAcrossTracks: false,
        },
        grid: "1/4",
        bpm: 120,
        beatsPerBar: 4,
        tempoMap: null,
        pxPerSec: 100,
        // 范围外候选：clip 起点 12.02（比范围内的 11.95 更近）。
        clips: [clip("other", 12.02, 1)],
        tracks: [track],
        selectedClipIds: [],
        // 范围内候选：播放头 11.95。
        playheadSec: 11.95,
        object: "clip",
        anchorTrackId: "t0",
    };
}

describe("★ 吸附候选范围过滤（candidateRangeSec）", () => {
    it("不传范围 ⇒ 吸附到**范围外**那个更近的候选（对照，证明范围确实起作用）", () => {
        const result = snapTimelinePosition(context(), 12.0);
        expect(result.snapped).toBe(true);
        expect(result.sec).toBeCloseTo(12.02, 9);
    });

    it("传入范围 ⇒ 范围外候选被排除，落到范围内的次近候选", () => {
        const result = snapTimelinePosition(
            { ...context(), candidateRangeSec: { lo: 10, hi: 12 } },
            12.0,
        );
        expect(result.snapped).toBe(true);
        expect(result.sec).toBeCloseTo(11.95, 9);
        // 命中点必须落在范围内 —— 否则"命中即落点"的前提不成立。
        expect(result.sec).toBeGreaterThanOrEqual(10);
        expect(result.sec).toBeLessThanOrEqual(12);
    });

    it("范围把两侧候选都排除 ⇒ 不吸附（退回原位置）", () => {
        const result = snapTimelinePosition(
            { ...context(), candidateRangeSec: { lo: 12.5, hi: 12.9 } },
            12.0,
        );
        expect(result.snapped).toBe(false);
        expect(result.sec).toBeCloseTo(12.0, 9);
        expect(result.candidate).toBeNull();
    });

    it("范围顺序颠倒（hi < lo）等价", () => {
        const result = snapTimelinePosition(
            { ...context(), candidateRangeSec: { lo: 12, hi: 10 } },
            12.0,
        );
        expect(result.sec).toBeCloseTo(11.95, 9);
    });

    it("范围内的候选仍照常吸附（过滤不能变成「永不吸附」）", () => {
        const result = snapTimelinePosition(
            { ...context(), candidateRangeSec: { lo: 11.9, hi: 12.1 } },
            11.98,
        );
        expect(result.snapped).toBe(true);
        expect(result.sec).toBeCloseTo(11.95, 9);
    });
});
