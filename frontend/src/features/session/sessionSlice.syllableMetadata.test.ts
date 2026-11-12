// @vitest-environment jsdom
import { afterEach, expect, test } from "vitest";
import reducer, { applyClipSyllableAnalysis, fetchTimeline } from "./sessionSlice";
import type { TimelineResult } from "../../types/api";

function timeline(): TimelineResult {
    return {
        ok: true,
        tracks: [
            {
                id: "root",
                name: "Root",
                volume: 1,
                compose_enabled: true,
                muted: false,
                solo: false,
                pitch_analysis_algo: "world",
                color: "#777777",
            },
        ],
        clips: ["a", "b"].map((id) => ({
            id,
            track_id: "root",
            name: id,
            start_sec: 0,
            length_sec: 1,
            color: "#777777",
            active_take_id: `take-${id}`,
            source_path: `${id}.wav`,
            takes: [
                {
                    id: `take-${id}`,
                    name: id,
                    reversed: false,
                    loop_enabled: false,
                    source_path: `${id}.wav`,
                    source_start_sec: 0,
                    source_end_sec: 1,
                    playback_rate: 1,
                    gain: 1,
                },
            ],
        })),
        selected_track_id: "root",
        selected_clip_id: "a",
        project_sec: 1,
        bpm: 120,
        playhead_sec: 0,
        disabled_group_ids: [],
    } as TimelineResult;
}
function initial() {
    window.__HFS_PLUGIN_BOOTSTRAP__ = { version: 1, viewId: "metadata" };
    return reducer(undefined, fetchTimeline.fulfilled(timeline(), "init", undefined));
}
const result: Parameters<typeof applyClipSyllableAnalysis>[0] = {
    clipId: "a",
    takeId: "take-a",
    sourcePath: "a.wav",
    analyzed: true,
    sourceSpansSec: [
        { startSec: 0, endSec: 0.4 },
        { startSec: 0.4, endSec: 1 },
    ],
    sourceNotes: [60, 62],
    analyzedWindowSec: [0, 1],
};
afterEach(() => {
    delete window.__HFS_PLUGIN_BOOTSTRAP__;
});

test("GAME updates only its Take and block metadata without invalidating curves", () => {
    const before = initial();
    const after = reducer(before, applyClipSyllableAnalysis(result));
    expect(after.clips[0].takes?.[0].syllableSpansSec).toEqual([
        { startSec: 0, endSec: 0.4, note: 60 },
        { startSec: 0.4, endSec: 1, note: 62 },
    ]);
    expect(after.clips[0].syllableSpansSec).toEqual([
        { startSec: 0, endSec: 0.4, note: 60 },
        { startSec: 0.4, endSec: 1, note: 62 },
    ]);
    expect(after.paramsEpoch).toBe(before.paramsEpoch);
    expect(after.clipPitchCurves).toBe(before.clipPitchCurves);
    expect(after.tracks).toBe(before.tracks);
    expect(after.clips[1]).toBe(before.clips[1]);
    expect(reducer(after, applyClipSyllableAnalysis(result))).toBe(after);
});

test("a timeline response issued before GAME completion cannot remove its blocks", () => {
    let state = initial();
    state = reducer(state, fetchTimeline.pending("old", undefined));
    state = reducer(state, applyClipSyllableAnalysis(result));
    state = reducer(state, fetchTimeline.fulfilled(timeline(), "old", undefined));
    expect(state.clips[0].syllableSpansSec).toEqual([
        { startSec: 0, endSec: 0.4, note: 60 },
        { startSec: 0.4, endSec: 1, note: 62 },
    ]);
});

test("late, failed, or invalid GAME results cannot overwrite a different source", () => {
    const state = reducer(initial(), applyClipSyllableAnalysis(result));
    for (const changed of [
        { ...result, sourcePath: "changed.wav" },
        { ...result, takeId: "removed" },
        { ...result, analyzed: false },
        { ...result, sourceSpansSec: [] },
        { ...result, sourceSpansSec: [{ startSec: 1, endSec: 0 }] },
        { ...result, sourceSpansSec: [{ startSec: -0.1, endSec: 0.1 }] },
        { ...result, sourceSpansSec: [{ startSec: 0, endSec: 0.4 }], sourceNotes: [] },
        { ...result, sourceNotes: [60, Number.NaN] },
    ])
        expect(reducer(state, applyClipSyllableAnalysis(changed))).toBe(state);
});

test("GAME commits the analyzed window and parallel source notes", () => {
    const state = reducer(initial(), applyClipSyllableAnalysis(result));
    expect(state.clips[0].takes?.[0].syllableAnalysisWindowSec).toEqual([0, 1]);
    expect(state.clips[0].takes?.[0].syllableNotes).toEqual([60, 62]);
    expect(state.clips[0].syllableSpansSec).toEqual([
        { startSec: 0, endSec: 0.4, note: 60 },
        { startSec: 0.4, endSec: 1, note: 62 },
    ]);
});

test("a malformed analyzed window is rejected", () => {
    const state = initial();
    expect(
        reducer(
            state,
            applyClipSyllableAnalysis({
                ...result,
                analyzedWindowSec: [1, 0],
            }),
        ),
    ).toBe(state);
});
