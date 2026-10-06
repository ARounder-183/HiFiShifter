// 插件几何回流竞态：旧自动刷新不得把B闪回A，失败/外部编辑仍能重新同步。
// @vitest-environment jsdom
import { configureStore } from "@reduxjs/toolkit";
import { afterEach, expect, test, vi } from "vitest";
import reducer, { beginInteraction, endInteraction, fetchTimeline } from "./sessionSlice";
import { moveClipRemote, setClipsStateBulkRemote } from "./thunks/timelineThunks";
import { webApi } from "../../services/webviewApi";
import { seekPlayhead, syncPlaybackState } from "./thunks/transportThunks";
import { importAudioFileAtPosition } from "./thunks/importThunks";
import type { TimelineResult } from "../../types/api";

const move = { clipId: "host-clip", startSec: 5, trackId: "host-track", moveLinkedParams: true };
/** 只造真实字段形状的最小快照，不把fake当REAPER验收。 */
function timeline(start: number): TimelineResult {
    return { ok: true, tracks: [{ id: "host-track", name: "Track", volume: 1,
        muted: false, solo: false, compose_enabled: true, pitch_analysis_algo: "world", color: "#74787e" }],
        clips: [{ id: "host-clip", track_id: "host-track", name: "clip", color: "#74787e", start_sec: start, length_sec: 1,
            source_start_sec: 0, source_end_sec: 1, playback_rate: 1, gain: 1, muted: false }],
        selected_track_id: "host-track", selected_clip_id: "host-clip", playhead_sec: 0,
        project_sec: 10, bpm: 120, disabled_group_ids: [] } as TimelineResult;
}
function initial() {
    window.__HFS_PLUGIN_BOOTSTRAP__ = { version: 1, viewId: "race", clipEditing: true };
    return reducer(undefined, fetchTimeline.fulfilled(timeline(0), "initial", undefined));
}
afterEach(() => { delete window.__HFS_PLUGIN_BOOTSTRAP__; vi.restoreAllMocks(); });

test("plugin seek ignores old-position polls during request and old polls after acknowledgement", () => {
    let state = { ...initial(), playheadSec: 5 };
    state = reducer(state, seekPlayhead.pending("seek-b", 5));
    const epoch = state._transportEpoch;
    const oldPoll = { ok: true as const, is_playing: false, target: null, base_sec: 0, position_sec: 1, playhead_sec: 1, duration_sec: 10 };
    state = reducer(state, syncPlaybackState.fulfilled(oldPoll, "poll", { epoch, dispatchedAtMs: 0 }));
    expect(state.playheadSec).toBe(5);
    state = reducer(state, seekPlayhead.fulfilled({ok:true,playhead_sec:1}, "seek-b", 5));
    expect(state.playheadSec).toBe(5);
    state = reducer(state, syncPlaybackState.fulfilled(oldPoll, "poll-late", { epoch, dispatchedAtMs: 0 }));
    expect(state.playheadSec).toBe(5);
    expect(state._pluginSeekRequestId).toBeNull();
});

test("import acknowledgement cannot be removed by a read started before import", () => {
    let state = initial();state = reducer(state, fetchTimeline.pending("before-import", undefined));
    const args={file:new File(["fixture"],"voice.wav"),trackId:"host-track",startSec:5};
    state = reducer(state, importAudioFileAtPosition.pending("import", args));
    const imported=timeline(0);imported.clips.push({...imported.clips[0],id:"imported",start_sec:5});
    state = reducer(state, importAudioFileAtPosition.fulfilled({ok:true,imported,newClipIds:["imported"],playheadSec:undefined},"import",args));
    state = reducer(state, fetchTimeline.fulfilled(timeline(0),"before-import",undefined));
    expect(state.clips.some(clip=>clip.id==="imported")).toBe(true);
});

test("old refresh is discarded during drag and after the B commit releases its lock", () => {
    let state = initial();
    state = reducer(state, fetchTimeline.pending("old", undefined));
    state = reducer(state, beginInteraction());
    state = reducer(state, moveClipRemote.pending("move", move));
    state = reducer(state, moveClipRemote.fulfilled(timeline(5), "move", move));
    state = reducer(state, fetchTimeline.fulfilled(timeline(0), "old", undefined));
    expect(state.clips[0].startSec).toBe(5);
    state = reducer(state, fetchTimeline.pending("inside-drag", undefined));
    state = reducer(state, endInteraction());
    state = reducer(state, fetchTimeline.fulfilled(timeline(0), "inside-drag", undefined));
    expect(state.clips[0].startSec).toBe(5);
    state = reducer(state, fetchTimeline.pending("external", undefined));
    state = reducer(state, fetchTimeline.fulfilled(timeline(7), "external", undefined));
    expect(state.clips[0].startSec).toBe(7);
    expect(state._pluginTimelineFetchEpochs).toEqual({});
});

test("width edits without a gesture lock defer refresh; rejection releases the guard", () => {
    let state = initial();
    const arg = { updates: [{ clipId: "host-clip", fadeInSec: 0.3 }] };
    state = reducer(state, setClipsStateBulkRemote.pending("width", arg));
    state = reducer(state, fetchTimeline.pending("old", undefined));
    state = reducer(state, fetchTimeline.fulfilled(timeline(0), "old", undefined));
    expect(state.clips[0].fadeInSec).toBe(0.3);
    state = reducer(state, setClipsStateBulkRemote.rejected(new Error("host changed"), "width", arg));
    expect(state._pluginClipEditRequests).toEqual({});
    state = reducer(state, fetchTimeline.pending("recover", undefined));
    state = reducer(state, fetchTimeline.fulfilled(timeline(2), "recover", undefined));
    expect(state.clips[0].startSec).toBe(2);
});

test("async stale refresh rejects so the apply panel does not consume the host version", async () => {
    const store = configureStore({ reducer: { session: reducer }, preloadedState: { session: initial() } });
    let reply!: (value: TimelineResult) => void;
    vi.spyOn(webApi, "getTimelineState").mockImplementationOnce(() => new Promise(resolve => { reply = resolve; }));
    const refresh = store.dispatch(fetchTimeline());
    store.dispatch(moveClipRemote.pending("move", move));
    store.dispatch(moveClipRemote.fulfilled(timeline(5), "move", move));
    reply(timeline(0));
    await expect(refresh.unwrap()).rejects.toContain("Stale host timeline");
    expect(store.getState().session.clips[0].startSec).toBe(5);
    expect(store.getState().session._pluginTimelineFetchEpochs).toEqual({});
});

test("standalone fetch retains its original authoritative force semantics", () => {
    let state = reducer(undefined, fetchTimeline.fulfilled(timeline(0), "init", undefined));
    state = reducer(state, beginInteraction());
    state = reducer(state, fetchTimeline.pending("fetch", undefined));
    state = reducer(state, fetchTimeline.fulfilled(timeline(3), "fetch", undefined));
    expect(state.clips[0].startSec).toBe(3);
});
