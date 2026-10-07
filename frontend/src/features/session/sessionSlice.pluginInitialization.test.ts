// 插件初始化必须等待真实宿主轨道，不能先向ARA请求独立app的默认Main参数。
// @vitest-environment jsdom
import { afterEach, expect, test } from "vitest";
import reducer, { fetchTimeline } from "./sessionSlice";
import type { TimelineResult } from "../../types/api";

afterEach(() => { delete window.__HFS_PLUGIN_BOOTSTRAP__; });

test("plugin starts without a fabricated selected track before host snapshot", () => {
    window.__HFS_PLUGIN_BOOTSTRAP__ = { version: 1, viewId: "new-copied-instance" };
    const state = reducer(undefined, { type: "@@INIT" });
    expect(state.tracks).toEqual([]);
    expect(state.selectedTrackId).toBeNull();
    expect(state.clips).toEqual([]);
});

test("standalone still starts with its original editable Main track", () => {
    const state = reducer(undefined, { type: "@@INIT" });
    expect(state.tracks.map(track => track.id)).toEqual(["track_main"]);
    expect(state.selectedTrackId).toBe("track_main");
});

test("real host snapshot selects the assigned track after empty plugin startup", () => {
    window.__HFS_PLUGIN_BOOTSTRAP__ = { version: 1, viewId: "assigned-instance" };
    const initial = reducer(undefined, { type: "@@INIT" });
    const host: TimelineResult = {
        ok: true,
        tracks: [{ id: "hfs-ui-123-2-ara-track-2", name: "REAPER Track 2", volume: 1,
            muted: false, solo: false, compose_enabled: true, pitch_analysis_algo: "world", color: "#74787e" }],
        clips: [], selected_track_id: "hfs-ui-123-2-ara-track-2", selected_clip_id: null,
        playhead_sec: 1, project_sec: 5, bpm: 180, disabled_group_ids: [],
    };
    const state = reducer(initial, fetchTimeline.fulfilled(host, "real-host", undefined));
    expect(state.tracks.map(track => track.id)).toEqual(["hfs-ui-123-2-ara-track-2"]);
    expect(state.selectedTrackId).toBe("hfs-ui-123-2-ara-track-2");
    expect(state.bpm).toBe(180);
});
