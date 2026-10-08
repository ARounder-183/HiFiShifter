// 宿主音频读数进 Redux 的门禁：只接受白名单分类，未知/缺失沿用上一次已知值。
// @vitest-environment jsdom
import { expect, test } from "vitest";

import reducer, { fetchTimeline } from "./sessionSlice";
import type { TimelineResult } from "../../types/api";

/** 只造真实字段形状的最小快照，不把 fake 当 REAPER 验收。 */
function timeline(extra: Record<string, unknown> = {}): TimelineResult {
    return {
        ok: true,
        tracks: [],
        clips: [],
        selected_track_id: null,
        selected_clip_id: null,
        playhead_sec: 0,
        project_sec: 0,
        bpm: 120,
        disabled_group_ids: [],
        ...extra,
    } as TimelineResult;
}

function loaded(extra: Record<string, unknown> = {}) {
    window.__HFS_PLUGIN_BOOTSTRAP__ = { version: 1, viewId: "host-audio" };
    return reducer(undefined, fetchTimeline.fulfilled(timeline(extra), "req", undefined));
}

test("the folder-parent reading reaches the session state", () => {
    const state = loaded({
        host_audio: { state: "folder_parent_without_regions", waiting_clips: 2 },
    });
    expect(state.hostAudio).toEqual({
        state: "folder_parent_without_regions",
        waiting_clips: 2,
    });
});

test("a snapshot without the field keeps the last known reading", () => {
    const first = loaded({ host_audio: { state: "awaiting_regions", waiting_clips: 1 } });
    expect(first.hostAudio).toEqual({ state: "awaiting_regions", waiting_clips: 1 });
    // 普通编辑类响应不带该字段（它只由时间线快照装饰），读数不能被清空。
    const second = reducer(first, fetchTimeline.fulfilled(timeline(), "req2", undefined));
    expect(second.hostAudio).toEqual({ state: "awaiting_regions", waiting_clips: 1 });
});

/*
 * 【为什么必须拒绝未知分类】后端将来加了新分类名时，旧前端若把它当成已知状态渲染，
 * 就会显示一条与真实原因不符的指引（用户照着改工程，问题更糟）。未知分类与"没带该
 * 字段"同义：沿用上一次已知值。
 */
test("an unknown classification never overwrites a known reading", () => {
    const first = loaded({ host_audio: { state: "ready", waiting_clips: 0 } });
    const second = reducer(
        first,
        fetchTimeline.fulfilled(
            timeline({ host_audio: { state: "some_future_state", waiting_clips: 9 } }),
            "req2",
            undefined,
        ),
    );
    expect(second.hostAudio).toEqual({ state: "ready", waiting_clips: 0 });
});

test("the standalone app never grows a host reading", () => {
    const state = reducer(undefined, fetchTimeline.fulfilled(timeline(), "req", undefined));
    expect(state.hostAudio).toBeNull();
});
