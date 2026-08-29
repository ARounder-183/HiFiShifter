import { test } from "vitest";

import reducer from "./sessionSlice.ts";
import { moveClipRemote } from "./thunks/timelineThunks.ts";

/**
 * "暂停后光标原地"回归测试：
 *
 * 后端快照的 playhead_sec 只在显式 seek/transport 时同步——播放期间它停留
 * 在本次播放的起始位置（暂停后由后端 stop_audio 回写为暂停点）。因此：
 * - 暂停中（runtime.isPlaying=false）：编辑回灌快照应采纳后端 playhead_sec
 *   （配合 stop_audio 的回写即为准确值）；
 * - 播放中（runtime.isPlaying=true）：光标由轮询驱动，必须保留本地值，
 *   否则任何编辑都会把播放头瞬时拉回播放起始位置。
 */
test("features/session/sessionSlice.playheadGuard.test.ts applyTimelineState playhead adoption", async () => {
    function assertEqual(actual: unknown, expected: unknown, label: string): void {
        if (actual !== expected) {
            throw new Error(`${label}: expected ${String(expected)}, received ${String(actual)}`);
        }
    }

    const timelinePayload = {
        ok: true,
        tracks: [],
        clips: [],
        selected_track_id: null,
        selected_clip_id: null,
        playhead_sec: 7,
        project_sec: 32,
        bpm: 120,
        disabled_group_ids: [],
    } as any;

    function initState(overrides: { playheadSec: number; isPlaying: boolean }) {
        const base = reducer(undefined, { type: "@@INIT" }) as any;
        return {
            ...base,
            playheadSec: overrides.playheadSec,
            runtime: { ...base.runtime, isPlaying: overrides.isPlaying },
        };
    }

    // 暂停中：采纳后端回写的暂停点。
    {
        const next = reducer(
            initState({ playheadSec: 50, isPlaying: false }),
            moveClipRemote.fulfilled(timelinePayload, "req", {
                clipId: "clip-a",
                startSec: 2,
                moveLinkedParams: true,
            }),
        );
        assertEqual(next.playheadSec, 7, "paused state adopts backend playhead_sec");
    }

    // 播放中：保留轮询驱动的本地光标，不被陈旧的起始位置拉回。
    {
        const next = reducer(
            initState({ playheadSec: 50, isPlaying: true }),
            moveClipRemote.fulfilled(timelinePayload, "req", {
                clipId: "clip-a",
                startSec: 2,
                moveLinkedParams: true,
            }),
        );
        assertEqual(next.playheadSec, 50, "playing state preserves polled playhead");
    }
});
