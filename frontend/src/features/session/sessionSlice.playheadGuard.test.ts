import { test } from "vitest";

import reducer from "./sessionSlice.ts";
import { moveClipRemote } from "./thunks/timelineThunks.ts";
import { stopAudioPlayback } from "./thunks/transportThunks.ts";

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

/**
 * 暂停位置对齐回归：前端轮询存在至多一个周期（~33ms）+ 往返的滞后，暂停
 * 时后端 stop_audio 记录的精确停止位置（stopped_at_sec）领先于最后一次
 * 采样的视觉位置。暂停必须把视觉光标对齐到该精确位置——否则视觉位置与
 * 后端记录的暂停点不一致，后续任何编辑回灌快照都会让光标再次右跳。
 */
test("features/session/sessionSlice.playheadGuard.test.ts pause aligns the playhead to the exact stop position", async () => {
    function assertEqual(actual: unknown, expected: unknown, label: string): void {
        if (actual !== expected) {
            throw new Error(`${label}: expected ${String(expected)}, received ${String(actual)}`);
        }
    }

    function initState(playheadSec: number) {
        const base = reducer(undefined, { type: "@@INIT" }) as any;
        return { ...base, playheadSec };
    }

    // 暂停（无锚点恢复）：视觉光标对齐到引擎的精确停止位置。
    {
        const next = reducer(
            initState(42.3),
            stopAudioPlayback.fulfilled(
                { ok: true, stopped_at_sec: 42.5, restoreAnchor: false, wasPlaying: true, anchorSec: 40 },
                "req",
                undefined,
            ),
        );
        assertEqual(next.playheadSec, 42.5, "pause adopts the exact stop position");
    }

    // 停止（恢复锚点）：锚点优先，不采用停止位置。
    {
        const next = reducer(
            initState(42.3),
            stopAudioPlayback.fulfilled(
                {
                    ok: true,
                    stopped_at_sec: 42.5,
                    restoreAnchor: true,
                    wasPlaying: true,
                    anchorSec: 10,
                },
                "req",
                { restoreAnchor: true },
            ),
        );
        assertEqual(next.playheadSec, 10, "stop restores the anchor position");
    }

    // 引擎本就未在播放（如录音收尾的 stop）：无停止位置，光标原地不动。
    {
        const next = reducer(
            initState(42.3),
            stopAudioPlayback.fulfilled(
                { ok: true, stopped_at_sec: null, restoreAnchor: false, wasPlaying: false, anchorSec: 0 },
                "req",
                undefined,
            ),
        );
        assertEqual(next.playheadSec, 42.3, "idle stop leaves the playhead untouched");
    }
});
