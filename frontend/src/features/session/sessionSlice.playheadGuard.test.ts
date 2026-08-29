import { test } from "vitest";

/* eslint-disable @typescript-eslint/no-explicit-any -- 测试夹具：
   reducer 初始化 action 与全量时间线载荷无法在不使用 any 的情况下构造 */

import reducer from "./sessionSlice.ts";
import { moveClipRemote } from "./thunks/timelineThunks.ts";
import { stopAudioPlayback, syncPlaybackState } from "./thunks/transportThunks.ts";

/**
 * 播放头所有权回归测试：
 *
 * 播放头只归传输层（30Hz 轮询 / seek / stop_audio / 显式跳转）所有。编辑
 * 命令返回的全量快照携带的 playhead_sec 是编辑未触及的旧值（播放期间停留
 * 在本次播放的起始位置），applyTimelineState 不得采纳它——否则播放中编辑
 * 或引擎瞬态未播放（等待重渲染自动暂停）时，光标会被拉回任意旧位置。
 */
test("features/session/sessionSlice.playheadGuard.test.ts applyTimelineState never adopts the snapshot playhead", async () => {
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

    // 暂停中：编辑回灌不改写光标（编辑不拥有播放头）。
    {
        const next = reducer(
            initState({ playheadSec: 50, isPlaying: false }),
            moveClipRemote.fulfilled(timelinePayload, "req", {
                clipId: "clip-a",
                startSec: 2,
                moveLinkedParams: true,
            }),
        );
        assertEqual(next.playheadSec, 50, "paused state keeps the playhead on edits");
    }

    // 播放中：光标由轮询驱动，同样不被编辑快照改写。
    {
        const next = reducer(
            initState({ playheadSec: 50, isPlaying: true }),
            moveClipRemote.fulfilled(timelinePayload, "req", {
                clipId: "clip-a",
                startSec: 2,
                moveLinkedParams: true,
            }),
        );
        assertEqual(next.playheadSec, 50, "playing state keeps the polled playhead");
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
                {
                    ok: true,
                    stopped_at_sec: 42.5,
                    restoreAnchor: false,
                    wasPlaying: true,
                    anchorSec: 40,
                },
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
                {
                    ok: true,
                    stopped_at_sec: null,
                    restoreAnchor: false,
                    wasPlaying: false,
                    anchorSec: 0,
                },
                "req",
                undefined,
            ),
        );
        assertEqual(next.playheadSec, 42.3, "idle stop leaves the playhead untouched");
    }
});

/**
 * 播放→停止跃迁对齐：引擎自然结束或等待重渲染自动暂停时，position 冻结在
 * 真实停止点，而本地光标停留在最后一次轮询采样（略落后）。轮询 reducer
 * 必须在跃迁时把光标对齐到引擎的冻结位置；之后（已停止）不再改写。
 */
test("features/session/sessionSlice.playheadGuard.test.ts sync aligns the playhead on the playing-stopped transition", async () => {
    function assertEqual(actual: unknown, expected: unknown, label: string): void {
        if (actual !== expected) {
            throw new Error(`${label}: expected ${String(expected)}, received ${String(actual)}`);
        }
    }

    function playingState(playheadSec: number) {
        const base = reducer(undefined, { type: "@@INIT" }) as any;
        return {
            ...base,
            playheadSec,
            runtime: {
                ...base.runtime,
                isPlaying: true,
                playbackPositionSec: playheadSec,
            },
        };
    }

    const syncPayload = (isPlaying: boolean, positionSec: number) =>
        ({
            ok: true,
            is_playing: isPlaying,
            target: "original",
            base_sec: 0,
            position_sec: positionSec,
            duration_sec: 200,
        }) as any;

    // 播放中：轮询推进光标。
    {
        const next = reducer(playingState(50), syncPlaybackState.fulfilled(
            syncPayload(true, 50.02),
            "req",
            undefined,
        ));
        assertEqual(next.playheadSec, 50.02, "playing poll advances the playhead");
    }

    // 播放→停止跃迁：光标对齐到引擎冻结的精确停止位置。
    {
        const next = reducer(playingState(50.02), syncPlaybackState.fulfilled(
            syncPayload(false, 50.09),
            "req",
            undefined,
        ));
        assertEqual(
            next.playheadSec,
            50.09,
            "transition aligns the playhead with the engine's stop position",
        );
    }

    // 已停止后的后续轮询：不再改写光标（例如 handle_stop 后 position 归零）。
    {
        const stopped = reducer(playingState(50.02), syncPlaybackState.fulfilled(
            syncPayload(false, 50.09),
            "req",
            undefined,
        ));
        const next = reducer(stopped, syncPlaybackState.fulfilled(
            syncPayload(false, 0),
            "req",
            undefined,
        ));
        assertEqual(
            next.playheadSec,
            50.09,
            "polls after the stop transition never move the playhead",
        );
    }
});
