import { test } from "vitest";

/* eslint-disable @typescript-eslint/no-explicit-any -- 测试夹具：
   reducer 初始化 action 与全量时间线载荷无法在不使用 any 的情况下构造 */

import reducer from "./sessionSlice.ts";
import { moveClipRemote } from "./thunks/timelineThunks.ts";
import { undoRemote, redoRemote } from "./thunks/projectThunks.ts";
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
        const next = reducer(
            playingState(50),
            syncPlaybackState.fulfilled(syncPayload(true, 50.02), "req", undefined),
        );
        assertEqual(next.playheadSec, 50.02, "playing poll advances the playhead");
    }

    // 播放→停止跃迁：光标对齐到引擎冻结的精确停止位置。
    {
        const next = reducer(
            playingState(50.02),
            syncPlaybackState.fulfilled(syncPayload(false, 50.09), "req", undefined),
        );
        assertEqual(
            next.playheadSec,
            50.09,
            "transition aligns the playhead with the engine's stop position",
        );
    }

    // 已停止后的后续轮询：不再改写光标（例如 handle_stop 后 position 归零）。
    {
        const stopped = reducer(
            playingState(50.02),
            syncPlaybackState.fulfilled(syncPayload(false, 50.09), "req", undefined),
        );
        const next = reducer(
            stopped,
            syncPlaybackState.fulfilled(syncPayload(false, 0), "req", undefined),
        );
        assertEqual(
            next.playheadSec,
            50.09,
            "polls after the stop transition never move the playhead",
        );
    }
});

/**
 * 撤销/重做播放光标归位回归：
 *
 * 后端 undo/redo 把时间线（含 playhead_sec）整体回退/恢复到检查点快照 ——
 * 快照里的 playhead_sec 就是回退后后端的实际光标位置，也是后续一切以光标
 * 为锚点的编辑操作（粘贴/分割等）的实际操作点。前端必须采纳它让视觉光标
 * 同步归位：若沿用撤销前的本地光标（可能是上一步操作挪过去的位置，如粘贴
 * 的 pasteEndSec），视觉停在 B、后端实际停在 A，下一次操作就会落在视觉之
 * 外的位置（视觉编辑点 ≠ 实际编辑点）。
 *
 * 这是"编辑快照不拥有播放头"原则的例外：编辑命令的 playhead_sec 是未触及
 * 的旧值，而撤销/重做的 playhead_sec 是后端权威状态的组成部分。
 */
test("features/session/sessionSlice.playheadGuard.test.ts undo/redo adopt the checkpoint playhead", async () => {
    function assertEqual(actual: unknown, expected: unknown, label: string): void {
        if (actual !== expected) {
            throw new Error(`${label}: expected ${String(expected)}, received ${String(actual)}`);
        }
    }

    function timelinePayload(playheadSec: number) {
        return {
            ok: true,
            tracks: [],
            clips: [],
            selected_track_id: null,
            selected_clip_id: null,
            playhead_sec: playheadSec,
            project_sec: 32,
            bpm: 120,
            disabled_group_ids: [],
        } as any;
    }

    function initState(overrides: { playheadSec: number; isPlaying: boolean }) {
        const base = reducer(undefined, { type: "@@INIT" }) as any;
        return {
            ...base,
            playheadSec: overrides.playheadSec,
            runtime: { ...base.runtime, isPlaying: overrides.isPlaying },
        };
    }

    // 暂停中撤销 B→A：视觉光标跟随回退后的快照光标（9 → 5），并登记
    // "聚焦播放光标"，离屏时由 TimelinePanel 滚动到可见。
    {
        const pended = reducer(initState({ playheadSec: 9, isPlaying: false }), undoRemote.pending("req-undo", undefined));
        const next = reducer(
            pended,
            undoRemote.fulfilled(timelinePayload(5), "req-undo", undefined),
        );
        assertEqual(next.playheadSec, 5, "paused undo adopts the checkpoint playhead");
        assertEqual(
            next.pendingPlayheadRevealSec,
            5,
            "moved playhead registers a reveal request",
        );
    }

    // 重做 A→B：对称地跟随恢复快照的光标（5 → 9）。
    {
        const pended = reducer(initState({ playheadSec: 5, isPlaying: false }), redoRemote.pending("req-redo", undefined));
        const next = reducer(
            pended,
            redoRemote.fulfilled(timelinePayload(9), "req-redo", undefined),
        );
        assertEqual(next.playheadSec, 9, "redo adopts the restored checkpoint playhead");
        assertEqual(next.pendingPlayheadRevealSec, 9, "redo registers a reveal request");
    }

    // 光标未挪动（该状态形成后光标未变）：不登记聚焦请求，无谓滚动。
    {
        const pended = reducer(initState({ playheadSec: 5, isPlaying: false }), undoRemote.pending("req-undo-2", undefined));
        const next = reducer(
            pended,
            undoRemote.fulfilled(timelinePayload(5), "req-undo-2", undefined),
        );
        assertEqual(next.playheadSec, 5, "identical playhead stays put");
        assertEqual(
            next.pendingPlayheadRevealSec,
            null,
            "no reveal request when the playhead did not move",
        );
    }

    // 播放中撤销：光标归传输层（音频时钟）所有，检查点值停留在本次播放的
    // 起始位置已过期 —— 保持轮询位置，不采纳快照值。
    {
        const pended = reducer(initState({ playheadSec: 50, isPlaying: true }), undoRemote.pending("req-undo-3", undefined));
        const next = reducer(
            pended,
            undoRemote.fulfilled(timelinePayload(7), "req-undo-3", undefined),
        );
        assertEqual(next.playheadSec, 50, "playing undo keeps the polled playhead");
        assertEqual(
            next.pendingPlayheadRevealSec,
            null,
            "playing undo never registers a reveal",
        );
    }

    // 乱序防护：过期 undo 响应（requestId 不匹配）不得改写光标。
    {
        const pended = reducer(initState({ playheadSec: 9, isPlaying: false }), undoRemote.pending("req-undo-new", undefined));
        const next = reducer(
            pended,
            undoRemote.fulfilled(timelinePayload(5), "req-undo-stale", undefined),
        );
        assertEqual(next.playheadSec, 9, "stale undo response is discarded");
    }
});
