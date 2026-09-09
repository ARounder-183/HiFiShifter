import { test } from "vitest";

/* eslint-disable @typescript-eslint/no-explicit-any -- 测试夹具：
   reducer 初始化 action 与全量时间线载荷无法在不使用 any 的情况下构造 */

import reducer, { setPlaybackRenderingState } from "./sessionSlice.ts";
import { moveClipRemote } from "./thunks/timelineThunks.ts";
import { undoRemote, redoRemote } from "./thunks/projectThunks.ts";
import { playOriginal, stopAudioPlayback, syncPlaybackState } from "./thunks/transportThunks.ts";

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
    function assertNear(actual: unknown, expected: number, label: string, tolSec = 0.05): void {
        const value = Number(actual);
        if (!Number.isFinite(value) || Math.abs(value - expected) > tolSec) {
            throw new Error(
                `${label}: expected ~${String(expected)} (±${tolSec}), received ${String(actual)}`,
            );
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

    // 播放中：轮询推进光标。播放分支的采样会按"派发→处理"时延外推，因此
    // 用容差断言（测试里时延 ≈ 0）。
    {
        const next = reducer(
            playingState(50),
            syncPlaybackState.fulfilled(syncPayload(true, 50.02), "req", {
                epoch: 0,
                dispatchedAtMs: performance.now(),
            }),
        );
        assertNear(next.playheadSec, 50.02, "playing poll advances the playhead");
    }

    // 采样外推：payload 是"后端执行命令那一刻"的位置，reducer 必须把它按时延
    // 外推到处理时刻 —— 100ms 前派发的采样应落在 采样值+0.1s。
    {
        const next = reducer(
            playingState(50),
            syncPlaybackState.fulfilled(syncPayload(true, 50.0), "req", {
                epoch: 0,
                dispatchedAtMs: performance.now() - 100,
            }),
        );
        assertNear(next.playheadSec, 50.1, "playing sample is extrapolated to arrival time", 0.05);
        // 采样时刻随写入了采样值一并记录，供视觉插值锚定。
        assertEqual(
            next.playheadSampledAtMs > 0,
            true,
            "polled playhead records its sampling timestamp",
        );
    }

    // 播放→停止跃迁：光标对齐到引擎冻结的精确停止位置（冻结位置不外推）。
    {
        const next = reducer(
            playingState(50.02),
            syncPlaybackState.fulfilled(syncPayload(false, 50.09), "req", {
                epoch: 0,
                dispatchedAtMs: performance.now() - 500,
            }),
        );
        assertEqual(
            next.playheadSec,
            50.09,
            "transition aligns the playhead with the engine's stop position",
        );
    }

    // 已停止后的后续轮询：不再改写光标（例如 handle_stop 后 position 归零）。
    // 跃迁应用时会推进传输纪元，因此第二次轮询必须携带最新纪元才被应用 ——
    // 这同时验证了纪元防护不会误伤"当前"轮询。
    {
        const stopped = reducer(
            playingState(50.02),
            syncPlaybackState.fulfilled(syncPayload(false, 50.09), "req", {
                epoch: 0,
                dispatchedAtMs: performance.now(),
            }),
        );
        const next = reducer(
            stopped,
            syncPlaybackState.fulfilled(syncPayload(false, 0), "req", {
                epoch: stopped._transportEpoch,
                dispatchedAtMs: performance.now(),
            }),
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
        const pended = reducer(
            initState({ playheadSec: 9, isPlaying: false }),
            undoRemote.pending("req-undo", undefined),
        );
        const next = reducer(
            pended,
            undoRemote.fulfilled(timelinePayload(5), "req-undo", undefined),
        );
        assertEqual(next.playheadSec, 5, "paused undo adopts the checkpoint playhead");
        assertEqual(next.pendingPlayheadRevealSec, 5, "moved playhead registers a reveal request");
    }

    // 重做 A→B：对称地跟随恢复快照的光标（5 → 9）。
    {
        const pended = reducer(
            initState({ playheadSec: 5, isPlaying: false }),
            redoRemote.pending("req-redo", undefined),
        );
        const next = reducer(
            pended,
            redoRemote.fulfilled(timelinePayload(9), "req-redo", undefined),
        );
        assertEqual(next.playheadSec, 9, "redo adopts the restored checkpoint playhead");
        assertEqual(next.pendingPlayheadRevealSec, 9, "redo registers a reveal request");
    }

    // 光标未挪动（该状态形成后光标未变）：不登记聚焦请求，无谓滚动。
    {
        const pended = reducer(
            initState({ playheadSec: 5, isPlaying: false }),
            undoRemote.pending("req-undo-2", undefined),
        );
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
        const pended = reducer(
            initState({ playheadSec: 50, isPlaying: true }),
            undoRemote.pending("req-undo-3", undefined),
        );
        const next = reducer(
            pended,
            undoRemote.fulfilled(timelinePayload(7), "req-undo-3", undefined),
        );
        assertEqual(next.playheadSec, 50, "playing undo keeps the polled playhead");
        assertEqual(next.pendingPlayheadRevealSec, null, "playing undo never registers a reveal");
    }

    // 乱序防护：过期 undo 响应（requestId 不匹配）不得改写光标。
    {
        const pended = reducer(
            initState({ playheadSec: 9, isPlaying: false }),
            undoRemote.pending("req-undo-new", undefined),
        );
        const next = reducer(
            pended,
            undoRemote.fulfilled(timelinePayload(5), "req-undo-stale", undefined),
        );
        assertEqual(next.playheadSec, 9, "stale undo response is discarded");
    }
});

/**
 * 播放轮询乱序防护（传输纪元）：后端预渲染（ONNX 推理）负载下 IPC 往返
 * 可延迟至数百毫秒，跨 播放/暂停/停止 边界的迟到轮询快照按新语义应用会把
 * isPlaying 复活/掐灭、并把光标改写成旧播放位置或 0。轮询在派发时携带传输
 * 纪元，fulfilled 与当前纪元比对，不一致的响应直接丢弃。
 */
test("features/session/sessionSlice.playheadGuard.test.ts sync responses crossing a transport operation are discarded", async () => {
    function assertEqual(actual: unknown, expected: unknown, label: string): void {
        if (actual !== expected) {
            throw new Error(`${label}: expected ${String(expected)}, received ${String(actual)}`);
        }
    }

    function playingState(playheadSec: number, epoch: number) {
        const base = reducer(undefined, { type: "@@INIT" }) as any;
        return {
            ...base,
            playheadSec,
            _transportEpoch: epoch,
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

    // 停止前的播放采样迟到：不得复活 isPlaying、不得把光标推回旧播放位置。
    {
        const stopped = reducer(
            playingState(50, 0),
            stopAudioPlayback.pending("req-stop", undefined),
        );
        const next = reducer(
            stopped,
            syncPlaybackState.fulfilled(syncPayload(true, 53), "req", {
                epoch: 0,
                dispatchedAtMs: performance.now(),
            }),
        );
        assertEqual(next.runtime.isPlaying, false, "stale playing snapshot cannot revive playback");
        assertEqual(next.playheadSec, 50, "stale playing snapshot cannot move the playhead");
    }

    // 停止→播放间隙采样（引擎已归零 base/position）迟到：不得把光标写成 0、
    // 不得掐灭刚建立的播放状态。
    {
        const pended = reducer(playingState(50, 0), playOriginal.pending("req-play", undefined));
        const played = reducer(
            pended,
            playOriginal.fulfilled(
                { ok: true, clipId: null, anchorSec: 42, playing: "original" },
                "req-play",
                undefined,
            ),
        );
        assertEqual(played.playheadSec, 50, "play adopts the anchor at dispatch, not the payload");
        const next = reducer(
            played,
            syncPlaybackState.fulfilled(syncPayload(false, 0), "req", {
                epoch: 1,
                dispatchedAtMs: performance.now(),
            }),
        );
        assertEqual(next.runtime.isPlaying, true, "stale stopped snapshot cannot kill playback");
        assertEqual(next.playheadSec, 50, "stale stopped snapshot cannot move the playhead");
    }

    // 纪元匹配的轮询照常推进光标。
    {
        const next = reducer(
            playingState(50, 7),
            syncPlaybackState.fulfilled(syncPayload(true, 50.02), "req", {
                epoch: 7,
                dispatchedAtMs: performance.now(),
            }),
        );
        if (!(Math.abs(Number(next.playheadSec) - 50.02) <= 0.05)) {
            throw new Error(
                `matching-epoch poll advances the playhead: expected ~50.02, received ${String(next.playheadSec)}`,
            );
        }
    }
});

/**
 * 跃迁防呆：引擎被用户停止后 base/position 归零。竞态下（阻塞预渲染窗口
 * 溜进的轮询、乱序响应）"播放→停止跃迁"分支拿到 0/0 不代表真实停止点，
 * 写 0 会把光标拽到工程开头；真实冻结点（自动暂停/自然结束）总是 > 0。
 */
test("features/session/sessionSlice.playheadGuard.test.ts zero frozen position never zeroes the playhead", async () => {
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
            _transportEpoch: 3,
            runtime: {
                ...base.runtime,
                isPlaying: true,
                playbackPositionSec: playheadSec,
            },
        };
    }

    const syncPayload = (baseSec: number, positionSec: number) =>
        ({
            ok: true,
            is_playing: false,
            target: null,
            base_sec: baseSec,
            position_sec: positionSec,
            duration_sec: 200,
        }) as any;

    // 0/0（引擎被 handle_stop 归零后）：跃迁应用，但光标原地不动。
    {
        const next = reducer(
            playingState(50.02),
            syncPlaybackState.fulfilled(syncPayload(0, 0), "req", {
                epoch: 3,
                dispatchedAtMs: performance.now(),
            }),
        );
        assertEqual(next.runtime.isPlaying, false, "transition still applies");
        assertEqual(next.playheadSec, 50.02, "zeroed engine state never zeroes the playhead");
    }

    // 真实冻结点 > 0（后台渲染自动暂停）：照常对齐。
    {
        const next = reducer(
            playingState(50.02),
            syncPlaybackState.fulfilled(syncPayload(0, 50.09), "req", {
                epoch: 3,
                dispatchedAtMs: performance.now(),
            }),
        );
        assertEqual(next.playheadSec, 50.09, "frozen position aligns the playhead");
    }
});

/**
 * 锚点保留：暂停只结束"进行中的播放"，不清除"本次播放的起始位置"。旧行为
 * 在暂停时把锚点清零，导致暂停后按 Stop 时 wasPlaying 启发式
 * （playheadSec ≠ anchor）仍判定"曾处于播放中"，把光标"恢复"到 0
 * （播放→暂停→停止 光标跳到工程开头）。保留锚点后 Stop 回到最近一次播放的
 * 起点，与播放中按 Stop 的行为一致（REAPER 语义），且再次按 Stop 幂等。
 */
test("features/session/sessionSlice.playheadGuard.test.ts pause keeps the playback anchor for the next stop", async () => {
    function assertEqual(actual: unknown, expected: unknown, label: string): void {
        if (actual !== expected) {
            throw new Error(`${label}: expected ${String(expected)}, received ${String(actual)}`);
        }
    }

    function playingState(playheadSec: number, anchorSec: number) {
        const base = reducer(undefined, { type: "@@INIT" }) as any;
        return {
            ...base,
            playheadSec,
            playbackAnchorSec: anchorSec,
            runtime: {
                ...base.runtime,
                isPlaying: true,
                playbackPositionSec: playheadSec,
            },
        };
    }

    // 暂停（无锚点恢复）：光标对齐精确停止点，锚点保留。
    {
        const next = reducer(
            playingState(42.3, 10),
            stopAudioPlayback.fulfilled(
                {
                    ok: true,
                    stopped_at_sec: 42.5,
                    restoreAnchor: false,
                    wasPlaying: true,
                    anchorSec: 10,
                },
                "req",
                undefined,
            ),
        );
        assertEqual(next.playheadSec, 42.5, "pause aligns to the exact stop position");
        assertEqual(next.playbackAnchorSec, 10, "pause keeps the playback anchor");
    }

    // 暂停后按 Stop（锚点恢复）：回到最近一次播放的起点（而不是 0）。
    {
        const next = reducer(
            playingState(42.3, 10),
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
        assertEqual(next.playbackAnchorSec, 10, "stop keeps the anchor for idempotent repeats");
    }
});

/**
 * 阻塞式前台预渲染窗口拒采：target="original" 的渲染 active 期间，引擎尚未
 * 真正进入 playing，`get_playback_state` 返回上一场播放/seek 的陈旧传输态。
 * 短渲染（全缓存命中）完成后实时轮询已推进光标，渲染期间派发的轮询响应
 * 此刻才迟到到达 —— 纪元未变（渲染期间无传输操作），乱序防护拦不住；
 * 若按"播放→停止跃迁"应用，会把光标拽回播放起点、掐灭 isPlaying 并让
 * 轮询停摆（光标卡死而音频继续播）。渲染状态镜像（Redux）让 reducer 在
 * 该窗口内一概拒采；渲染结束（active=false）后恢复采信。
 */
test("features/session/sessionSlice.playheadGuard.test.ts polls are refused while a blocking foreground render is active", async () => {
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

    const syncArgs = { epoch: 0, dispatchedAtMs: performance.now() };

    // 阻塞式预渲染进行中：陈旧"未播放"采样被拒采，播放状态与光标不受影响。
    {
        const started = reducer(
            playingState(50),
            setPlaybackRenderingState({ active: true, target: "original" }),
        );
        const next = reducer(
            started,
            syncPlaybackState.fulfilled(syncPayload(false, 50), "req", syncArgs),
        );
        assertEqual(next.runtime.isPlaying, true, "blocking prerender window refuses stale polls");
        assertEqual(next.playheadSec, 50, "blocking prerender window never moves the playhead");
    }

    // 后台预渲染（target="background"）不拒采：播放是实时的，轮询照常推进。
    {
        const started = reducer(
            playingState(50),
            setPlaybackRenderingState({ active: true, target: "background" }),
        );
        const next = reducer(
            started,
            syncPlaybackState.fulfilled(syncPayload(true, 50.02), "req", syncArgs),
        );
        if (!(Math.abs(Number(next.playheadSec) - 50.02) <= 0.05)) {
            throw new Error(
                `background render keeps polls flowing: expected ~50.02, received ${String(next.playheadSec)}`,
            );
        }
    }

    // 渲染结束（active=false）：恢复采信，跃迁对齐照常工作。
    {
        const started = reducer(
            playingState(50),
            setPlaybackRenderingState({ active: true, target: "original" }),
        );
        const ended = reducer(
            started,
            setPlaybackRenderingState({ active: false, target: "original" }),
        );
        const next = reducer(
            ended,
            syncPlaybackState.fulfilled(syncPayload(false, 50.09), "req", syncArgs),
        );
        assertEqual(next.playheadSec, 50.09, "after the render ends polls apply again");
    }
});
