// @vitest-environment jsdom
import { afterEach, expect, test } from "vitest";

import reducer, { setPlayheadSec } from "./sessionSlice";
import { pasteTimelineClipboardRemote } from "./thunks/timelineThunks";
import { seekPlayhead } from "./thunks/transportThunks";
import type { SessionState } from "./sessionSlice/types";

/**
 * 播放光标**所有权**回归：宿主拥有它时，插件不写它。
 *
 * 【为什么这条与 playheadGuard 分开】那个文件回答"谁的**采样**算数"（编辑快照不得
 * 抢走传输层的所有权）；这个文件回答"插件**能不能**把光标搬走"。非 ARA 的 VST 宿主
 * 里后者恒为不能 —— `set_transport` 需要 ARA 播放请求，拿不到就必然失败。
 *
 * 【症状是什么】每一次跳转都是"乐观写 store + 打后端"两步：光标先跳过去，随后
 * 30 Hz 的播放轮询用宿主真实位置把它拽回来。用户看到"先跳过去、再跳回来"。
 */

afterEach(() => {
    delete window.__HFS_PLUGIN_BOOTSTRAP__;
});

/** 初始状态（reducer 自带完整默认值，这里只覆盖与本文件相关的两项）。 */
function initialState(): SessionState {
    return reducer(undefined, { type: "@@INIT" }) as SessionState;
}

function withPlayhead(sec: number): SessionState {
    const base = initialState();
    return { ...base, playheadSec: sec, playheadLocalAtMs: 111 };
}

/** 插件形态的 bootstrap。`transportControl` = 宿主是否给了 ARA 播放请求。 */
function pluginBootstrap(transportControl: boolean): void {
    window.__HFS_PLUGIN_BOOTSTRAP__ = {
        version: 1,
        viewId: "playhead-ownership",
        transportControl,
        hostCapabilities: {},
    } as never;
}

test("standalone keeps writing the playhead: the engine really does move", () => {
    const next = reducer(withPlayhead(1), setPlayheadSec(4.5)) as SessionState;
    expect(next.playheadSec).toBe(4.5);
});

test("a host-owned playhead is never written by an intent", () => {
    pluginBootstrap(false);
    const next = reducer(withPlayhead(1), setPlayheadSec(4.5)) as SessionState;
    expect(next.playheadSec).toBe(1);
    // 乐观写同时会给视觉插值打一个本地锚点；闸住时它也不能前进，
    // 否则"最后一次本地写入"的时间戳会骗过播放轮询的 300ms 避让窗口。
    expect(next.playheadLocalAtMs).toBe(111);
});

test("an ARA host that can seek keeps the optimistic write", () => {
    pluginBootstrap(true);
    const next = reducer(withPlayhead(1), setPlayheadSec(4.5)) as SessionState;
    expect(next.playheadSec).toBe(4.5);
});

test("paste does not drag a host-owned playhead to the pasted range", () => {
    const timeline = {
        tracks: [],
        clips: [],
        selected_track_id: null,
        selected_clip_id: null,
        playhead_sec: 1,
        project_sec: 32,
        bpm: 120,
        disabled_group_ids: [],
    } as never;
    const paste = (state: SessionState) =>
        reducer(
            reducer(state, pasteTimelineClipboardRemote.pending("paste-1", undefined)),
            pasteTimelineClipboardRemote.fulfilled(
                {
                    ok: true,
                    timeline,
                    newClipIds: ["clip-a"],
                    pasteEndSec: 9,
                } as never,
                "paste-1",
                undefined,
            ),
        ) as SessionState;

    pluginBootstrap(false);
    const owned = paste(withPlayhead(1));
    expect(owned.playheadSec).toBe(1);
    // 视图也不该被滚到一个光标根本没去的地方。
    expect(owned.pendingPlayheadRevealSec).toBeNull();

    delete window.__HFS_PLUGIN_BOOTSTRAP__;
    const standalone = paste(withPlayhead(1));
    expect(standalone.playheadSec).toBe(9);
    expect(standalone.pendingPlayheadRevealSec).toBe(9);
});

test("seekPlayhead short-circuits instead of posting a doomed transport command", async () => {
    pluginBootstrap(false);
    const dispatched = seekPlayhead(12);
    const result = await dispatched(
        ((action: unknown) => action) as never,
        (() => withPlayhead(1)) as never,
        undefined as never,
    );
    expect(result.payload).toMatchObject({ ok: true, playhead_sec: 12, skipped: true });
    expect(result.meta.requestStatus).toBe("fulfilled");
});
