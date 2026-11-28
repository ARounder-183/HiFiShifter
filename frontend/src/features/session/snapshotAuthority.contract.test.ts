// 快照权威契约：版本号仲裁 + 手势会话推迟。
//
// 这两条是本轮修复的地基 —— 它们同时决定「拖动时会不会弹回去」与
// 「分割/撤销的回包会不会被别的动作挤掉」。因此断言必须落在**可观察的状态**上
// （`clips[i].startSec` / `timelineRevision`），而不是某个内部标志位。
import { describe, expect, it } from "vitest";
import reducer, { beginGesture, endGesture } from "./sessionSlice";
import { applyTimelineState } from "./sessionSlice/helpersTimeline";
import { initialState } from "./sessionSlice/initialState";
import type { SessionState } from "./sessionSlice/types";
import type { TimelineState } from "../../types/api";

/** 造一份最小可用载荷；`revision` 缺省时不带该字段（模拟旧后端）。 */
function payload(startSec: number, revision?: number): TimelineState {
    const base = {
        ok: true,
        tracks: [
            {
                id: "track",
                name: "Track",
                volume: 1,
                muted: false,
                solo: false,
                compose_enabled: false,
                pitch_analysis_algo: "world",
                color: "#74787e",
            },
        ],
        clips: [
            {
                id: "clip",
                track_id: "track",
                name: "clip",
                color: "#74787e",
                start_sec: startSec,
                length_sec: 1,
                source_start_sec: 0,
                source_end_sec: 1,
                playback_rate: 1,
                gain: 1,
                muted: false,
            },
        ],
        selected_track_id: "track",
        selected_clip_id: "clip",
        playhead_sec: 0,
        project_sec: 10,
        bpm: 120,
        disabled_group_ids: [],
    } as unknown as TimelineState;
    if (revision !== undefined) (base as { revision?: number }).revision = revision;
    return base;
}

/** 从零态构造一份干净的会话状态（避免测试间互相污染）。 */
function freshState(): SessionState {
    return JSON.parse(JSON.stringify(initialState)) as SessionState;
}

/**
 * 应用一份快照并返回**新的顶层对象**。
 *
 * 【为什么要浅拷贝】RTK/Immer 产出的 state 是冻结的，而 `applyTimelineState`
 * 直接对入参赋值（它平时拿到的是 Immer draft）。测试里用浅拷贝模拟 draft 的
 * 「顶层可写」语义，被改写的嵌套容器也全都是整块重新赋值，不会碰到冻结子对象。
 */
function apply(state: SessionState, startSec: number, revision?: number) {
    const next = { ...state };
    applyTimelineState(next, payload(startSec, revision), {});
    return next;
}

/** 派发一个 slice action，并把结果解冻成可继续改写的普通对象。 */
function dispatch(state: SessionState, action: { type: string }): SessionState {
    return { ...(reducer(state, action as never) as SessionState) };
}

describe("版本号仲裁（替代单槽 _latestEditRequestId）", () => {
    it("迟到的旧版本快照被丢弃", () => {
        let state = apply(freshState(), 0, 10);
        expect(state.clips[0].startSec).toBe(0);
        expect(state.timelineRevision).toBe(10);

        state = apply(state, 3, 12);
        expect(state.clips[0].startSec).toBe(3);

        state = apply(state, 9, 11); // 旧版本迟到
        expect(state.clips[0].startSec).toBe(3);
        expect(state.timelineRevision).toBe(12);
    });

    it("★ 新版本一定落地，即使另一次编辑仍在途（不同动作不再互相作废）", () => {
        let state = apply(freshState(), 0, 1);
        // A 请求先发（版本 2），B 请求后发（版本 3）；B 先回，A 后回。
        state = apply(state, 5, 3); // B：分割的结果
        expect(state.clips[0].startSec).toBe(5);
        state = apply(state, 7, 2); // A：迟到的音量编辑
        expect(state.clips[0].startSec).toBe(5); // 旧的不许覆盖新的
        state = apply(state, 8, 4); // 之后任何更新的一条都必须落地
        expect(state.clips[0].startSec).toBe(8);
    });

    it("同版本号可以覆盖（内容相同，不是「旧」）", () => {
        let state = apply(freshState(), 0, 5);
        state = apply(state, 2, 5);
        expect(state.clips[0].startSec).toBe(2);
    });

    it("没有版本号的载荷（旧后端）不参与版本仲裁", () => {
        let state = apply(freshState(), 0, 10);
        state = apply(state, 4); // 无 revision
        expect(state.clips[0].startSec).toBe(4);
        // 版本号不被无版本载荷污染。
        expect(state.timelineRevision).toBe(10);
    });
});

describe("手势会话：推迟而不是丢弃", () => {
    it("★ 手势期间 state.clips 不变，结束后最新的那份落地一次", () => {
        let state = apply(freshState(), 0, 1);
        state = dispatch(state, beginGesture({ kind: "clip-move" }));
        expect(state._gestureDepth).toBe(1);

        state = apply(state, 5, 2);
        expect(state.clips[0].startSec).toBe(0); // 没落地
        expect(state._deferredSnapshot?.revision).toBe(2);

        state = apply(state, 6, 3);
        expect(state._deferredSnapshot?.revision).toBe(3); // 只留最新

        state = apply(state, 9, 2); // 迟到的旧版本不覆盖推迟项
        expect(state._deferredSnapshot?.revision).toBe(3);

        state = dispatch(state, endGesture());
        expect(state._gestureDepth).toBe(0);
        expect(state.clips[0].startSec).toBe(6); // 落地的是最新那份
        expect(state._deferredSnapshot).toBeNull();
    });

    it("★ 手势结束时不落地「已经被更权威快照取代」的那份", () => {
        let state = apply(freshState(), 0, 1);
        state = dispatch(state, beginGesture({ kind: "param-paint" }));
        state = apply(state, 5, 2); // 被推迟
        // 手势期间一条**权威**载荷落地（force：工程生命周期 / 撤销重做）。
        state = { ...state };
        applyTimelineState(state, payload(8, 3), { force: true });
        expect(state.clips[0].startSec).toBe(8);
        state = dispatch(state, endGesture());
        // 推迟项版本 2 < 已落地 3 ⇒ 不落地，界面不会退回 5。
        expect(state.clips[0].startSec).toBe(8);
    });

    it("嵌套手势：内层结束不落地，最外层结束才落地", () => {
        let state = apply(freshState(), 0, 1);
        state = dispatch(state, beginGesture({ kind: "clip-move" }));
        state = dispatch(state, beginGesture({ kind: "pan" }));
        state = apply(state, 5, 2);
        state = dispatch(state, endGesture());
        expect(state._gestureDepth).toBe(1);
        expect(state.clips[0].startSec).toBe(0);
        state = dispatch(state, endGesture());
        expect(state.clips[0].startSec).toBe(5);
    });

    it("没有开始就结束：不抛错、不落地任何东西", () => {
        const state = apply(freshState(), 0, 1);
        const next = dispatch(state, endGesture());
        expect(next._gestureDepth).toBe(0);
        expect(next._deferredSnapshot).toBeNull();
        expect(next.clips[0].startSec).toBe(0);
    });
});
