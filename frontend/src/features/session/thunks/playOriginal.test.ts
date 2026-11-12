/**
 * 起播权威：`play_original` 不再接收前端锚点。
 *
 * 【为什么这条必须存在】第一轮的修法是把 `state.session.playheadSec` 作为
 * `startSec` 传给后端。它看起来"修好了跳变"，实际上把 Redux 镜像（30Hz 轮询 +
 * IPC 往返外推）当成了权威 —— 镜像与引擎差一拍时，起播点就跟着错。
 * 这条测试直接钉住契约：**调用不带任何参数**，起播点由引擎决定。
 *
 * 在第一轮的实现上它会失败（那里必定带一个数字），因此它是"修好了没有"的判据，
 * 而不是"代码长这样"的回声。
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { configureStore } from "@reduxjs/toolkit";

import reducer from "../sessionSlice";
import { playOriginal } from "./transportThunks";
import { webApi } from "../../../services/webviewApi";

type SessionState = ReturnType<typeof reducer>;

/** 造一个只带会话切片的 store；播放相关字段用 preloadedState 精确摆位。 */
function makeStore(options: { playheadSec: number; isPlaying?: boolean }) {
    const base = reducer(undefined, { type: "@@INIT" }) as SessionState;
    const preloaded: SessionState = {
        ...base,
        playheadSec: options.playheadSec,
        runtime: { ...base.runtime, isPlaying: Boolean(options.isPlaying) },
    };
    return configureStore({
        reducer: { session: reducer },
        preloadedState: { session: preloaded },
    });
}

afterEach(() => {
    vi.restoreAllMocks();
});

describe("playOriginal", () => {
    it("不把本地光标当作起播点传给后端", async () => {
        const spy = vi
            .spyOn(webApi, "playOriginal")
            .mockResolvedValue({ ok: true, playing: "original", start_sec: 12.5 });

        const store = makeStore({ playheadSec: 3 });
        const result = await store.dispatch(playOriginal()).unwrap();

        expect(spy).toHaveBeenCalledTimes(1);
        // ★ 关键断言：零个位置参数。多一个数字就会把镜像误差固化进起播点。
        expect(spy.mock.calls[0]).toEqual([]);
        // 锚点来自引擎回执（12.5），而不是本地状态（3）。
        expect(result.anchorSec).toBe(12.5);
        expect(store.getState().session.playbackAnchorSec).toBe(12.5);
    });

    it("引擎回执缺少位置时退回本地光标（仅用于 Stop 的锚点恢复）", async () => {
        vi.spyOn(webApi, "playOriginal").mockResolvedValue({ ok: true, playing: "original" });

        const store = makeStore({ playheadSec: 7.25 });
        const result = await store.dispatch(playOriginal()).unwrap();
        expect(result.anchorSec).toBe(7.25);
    });

    it("已在播放时的重复触发完全不打扰后端", async () => {
        const spy = vi.spyOn(webApi, "playOriginal");
        const store = makeStore({ playheadSec: 7.25, isPlaying: true });

        const result = await store.dispatch(playOriginal()).unwrap();
        expect(result.noop).toBe(true);
        expect(spy).not.toHaveBeenCalled();
    });
});
