// @vitest-environment jsdom
// 传输推送通道与"谁给的读数"：插件形态下光标不再靠轮询、也不再被残值拽回。
import { afterEach, expect, test } from "vitest";
import reducer from "./sessionSlice";
import { transportPushed, type TransportSample } from "./sessionSlice/extraReducers/transport";
import { stopAudioPlayback, syncPlaybackState } from "./thunks/transportThunks";

const PLUGIN = { version: 1 as const, viewId: "transport-push", transportControl: true };

function pluginMode() {
    (window as unknown as { __HFS_PLUGIN_BOOTSTRAP__?: unknown }).__HFS_PLUGIN_BOOTSTRAP__ = PLUGIN;
}

function standalone() {
    delete (window as unknown as { __HFS_PLUGIN_BOOTSTRAP__?: unknown }).__HFS_PLUGIN_BOOTSTRAP__;
}

afterEach(standalone);

const sample = (over: Partial<TransportSample> = {}): TransportSample => ({
    ok: true,
    is_playing: true,
    base_sec: 0,
    position_sec: 0,
    duration_sec: 60,
    host_authoritative: true,
    position_source: "host",
    position_age_ms: 4,
    seek_pending: false,
    ...over,
});

test("★ 一条推送就能移动光标，不需要任何轮询", () => {
    const base = reducer(undefined, { type: "@@INIT" });
    const next = reducer(base, transportPushed(sample({ position_sec: 12.5 })));
    expect(next.playheadSec).toBe(12.5);
    expect(next.runtime.isPlaying).toBe(true);
    expect(next.runtime.hostAuthoritative).toBe(true);
});

test("★ 残值（stale）绝不把光标拽回去", () => {
    const base = reducer(undefined, { type: "@@INIT" });
    const playing = reducer(base, transportPushed(sample({ position_sec: 30 })));
    // 宿主这一轮什么都没给：位置是上一次观测的残值。
    const stale = reducer(
        playing,
        transportPushed(
            sample({
                is_playing: false,
                position_sec: 3,
                position_source: "stale",
                position_age_ms: 4_294_967_295,
            }),
        ),
    );
    expect(stale.playheadSec).toBe(30);
});

test("★ 定位窗口内的读数就是目标位置，不会被下一条旧采样撤销", () => {
    const base = reducer(undefined, { type: "@@INIT" });
    // 后端在宿主落地之前把读数钉在目标上（position_source = "seek"）。
    const seeking = reducer(
        base,
        transportPushed(sample({ position_sec: 20, position_source: "seek", seek_pending: true })),
    );
    expect(seeking.playheadSec).toBe(20);
    expect(seeking.runtime.isPlaying).toBe(true);
    // 同一时刻一条"宿主还在旧位置"的采样不得改写它（后端已经挡住了，
    // 这里钉的是前端不会自己再拽一次）。
    const again = reducer(
        seeking,
        transportPushed(sample({ position_sec: 20, position_source: "seek", seek_pending: true })),
    );
    expect(again.playheadSec).toBe(20);
});

test("★ 插件形态下阻塞式渲染不再冻结光标（那里的渲染不改宿主位置）", () => {
    pluginMode();
    const base = reducer(undefined, { type: "@@INIT" });
    const rendering = {
        ...base,
        playbackBlockingRenderActive: true,
    };
    const next = reducer(rendering, transportPushed(sample({ position_sec: 7.25 })));
    expect(next.playheadSec).toBe(7.25);
});

test("独立 App 的阻塞式渲染仍然拒采（那里的位置确实被冻结）", () => {
    standalone();
    const base = reducer(undefined, { type: "@@INIT" });
    const rendering = {
        ...base,
        playbackBlockingRenderActive: true,
        playheadSec: 3,
    };
    const next = reducer(rendering, transportPushed(sample({ position_sec: 7.25 })));
    expect(next.playheadSec).toBe(3);
});

test("★ 位置真的前进了才置 positionAdvancing（视觉推进的事实位）", () => {
    const base = reducer(undefined, { type: "@@INIT" });
    // 起播第一条采样：位置从 0 走到 1 ⇒ 有证据说在前进。这正是"起播就要动"。
    const first = reducer(base, transportPushed(sample({ position_sec: 1 })));
    expect(first.runtime.positionAdvancing).toBe(true);
    // 同一个位置再来一条 ⇒ 冻结，不是前进（宿主报"在播"但位置没动）。
    const frozen = reducer(first, transportPushed(sample({ position_sec: 1 })));
    expect(frozen.runtime.positionAdvancing).toBe(false);
    // 又动了 ⇒ 恢复。
    const moving = reducer(frozen, transportPushed(sample({ position_sec: 1.033 })));
    expect(moving.runtime.positionAdvancing).toBe(true);
    // 停播 ⇒ 永远不是前进。
    const stopped = reducer(moving, transportPushed(sample({ is_playing: false })));
    expect(stopped.runtime.positionAdvancing).toBe(false);
});

test("★ 同一序号的后读采样必须仍然生效：宿主停发音频块时播放位要能过期", () => {
    // REAPER 的 Item（take FX）停播后**不再送音频块**，后端因此不再产生新的
    // `sample_seq`；而"播放位已过期"只在随后的同序号读数里表达。把同序号读数当
    // 重复丢掉，界面就会永远停在"宿主还在播"——光标能动、但不会停止。
    const base = reducer(undefined, { type: "@@INIT" });
    const playing = reducer(
        base,
        transportPushed(sample({ position_sec: 30, sample_seq: 5, position_age_ms: 4 })),
    );
    expect(playing.runtime.isPlaying).toBe(true);
    expect(playing.runtime.positionAdvancing).toBe(true);

    // 同一条发布，再读一次：块已过期，位置冻结在 30 s。
    const expired = reducer(
        playing,
        transportPushed(
            sample({
                is_playing: false,
                position_sec: 30,
                sample_seq: 5,
                position_source: "stale",
                position_age_ms: 4_294_967_295,
            }),
        ),
    );
    expect(expired.runtime.isPlaying).toBe(false);
    expect(expired.runtime.positionAdvancing).toBe(false);
    // 残值只负责熄灭播放位，绝不把光标拽回旧位置。
    expect(expired.playheadSec).toBe(30);
});

test("★ 严格更旧的序号仍然被丢弃（乱序防护没有被放宽）", () => {
    const base = reducer(undefined, { type: "@@INIT" });
    const playing = reducer(base, transportPushed(sample({ position_sec: 30, sample_seq: 5 })));
    const older = reducer(playing, transportPushed(sample({ position_sec: 12, sample_seq: 4 })));
    expect(older.playheadSec).toBe(30);
    expect(older.runtime.isPlaying).toBe(true);
});

test("★ 停止 / 暂停命令 pending 立即清掉视觉推进位", () => {
    const base = reducer(undefined, { type: "@@INIT" });
    const playing = reducer(base, transportPushed(sample({ position_sec: 1 })));
    expect(playing.runtime.isPlaying).toBe(true);
    expect(playing.runtime.positionAdvancing).toBe(true);

    const stopping = reducer(playing, stopAudioPlayback.pending("test-stop", undefined));
    expect(stopping.runtime.isPlaying).toBe(false);
    expect(stopping.runtime.positionAdvancing).toBe(false);
});

test("轮询与推送共用同一份语义：同一条采样经两条通道得到同一个状态", () => {
    const base = reducer(undefined, { type: "@@INIT" });
    const pushed = reducer(base, transportPushed(sample({ position_sec: 9.5 })));
    const polled = reducer(
        base,
        syncPlaybackState.fulfilled(sample({ position_sec: 9.5 }) as never, "poll", {
            epoch: base._transportEpoch,
            dispatchedAtMs: performance.now(),
        } as never),
    );
    expect(pushed.playheadSec).toBe(polled.playheadSec);
    expect(pushed.runtime.isPlaying).toBe(polled.runtime.isPlaying);
    expect(pushed.runtime.positionAdvancing).toBe(polled.runtime.positionAdvancing);
});
