/**
 * 试听引擎的并发与失败契约（用户报告"重复播放并叠加"的根因）。
 *
 * 【要钉死的三条】
 * 1. 首次播放要经过一次 `await` 取数；若期间又发起一次播放，**旧的那次必须原地
 *    放弃**，不能留下"已经发声但不在 `this.source` 里"的孤儿音源（`stop()` 停不掉
 *    它、它的 onended 也不再回调，于是高亮永远清不掉）。
 * 2. `stop()` 必须停掉**所有**已登记的源。
 * 3. 取数/解码失败必须收敛：不抛给调用方、不留在"正在播放"。
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const readAudioPreview = vi.fn();
vi.mock("../../services/api/fileBrowser", () => ({
    fileBrowserApi: {
        readAudioPreview: (...args: unknown[]) => readAudioPreview(...args),
    },
}));

import { audioPreview } from "./audioPreview";

/** 可手动放行的 promise（用于把一次 play 挂在 await 上）。 */
function makeDeferred(): { promise: Promise<void>; resolve: () => void } {
    let resolveFn: () => void = () => {};
    const promise = new Promise<void>((resolve) => {
        resolveFn = resolve;
    });
    return { promise, resolve: () => resolveFn() };
}

/** 一帧静音 PCM 的后端载荷。 */
function silentPreview() {
    const bytes = new Uint8Array(new Float32Array([0]).buffer);
    let binary = "";
    for (const byte of bytes) binary += String.fromCharCode(byte);
    return { sampleRate: 48000, channels: 1, pcmBase64: btoa(binary) };
}

/** 假音源：记录 start/stop 调用，便于断言"有没有孤儿在响"。 */
function makeFakeSource() {
    return {
        buffer: null as unknown,
        onended: null as (() => void) | null,
        started: 0,
        stopped: 0,
        disconnected: 0,
        connect() {},
        start() {
            this.started += 1;
        },
        stop() {
            this.stopped += 1;
        },
        disconnect() {
            this.disconnected += 1;
        },
    };
}

describe("audioPreview（试听引擎）", () => {
    let sources: ReturnType<typeof makeFakeSource>[];

    beforeEach(() => {
        sources = [];
        readAudioPreview.mockReset();
        // 1 帧单声道静音：解码路径只需可跑通。
        readAudioPreview.mockResolvedValue(silentPreview());

        const fakeCtx = {
            state: "running",
            destination: {},
            resume: () => Promise.resolve(),
            createGain: () => ({ connect() {}, gain: { value: 1 } }),
            createBuffer: (_channels: number, frames: number, sampleRate: number) => ({
                length: frames,
                sampleRate,
                getChannelData: () => new Float32Array(frames),
            }),
            createBufferSource: () => {
                const source = makeFakeSource();
                sources.push(source);
                return source;
            },
        };
        (globalThis as { AudioContext?: unknown }).AudioContext = function AudioContextStub() {
            return fakeCtx;
        } as unknown as typeof AudioContext;
        audioPreview.clearCache();
        audioPreview.stop();
    });

    afterEach(() => {
        audioPreview.stop();
        delete (globalThis as { AudioContext?: unknown }).AudioContext;
    });

    it("★ 并发 play：旧的一次必须放弃，且不留孤儿音源", async () => {
        // 让第一次取数挂在 await 上，期间发起第二次播放。
        const gate = makeDeferred();
        readAudioPreview.mockImplementationOnce(async () => {
            await gate.promise;
            return silentPreview();
        });

        const first = audioPreview.play("a.wav");
        const second = audioPreview.play("b.wav");
        gate.resolve();

        const [firstStarted, secondStarted] = await Promise.all([first, second]);
        expect(firstStarted).toBe(false); // 被抢占，未开始
        expect(secondStarted).toBe(true);
        expect(audioPreview.getCurrentFile()).toBe("b.wav");

        // 只有一个源真的 start 过（不存在"第二路叠加"）。
        expect(sources.filter((source) => source.started > 0)).toHaveLength(1);

        // stop 之后所有源都被停掉，没有任何遗留。
        audioPreview.stop();
        expect(audioPreview.isPlaying()).toBe(false);
        expect(audioPreview.getCurrentFile()).toBeNull();
        for (const source of sources) {
            if (source.started > 0) expect(source.stopped).toBeGreaterThan(0);
        }
    });

    it("★ 取数失败：不抛给调用方、不留在播放态", async () => {
        readAudioPreview.mockRejectedValueOnce(new Error("decode failed"));
        const started = await audioPreview.play("broken.wav");
        expect(started).toBe(false);
        expect(audioPreview.isPlaying()).toBe(false);
        expect(audioPreview.getCurrentFile()).toBeNull();
        expect(sources.every((source) => source.started === 0)).toBe(true);
    });

    it("缓存命中路径的连续两次 play 不会叠加", async () => {
        expect(await audioPreview.play("c.wav")).toBe(true);
        expect(await audioPreview.play("c.wav")).toBe(true);
        // 第二次 play 先 stop 了第一次的源。
        expect(sources[0].stopped).toBeGreaterThan(0);
        expect(sources[1].started).toBe(1);
        audioPreview.stop();
        expect(audioPreview.isPlaying()).toBe(false);
    });

    it("自然结束会清理状态并回调", async () => {
        let ended = 0;
        await audioPreview.play("d.wav", () => {
            ended += 1;
        });
        // 模拟播放自然结束。
        sources[0].onended?.();
        expect(ended).toBe(1);
        expect(audioPreview.getCurrentFile()).toBeNull();
    });

    it("stop 会作废在飞的 play（会话号推进）", async () => {
        const gate = makeDeferred();
        readAudioPreview.mockImplementationOnce(async () => {
            await gate.promise;
            return silentPreview();
        });
        const pending = audioPreview.play("e.wav");
        audioPreview.stop();
        gate.resolve();
        expect(await pending).toBe(false);
        expect(audioPreview.isPlaying()).toBe(false);
        expect(sources.every((source) => source.started === 0)).toBe(true);
    });
});
