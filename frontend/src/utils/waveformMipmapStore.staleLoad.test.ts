/**
 * 波形 mipmap 缓存「在途响应作废」回归测试。
 *
 * 覆盖的是同一症状的两个历史根源（打开工程 / 导入音频后波形空白，要等用户
 * 滚动或缩放才出现）：
 *
 * 1. **代次必须是按文件的**。`refresh()`（后端 `waveform_analysis_progress`
 *    done/cached）对每个文件各调一次，内部走 `invalidate()`。若代次是全局
 *    的，第 k 个文件的作废会连带把其余文件正在途的响应全部丢弃 —— 那些文件
 *    既没有新请求也没有通知，波形永久空白。
 * 2. **作废必须同时清掉在途登记**。`invalidate()` 只删缓存条目、不摘
 *    `loadingPromises` 的键，紧随其后的 `batchPreload()` 就会被自己的判重
 *    挡住（registered 为空 → 一个请求都不发），而那个在途 Promise 又因为
 *    代次不匹配被整包丢弃。
 * 3. **丢弃不是终态**：过期响应落地后若该文件仍在缓存中且级别依旧缺失，
 *    必须补发一次请求，否则同样落回"没人再要、没人通知"的死局。
 *
 * 同时保留 `clear()` 的原有语义：全局作废后，过期响应不得让已清空的缓存
 * 复活。
 */

import { beforeEach, describe, expect, it, vi } from "vitest";

import { waveformApi } from "../services/api/waveform";

import { waveformMipmapStore } from "./waveformMipmapStore.js";

vi.mock("../services/api/waveform", () => ({
    waveformApi: {
        getWaveformMipmapBinary: vi.fn(),
        preloadWaveformMipmap: vi.fn(async () => ({ ok: true })),
        batchGetWaveformMipmap: vi.fn(async () => ({})),
        getWaveformManifest: vi.fn(),
        getWaveformTilesBinary: vi.fn(),
        getRootMixWaveformPeaksSegment: vi.fn(),
        getTrackMixWaveformPeaksSegment: vi.fn(),
    },
}));

const singleMock = vi.mocked(waveformApi.getWaveformMipmapBinary);
const batchMock = vi.mocked(waveformApi.batchGetWaveformMipmap);

/**
 * 与 store 内的 RETRY_NOT_READY_COOLDOWN_MS 保持一致（该常量未导出；
 * 测试只需一个"大于冷却"的时间刻度）。
 */
const RETRY_COOLDOWN_MS = 3000;
/** 与 store 内的 NOT_READY_MAX_RETRY 保持一致。 */
const NOT_READY_MAX_RETRY = 5;

/** 构造一份合法的 mipmap 二进制载荷（Base64）。 */
function encodeMipmap(level: number, peakCount = 4): string {
    const bytes = new Uint8Array(20 + peakCount * 8);
    bytes.set([0x57, 0x46, 0x50, 0x4b], 0); // "WFPK"
    const view = new DataView(bytes.buffer);
    view.setUint32(4, 44100, true); // sample_rate
    view.setUint32(8, 4096, true); // division_factor
    view.setUint32(12, peakCount, true); // peak_count
    view.setUint32(16, level, true); // level
    let binary = "";
    for (let i = 0; i < bytes.length; i += 1) binary += String.fromCharCode(bytes[i]);
    return btoa(binary);
}

interface Deferred<T> {
    promise: Promise<T>;
    resolve: (value: T) => void;
}

function deferred<T>(): Deferred<T> {
    let resolve!: (value: T) => void;
    const promise = new Promise<T>((res) => {
        resolve = res;
    });
    return { promise, resolve };
}

/** 让所有已排队的微任务跑完。 */
function flush(): Promise<void> {
    return new Promise((resolve) => {
        setTimeout(resolve, 0);
    });
}

beforeEach(() => {
    waveformMipmapStore.clear();
    vi.clearAllMocks();
    batchMock.mockResolvedValue({});
});

describe("waveformMipmapStore 在途响应作废", () => {
    it("有加载在途时 refresh() 不得作废该文件（否则事件→作废→丢响应→再请求死循环）", async () => {
        // 复现自诊断日志的死循环：后端每次被调用都会 emit `cached`，
        // App 把它转成 refresh()。若 refresh() 在批量响应落地前 invalidate，
        // 批量响应按代次被丢弃 → L2 永远写不进缓存 → batchPreload 再发 →
        // 后端再 emit `cached` → 再 refresh() …… 波形永不出现且 IPC 风暴。
        const pendingBatch = deferred<Record<string, [string, string, string]>>();
        batchMock.mockReturnValue(pendingBatch.promise);
        // 首次抢跑拿空串（后端忙）：落地后只剩冷却，无任何在途请求。
        singleMock.mockResolvedValue("");
        expect(waveformMipmapStore.getPeaks("/loop.wav", 2)).toBeNull();
        await flush();

        // 首次 refresh：无在途 → invalidate + batchPreload，发出批量请求。
        waveformMipmapStore.refresh("/loop.wav");
        expect(batchMock).toHaveBeenCalledTimes(1);

        // 响应尚未落地时，后端的 `cached` 事件先到 → 再次 refresh()。
        // 此时批量请求在途，refresh() 必须是 no-op，而不是再作废一轮。
        waveformMipmapStore.refresh("/loop.wav");

        // 此时批量响应才落地：带有效 L2。
        pendingBatch.resolve({ "/loop.wav": ["", "", encodeMipmap(2)] });
        await flush();

        // L2 必须成功写入缓存（旧实现会被代次作废整包丢弃 → 死循环）。
        expect(waveformMipmapStore.hasLevel("/loop.wav", 2)).toBe(true);
        // 且不再发起第二轮批量请求。
        expect(batchMock).toHaveBeenCalledTimes(1);

        // 收尾：清掉"未就绪"重试定时器，避免悬挂计时器跨测试触发。
        waveformMipmapStore.clear();
    });

    it("作废一个文件不得丢弃另一个文件正在途的响应", async () => {
        const a = deferred<string>();
        const b = deferred<string>();
        singleMock.mockImplementation((path) => (path === "/a.wav" ? a.promise : b.promise));

        expect(waveformMipmapStore.getPeaks("/a.wav", 2)).toBeNull();
        expect(waveformMipmapStore.getPeaks("/b.wav", 2)).toBeNull();

        // 后端分析完成事件先落到 b：refresh(b) → invalidate(b)。
        // 全局代次实现会在这里把 a 的在途响应一并作废。
        waveformMipmapStore.invalidate("/b.wav");

        a.resolve(encodeMipmap(2));
        await flush();

        expect(waveformMipmapStore.hasLevel("/a.wav", 2)).toBe(true);
    });

    it("多文件 refresh 互不影响：前一个文件的响应仍然生效", async () => {
        const a = deferred<string>();
        singleMock.mockImplementation(() => a.promise);

        expect(waveformMipmapStore.getPeaks("/a.wav", 2)).toBeNull();
        // b 的请求已返回空（后端未就绪），随后分析完成 → refresh(b)。
        singleMock.mockResolvedValueOnce("");
        expect(waveformMipmapStore.getPeaks("/b.wav", 2)).toBeNull();
        await flush();

        waveformMipmapStore.refresh("/b.wav");
        a.resolve(encodeMipmap(2));
        await flush();

        expect(waveformMipmapStore.hasLevel("/a.wav", 2)).toBe(true);
    });

    it("invalidate 清掉在途登记，随后的 batchPreload 才能真的发出请求", async () => {
        const pending = deferred<string>();
        singleMock.mockReturnValue(pending.promise);

        expect(waveformMipmapStore.getPeaks("/c.wav", 2)).toBeNull();
        waveformMipmapStore.invalidate("/c.wav");

        await waveformMipmapStore.batchPreload(["/c.wav"]);

        expect(batchMock).toHaveBeenCalledWith(["/c.wav"]);
    });

    it("过期响应被丢弃后会补发一次请求（丢弃不是终态）", async () => {
        const stale = deferred<string>();
        singleMock.mockReturnValueOnce(stale.promise);
        singleMock.mockResolvedValue(encodeMipmap(2));

        // 1) 发起请求 #1，随后整个文件被作废。
        expect(waveformMipmapStore.getPeaks("/d.wav", 2)).toBeNull();
        waveformMipmapStore.invalidate("/d.wav");

        // 2) 作废后该文件又被重新登记（模拟紧随其后的 batchPreload / 下一帧
        //    draw），但批量响应里没有它 —— 条目存在、无数据、无在途请求。
        batchMock.mockResolvedValue({});
        await waveformMipmapStore.batchPreload(["/d.wav"]);
        expect(waveformMipmapStore.hasLevel("/d.wav", 2)).toBe(false);

        // 3) 请求 #1 姗姗来迟 → 判定过期 → 丢弃后必须补发。
        const callsBefore = singleMock.mock.calls.length;
        stale.resolve(encodeMipmap(2));
        await flush();

        expect(singleMock.mock.calls.length).toBeGreaterThan(callsBefore);
        expect(waveformMipmapStore.hasLevel("/d.wav", 2)).toBe(true);
    });

    it("后端暂未就绪的冷却会自愈重试，不必等用户滚动/缩放", async () => {
        vi.useFakeTimers();
        try {
            // 第一次抢跑（早于后端分析完成）拿到空串，之后后端就绪。
            singleMock.mockResolvedValueOnce("").mockResolvedValue(encodeMipmap(2));

            expect(waveformMipmapStore.getPeaks("/f.wav", 2)).toBeNull();
            await vi.advanceTimersByTimeAsync(0);

            // 空结果：数据未到，且冷却期内不重复打后端。
            expect(waveformMipmapStore.hasLevel("/f.wav", 2)).toBe(false);
            expect(singleMock).toHaveBeenCalledTimes(1);

            // 冷却到点后**本模块自己**补发一次 —— 这是修复的核心：
            // 后端只在被调用时才 emit 进度事件，若前端干等 refresh()，
            // 波形就会停在空白上，直到用户滚动/缩放。
            await vi.advanceTimersByTimeAsync(RETRY_COOLDOWN_MS + 400);

            expect(singleMock.mock.calls.length).toBeGreaterThan(1);
            expect(waveformMipmapStore.hasLevel("/f.wav", 2)).toBe(true);
        } finally {
            vi.useRealTimers();
        }
    });

    it("未就绪重试有上限，真正缺失的文件不会被无限重试", async () => {
        vi.useFakeTimers();
        try {
            singleMock.mockResolvedValue("");
            waveformMipmapStore.getPeaks("/g.wav", 2);
            await vi.advanceTimersByTimeAsync((RETRY_COOLDOWN_MS + 400) * 10);

            // 首次 + 最多 NOT_READY_MAX_RETRY 次补发。
            expect(singleMock.mock.calls.length).toBeLessThanOrEqual(1 + NOT_READY_MAX_RETRY);
        } finally {
            vi.useRealTimers();
        }
    });

    it("clear() 仍然作废全部在途响应，已清空的缓存不得复活", async () => {
        const pending = deferred<string>();
        singleMock.mockReturnValue(pending.promise);

        expect(waveformMipmapStore.getPeaks("/e.wav", 2)).toBeNull();
        waveformMipmapStore.clear();

        pending.resolve(encodeMipmap(2));
        await flush();

        expect(waveformMipmapStore.hasLevel("/e.wav", 2)).toBe(false);
        expect(waveformMipmapStore.size).toBe(0);
    });
});
