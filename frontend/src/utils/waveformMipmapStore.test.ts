import { beforeEach, describe, expect, it, vi } from "vitest";

import { waveformApi } from "../services/api/waveform";

import {
    getDivisionFactors,
    getSppThresholds,
    waveformMipmapStore,
    type WaveformMipmapLevel,
} from "./waveformMipmapStore.js";

vi.mock("../services/api/waveform", () => ({
    waveformApi: {
        getWaveformMipmapBinary: vi.fn(),
        preloadWaveformMipmap: vi.fn(async () => ({ ok: true })),
        batchGetWaveformMipmap: vi.fn(async () => ({})),
        getRootMixWaveformPeaksSegment: vi.fn(),
        getTrackMixWaveformPeaksSegment: vi.fn(),
    },
}));

const singleMock = vi.mocked(waveformApi.getWaveformMipmapBinary);

it("utils/waveformMipmapStore.test.ts scripted checks", async () => {
    function assertEqual(actual: unknown, expected: unknown, label: string): void {
        const actualJson = JSON.stringify(actual);
        const expectedJson = JSON.stringify(expected);
        if (actualJson !== expectedJson) {
            throw new Error(`${label}: expected ${expectedJson}, received ${actualJson}`);
        }
    }

    assertEqual(getSppThresholds(), [512, 1024], "updated spp thresholds");
    assertEqual(getDivisionFactors(), [16, 512, 4096], "updated division factors");

    assertEqual(waveformMipmapStore.selectLevel(512), 0, "L0 covers spp <= 512");
    assertEqual(waveformMipmapStore.selectLevel(513), 1, "L1 starts above 512");
    assertEqual(waveformMipmapStore.selectLevel(1024), 1, "L1 covers spp <= 1024");
    assertEqual(waveformMipmapStore.selectLevel(1025), 2, "L2 starts above 1024");

    assertEqual(
        waveformMipmapStore.selectLevelStable(641, 0 as WaveformMipmapLevel),
        1,
        "stable selector enters L1 at updated hysteresis boundary",
    );
    assertEqual(
        waveformMipmapStore.selectLevelStable(1281, 1 as WaveformMipmapLevel),
        2,
        "stable selector enters L2 at updated hysteresis boundary",
    );
});

/**
 * 构造一份 v2 立体声（channels=2）mipmap 二进制载荷（Base64）。
 *
 * 两个声道用可区分的幅度（ch0 = ±0.5，ch1 = ±0.9），供双带判定断言
 * 「返回的两个平面确实是不同声道的数据」。
 */
function encodeStereoMipmapV2(level: number, peakCount = 4): string {
    const bytes = new Uint8Array(28 + peakCount * 4 * 4);
    bytes.set([0x57, 0x46, 0x50, 0x4b], 0); // "WFPK"
    const view = new DataView(bytes.buffer);
    view.setUint32(4, 2, true); // format_version = 2
    view.setUint32(8, 44100, true); // sample_rate
    view.setUint32(12, 4096, true); // division_factor
    view.setUint32(16, peakCount, true); // peak_count
    view.setUint32(20, level, true); // level
    view.setUint32(24, 2, true); // channels = 2
    const ch0Min = new Float32Array(bytes.buffer, 28, peakCount);
    const ch0Max = new Float32Array(bytes.buffer, 28 + peakCount * 4, peakCount);
    const ch1Min = new Float32Array(bytes.buffer, 28 + peakCount * 8, peakCount);
    const ch1Max = new Float32Array(bytes.buffer, 28 + peakCount * 12, peakCount);
    for (let i = 0; i < peakCount; i += 1) {
        ch0Min[i] = -0.5;
        ch0Max[i] = 0.5;
        ch1Min[i] = -0.9;
        ch1Max[i] = 0.9;
    }
    let binary = "";
    for (let i = 0; i < bytes.length; i += 1) binary += String.fromCharCode(bytes[i]);
    return btoa(binary);
}

/** 让所有已排队的微任务跑完。 */
function flush(): Promise<void> {
    return new Promise((resolve) => {
        setTimeout(resolve, 0);
    });
}

/**
 * 立体声双带兜底回归（P2）。
 *
 * 【为什么必须有它】`getBestSliceView` 的 sourceChannels 在生产路径上**不是
 * undefined 就是 0**（sceneBuilder 把缺失元数据强转为 0、工程加载把
 * `source_channels <= 0` 映射为 undefined），`sourceChannels ?? peaks.channels`
 * 拦不住 0 —— effectiveChannels 会把它当单声道，所有 L/R 双带渲染静默退化
 * 为合并单带。修复后必须先归一化（非正数回退峰值数据自身的 channels）。
 */
describe("getBestSliceView 立体声双带兜底", () => {
    beforeEach(() => {
        waveformMipmapStore.clear();
        vi.clearAllMocks();
        singleMock.mockResolvedValue(encodeStereoMipmapV2(2));
    });

    it("sourceChannels=0（元数据缺失的乐观 clip）+ 峰值数据 channels=2 → 双带", async () => {
        expect(waveformMipmapStore.getPeaks("/stereo.wav", 2)).toBeNull();
        await flush();

        const view = waveformMipmapStore.getBestSliceView("/stereo.wav", 2, 0, 1, 0, 0);
        expect(view?.channels).toBe(2);
        expect(view?.ch1Min).toBeDefined();
        expect(view?.ch1Max).toBeDefined();
        // 两个平面必须是不同声道的数据：ch0 = ±0.5，ch1 = ±0.9
        // （0.9 非二进制精确值，Float32 存储后用 fround 对齐）。
        expect(view?.min[0]).toBe(-0.5);
        expect(view?.ch1Min?.[0]).toBe(Math.fround(-0.9));
    });

    it("sourceChannels=undefined（工程加载未映射）+ 峰值数据 channels=2 → 双带", async () => {
        expect(waveformMipmapStore.getPeaks("/stereo.wav", 2)).toBeNull();
        await flush();

        const view = waveformMipmapStore.getBestSliceView("/stereo.wav", 2, 0, 1, 0, undefined);
        expect(view?.channels).toBe(2);
        expect(view?.ch1Min?.[0]).toBe(Math.fround(-0.9));
    });

    it("sourceChannels=1（真单声道源）→ 单带（不进双带分支）", async () => {
        expect(waveformMipmapStore.getPeaks("/stereo.wav", 2)).toBeNull();
        await flush();

        const view = waveformMipmapStore.getBestSliceView("/stereo.wav", 2, 0, 1, 0, 1);
        expect(view?.channels).toBe(1);
        expect(view?.ch1Min).toBeUndefined();
        expect(view?.ch1Max).toBeUndefined();
    });
});
