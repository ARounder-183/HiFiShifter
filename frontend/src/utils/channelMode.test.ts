import { describe, expect, it } from "vitest";

import {
    channelModeI18nKey,
    channelModeShortLabel,
    effectiveChannels,
    nextChannelMode,
    normalizeChannelMode,
} from "./channelMode";
import {
    base64ToArrayBuffer,
    decodeWaveformBinary,
} from "./waveformBinaryCodec";

function encodeWfpkV2(channels: number, peakCount: number, blocks: Float32Array[]): string {
    const header = new ArrayBuffer(28);
    const view = new DataView(header);
    // magic "WFPK"
    view.setUint8(0, 0x57);
    view.setUint8(1, 0x46);
    view.setUint8(2, 0x50);
    view.setUint8(3, 0x4b);
    view.setUint32(4, 2, true); // format_version
    view.setUint32(8, 44100, true); // sample_rate
    view.setUint32(12, 16, true); // division_factor
    view.setUint32(16, peakCount, true);
    view.setUint32(20, 0, true); // level
    view.setUint32(24, channels, true);

    const dataSize = blocks.reduce((sum, b) => sum + b.byteLength, 0);
    const out = new Uint8Array(header.byteLength + dataSize);
    out.set(new Uint8Array(header), 0);
    let offset = header.byteLength;
    for (const block of blocks) {
        out.set(new Uint8Array(block.buffer, block.byteOffset, block.byteLength), offset);
        offset += block.byteLength;
    }
    let binary = "";
    const bytes = out;
    for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
    return btoa(binary);
}

describe("channelMode utils", () => {
    it("normalizes unknown raw values to Normal", () => {
        expect(normalizeChannelMode(0)).toBe(0);
        expect(normalizeChannelMode(4)).toBe(4);
        expect(normalizeChannelMode(-1)).toBe(0);
        expect(normalizeChannelMode(5)).toBe(0);
        expect(normalizeChannelMode(null)).toBe(0);
        expect(normalizeChannelMode(undefined)).toBe(0);
        expect(normalizeChannelMode(Number.NaN)).toBe(0);
    });

    it("computes effective channels per mode × source table", () => {
        expect(effectiveChannels(2, 0)).toBe(2);
        expect(effectiveChannels(2, 1)).toBe(2);
        expect(effectiveChannels(2, 2)).toBe(1);
        expect(effectiveChannels(2, 3)).toBe(1);
        expect(effectiveChannels(2, 4)).toBe(1);
        expect(effectiveChannels(1, 0)).toBe(1);
        expect(effectiveChannels(6, 0)).toBe(2);
        expect(effectiveChannels(undefined, 0)).toBe(1);
        expect(effectiveChannels(0, 0)).toBe(1);
    });

    it("cycles modes in REAPER CHANMODE order", () => {
        expect(nextChannelMode(0)).toBe(1);
        expect(nextChannelMode(1)).toBe(2);
        expect(nextChannelMode(2)).toBe(3);
        expect(nextChannelMode(3)).toBe(4);
        expect(nextChannelMode(4)).toBe(0);
        expect(nextChannelMode(99)).toBe(1); // 未知名先归一为 0，再循环
    });

    it("maps modes to i18n keys and stable short labels", () => {
        expect(channelModeI18nKey(0)).toBe("clip_channel_mode_normal");
        expect(channelModeI18nKey(1)).toBe("clip_channel_mode_swap");
        expect(channelModeI18nKey(2)).toBe("clip_channel_mode_mono_mix");
        expect(channelModeI18nKey(3)).toBe("clip_channel_mode_mono_left");
        expect(channelModeI18nKey(4)).toBe("clip_channel_mode_mono_right");
        expect(channelModeShortLabel(0)).toBe("L·R");
        expect(channelModeShortLabel(1)).toBe("⇄");
        expect(channelModeShortLabel(2)).toBe("MIX");
        expect(channelModeShortLabel(3)).toBe("L");
        expect(channelModeShortLabel(4)).toBe("R");
    });
});

describe("waveformBinaryCodec v2", () => {
    it("decodes v2 stereo payload into per-channel views and merged envelope", () => {
        // ch0: min [-1, -0.5] max [1, 0.5]; ch1: min [-0.2, 0] max [0.2, 0.1]
        const ch0Min = new Float32Array([-1, -0.5]);
        const ch0Max = new Float32Array([1, 0.5]);
        const ch1Min = new Float32Array([-0.2, 0]);
        const ch1Max = new Float32Array([0.2, 0.1]);
        const base64 = encodeWfpkV2(2, 2, [ch0Min, ch0Max, ch1Min, ch1Max]);
        const decoded = decodeWaveformBinary(base64ToArrayBuffer(base64));
        expect(decoded).not.toBeNull();
        expect(decoded!.channels).toBe(2);
        expect(decoded!.peakCount).toBe(2);
        expect(decoded!.ch0Min).toBeInstanceOf(Float32Array);
        expect(Array.from(decoded!.ch0Min)).toEqual(expect.arrayContaining([expect.closeTo(-1), expect.closeTo(-0.5)]));
        expect(Array.from(decoded!.ch1Min)).toEqual([
            expect.closeTo(-0.2),
            0,
        ]);
        // 包络取各声道极值的合并
        expect(Array.from(decoded!.min)).toEqual([-1, -0.5]);
        expect(Array.from(decoded!.max)).toEqual([expect.closeTo(1), expect.closeTo(0.5)]);
    });

    it("decodes v2 mono payload with identical envelope and channel views", () => {
        const min = new Float32Array([-0.3]);
        const max = new Float32Array([0.7]);
        const base64 = encodeWfpkV2(1, 1, [min, max]);
        const decoded = decodeWaveformBinary(base64ToArrayBuffer(base64));
        expect(decoded).not.toBeNull();
        expect(decoded!.channels).toBe(1);
        expect(decoded!.min[0]).toBeCloseTo(-0.3);
        expect(decoded!.ch1Min[0]).toBeCloseTo(-0.3);
    });

    it("decodes legacy v1 payload as single-channel", () => {
        // v1 头 20B：magic + sample_rate + division + peak_count + level，无 format_version。
        const buffer = new ArrayBuffer(20 + 2 * 4 * 2);
        const view = new DataView(buffer);
        view.setUint8(0, 0x57);
        view.setUint8(1, 0x46);
        view.setUint8(2, 0x50);
        view.setUint8(3, 0x4b);
        view.setUint32(4, 48000, true);
        view.setUint32(8, 512, true);
        view.setUint32(12, 2, true);
        view.setUint32(16, 1, true);
        const f32 = new Float32Array(buffer, 20, 4);
        f32.set([-0.9, -0.4, 0.9, 0.4]);
        let binary = "";
        const bytes = new Uint8Array(buffer);
        for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
        const decoded = decodeWaveformBinary(buffer);
        expect(decoded).not.toBeNull();
        expect(decoded!.channels).toBe(1);
        expect(decoded!.sampleRate).toBe(48000);
        expect(decoded!.min[0]).toBeCloseTo(-0.9);
        expect(decoded!.min[1]).toBeCloseTo(-0.4);
        expect(decoded!.max[0]).toBeCloseTo(0.9);
        expect(decoded!.max[1]).toBeCloseTo(0.4);
        void f32;
    });
});
