import type { WaveformPeaksSegmentPayload } from "../../types/api";

import { invoke } from "../invoke";

export const waveformApi = {
    // ============== Mipmap 二进制 API ==============

    /** 获取指定级别的 mipmap 数据（二进制格式，返回 number[] 需转 ArrayBuffer） */
    getWaveformMipmapBinary: (sourcePath: string, level: number) =>
        invoke<number[]>("get_waveform_mipmap_binary", sourcePath, level),

    /** 预加载所有级别的 mipmap 数据（音频加载时调用） */
    preloadWaveformMipmap: (sourcePath: string) =>
        invoke<{ ok: boolean; error?: string }>("preload_waveform_mipmap", sourcePath),

    // ============== Mix 波形 API ==============

    getRootMixWaveformPeaksSegment: (
        trackId: string,
        startSec: number,
        durationSec: number,
        columns: number,
    ) =>
        invoke<WaveformPeaksSegmentPayload>(
            "get_root_mix_waveform_peaks_segment",
            trackId,
            startSec,
            durationSec,
            columns,
        ),

    getTrackMixWaveformPeaksSegment: (
        trackId: string,
        startSec: number,
        durationSec: number,
        columns: number,
    ) =>
        invoke<WaveformPeaksSegmentPayload>(
            "get_track_mix_waveform_peaks_segment",
            trackId,
            startSec,
            durationSec,
            columns,
        ),
};
