import type { WaveformPeaksSegmentPayload } from "../../types/api";

import { invoke } from "../invoke";

export const waveformApi = {
    // ============== Mipmap 二进制 API ==============

    /** 获取指定级别的 mipmap 数据（Base64 编码的二进制格式） */
    getWaveformMipmapBinary: (sourcePath: string, level: number) =>
        invoke<string>("get_waveform_mipmap_binary", sourcePath, level),

    /** 预加载所有级别的 mipmap 数据（音频加载时调用） */
    preloadWaveformMipmap: (sourcePath: string) =>
        invoke<{ ok: boolean; error?: string }>("preload_waveform_mipmap", sourcePath),

    /**
     * 批量获取多个文件的 mipmap 数据（Base64 编码）
     *
     * 将 N×3 次 IPC 合并为 1 次，大幅减少 IPC 往返开销。
     * 返回 Record<sourcePath, [L0_base64, L1_base64, L2_base64]>。
     *
     * `levels` 为需要传输的级别白名单（缺省 = 全部三级）。批量预载只落地 L2，
     * 传 `[2]` 可避免后端把 L0（单级 ≈ 159MB/小时素材）也编码传输后前端丢弃；
     * 未请求的级别在返回中为空字符串。
     */
    batchGetWaveformMipmap: (sourcePaths: string[], levels?: number[]) =>
        invoke<Record<string, [string, string, string]>>(
            "batch_get_waveform_mipmap",
            sourcePaths,
            levels,
        ),

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
