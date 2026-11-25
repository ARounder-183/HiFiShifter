import type { WaveformPeaksSegmentPayload } from "../../types/api";

import { invoke } from "../invoke";

/**
 * 批量 mipmap 里**一条源**的结果。
 *
 * 【为什么是三条状态而不是"有数据/没数据"】"还没就绪"必须能被重试（新导入的源在宿主
 * 授权之前、峰值还在算），"被拒绝/读不出来"是终态，重试只会把同一个错误刷满日志。
 * 两者混成一个"没数据"，前端就只能二选一：要么永远不重试（波形永久空白），要么对
 * 每个坏源无限重试。
 */
export interface BatchWaveformMipmapEntry {
    path: string;
    ok: boolean;
    /** 尚未就绪（瞬时）：`ok=false` 且 `error` 为空。 */
    pending: boolean;
    /** `[L0_base64, L1_base64, L2_base64]`；未就绪/失败为 `null`。 */
    data: [string, string, string] | null;
    /** 终态失败的原因；未就绪与成功都为 `null`。 */
    error: string | null;
}

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
     *
     * 返回**逐源结果数组**（`BatchWaveformMipmapEntry`），顺序与 `sourcePaths` 一致。
     * 一条源出错只影响它自己那条 —— 整批失败会让同批其余文件退回逐个重试，
     * 那正是"一个坏源拖垮整批"的老缺陷。
     *
     * `levels` 为需要传输的级别白名单（缺省 = 全部三级）。批量预载只落地 L2，
     * 传 `[2]` 可避免后端把 L0（单级 ≈ 159MB/小时素材）也编码传输后前端丢弃；
     * 未请求的级别在 `data` 中为空字符串。
     */
    batchGetWaveformMipmap: (sourcePaths: string[], levels?: number[]) =>
        invoke<BatchWaveformMipmapEntry[]>("batch_get_waveform_mipmap", sourcePaths, levels),

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
