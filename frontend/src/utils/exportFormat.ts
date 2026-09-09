/*
 * 导出格式纯工具：扩展名联动与采样率合法性。
 * 与后端 crate::encode 的 with_format_extension / MP3_SAMPLE_RATES 语义一致。
 */

import type { ExportFormat } from "../services/api/core";

/** 已知可替换的音频扩展名（ASCII 大小写不敏感）。 */
const KNOWN_AUDIO_EXTENSIONS = ["wav", "mp3", "flac"] as const;

/** 全量导出采样率（WAV / FLAC 支持；与后端 normalize_export_sample_rate 一致）。 */
export const WAVE_SAMPLE_RATES = [
    8000, 11025, 12000, 16000, 22050, 32000, 44100, 48000, 88200, 96000, 176400, 192000,
] as const;

/** MP3（MPEG-1/2/2.5 Layer III）支持的采样率。 */
export const MP3_SAMPLE_RATES = [
    8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000,
] as const;

/** MP3 CBR 合法比特率（kbps，MPEG-1/2/2.5 并集）。 */
export const MP3_BITRATES = [
    8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 192, 224, 256, 320,
] as const;

/** rusty_mp3 的 VBR 平均码率映射表（与 vbr_quality_index 一致，仅用于展示）。 */
export const MP3_VBR_AVG_KBPS = [245, 225, 190, 175, 165, 130, 115, 100, 85, 65] as const;

/** FLAC 压缩级别范围（rusty_flac set_compression_level 0..=8）。 */
export const FLAC_COMPRESSION_RANGE = { min: 0, max: 8, default: 5 } as const;

export function sampleRateOptions(format: ExportFormat): number[] {
    return format === "mp3" ? [...MP3_SAMPLE_RATES] : [...WAVE_SAMPLE_RATES];
}

export function isSampleRateAllowed(format: ExportFormat, rate: number): boolean {
    return sampleRateOptions(format).includes(rate);
}

/**
 * 返回当前采样率在目标格式下的最近合法档位；已合法则返回 null（无需纠正）。
 * 纠正策略：对高于表内档位的值逐级减半（88.2k→44.1k、96k→48k、192k→48k），
 * 若仍不在表内则取对数距离最近的一档。
 */
export function nearestAllowedSampleRate(
    format: ExportFormat,
    rate: number,
): number | null {
    const options = sampleRateOptions(format);
    if (!Number.isFinite(rate) || rate <= 0 || options.includes(rate)) return null;

    let candidate = Math.round(rate);
    for (let i = 0; i < 8 && !options.includes(candidate); i += 1) {
        candidate = Math.round(candidate / 2);
    }
    if (options.includes(candidate)) return candidate;

    return options.reduce((best, value) =>
        Math.abs(Math.log2(value / rate)) < Math.abs(Math.log2(best / rate)) ? value : best,
    options[0]);
}

/**
 * 把文件名 / 命名模板末段的音频扩展名替换为目标格式；
 * 已有 wav/mp3/flac 扩展名 → 原地替换，其余情况 → 追加。
 */
export function applyExtensionToFileName(name: string, format: ExportFormat): string {
    const lower = name.toLowerCase();
    for (const ext of KNOWN_AUDIO_EXTENSIONS) {
        const suffix = `.${ext}`;
        if (lower.length > suffix.length && lower.endsWith(suffix)) {
            return `${name.slice(0, name.length - suffix.length)}.${format}`;
        }
    }
    return `${name}.${format}`;
}
