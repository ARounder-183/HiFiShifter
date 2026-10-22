export interface ClipNormalizationInput {
    sourcePath?: string;
    durationSec?: number;
    lengthSec: number;
    sourceStartSec?: number;
    sourceEndSec?: number;
    playbackRate?: number;
}

type InterleavedSlice = {
    interleaved: ArrayLike<number>;
};

/**
 * 归一化增益的钳制区间（±12 dB）。
 *
 * 【⚠ 这不是 Clip 增益的值域】Clip / 轨道增益的值域是 `utils/gainDb`（−inf ~ +24.08 dB，
 * 与 REAPER 的 item 增益上限对齐）。这里的 ±12 dB 是**另一个语义**：它是"把峰值提到
 * 0 dBFS"这条自动增益的**上限策略** —— 一段 −40 dBFS 的素材若被放大到 +24 dB，
 * 输出必然削顶，那不是用户要的"归一化"。
 *
 * 因此两处**不该**被一起改。上一次值域调整时这里被顺手改过一版，结果是归一化开始
 * 把安静的素材推爆。
 */
const MIN_NORMALIZED_GAIN = dbToGain(-12);
const MAX_NORMALIZED_GAIN = dbToGain(12);

function dbToGain(db: number): number {
    return 10 ** (db / 20);
}

export function computeNormalizationGainFromInterleaved(data: ArrayLike<number>): number | null {
    let peak = 0;
    for (let i = 0; i < data.length; i += 1) {
        const value = Math.abs(Number(data[i]) || 0);
        if (value > peak) peak = value;
    }
    if (peak <= 0) return null;
    return Math.min(Math.max(1 / peak, MIN_NORMALIZED_GAIN), MAX_NORMALIZED_GAIN);
}

export function computeClipNormalizationGain(
    clip: ClipNormalizationInput,
    deps: {
        getInterleavedSlice: (
            sourcePath: string,
            channel: number,
            sourceStartSec: number,
            sourceSpanSec: number,
        ) => InterleavedSlice | null;
        releaseInterleaved: (data: ArrayLike<number>) => void;
    },
): number | null {
    if (!clip.sourcePath || !clip.durationSec || clip.durationSec <= 0) {
        return null;
    }

    const sourceStartSec = Number(clip.sourceStartSec ?? 0) || 0;
    const sourceEndSec = Number(clip.sourceEndSec ?? clip.durationSec) || clip.durationSec;
    const playbackRate = Math.max(1e-6, Number(clip.playbackRate ?? 1) || 1);
    const clipSourceSpanSec = Math.max(
        0,
        Math.min(clip.lengthSec * playbackRate, sourceEndSec - sourceStartSec),
    );
    if (clipSourceSpanSec <= 0) {
        return null;
    }

    const slice = deps.getInterleavedSlice(clip.sourcePath, 0, sourceStartSec, clipSourceSpanSec);
    if (!slice || slice.interleaved.length < 2) {
        return null;
    }

    try {
        return computeNormalizationGainFromInterleaved(slice.interleaved);
    } finally {
        deps.releaseInterleaved(slice.interleaved);
    }
}
