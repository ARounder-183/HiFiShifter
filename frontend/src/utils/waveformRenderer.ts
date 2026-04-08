/**
 * 波形渲染工具模块（Canvas per-pixel 渲染）
 *
 * 本模块负责将已降采样的 peaks 数据绘制到 Canvas 上，是波形可视化的最后一环。
 *
 * ## 数据流
 *   waveformMipmapStore（降采样 / resample）
 *     → applyGainsToPeaks（叠加音量 + 淡入淡出增益）
 *       → renderWaveform（Canvas per-pixel 绘制）
 *
 * ## 坐标系映射链
 *   canvas 本地像素 → clip 全局像素 → timeline 时间 → 源文件时间 → peaks 数据索引
 *
 * ## 导出
 * - {@link WaveformRenderParams} — 渲染参数接口
 * - {@link applyGainsToPeaks}    — 增益应用（音量 × 淡入淡出曲线）
 * - {@link renderWaveform}       — Canvas 绘制（line 竖线 / jitter 抖动线）
 *
 * @module waveformRenderer
 */

import type { FadeCurveType } from "../components/layout/timeline/paths";
import { fadeCurveGain } from "../components/layout/timeline/paths";

// ============================================================================
// 类型定义
// ============================================================================

/**
 * 波形渲染参数
 *
 * 包含三类信息：
 * 1. Canvas 物理尺寸（canvasWidth / canvasHeight / centerY）
 * 2. 时间域参数（sourceStartSec / clipDuration / playbackRate / fade 等）
 * 3. 可视区裁剪参数（clipPixelOffset / clipTotalWidthPx，由调用方传入）
 */
export interface WaveformRenderParams {
    /** canvas 宽度（像素） */
    canvasWidth: number;
    /** canvas 高度（像素） */
    canvasHeight: number;
    /** 波形中心 Y 坐标 */
    centerY: number;
    /** clip 在源文件中的起始位置（秒） */
    sourceStartSec: number;
    /** clip 时长（秒，考虑 playbackRate 后） */
    clipDuration: number;
    /** 播放速度 */
    playbackRate: number;
    /** 是否倒放 */
    reversed?: boolean;
    /** 源文件时长（秒） */
    sourceDurationSec: number;
    /** clip 音量增益（线性值，0~2） */
    volumeGain: number;
    /** 0dB 对应的半高（像素），未传时默认使用 canvasHeight / 2 */
    zeroDbHalfHeight?: number;
    /** 淡入时长（秒） */
    fadeInSec: number;
    /** 淡出时长（秒） */
    fadeOutSec: number;
    /** 淡入曲线类型 */
    fadeInCurve: FadeCurveType;
    /** 淡出曲线类型 */
    fadeOutCurve: FadeCurveType;
    /** 数据起始时间（秒，源文件坐标系） */
    dataStartSec?: number;
    /** 数据持续时间（秒） */
    dataDurationSec?: number;

    // ========================================
    // 可视区裁剪参数（由调用方传入）
    // ========================================
    /** 当前 canvas 在 clip 内的像素偏移量（canvas 左边缘对应 clip 的第几个像素） */
    clipPixelOffset?: number;
    /** clip 完整像素宽度（用于将像素位置映射到 timeline 时间） */
    clipTotalWidthPx?: number;
}

// ============================================================================
// applyGainsToPeaks — 增益应用（带 buffer 复用池）
// ============================================================================

/**
 * applyGainsToPeaks 内部复用缓冲池
 * 避免每帧 new Float32Array 导致 GC 压力
 */
let _gainBufferPool: Float32Array[] = [];
const _GAIN_POOL_MAX = 4;

function acquireGainBuffer(len: number): Float32Array {
    for (let i = 0; i < _gainBufferPool.length; i++) {
        if (_gainBufferPool[i].length === len) {
            return _gainBufferPool.splice(i, 1)[0];
        }
    }
    return new Float32Array(len);
}

/** 归还增益 buffer 到池中 */
export function releaseGainBuffer(buf: Float32Array): void {
    if (buf.length > 0 && _gainBufferPool.length < _GAIN_POOL_MAX) {
        _gainBufferPool.push(buf);
    }
}

/**
 * 将音量增益和淡入淡出曲线应用到波形 peaks 数据上
 *
 * 对每个采样点：
 *   1. 根据 position 线性插值算出该点对应的 **源文件时间**
 *   2. 将源文件时间映射为 **timeline 时间**（÷ playbackRate）
 *   3. 根据 timeline 时间判断是否处于淡入/淡出区间，计算淡入淡出增益
 *   4. 最终增益 = volumeGain × fadeGain，同时乘到 min 和 max 上
 *
 * 快速路径：若无淡入淡出且 volumeGain ≈ 1，直接返回原数组（零拷贝）。
 *
 * @param peaks  - Float32Array，交错格式 [min0, max0, min1, max1, ...]
 * @param params - 渲染参数（需要时间域 + fade 相关字段）
 * @returns Float32Array，与 peaks 等长，已叠加增益（可能是原数组引用）
 */
export function applyGainsToPeaks(peaks: Float32Array, params: WaveformRenderParams): Float32Array {
    const {
        sourceStartSec,
        clipDuration,
        playbackRate,
        reversed = false,
        volumeGain,
        fadeInSec,
        fadeOutSec,
        fadeInCurve,
        fadeOutCurve,
        dataStartSec,
        dataDurationSec,
    } = params;

    const totalSamples = peaks.length / 2;

    // 计算数据的时间范围（与 renderWaveform 保持一致）
    const effectiveDataStartSec = dataStartSec ?? sourceStartSec;
    const effectiveDataDurationSec = dataDurationSec ?? clipDuration * playbackRate;

    // 快速路径：无淡入淡出且增益为 1 时直接返回原数组（零拷贝）
    const hasFade = fadeInSec > 0 || fadeOutSec > 0;
    if (!hasFade && Math.abs(volumeGain - 1) < 1e-6) {
        return peaks;
    }

    const result = acquireGainBuffer(peaks.length);

    // 仅有音量增益、无淡入淡出时的快速路径
    if (!hasFade) {
        for (let i = 0, len = peaks.length; i < len; i++) {
            result[i] = peaks[i] * volumeGain;
        }
        return result;
    }

    // 预计算常数
    const invTotalSamplesM1 = totalSamples > 1 ? 1 / (totalSamples - 1) : 0;
    const invPlaybackRate = 1 / playbackRate;
    const clipSourceEndSec = sourceStartSec + clipDuration * playbackRate;
    const fadeOutStart = clipDuration - fadeOutSec;
    const invFadeInSec = fadeInSec > 0 ? 1 / fadeInSec : 0;
    const invFadeOutSec = fadeOutSec > 0 ? 1 / fadeOutSec : 0;

    // 提取时间映射的线性步长与基准值
    const baseTime = reversed
        ? (clipSourceEndSec - effectiveDataStartSec) * invPlaybackRate
        : (effectiveDataStartSec - sourceStartSec) * invPlaybackRate;
    const timeStep =
        totalSamples > 1
            ? reversed
                ? -(effectiveDataDurationSec * invTotalSamplesM1 * invPlaybackRate)
                : effectiveDataDurationSec * invTotalSamplesM1 * invPlaybackRate
            : 0;

    for (let i = 0; i < totalSamples; i++) {
        // 使用预计算的等差数列求解 time
        const time = baseTime + i * timeStep;

        // 计算综合增益
        let gain = volumeGain;

        // 淡入：时间 0 -> fadeInSec，增益 0 -> 1
        if (fadeInSec > 0 && time < fadeInSec) {
            gain *= fadeCurveGain(time * invFadeInSec, fadeInCurve);
        }

        // 淡出：时间 (clipDuration - fadeOutSec) -> clipDuration，增益 1 -> 0
        if (fadeOutSec > 0 && time > fadeOutStart) {
            gain *= 1 - fadeCurveGain((time - fadeOutStart) * invFadeOutSec, fadeOutCurve);
        }

        // 应用增益
        result[i * 2] = peaks[i * 2] * gain;
        result[i * 2 + 1] = peaks[i * 2 + 1] * gain;
    }

    return result;
}

// ============================================================================
// renderWaveform — Canvas per-pixel 绘制（预计算常数优化版）
// ============================================================================

/**
 * 将 interleaved peaks 数组按采样点倒序。
 *
 * 输入/输出格式均为 [min0, max0, min1, max1, ...]。
 */
export function reverseInterleavedPeaks(peaks: Float32Array): Float32Array {
    const n = Math.floor(peaks.length / 2);
    if (n <= 1) return peaks;

    const out = new Float32Array(peaks.length);
    for (let i = 0; i < n; i++) {
        const src = (n - 1 - i) * 2;
        const dst = i * 2;
        out[dst] = peaks[src];
        out[dst + 1] = peaks[src + 1];
    }
    return out;
}

/**
 * 将 peaks 数据绘制到 Canvas 上（per-pixel 模式）
 *
 * ## 渲染模式
 * - **line**（默认）：per-pixel min/max 竖线 —— DAW 标准做法，每像素列一条从 yTop 到 yBot 的竖线
 * - **jitter**：抖动线 —— 偶数列取包络 25% 位置、奇数列取 75% 位置，连成折线，视觉更平滑
 *
 * ## 核心流程
 * 1. **可视区裁剪**：根据 clipPixelOffset / clipTotalWidthPx 确定 canvas 与 clip 的映射关系，
 *    再与数据的时间范围求交集，只遍历有数据覆盖的像素列
 * 2. **像素→时间→索引**：每个像素列 px 覆盖 [px-0.5, px+0.5) 的时间段，
 *    通过预计算的线性系数直接映射到 peaks 数据索引范围
 * 3. **滑动指针扫描**：cursor 只前进不后退，保证整体 O(W + N) 复杂度
 * 4. **绘制**：line 模式 moveTo/lineTo 竖线；jitter 模式连续 lineTo 折线
 *
 * ## 性能特性
 * - 预计算 pxToIndex 线性系数，避免 per-pixel 闭包调用
 * - 数据密度高时自动聚合（多采样点 → 一像素取 min/max）
 * - 数据密度低时优雅降级（相邻像素复用同一采样点）
 * - 静音段保证最小 0.5px 可见高度
 *
 * @param ctx         - Canvas 2D 上下文
 * @param peaks       - Float32Array，交错格式 [min0, max0, min1, max1, ...]
 * @param params      - 渲染参数（含 canvas 尺寸 + 时间域 + 裁剪信息）
 * @param strokeColor - 描边颜色（默认 "currentColor"）
 * @param strokeWidth - 描边宽度（默认 1）
 * @param mode        - 渲染模式："line"（竖线）或 "jitter"（抖动线），默认 "line"
 */
export function renderWaveform(
    ctx: CanvasRenderingContext2D,
    peaks: Float32Array,
    params: WaveformRenderParams,
    strokeColor: string = "currentColor",
    strokeWidth: number = 1,
    mode: "line" | "jitter" = "line",
): void {
    const {
        canvasWidth,
        canvasHeight,
        centerY,
        sourceStartSec,
        clipDuration,
        playbackRate,
        reversed = false,
        dataStartSec,
        dataDurationSec,
        clipPixelOffset = 0,
        clipTotalWidthPx,
    } = params;
    const totalSamples = peaks.length / 2;

    if (totalSamples < 2 || canvasWidth < 1) return;

    // ========================================
    // 可视区裁剪核心逻辑
    // ========================================
    const clipTotalW = clipTotalWidthPx ?? canvasWidth;

    // 振幅比例：0 电平在中心（静音），±1 电平顶到底边界（0dB）
    const amplitudeScale = params.zeroDbHalfHeight ?? canvasHeight / 2;

    // 计算数据的时间范围（源文件坐标系）
    const effectiveDataStartSec = dataStartSec ?? sourceStartSec;
    const effectiveDataDurationSec = dataDurationSec ?? clipDuration * playbackRate;
    const dataEndSec = effectiveDataStartSec + effectiveDataDurationSec;

    // 计算 clip 在源文件中的时间范围
    const clipSourceStartSec = sourceStartSec;
    const clipSourceEndSec = sourceStartSec + clipDuration * playbackRate;

    // 计算数据与 clip 的重叠范围（在源文件坐标系中）
    const overlapStartSec = Math.max(effectiveDataStartSec, clipSourceStartSec);
    const overlapEndSec = Math.min(dataEndSec, clipSourceEndSec);

    // 如果没有重叠，不渲染
    if (overlapEndSec <= overlapStartSec) {
        return;
    }

    // 重叠范围映射到 timeline 时间
    const invPlaybackRate = 1 / playbackRate;
    const timelineOverlapStart = reversed
        ? (clipSourceEndSec - overlapEndSec) * invPlaybackRate
        : (overlapStartSec - sourceStartSec) * invPlaybackRate;
    const timelineOverlapEnd = reversed
        ? (clipSourceEndSec - overlapStartSec) * invPlaybackRate
        : (overlapEndSec - sourceStartSec) * invPlaybackRate;

    // 重叠范围映射到 clip 全局像素
    const invClipDuration = 1 / clipDuration;
    const globalPxStart = timelineOverlapStart * invClipDuration * clipTotalW;
    const globalPxEnd = timelineOverlapEnd * invClipDuration * clipTotalW;

    // 裁剪到当前 canvas 范围 [0, canvasWidth)
    const localPxStart = Math.max(0, Math.floor(globalPxStart - clipPixelOffset));
    const localPxEnd = Math.min(canvasWidth - 1, Math.ceil(globalPxEnd - clipPixelOffset) - 1);

    if (localPxEnd <= localPxStart) return;

    // ========================================
    // 预计算 pxToIndex 线性系数（消除 per-pixel 闭包调用）
    // ========================================
    // pxToSourceTime(localPx) = sourceStartSec + (localPx + clipPixelOffset) / clipTotalW * clipDuration * playbackRate
    //                         = sourceStartSec + (localPx + clipPixelOffset) * pxToTimeScale
    // timeToIndex(srcTime)    = (srcTime - effectiveDataStartSec) / effectiveDataDurationSec * (totalSamples - 1)
    //                         = (srcTime - effectiveDataStartSec) * timeToIdxScale
    //
    // 合并：pxToIndex(localPx) = ((localPx + clipPixelOffset) * pxToTimeScale + sourceStartSec - effectiveDataStartSec) * timeToIdxScale
    //                          = localPx * pxToIdxScale + pxToIdxBase
    const pxToTimeScale = (clipDuration * playbackRate) / clipTotalW;
    const invDataDuration = 1 / effectiveDataDurationSec;
    const timeToIdxScale = (totalSamples - 1) * invDataDuration;
    const pxToIdxScale = (reversed ? -1 : 1) * pxToTimeScale * timeToIdxScale;
    const pxToIdxBase = reversed
        ? (clipSourceEndSec - clipPixelOffset * pxToTimeScale - effectiveDataStartSec) *
          timeToIdxScale
        : (clipPixelOffset * pxToTimeScale + sourceStartSec - effectiveDataStartSec) *
          timeToIdxScale;
    // halfPixelIdx 对应 0.5 像素覆盖的索引偏移量
    const halfPixelIdx = Math.abs(0.5 * pxToIdxScale);

    // 数据边界（预计算，避免循环内重复调用 timeToIndex / Math.min/max）
    const idxAtDataEnd = (dataEndSec - effectiveDataStartSec) * timeToIdxScale;
    const maxIdx = totalSamples - 1;

    // 设置绘制样式
    ctx.strokeStyle = strokeColor;
    ctx.fillStyle = strokeColor;
    ctx.lineWidth = strokeWidth;
    ctx.lineJoin = "round";
    ctx.lineCap = mode === "jitter" ? "round" : "butt";

    // ========================================
    // 滑动指针优化：O(W + N) 复杂度
    // ========================================
    // 仅在连续折线模式下才开启矢量 Path
    if (mode === "jitter") {
        ctx.beginPath();
    }

    let jitterStarted = false;

    for (let px = localPxStart; px <= localPxEnd; px++) {
        // 用预计算系数计算该像素列覆盖的索引范围
        const centerIdx = px * pxToIdxScale + pxToIdxBase;
        const rawIdxLeft = centerIdx - halfPixelIdx;
        const rawIdxRight = centerIdx + halfPixelIdx;

        // 裁剪到有效数据范围
        const idxLeft = rawIdxLeft < 0 ? 0 : rawIdxLeft;
        const idxRight =
            rawIdxRight > maxIdx ? maxIdx : rawIdxRight > idxAtDataEnd ? idxAtDataEnd : rawIdxRight;

        // 取该范围内所有采样点的 min/max
        const iStart = idxLeft < 0 ? 0 : idxLeft | 0;
        const iEnd = idxRight > maxIdx ? maxIdx : Math.ceil(idxRight);

        let pixelMin = Infinity;
        let pixelMax = -Infinity;

        // 直接干脆地遍历属于该像素的数据段
        for (let i = iStart; i <= iEnd; i++) {
            const idx2 = i * 2;
            const sMin = peaks[idx2];
            const sMax = peaks[idx2 + 1];
            if (sMin < pixelMin) pixelMin = sMin;
            if (sMax > pixelMax) pixelMax = sMax;
        }

        if (pixelMin === Infinity) continue;

        if (mode === "jitter") {
            // ========================================
            // 抖动线模式
            // ========================================
            const t = px & 1 ? 0.75 : 0.25;
            const value = pixelMax + (pixelMin - pixelMax) * t;
            const y = centerY - value * amplitudeScale;

            if (!jitterStarted) {
                ctx.moveTo(px, y);
                jitterStarted = true;
            } else {
                ctx.lineTo(px, y);
            }
        } else {
            const yTop = centerY - pixelMax * amplitudeScale;
            const yBot = centerY - pixelMin * amplitudeScale;

            // 确保静音段至少有最小可见高度（0.5px）
            if (yBot - yTop < 0.5) {
                const midY = (yTop + yBot) * 0.5;
                // 用 px - strokeWidth / 2 确保矩形的中心点与原本线条的中轴完美对齐
                ctx.fillRect(px - strokeWidth / 2, midY - 0.25, strokeWidth, 0.5);
            } else {
                ctx.fillRect(px - strokeWidth / 2, yTop, strokeWidth, yBot - yTop);
            }
        }
    }

    if (mode === "jitter") {
        ctx.stroke();
    }
}
