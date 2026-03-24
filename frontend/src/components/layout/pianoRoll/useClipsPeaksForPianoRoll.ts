/**
 * Piano Roll Per-Clip 波形 Peaks Hook (Mipmap 优化版)
 *
 * 使用 Mipmap 多级缓存机制，根据缩放级别自动选择最佳峰值分辨率
 * 替代原来对单个track mix后取波形的方式
 * 每个可见 clip 独立获取 peaks，按时间位置叠加绘制
 */
import { useEffect, useMemo, useRef, useState } from "react";

import type { ClipInfo, FadeCurveType } from "../../../features/session/sessionTypes";
import { lruGet, lruSet } from "./peaksCache";
import { waveformMipmapStore } from "../../../utils/waveformMipmapStore";

/** 单个 clip 的 peaks 数据条目 */
/** 单个 clip 的 peaks 数据条目 */
export interface ClipPeaksEntry {
    /** clip ID */
    clipId: string;
    /** clip 在 timeline 上的起始位置（秒，来自 ClipInfo.startSec） */
    startSec: number;
    /** clip 的长度（秒，来自 ClipInfo.lengthSec），用于绘制宽度，不影响 peaks 请求 */
    lengthSec: number;
    /** clip 的 sourceStartSec（秒），渲染时用于计算波形偏移 */
    sourceStartSec: number;
    /** source 文件总时长（秒），peaks 覆盖整个 source */
    sourceDurationSec: number;
    /** 播放速率 */
    playbackRate: number;
    /** clip 增益（线性值，0~4） */
    gain: number;
    /** 淡入时长（秒） */
    fadeInSec: number;
    /** 淡出时长（秒） */
    fadeOutSec: number;
    /** 淡入曲线类型 */
    fadeInCurve: FadeCurveType;
    /** 淡出曲线类型 */
    fadeOutCurve: FadeCurveType;
    peaks: {
        min: number[];
        max: number[];
        startSec: number;
        durSec: number;
        /** peaks 实际列数，用于绘制时避免拉伸 */
        columns: number;
    } | null;
}

/** 缓存条目 */
type CachedEntry = {
    min: number[];
    max: number[];
    startSec: number;
    durSec: number;
    t: number;
};

// 模块级缓存，各hook实例共享
const clipPeaksCache = new Map<string, CachedEntry>();
const clipPeaksInflight = new Map<string, Promise<CachedEntry | null>>();
const CLIP_PEAKS_CACHE_LIMIT = 128;

/** 量化秒数，减少重复请�?*/
function qsec(x: number, step = 0.005): number {
    if (!Number.isFinite(x)) return 0;
    return Math.round(x / step) * step;
}

/**
 * 计算目标samplesPerPixel，用于选择Mipmap级别
 * @param clipLengthSec clip长度（秒）
 * @param pxWidth clip在canvas上的像素宽度
 * @param sourceSampleRate 源文件采样率
 * @returns samplesPerPixel 每个像素对应的采样点数
 */
function calculateSamplesPerPixel(
    clipLengthSec: number,
    pxWidth: number,
    sourceSampleRate: number,
): number {
    if (!Number.isFinite(clipLengthSec) || clipLengthSec <= 0 || pxWidth <= 0) {
        return 256; // 默认值对应Mipmap Level 0/1边界
    }
    const totalSamples = clipLengthSec * sourceSampleRate;
    return totalSamples / pxWidth;
}

/** 从clip信息构建peaks请求参数 (Mipmap优化版) */
function buildClipPeaksRequest(
    clip: ClipInfo,
    pxWidth?: number,
): {
    sourcePath: string;
    startSec: number;
    durSec: number;
    samplesPerPixel: number;
    cacheKey: string;
} | null {
    const sourcePath = clip.sourcePath;
    if (!sourcePath) return null;

    // 优先使用精确的frame计算
    let durationSec: number;
    let sourceSampleRate = 44100; // 默认采样率
    if (
        clip.durationFrames &&
        clip.sourceSampleRate &&
        clip.sourceSampleRate > 0
    ) {
        durationSec = clip.durationFrames / clip.sourceSampleRate;
        sourceSampleRate = clip.sourceSampleRate;
    } else {
        durationSec = Number(clip.durationSec ?? 0);
    }

    if (!Number.isFinite(durationSec) || durationSec <= 0) return null;

    const lengthSec = Math.max(0, Number(clip.lengthSec ?? 0) || 0);
    if (lengthSec <= 1e-9) return null;

    // 固定请求整个 source 文件的 peaks，不依赖 trim 值
    const startSecQ = 0;
    const durSecQ = Math.max(0.005, qsec(durationSec));

    // 计算samplesPerPixel用于选择Mipmap级别
    // 如果提供了像素宽度，则使用实际宽度计算；否则使用默认中等值
    const samplesPerPixel = pxWidth && pxWidth > 0
        ? calculateSamplesPerPixel(durSecQ, pxWidth, sourceSampleRate)
        : 512; // 默认中等精度

    // 缓存键包含 samplesPerPixel 的量化值，减少重复请求
    const sppQuantized = Math.round(samplesPerPixel / 50) * 50;
    const cacheKey = `${sourcePath}|${startSecQ.toFixed(3)}|${durSecQ.toFixed(3)}|spp${sppQuantized}`;

    return {
        sourcePath,
        startSec: startSecQ,
        durSec: durSecQ,
        samplesPerPixel,
        cacheKey,
    };
}

/**
 * Piano Roll 获取当前 track 下所有可见 clip 的 peaks 数据 (Mipmap 优化版)
 *
 * 特性：
 * 1. 使用 Mipmap 多级缓存，根据缩放级别自动选择最佳峰值分辨率
 * 2. Level 0 (div ~128): 放大显示，高精度
 * 3. Level 1 (div ~512): 中等缩放，平衡性能
 * 4. Level 2 (div ~2048): 小缩放，适合概览
 * 5. Level 3 (div ~8192): 全景视图，最低精度
 *
 * @param args.clips - 当前 track 下的所有 clip
 * @param args.visibleStartSec - 可见区域起始时间（秒）
 * @param args.visibleEndSec - 可见区域结束时间（秒）
 * @param args.pxPerSec - 像素/秒比例，用于计算 samplesPerPixel 选择 Mipmap 级别
 * @returns ClipPeaksEntry 数组，每个 entry 对应一个可见 clip
 */
export function useClipsPeaksForPianoRoll(args: {
    clips: ClipInfo[];
    visibleStartSec: number;
    visibleEndSec: number;
    pxPerSec?: number;
}): ClipPeaksEntry[] {
    const { clips, visibleStartSec, visibleEndSec, pxPerSec } = args;
    const [peaksMap, setPeaksMap] = useState<Map<string, CachedEntry>>(
        new Map(),
    );
    const requestIdRef = useRef(0);
    const mountedRef = useRef(true);
    const lastLevelByPathRef = useRef<Record<string, 0 | 1 | 2>>({});

    useEffect(() => {
        mountedRef.current = true;
        return () => {
            mountedRef.current = false;
        };
    }, []);

    /**
     * 计算稳定的 peaks 请求 key 列表（不含 lengthSec）。
     * trim 拖动只改变 lengthSec，不改变 sourcePath/trimStart/trimEnd/playbackRate。
     * 因此 cacheKey 不变，不会触发重新请求。
     * useMemo 确保只在依赖真正变化时才重新计算，避免每次 render 都调用 buildClipPeaksRequest。
     * 
     * Mipmap优化：缓存键包含 pxPerSec 的量化值，确保缩放时选择合适的 Mipmap 级别
     */
    const peaksRequestKeys = useMemo(
        () =>
            clips
                .filter((clip) => {
                    return (
                        clip.startSec + clip.lengthSec > visibleStartSec &&
                        clip.startSec < visibleEndSec
                    );
                })
                .map((clip) => {
                    // 估算 clip 在屏幕上的像素宽度
                    const clipWidthPx = pxPerSec ? clip.lengthSec * pxPerSec : undefined;
                    const req = buildClipPeaksRequest(clip, clipWidthPx);
                    if (!req) return null;
                    const previousLevel = lastLevelByPathRef.current[req.sourcePath];
                    const level = waveformMipmapStore.selectLevelStable(
                        req.samplesPerPixel,
                        previousLevel,
                    );
                    return `${clip.id}:${req.sourcePath}|${req.startSec.toFixed(3)}|${req.durSec.toFixed(3)}|lvl${level}`;
                })
                .filter(Boolean)
                .join(","),
        // eslint-disable-next-line react-hooks/exhaustive-deps
        [clips, visibleStartSec, visibleEndSec, pxPerSec],
    );

    useEffect(() => {
        // 过滤出与可见区域有交叠的 clip（clip.startSec/lengthSec 是秒，直接比较）
        const visibleClips = clips.filter((clip) => {
            return (
                clip.startSec + clip.lengthSec > visibleStartSec &&
                clip.startSec < visibleEndSec
            );
        });

        if (visibleClips.length === 0) {
            setPeaksMap(new Map());
            return;
        }

        const currentRequestId = ++requestIdRef.current;

        // 先用缓存数据立即更新
        const initialMap = new Map<string, CachedEntry>();
        const toFetch: Array<{
            clip: ClipInfo;
            req: NonNullable<ReturnType<typeof buildClipPeaksRequest>>;
            columns: number;
            level: 0 | 1 | 2;
            cacheKey: string;
        }> = [];

        for (const clip of visibleClips) {
            // 列数按 clip 时长自适应，长 clip 提升精度，同时保持请求上限。
            const clipWidthPx = pxPerSec ? clip.lengthSec * pxPerSec : undefined;
            const req = buildClipPeaksRequest(clip, clipWidthPx);
            if (!req) continue;
            const previousLevel = lastLevelByPathRef.current[req.sourcePath];
            const level = waveformMipmapStore.selectLevelStable(
                req.samplesPerPixel,
                previousLevel,
            );
            lastLevelByPathRef.current[req.sourcePath] = level;
            const cacheKey =
                `${req.sourcePath}|${req.startSec.toFixed(3)}|${req.durSec.toFixed(3)}|lvl${level}`;

            // 计算 columns：基于 clip 像素宽度，确保波形清晰
            const columns = clipWidthPx ? Math.max(16, Math.round(clipWidthPx * 2)) : 256;

            const cached = lruGet(clipPeaksCache, cacheKey);
            if (cached) {
                initialMap.set(clip.id, cached);
            } else {
                const bestEffort = waveformMipmapStore.getBestSlice(
                    req.sourcePath,
                    level,
                    req.startSec,
                    req.durSec,
                );
                if (bestEffort) {
                    const entry: CachedEntry = {
                        min: Array.from(bestEffort.min),
                        max: Array.from(bestEffort.max),
                        startSec: req.startSec,
                        durSec: req.durSec,
                        t: Date.now(),
                    };
                    lruSet(
                        clipPeaksCache,
                        cacheKey,
                        entry,
                        CLIP_PEAKS_CACHE_LIMIT,
                    );
                    initialMap.set(clip.id, entry);
                } else {
                    toFetch.push({ clip, req, columns, level, cacheKey });
                }
            }
        }

        setPeaksMap((prev) => {
            const next = new Map<string, CachedEntry>();
            for (const clip of visibleClips) {
                const entry = initialMap.get(clip.id) ?? prev.get(clip.id);
                if (entry) {
                    next.set(clip.id, entry);
                }
            }
            return next;
        });

        if (toFetch.length === 0) return;

        // 异步获取未缓存的 clip peaks (Mipmap优化版)
        void (async () => {
            const results = await Promise.allSettled(
                toFetch.map(async ({ clip, req, level, cacheKey }) => {
                    // 使用 waveformMipmapStore（三级整文件缓存）获取峰值数据
                    const inflightKey = `${req.sourcePath}|${level}`;
                    let p = clipPeaksInflight.get(inflightKey);
                    if (!p) {
                        p = (async () => {
                            // 确保数据已预加载（preload 内部会去重并等待正在进行的加载）
                            await waveformMipmapStore.preload(req.sourcePath);
                            const slice = waveformMipmapStore.getSlice(
                                req.sourcePath,
                                level,
                                req.startSec,
                                req.durSec,
                            );
                            if (!slice) return null;
                            const entry: CachedEntry = {
                                min: Array.from(slice.min),
                                max: Array.from(slice.max),
                                startSec: req.startSec,
                                durSec: req.durSec,
                                t: Date.now(),
                            };
                            lruSet(
                                clipPeaksCache,
                                cacheKey,
                                entry,
                                CLIP_PEAKS_CACHE_LIMIT,
                            );
                            return entry;
                        })().finally(() => {
                            clipPeaksInflight.delete(inflightKey);
                        });
                        clipPeaksInflight.set(inflightKey, p);
                    }

                    const entry = await p;
                    return { clipId: clip.id, entry };
                }),
            );

            if (
                !mountedRef.current ||
                currentRequestId !== requestIdRef.current
            )
                return;

            setPeaksMap((prev) => {
                const next = new Map(prev);
                for (const result of results) {
                    if (result.status === "fulfilled" && result.value) {
                        const { clipId, entry } = result.value;
                        if (entry) {
                            next.set(clipId, entry);
                        }
                    }
                }
                return next;
            });
        })();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [peaksRequestKeys, visibleStartSec, visibleEndSec, pxPerSec]);

    // 构建返回值：过滤可见 clip，附加 peaks 数据。
    // useMemo 确保只在 peaksMap 或可见 clips 真正变化时才返回新数组引用，
    // 避免每次 render 都天然产生新引用导致 PianoRollPanel 的 invalidate effect 每帧执行。
    return useMemo(() => {
        const visibleClips = clips.filter((clip) => {
            return (
                clip.startSec + clip.lengthSec > visibleStartSec &&
                clip.startSec < visibleEndSec
            );
        });

        return visibleClips.map((clip): ClipPeaksEntry => {
            const entry = peaksMap.get(clip.id) ?? null;

            // 计算 source 文件总时长
            let sourceDurationSec: number;
            if (
                clip.durationFrames &&
                clip.sourceSampleRate &&
                clip.sourceSampleRate > 0
            ) {
                sourceDurationSec = clip.durationFrames / clip.sourceSampleRate;
            } else {
                sourceDurationSec = Number(clip.durationSec ?? 0);
            }

            const playbackRate = Number(clip.playbackRate ?? 1);
            const pr =
                Number.isFinite(playbackRate) && playbackRate > 0
                    ? playbackRate
                    : 1;

            return {
                clipId: clip.id,
                startSec: clip.startSec,
                lengthSec: clip.lengthSec,
                sourceStartSec: Math.max(
                    0,
                    Number(clip.sourceStartSec ?? 0) || 0,
                ),
                sourceDurationSec:
                    sourceDurationSec > 0 ? sourceDurationSec : 0,
                playbackRate: pr,
                gain: clip.gain ?? 1,
                fadeInSec: clip.fadeInSec ?? 0,
                fadeOutSec: clip.fadeOutSec ?? 0,
                fadeInCurve: clip.fadeInCurve ?? "linear",
                fadeOutCurve: clip.fadeOutCurve ?? "linear",
                peaks: entry
                    ? {
                          min: entry.min,
                          max: entry.max,
                          startSec: entry.startSec,
                          durSec: entry.durSec,
                          columns: entry.min.length,
                      }
                    : null,
            };
        });
        // peaksMap 内容变化时才重新计算；可见区域或 clips 变化时同步更新。
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [peaksMap, clips, visibleStartSec, visibleEndSec]);
}
