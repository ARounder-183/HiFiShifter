/**
 * Piano Roll Per-Clip 波形 Peaks Hook (v2 重构版)
 *
 * 与 WaveformTrackCanvas 保持一致的数据路径：
 * 直接使用 waveformMipmapStore.getInterleavedSlice() 获取 interleaved Float32Array，
 * 无需独立的 min[]/max[] 路径和手动 resample 逻辑。
 *
 * 数据流：
 *   waveformMipmapStore.getInterleavedSlice() → interleaved Float32Array → render.ts
 */
import { useEffect, useMemo, useRef, useState } from "react";

import type { ClipInfo, FadeCurveType } from "../../../features/session/sessionTypes";
import { waveformMipmapStore } from "../../../utils/waveformMipmapStore";

/** 单个 clip 的波形数据条目（v2：interleaved 格式，与 WaveformTrackCanvas 一致） */
export interface ClipPeaksEntry {
    /** clip ID */
    clipId: string;
    /** clip 在 timeline 上的起始位置（秒，来自 ClipInfo.startSec） */
    startSec: number;
    /** clip 的长度（秒，来自 ClipInfo.lengthSec），用于绘制宽度 */
    lengthSec: number;
    /** clip 的 sourceStartSec（秒），渲染时用于计算波形偏移 */
    sourceStartSec: number;
    /** source 文件总时长（秒） */
    sourceDurationSec: number;
    /** clip 在源文件中的结束位置（秒），裁剪后的终点 */
    sourceEndSec: number;
    /** source 文件采样率 */
    sourceSampleRate: number;
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
    /** source 文件路径（用于从 mipmap store 获取数据） */
    sourcePath: string;
    /** clip 是否静音 */
    muted: boolean;
}

/**
 * Piano Roll 获取当前 track 下所有可见 clip 的信息
 *
 * v2 重构：不再在 hook 内获取 peaks 数据，只返回 clip 元数据。
 * 波形数据在 render.ts 的绘制循环中通过 waveformMipmapStore.getInterleavedSlice()
 * 同步获取，与 WaveformTrackCanvas 保持相同的渲染模式。
 *
 * @param args.clips - 当前 track 下的所有 clip
 * @param args.visibleStartSec - 可见区域起始时间（秒）
 * @param args.visibleEndSec - 可见区域结束时间（秒）
 * @param args.pxPerSec - 像素/秒比例，用于选择 Mipmap 级别
 * @returns ClipPeaksEntry 数组，每个 entry 对应一个可见 clip
 */
export function useClipsPeaksForPianoRoll(args: {
    clips: ClipInfo[];
    visibleStartSec: number;
    visibleEndSec: number;
    pxPerSec?: number;
}): ClipPeaksEntry[] {
    const { clips, visibleStartSec, visibleEndSec } = args;

    // 强制重绘计数器（mipmap 数据加载完成时 +1 触发重绘）
    const [redrawTick, setRedrawTick] = useState(0);

    // 监听 mipmap 缓存加载完成事件，触发重绘
    useEffect(() => {
        const neededPaths = new Set<string>();
        for (const clip of clips) {
            if (clip.sourcePath) neededPaths.add(clip.sourcePath);
        }

        const unsub = waveformMipmapStore.addListener((sourcePath, status) => {
            if (status === "done" && neededPaths.has(sourcePath)) {
                setRedrawTick((t) => t + 1);
            }
        });

        return unsub;
    }, [clips]);

    // 触发预加载所有可见 clip 的 mipmap 数据（使用 batchPreload 合并 IPC 调用）
    const preloadedPathsRef = useRef(new Set<string>());
    useEffect(() => {
        const newPaths: string[] = [];
        for (const clip of clips) {
            if (clip.sourcePath && !preloadedPathsRef.current.has(clip.sourcePath)) {
                preloadedPathsRef.current.add(clip.sourcePath);
                newPaths.push(clip.sourcePath);
            }
        }
        if (newPaths.length > 0) {
            void waveformMipmapStore.batchPreload(newPaths);
        }
    }, [clips]);

    // 构建返回值：过滤可见 clip，返回元数据
    return useMemo(() => {
        // 引用 redrawTick 以便 mipmap 加载完成后重新计算
        void redrawTick;

        const visibleClips = clips.filter((clip) => {
            return (
                clip.startSec + clip.lengthSec > visibleStartSec && clip.startSec < visibleEndSec
            );
        });

        return visibleClips.map((clip): ClipPeaksEntry => {
            // 计算 source 文件总时长
            let sourceDurationSec: number;
            let sourceSampleRate = 44100;
            if (clip.durationFrames && clip.sourceSampleRate && clip.sourceSampleRate > 0) {
                sourceDurationSec = clip.durationFrames / clip.sourceSampleRate;
                sourceSampleRate = clip.sourceSampleRate;
            } else {
                sourceDurationSec = Number(clip.durationSec ?? 0);
            }

            const playbackRate = Number(clip.playbackRate ?? 1);
            const pr = Number.isFinite(playbackRate) && playbackRate > 0 ? playbackRate : 1;

            // sourceEndSec：与 WaveformTrackCanvas 一致，优先使用 clip.sourceEndSec
            const clipSourceEndSec =
                Number(clip.sourceEndSec ?? sourceDurationSec) || sourceDurationSec;

            return {
                clipId: clip.id,
                startSec: clip.startSec,
                lengthSec: clip.lengthSec,
                sourceStartSec: Math.max(0, Number(clip.sourceStartSec ?? 0) || 0),
                sourceDurationSec: sourceDurationSec > 0 ? sourceDurationSec : 0,
                sourceEndSec: clipSourceEndSec,
                sourceSampleRate,
                playbackRate: pr,
                gain: clip.gain ?? 1,
                fadeInSec: clip.fadeInSec ?? 0,
                fadeOutSec: clip.fadeOutSec ?? 0,
                fadeInCurve: clip.fadeInCurve ?? "linear",
                fadeOutCurve: clip.fadeOutCurve ?? "linear",
                sourcePath: clip.sourcePath ?? "",
                muted: clip.muted ?? false,
            };
        });
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [clips, visibleStartSec, visibleEndSec, redrawTick]);
}
