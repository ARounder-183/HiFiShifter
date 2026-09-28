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

import type { ClipInfo } from "../../../features/session/sessionTypes";
import { resolveSourceEndSec } from "../../../utils/loopRender";
import { waveformMipmapStore } from "../../../utils/waveformMipmapStore";

/**
 * 水平取窗余量（CSS px）。
 *
 * React 的 `scrollLeft` 是 256px 量化提交的（内核 `SCROLL_COMMIT_STEP_PX`），
 * 滚动中最多滞后内核真值 255px；波形面的几何另带最多 512px 的水平余量窗口
 * （`WaveformSurface.draw`）。取窗因此向两侧各放宽约 1.3 个视口宽，把「量化
 * 滞后 + 几何余量」整段兜住——滚动期间波形面取到的 clip 集合**始终**覆盖当前
 * 视口，「进入视口的 clip 波形消失、随滚动恢复、拖回去又消失」在数据源头关闭。
 * 余量以 px 计、由可见区间的秒宽折算（见 `windowMarginSec`），随缩放自动缩放。
 */
const WINDOW_MARGIN_PX = 2048;

/** 折算余量时假定的典型视口宽（CSS px）。仅影响余量大小，不影响正确性。 */
const TYPICAL_VIEWPORT_W_PX = 1600;

/**
 * 水平余量的秒数：由可见区间的秒宽反推 pxPerSec 后折算。
 *
 * 面板传入的是**量化可见区间**（秒），而量化滞后与波形面的几何余量都以 px
 * 发生。`visibleSpanSec × pxPerSec ≈ viewportWidthPx`（误差即量化偏差 ≤255px，
 * 远小于余量本身），因此 `pxPerSec ≈ TYPICAL_VIEWPORT_W_PX / visibleSpanSec`，
 * 余量秒 = `WINDOW_MARGIN_PX / pxPerSec`。区间非法时退回 0（精确窗），行为与
 * 旧实现一致——退化只发生在不可渲染的输入上，不影响正常路径。
 */
function windowMarginSec(visibleStartSec: number, visibleEndSec: number): number {
    const spanSec = visibleEndSec - visibleStartSec;
    if (!(spanSec > 1e-9)) return 0;
    return (spanSec * WINDOW_MARGIN_PX) / TYPICAL_VIEWPORT_W_PX;
}

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
    /** 自动交叉淡化时长（秒）；>0 时有效淡化 = 自动值，否则用手动值。 */
    autoFadeInSec: number;
    autoFadeOutSec: number;
    /** 淡入形状（REAPER 形状 id）与曲率。 */
    fadeInShape: number;
    fadeInDir: number;
    /** 淡出形状（REAPER 形状 id）与曲率。 */
    fadeOutShape: number;
    fadeOutDir: number;
    /** source 文件路径（用于从 mipmap store 获取数据） */
    sourcePath: string;
    /** clip 是否静音 */
    muted: boolean;
    /** 是否倒放 */
    reversed: boolean;
    /** Loop（循环源）：超出源窗口的内容按周期回绕重复 */
    loopEnabled: boolean;
    /** Take 声道模式（0..=4，对齐 REAPER CHANMODE）。 */
    channelMode: number;
    /** 源文件声道数（未知时 0）。 */
    sourceChannels: number;
}

/**
 * Piano Roll 获取当前 track 下所有可见 clip 的信息
 *
 * v2 重构：不再在 hook 内获取 peaks 数据，只返回 clip 元数据。
 * 波形数据在 render.ts 的绘制循环中通过 waveformMipmapStore.getInterleavedSlice()
 * 同步获取，与 WaveformTrackCanvas 保持相同的渲染模式。
 *
 * React 的 `visibleStartSec / visibleEndSec` 来自 256px 量化提交的 scrollLeft，
 * 滚动中最多滞后内核真值 255px；而波形面由视口总线（内核真值）同帧驱动。若按
 * 可见区间精确取窗，内核刚带进视口的 clip 在「新视口 × 旧 rows」的组合帧里没有
 * 几何——波形消失，直到 React 提交后 rows 才跟上（快速往返拖拽时反复出现，
 * 且与「同步到时间轴」开关无关：两条滚动路径汇入同一条总线）。因此本 hook 按
 * **放宽窗**过滤：量化可见区间向两侧各扩约 1.3 个视口宽，保证滞后帧的视口
 * （含波形面自身 ≤512px 的几何余量）仍被 clip 集合覆盖。宽窗让单次重建多纳入
 * 少量 clip——远离视口的会在 `buildWaveformScene` 的视口裁剪中被剔除，不产生
 * 几何也不发起取数，成本远低于「每个滚动方向闪一次」。
 *
 * 【竖直方向】参数编辑器是单行（全轨道混合）视图，无竖直窗口化，无需处理。
 *
 * @param args.clips - 当前 track 下的所有 clip
 * @param args.visibleStartSec - 可见区域起始时间（秒，256px 量化提交）
 * @param args.visibleEndSec - 可见区域结束时间（秒，256px 量化提交）
 * @returns ClipPeaksEntry 数组，每个 entry 对应一个可见 clip
 */
export function useClipsPeaksForPianoRoll(args: {
    clips: ClipInfo[];
    visibleStartSec: number;
    visibleEndSec: number;
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
            // done = 数据就绪；evicted = 可见数据被内存压力淘汰（渲染循环
            // 会重新发起加载，这里只负责触发一次重绘）。
            if ((status === "done" || status === "evicted") && neededPaths.has(sourcePath)) {
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

    // 构建返回值：按放宽的水平窗口过滤可见 clip，返回元数据
    return useMemo(() => {
        // 引用 redrawTick 以便 mipmap 加载完成后重新计算
        void redrawTick;

        // 水平窗口：量化可见区间向两侧各放宽（见 `WINDOW_MARGIN_PX` 的说明）。
        const windowStartSec = visibleStartSec - windowMarginSec(visibleStartSec, visibleEndSec);
        const windowEndSec = visibleEndSec + windowMarginSec(visibleStartSec, visibleEndSec);

        const visibleClips = clips.filter((clip) => {
            return clip.startSec + clip.lengthSec > windowStartSec && clip.startSec < windowEndSec;
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

            // sourceEndSec：派生窗口（REAPER 语义）—— 非 Loop 正放取
            // 起点+长度×速率，与 WaveformTrackCanvas 一致；陈旧存储窗口
            // 不再冻结静音区。Loop/倒放保持原字段。
            const clipSourceEndSec = resolveSourceEndSec({
                loopEnabled: Boolean(clip.loopEnabled),
                reversed: Boolean(clip.reversed),
                sourceStartSec: Number(clip.sourceStartSec ?? 0) || 0,
                playbackRate: pr,
                lengthSec: clip.lengthSec,
                sourceEndSec: Number(clip.sourceEndSec ?? sourceDurationSec) || sourceDurationSec,
            });

            return {
                clipId: clip.id,
                startSec: clip.startSec,
                lengthSec: clip.lengthSec,
                // 保留原始值（可为负 / 超界）：渲染端按 loopRender 约定用
                // floor_mod（modEuclid）归一化，不能在此处 clamp —— 否则
                // slip/左延伸产生的域外锚点会与 arrange 画布相位错位。
                sourceStartSec: Number(clip.sourceStartSec ?? 0) || 0,
                sourceDurationSec: sourceDurationSec > 0 ? sourceDurationSec : 0,
                sourceEndSec: clipSourceEndSec,
                sourceSampleRate,
                playbackRate: pr,
                gain: clip.gain ?? 1,
                fadeInSec: clip.fadeInSec ?? 0,
                fadeOutSec: clip.fadeOutSec ?? 0,
                autoFadeInSec: clip.autoFadeInSec ?? 0,
                autoFadeOutSec: clip.autoFadeOutSec ?? 0,
                fadeInShape: Number.isFinite(clip.fadeInShape) ? clip.fadeInShape : 0,
                fadeOutShape: Number.isFinite(clip.fadeOutShape) ? clip.fadeOutShape : 0,
                fadeInDir: clip.fadeInDir ?? 0,
                fadeOutDir: clip.fadeOutDir ?? 0,
                sourcePath: clip.sourcePath ?? "",
                muted: clip.muted ?? false,
                reversed: Boolean(clip.reversed),
                loopEnabled: Boolean(clip.loopEnabled),
                channelMode: clip.channelMode ?? 0,
                sourceChannels: clip.sourceChannels ?? 0,
            };
        });
    }, [clips, visibleStartSec, visibleEndSec, redrawTick]);
}
