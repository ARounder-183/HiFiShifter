/**
 * 参数编辑器（Piano Roll）波形面适配层。
 *
 * 【主要内容】把 clip 峰值条目组装成单行 `WaveformSceneRow[]`，并把参数编辑器
 * 的水平缩放 / 滚动位置包装成 `TimelineAxis` 交给共享的 `WaveformSurface`。
 *
 * 【作用】参数编辑器与时间线共用同一块波形面，本组件只负责「单波形带」这一
 * 布局差异与坐标投影的构造。
 *
 * 【与其他模块的关系】
 * - 上游：`PianoRollPanel` 传入 clip 峰值、宽度与 scrollLeft / pxPerSec。
 * - 横向：视口实时同步走 `pianoRollViewportBus`；时间↔像素换算一律由
 *   `timelineAxis.ts` 提供。
 * - 下游：`waveform/WaveformSurface`。
 */

import React from "react";

import { WaveformSurface } from "../../../waveform/WaveformSurface";
import type { WaveformSceneClip, WaveformSceneRow } from "../../../waveform/sceneBuilder";
import type { WaveformColors } from "../../../theme/waveformColors";
import { createTimelineAxis } from "../timeline/runtime/timelineAxis.js";
import type { ClipPeaksEntry } from "./useClipsPeaksForPianoRoll";
import { pianoRollViewportBus } from "./pianoRollViewportBus";

function toSceneClip(entry: ClipPeaksEntry): WaveformSceneClip | null {
    if (!entry.sourcePath || entry.muted) return null;
    return {
        id: entry.clipId,
        sourcePath: entry.sourcePath,
        startSec: entry.startSec,
        lengthSec: entry.lengthSec,
        sourceStartSec: entry.sourceStartSec,
        sourceEndSec: entry.sourceEndSec,
        durationSec: entry.sourceDurationSec,
        sourceSampleRate: entry.sourceSampleRate,
        playbackRate: entry.playbackRate,
        reversed: entry.reversed,
        loopEnabled: entry.loopEnabled,
        gain: entry.gain,
        muted: false,
        fadeInSec: entry.fadeInSec,
        fadeOutSec: entry.fadeOutSec,
        autoFadeInSec: entry.autoFadeInSec,
        autoFadeOutSec: entry.autoFadeOutSec,
        fadeInShape: entry.fadeInShape,
        fadeInDir: entry.fadeInDir,
        fadeOutShape: entry.fadeOutShape,
        fadeOutDir: entry.fadeOutDir,
    };
}

export const PianoRollWaveformSurface = React.memo(function PianoRollWaveformSurface(props: {
    clips: readonly ClipPeaksEntry[];
    widthPx: number;
    heightPx: number;
    scrollLeftPx: number;
    pxPerSec: number;
    colors: WaveformColors;
}) {
    const rows = React.useMemo<WaveformSceneRow[]>(
        () => [
            {
                topPx: 0,
                waveformTopPx: 0,
                waveformHeightPx: props.heightPx,
                clips: props.clips
                    .map(toSceneClip)
                    .filter((clip): clip is WaveformSceneClip => clip != null),
            },
        ],
        [props.clips, props.heightPx],
    );
    // 统一坐标投影：缓存以避免每次渲染新建对象导致波形面 memo 失效。
    // 总线驱动时 WaveformSurface 会用总线快照覆盖 scrollLeftPx / pxPerSec。
    const axis = React.useMemo(
        () =>
            createTimelineAxis({
                pxPerSec: props.pxPerSec,
                scrollLeftPx: props.scrollLeftPx,
                viewportWidthPx: props.widthPx,
                dpr: window.devicePixelRatio || 1,
            }),
        [props.pxPerSec, props.scrollLeftPx, props.widthPx],
    );

    return (
        <WaveformSurface
            rows={rows}
            widthPx={props.widthPx}
            heightPx={props.heightPx}
            axis={axis}
            color={props.colors.stroke}
            style={{ opacity: 0.86 }}
            viewportSource={pianoRollViewportBus}
        />
    );
});
