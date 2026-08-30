import React from "react";

import { useAppSelector } from "../../../app/hooks";
import type { ClipInfo, TrackInfo } from "../../../features/session/sessionTypes";
import { useAppTheme } from "../../../theme/AppThemeProvider";
import { getWaveformColors } from "../../../theme/waveformColors";
import { timelineViewportBus } from "../../../utils/timelineViewportBus";
import { WaveformSurface } from "../../../waveform/WaveformSurface";
import type { WaveformSceneClip, WaveformSceneRow } from "../../../waveform/sceneBuilder";
import { CLIP_BODY_PADDING_Y, CLIP_HEADER_HEIGHT } from "./constants";
import { computeLeadingOverlapSecByClipId } from "./TrackLane";
import { clipToSceneClip, expandClipToTakeSceneClips } from "./takeLanes";

/**
 * 时间线轨道波形面适配层。
 *
 * 【主要内容】把轨道 / clip 数据组装成波形面所需的 `WaveformSceneRow[]`（含
 * 多 Take 展开），并把时间线的坐标投影交给共享的 `WaveformSurface`。
 *
 * 【作用】时间线侧与波形渲染器之间的唯一适配点：负责「轨道行 → 波形行」的
 * 竖直布局与多 Take lane 投影，其余一律下沉到 `WaveformSurface`。
 *
 * 【与其他模块的关系】
 * - 上游：`TimelineSurface` 传入轨道窗口与 `TimelineAxis`。
 * - 横向：多 Take 的 lane 几何复用 `takeLanes.ts`，与 DOM 命中区同源；
 *   时间↔像素换算全部由 axis 提供，本文件不做任何乘法。
 * - 下游：`waveform/WaveformSurface`。
 */

export const TimelineWaveformSurface = React.memo(function TimelineWaveformSurface(props: {
    tracks: readonly TrackInfo[];
    /** 窗口首行的绝对轨道索引：行 topPx 使用内容绝对坐标，
     * 竖直滚动时由总线 scrollTopPx 统一平移（与 DOM 内容层同帧提交）。 */
    startTrackIndex: number;
    clipsByTrackId: Readonly<Record<string, readonly ClipInfo[]>>;
    rowHeight: number;
    widthPx: number;
    heightPx: number;
    /** 统一坐标投影：视口起点与缩放的唯一来源。 */
    axis: import("./runtime/timelineAxis.js").TimelineAxis;
}) {
    const { mode } = useAppTheme();
    // 与 DOM 交互层（ClipItem/TrackLane）同一份持久化设置：开关切换即重建场景。
    const showAllTakes = useAppSelector((state) => state.session.showAllTakes);
    const color = React.useMemo(() => getWaveformColors(mode, "timeline").stroke, [mode]);
    const rows = React.useMemo<WaveformSceneRow[]>(
        () =>
            props.tracks.map((track, index) => {
                const clips = props.clipsByTrackId[track.id] ?? [];
                const waveformHeightPx = Math.max(
                    1,
                    props.rowHeight - CLIP_BODY_PADDING_Y - CLIP_HEADER_HEIGHT,
                );
                // 多 Take clip 展开为每音频 Take 一个 lane 场景 clip；空间不足
                // 或开关关闭时回退为单 active-take 投影（与 DOM 命中区同套
                // takeLanes 数学，lane 边界逐像素一致）。
                const sceneClips: WaveformSceneClip[] = [];
                for (const clip of clips) {
                    const laneClips = expandClipToTakeSceneClips(
                        clip,
                        showAllTakes,
                        waveformHeightPx,
                    );
                    if (laneClips) {
                        sceneClips.push(...laneClips);
                        continue;
                    }
                    const base = clipToSceneClip(clip);
                    if (base) sceneClips.push(base);
                }
                return {
                    topPx: (props.startTrackIndex + index) * props.rowHeight,
                    waveformTopPx: CLIP_HEADER_HEIGHT,
                    waveformHeightPx,
                    clips: sceneClips,
                    leadingOverlapSecByClipId: computeLeadingOverlapSecByClipId([...clips]),
                };
            }),
        [props.clipsByTrackId, props.rowHeight, props.startTrackIndex, props.tracks, showAllTakes],
    );

    return (
        <WaveformSurface
            rows={rows}
            widthPx={props.widthPx}
            heightPx={props.heightPx}
            axis={props.axis}
            viewportTopPx={props.startTrackIndex * props.rowHeight}
            color={color}
            viewportSource={timelineViewportBus}
        />
    );
});
