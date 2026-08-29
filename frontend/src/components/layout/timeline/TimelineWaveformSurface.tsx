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

export const TimelineWaveformSurface = React.memo(function TimelineWaveformSurface(props: {
    tracks: readonly TrackInfo[];
    /** 窗口首行的绝对轨道索引：行 topPx 使用内容绝对坐标，
     * 竖直滚动时由总线 scrollTopPx 统一平移（与 DOM 内容层同帧提交）。 */
    startTrackIndex: number;
    clipsByTrackId: Readonly<Record<string, readonly ClipInfo[]>>;
    rowHeight: number;
    widthPx: number;
    heightPx: number;
    viewportStartSec: number;
    viewportEndSec: number;
    pxPerSec: number;
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
        [
            props.clipsByTrackId,
            props.rowHeight,
            props.startTrackIndex,
            props.tracks,
            showAllTakes,
        ],
    );

    return (
        <WaveformSurface
            rows={rows}
            widthPx={props.widthPx}
            heightPx={props.heightPx}
            viewportStartSec={props.viewportStartSec}
            viewportEndSec={props.viewportEndSec}
            pxPerSec={props.pxPerSec}
            viewportTopPx={props.startTrackIndex * props.rowHeight}
            color={color}
            viewportSource={timelineViewportBus}
        />
    );
});
