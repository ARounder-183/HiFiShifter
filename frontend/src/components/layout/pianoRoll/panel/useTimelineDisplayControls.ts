/**
 * 标尺与时间显示的**派生值**：刻度、时间格式上下文、单位切换与"复制播放头时间"。
 *
 * 【为什么是独立的一个 hook】这一组原本是模型尾部的七个绑定，全部
 * 由同一份会话切片（BPM / 拍号 / 网格 / Tempo Map / 时间单位）推出，没有一条读写
 * 面板的其它状态。它们此前散在工具图标与返回对象之间，读起来像"顺手放在这里"。
 *
 * 【刻度与网格同源】`timelineTicks` 是标尺刻度与背景网格线的**唯一**来源，与时间线
 * 面板走同一个 `buildTimelineTicks`；两个面板的刻度因此严格同源于 axis 投影。
 *
 * 【所有权不变】会话状态与 dispatch 仍属于面板，这里只是把七个声明搬进子 hook：
 * 调用点、hook 顺序与依赖数组全部保持原位。
 */
import { useCallback, useMemo } from "react";
import {
    persistUiSettings,
    setPrimaryTimeUnit,
    setSecondaryTimeUnit,
    setTempoMap,
} from "../../../../features/session/sessionSlice";
import { setTempoMapRemote } from "../../../../features/session/thunks/tempoMapThunks";
import type { TempoMap } from "../../../../utils/tempoMap";
import { buildTimelineTicks } from "../../timeline/runtime/buildTimelineTicks.js";
import type { createTickAxis } from "../../timeline/runtime/tickAxis.js";
import { formatCursorTime } from "../../timeline";
import type { TimeFormatContext, TimeUnit, TimeUnitChoice } from "../../timeline";
import type { ModelDeps } from "./modelDeps";

type TimelineDisplayInputs = Pick<ModelDeps, "dispatch" | "s"> & {
    /** 刻度专用轴（比绘制轴更保守的窗口，见模型里的说明）。 */
    prTickAxis: ReturnType<typeof createTickAxis>;
};

/**
 * 标尺与时间显示的派生值。
 *
 * 调用点必须固定在模型里原来的位置：这里注册的 `useMemo` / `useCallback` 数量与顺序
 * 是 hook 序列的一部分。
 */
export function useTimelineDisplayControls({ dispatch, s, prTickAxis }: TimelineDisplayInputs) {
    // 统一刻度源：标尺刻度与背景网格线共用，与时间线侧同一实现，
    // 保证两个面板的网格/标尺位置严格同源于 axis 投影。
    const timelineTicks = useMemo(
        () =>
            buildTimelineTicks({
                axis: prTickAxis.axis,
                bpm: s.bpm,
                beatsPerBar: Math.max(1, Math.round(s.beats || 4)),
                grid: s.grid,
                primaryUnit: s.primaryTimeUnit,
                secondaryUnit: s.secondaryTimeUnit,
                minLabelSpacingPx: s.rulerLabelSpacingPx,
                minGridSpacingPx: s.timelineSnap.gridMinSpacingPx,
                swingPercent: s.timelineSnap.swingEnabled ? s.timelineSnap.swingPercent : 0,
                tempoMap: s.tempoMap,
            }),
        [
            prTickAxis,
            s.bpm,
            s.beats,
            s.grid,
            s.primaryTimeUnit,
            s.secondaryTimeUnit,
            s.rulerLabelSpacingPx,
            s.timelineSnap,
            s.tempoMap,
        ],
    );
    const timeContext = useMemo<TimeFormatContext>(
        () => ({
            bpm: s.bpm,
            beatsPerBar: Math.max(1, Math.round(s.beats || 4)),
            grid: s.grid,
            tempoMap: s.tempoMap,
        }),
        [s.bpm, s.beats, s.grid, s.tempoMap],
    );

    const handlePrimaryUnitChange = useCallback(
        (unit: TimeUnit) => {
            dispatch(setPrimaryTimeUnit(unit));
            void dispatch(persistUiSettings());
        },
        [dispatch],
    );
    const handleSecondaryUnitChange = useCallback(
        (unit: TimeUnitChoice) => {
            dispatch(setSecondaryTimeUnit(unit));
            void dispatch(persistUiSettings());
        },
        [dispatch],
    );

    const handleTempoMapChange = useCallback(
        (next: TempoMap | null) => {
            dispatch(setTempoMap(next));
        },
        [dispatch],
    );
    const handleTempoMapCommit = useCallback(
        (next: TempoMap | null) => {
            dispatch(setTempoMap(next));
            void dispatch(setTempoMapRemote(next));
        },
        [dispatch],
    );
    const handleCopyPlayheadTime = useCallback(async () => {
        const text = formatCursorTime(
            s.primaryTimeUnit,
            s.secondaryTimeUnit,
            Number(s.playheadSec ?? 0),
            timeContext,
        ).combined;
        try {
            await navigator.clipboard.writeText(text);
        } catch {
            // 忽略复制失败
        }
    }, [s.primaryTimeUnit, s.secondaryTimeUnit, s.playheadSec, timeContext]);
    return {
        timelineTicks,
        timeContext,
        handlePrimaryUnitChange,
        handleSecondaryUnitChange,
        handleTempoMapChange,
        handleTempoMapCommit,
        handleCopyPlayheadTime,
    };
}
