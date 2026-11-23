/**
 * 播放光标时间的**格式化上下文**：主/副单位、时间格式上下文、两段文本与预留宽度。
 *
 * 【为什么抽出来】读数（每帧插值）与"复制播放光标时间"（已提交位置）要的是同一套
 * 计算：同样的单位、同样的 Tempo Map、同样的"至少预留 1 小时"的宽度。两处各算一遍
 * 的下场是复制出来的字符串与屏幕上看到的不是同一个格式（副单位漏了、Tempo Map 没
 * 参与），而这种偏差没有任何报错，只能靠用户发现粘贴出来不对。
 */
import { useMemo } from "react";
import { shallowEqual } from "react-redux";

import { useAppSelector } from "../../../app/hooks";
import { formatCursorTime, formatCursorUnit } from "../timeline/timeFormat";
import type { TimeFormatContext } from "../timeline/timeFormat";

/** 预留宽度的下限：一小时。见下方 `reserveSec`。 */
const MIN_RESERVE_SEC = 3600;

export interface PlayheadTimeFormatState {
    formatted: { primaryLabel: string; secondaryLabel: string | null; combined: string };
    reservePrimary: string;
    reserveSecondary: string | null;
}

/**
 * @param sec 要显示的秒数。读数传**插值后**的视觉位置，复制传已提交位置 ——
 *            两者的差别只在播放中（插值领先一个采样周期）。
 */
export function usePlayheadTimeFormat(sec: number): PlayheadTimeFormatState {
    const s = useAppSelector(
        (state) => ({
            primaryUnit: state.session.primaryTimeUnit,
            secondaryUnit: state.session.secondaryTimeUnit,
            bpm: state.session.bpm,
            beats: state.session.beats,
            grid: state.session.grid,
            tempoMap: state.session.tempoMap,
            projectSec: state.session.projectSec,
            playheadSec: state.session.playheadSec,
        }),
        shallowEqual,
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

    const formatted = useMemo(
        () => formatCursorTime(s.primaryUnit, s.secondaryUnit, sec, timeContext),
        [s.primaryUnit, s.secondaryUnit, sec, timeContext],
    );

    // 至少预留 1 小时（或当前工程/光标更大者）的宽度；超长工程只会在首次跨过
    // 预留值时变宽一次，不会每秒抖动。
    const reserveSec = Math.max(MIN_RESERVE_SEC, Math.ceil(s.projectSec), Math.ceil(s.playheadSec));
    const reservePrimary = useMemo(
        () => formatCursorUnit(s.primaryUnit, reserveSec, timeContext),
        [s.primaryUnit, reserveSec, timeContext],
    );
    const reserveSecondary = useMemo(
        () =>
            s.secondaryUnit !== "none" && s.secondaryUnit !== s.primaryUnit
                ? formatCursorUnit(s.secondaryUnit, reserveSec, timeContext)
                : null,
        [s.primaryUnit, s.secondaryUnit, reserveSec, timeContext],
    );

    return { formatted, reservePrimary, reserveSecondary };
}
