/**
 * 播放光标时间读数。
 *
 * 三处挂载点：
 * - toolbar：主窗口 ActionBar 的 BPM & Time 组。始终存在，不依赖面板布局；
 * - trackHeader：轨道头里的可选紧凑副本，只显示主单位；
 * - detached：独立窗口顶部条，直接用 30Hz Redux 采样，不做跨窗口视觉插值。
 *
 * 时间选择器与旧 TrackHeaderPlayheadTime 同源（isTransportAdvancing 的统一判据），
 * 只是把读数从“轨道面板的属性”提升成了传输层信息。
 */
import { useCallback, useMemo, useState } from "react";
import { shallowEqual } from "react-redux";

import { useAppSelector } from "../../../app/hooks";
import { isTransportAdvancing } from "../../../features/session/selectors/transport";
import { useI18n } from "../../../i18n/I18nProvider";
import { useVisualPlayhead } from "../../../hooks/useVisualPlayhead";
import { SlotText } from "../../../ui/SlotText";
import { formatCursorTime, formatCursorUnit } from "../timeline/timeFormat";
import type { TimeFormatContext } from "../timeline/timeFormat";

export interface PlayheadTimeReadoutProps {
    variant: "toolbar" | "trackHeader" | "detached";
    /** 只显示主单位（轨道头 / 独立窗口）。 */
    compact?: boolean;
    /** 是否使用共享视觉播放头插值；独立窗口传 false。 */
    interpolate?: boolean;
}

export function PlayheadTimeReadout({
    variant,
    compact = false,
    interpolate = true,
}: PlayheadTimeReadoutProps) {
    const { t } = useI18n();
    const selector = useAppSelector(
        (state) => ({
            playheadSec: state.session.playheadSec,
            playheadSampledAtMs: state.session.playheadSampledAtMs,
            isPlaying: state.session.runtime.isPlaying,
            playbackWaitingForRender: state.session.runtime.playbackWaitingForRender,
            playbackPositionSec: state.session.runtime.playbackPositionSec,
            positionAdvancing: state.session.runtime.positionAdvancing,
            positionAgeMs: state.session.runtime.positionAgeMs,
            primaryTimeUnit: state.session.primaryTimeUnit,
            secondaryTimeUnit: state.session.secondaryTimeUnit,
            bpm: state.session.bpm,
            beats: state.session.beats,
            grid: state.session.grid,
            projectSec: state.session.projectSec,
            tempoMap: state.session.tempoMap,
        }),
        shallowEqual,
    );
    const [interpolatedSec, setInterpolatedSec] = useState(selector.playheadSec);
    const transportAdvancing = isTransportAdvancing(selector);
    useVisualPlayhead({
        enabled: interpolate,
        syncedPlayheadSec: selector.playheadSec,
        syncedAtMs: selector.playheadSampledAtMs,
        isTransportAdvancing: transportAdvancing,
        onFrame: useCallback((sec: number) => setInterpolatedSec(sec), []),
    });
    const visualSec = interpolate ? interpolatedSec : selector.playheadSec;

    const timeContext = useMemo<TimeFormatContext>(
        () => ({
            bpm: selector.bpm,
            beatsPerBar: Math.max(1, Math.round(selector.beats || 4)),
            grid: selector.grid,
            tempoMap: selector.tempoMap,
        }),
        [selector.bpm, selector.beats, selector.grid, selector.tempoMap],
    );
    const formatted = useMemo(
        () =>
            formatCursorTime(
                selector.primaryTimeUnit,
                selector.secondaryTimeUnit,
                visualSec,
                timeContext,
            ),
        [selector.primaryTimeUnit, selector.secondaryTimeUnit, visualSec, timeContext],
    );

    // 至少预留 1 小时（或当前工程/光标更大者）的宽度；超长工程只会在首次跨过
    // 预留值时变宽一次，不会每秒抖动。
    const reserveSec = Math.max(3600, selector.projectSec, selector.playheadSec);
    const reservePrimary = useMemo(
        () => formatCursorUnit(selector.primaryTimeUnit, reserveSec, timeContext),
        [selector.primaryTimeUnit, reserveSec, timeContext],
    );
    const reserveSecondary = useMemo(
        () =>
            selector.secondaryTimeUnit !== "none" &&
            selector.secondaryTimeUnit !== selector.primaryTimeUnit
                ? formatCursorUnit(selector.secondaryTimeUnit, reserveSec, timeContext)
                : null,
        [selector.primaryTimeUnit, selector.secondaryTimeUnit, reserveSec, timeContext],
    );

    const ariaLabel = t("playhead_time_readout") + ": " + formatted.combined;

    if (variant === "toolbar") {
        return (
            <div
                className="flex flex-col justify-center px-1 leading-none"
                data-playhead-time-readout="toolbar"
                data-playhead-sec={visualSec.toFixed(6)}
                data-tooltip={formatted.combined}
                aria-label={ariaLabel}
            >
                <SlotText
                    text={formatted.primaryLabel}
                    reserveText={reservePrimary}
                    className="hs-type-mono text-qt-sm font-medium tabular-nums"
                />
                {!compact && formatted.secondaryLabel !== null ? (
                    <SlotText
                        text={formatted.secondaryLabel}
                        reserveText={reserveSecondary ?? undefined}
                        className="hs-type-caption text-qt-text-muted tabular-nums"
                    />
                ) : null}
            </div>
        );
    }

    if (variant === "trackHeader") {
        return (
            <div
                className="min-w-0 flex-1 flex justify-end overflow-hidden pr-5"
                data-playhead-time-readout="trackHeader"
                data-playhead-sec={visualSec.toFixed(6)}
                data-tooltip={formatted.combined}
                aria-label={ariaLabel}
            >
                <SlotText
                    text={formatted.primaryLabel}
                    reserveText={reservePrimary}
                    tooltip={formatted.combined}
                    selectable
                    className="tabular-nums text-qt-text text-right leading-none whitespace-nowrap shrink-0"
                />
            </div>
        );
    }

    return (
        <div
            className="flex h-qt-bar-status items-center justify-end px-2"
            data-playhead-time-readout="detached"
            data-playhead-sec={visualSec.toFixed(6)}
            data-tooltip={formatted.combined}
            aria-label={ariaLabel}
        >
            <SlotText
                text={formatted.primaryLabel}
                reserveText={reservePrimary}
                tooltip={formatted.combined}
                className="hs-type-mono text-qt-xs tabular-nums"
            />
        </div>
    );
}
