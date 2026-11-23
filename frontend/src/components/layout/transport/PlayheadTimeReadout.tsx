/**
 * 播放光标时间读数。
 *
 * 四个挂载槽位（同一时刻**只有一个**主实例，由 `PlayheadTimeReadoutSlot` 决定）：
 * - toolbar：主窗口 ActionBar 的 BPM & Time 组；
 * - statusBar：主窗口底部状态栏右侧；
 * - trackHeader：轨道头的紧凑副本，只显示主单位；
 * - detached：独立窗口顶部条，直接用 30Hz Redux 采样，不做跨窗口视觉插值。
 *
 * 【为什么改成单行】两行块在工具栏里是**高度**上的浪费：主工具栏按功能组换行，
 * 一个两行读数会把整条工具栏顶高一行，而它显示的信息与一行完全相同。单行还让
 * "主单位 · 副单位"读起来是一句话，而不是两段互不相干的数字。
 *
 * 【为什么副单位不再用 `text-qt-text-muted`】它此前被压到近背景色：用户看到的是
 * "副单位不见了"，而不是"它是次要信息"。次要由**分隔符**表达，不由降低对比度表达 ——
 * 低对比度在这个界面上已经承担了"禁用"的语义。
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
    variant: "toolbar" | "statusBar" | "trackHeader" | "detached";
    /** 只显示主单位（轨道头 / 独立窗口）。 */
    compact?: boolean;
    /** 是否使用共享视觉播放头插值；独立窗口与轨道头传 false。 */
    interpolate?: boolean;
    /** 右键菜单等外部交互（读数本身不认识"放置位置"这件事）。 */
    onContextMenu?: (event: React.MouseEvent<HTMLDivElement>) => void;
}

export function PlayheadTimeReadout({
    variant,
    compact = false,
    interpolate = true,
    onContextMenu,
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
    const reserveSec = Math.max(
        3600,
        Math.ceil(selector.projectSec),
        Math.ceil(selector.playheadSec),
    );
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
    const showSecondary = !compact && formatted.secondaryLabel !== null;

    if (variant === "toolbar" || variant === "statusBar") {
        return (
            <div
                className={
                    variant === "statusBar"
                        ? "flex h-qt-bar-status shrink-0 items-center gap-1.5 px-2 leading-none"
                        : "flex shrink-0 items-center gap-1.5 px-1 leading-none"
                }
                data-playhead-time-readout={variant}
                data-playhead-sec={visualSec.toFixed(6)}
                data-tooltip={formatted.combined}
                aria-label={ariaLabel}
                onContextMenu={onContextMenu}
            >
                <SlotText
                    text={formatted.primaryLabel}
                    reserveText={reservePrimary}
                    className="hs-type-mono text-qt-sm font-medium tabular-nums text-qt-text"
                />
                {showSecondary ? (
                    <>
                        {/* 分隔符只做视觉切分：用边框色，不用低对比文字。 */}
                        <span aria-hidden="true" className="text-qt-border">
                            ·
                        </span>
                        <SlotText
                            text={formatted.secondaryLabel ?? ""}
                            reserveText={reserveSecondary ?? undefined}
                            className="hs-type-mono text-qt-sm font-normal tabular-nums text-qt-text"
                        />
                    </>
                ) : null}
            </div>
        );
    }

    if (variant === "trackHeader") {
        return (
            <div
                className="min-w-0 flex flex-1 justify-end overflow-hidden pr-5"
                data-playhead-time-readout="trackHeader"
                data-playhead-sec={visualSec.toFixed(6)}
                data-tooltip={formatted.combined}
                aria-label={ariaLabel}
                onContextMenu={onContextMenu}
            >
                <SlotText
                    text={formatted.primaryLabel}
                    reserveText={reservePrimary}
                    tooltip={formatted.combined}
                    selectable
                    className="hs-type-body font-medium tabular-nums text-qt-text text-right leading-none whitespace-nowrap shrink-0"
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
            onContextMenu={onContextMenu}
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
