/**
 * 播放光标时间读数。
 *
 * 四个挂载槽位（同一时刻**只有一个**主实例，由 `PlayheadTimeReadoutSlot` 决定）：
 * - toolbar：主窗口 ActionBar 的 BPM & Time 组；
 * - statusBar：主窗口底部状态栏右侧；
 * - trackHeader：轨道头的紧凑副本（字号更小，但**同样显示主副两段**）；
 * - detached：独立窗口顶部条，直接用 30Hz Redux 采样，不做跨窗口视觉插值。
 *
 * 【为什么改成单行】两行块在工具栏里是**高度**上的浪费：主工具栏按功能组换行，
 * 一个两行读数会把整条工具栏顶高一行，而它显示的信息与一行完全相同。单行还让
 * "主单位 · 副单位"读起来是一句话，而不是两段互不相干的数字。
 *
 * 【为什么副单位不再用 `text-qt-text-muted`】它此前被压到近背景色：用户看到的是
 * "副单位不见了"，而不是"它是次要信息"。次要由**分隔符**表达，不由降低对比度表达 ——
 * 低对比度在这个界面上已经承担了"禁用"的语义。
 *
 * 【为什么轨道头也显示副单位】主/副单位是用户**分别**设定的：设了两段就说明两段都
 * 要看。曾经轨道头只显示主单位（当时叫 `compact`），结果是同一个偏好在不同位置
 * 给出不同的信息 —— 用户挪一下读数就少了一半内容，却没有任何提示说这是故意的。
 */
import { useCallback, useState } from "react";
import { shallowEqual } from "react-redux";

import { useAppSelector } from "../../../app/hooks";
import { isTransportAdvancing } from "../../../features/session/selectors/transport";
import { useI18n } from "../../../i18n/I18nProvider";
import { useVisualPlayhead } from "../../../hooks/useVisualPlayhead";
import { SlotText } from "../../../ui/SlotText";
import { usePlayheadTimeFormat } from "./usePlayheadTimeFormat";

export interface PlayheadTimeReadoutProps {
    variant: "toolbar" | "statusBar" | "trackHeader" | "detached";
    /** 是否使用共享视觉播放头插值；独立窗口与轨道头传 false。 */
    interpolate?: boolean;
    /** 右键菜单等外部交互（读数本身不认识"放置位置"这件事）。 */
    onContextMenu?: (event: React.MouseEvent<HTMLDivElement>) => void;
}

export function PlayheadTimeReadout({
    variant,
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

    const { formatted, reservePrimary, reserveSecondary } = usePlayheadTimeFormat(visualSec);
    const ariaLabel = t("playhead_time_readout") + ": " + formatted.combined;
    const showSecondary = formatted.secondaryLabel !== null;

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
        /*
         * 【为什么比工具栏小一号】轨道头列宽是用户可拖的（出厂 256px，插件首用 140px），
         * 而两段时间串在 13px 下已经接近这个宽度。11px + 等宽数字让两段在出厂宽度下
         * 一次放下，用户把列拖窄时也还有余量。
         *
         * 【为什么主单位 `shrink-0`、副单位可收缩】列再窄下去总得裁掉一点：裁主单位
         * 会连"这是第几小节"都读不出来，裁副单位只是少了那一段。因此主单位永不收缩，
         * 副单位按剩余宽度收缩并在尾部裁切（全文始终在气泡与 `aria-label` 里）。
         *
         * 【为什么不预留宽度】预留会把两段一起顶到裁切边缘，反而先牺牲主单位。
         */
        return (
            <div
                className="min-w-0 flex flex-1 justify-end overflow-hidden pr-5"
                data-playhead-time-readout="trackHeader"
                data-playhead-sec={visualSec.toFixed(6)}
                data-tooltip={formatted.combined}
                aria-label={ariaLabel}
                onContextMenu={onContextMenu}
            >
                <span className="flex min-w-0 items-baseline justify-end gap-1">
                    <SlotText
                        text={formatted.primaryLabel}
                        tooltip={formatted.combined}
                        selectable
                        className="text-qt-xs font-medium tabular-nums text-qt-text text-right leading-none whitespace-nowrap shrink-0"
                    />
                    {showSecondary ? (
                        <>
                            <span
                                aria-hidden="true"
                                className="text-qt-border text-qt-xs leading-none shrink-0"
                            >
                                ·
                            </span>
                            {/*
                              副单位外面多一层可收缩的盒子：`SlotText` 用内联
                              `min-width` 钉住自己的宽度（那是它防抖的机制），
                              所以收缩只能发生在它**外面**——外层收窄，内层照旧
                              等宽，多出来的部分被裁掉。
                            */}
                            <span className="flex min-w-0 overflow-hidden">
                                <SlotText
                                    text={formatted.secondaryLabel ?? ""}
                                    tooltip={formatted.combined}
                                    selectable
                                    className="text-qt-xs font-normal tabular-nums text-qt-text text-right leading-none whitespace-nowrap"
                                />
                            </span>
                        </>
                    ) : null}
                </span>
            </div>
        );
    }

    return (
        <div
            className="flex h-qt-bar-status items-center justify-end gap-1.5 px-2"
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
                className="hs-type-mono text-qt-xs tabular-nums text-qt-text"
            />
            {showSecondary ? (
                <>
                    <span aria-hidden="true" className="text-qt-border text-qt-xs leading-none">
                        ·
                    </span>
                    <SlotText
                        text={formatted.secondaryLabel ?? ""}
                        reserveText={reserveSecondary ?? undefined}
                        tooltip={formatted.combined}
                        className="hs-type-mono text-qt-xs tabular-nums text-qt-text"
                    />
                </>
            ) : null}
        </div>
    );
}
