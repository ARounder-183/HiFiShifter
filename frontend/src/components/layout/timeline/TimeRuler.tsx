import React, { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { Box } from "@radix-ui/themes";
import { screenXToWorldSec } from "./runtime/timelineWorld.js";
import { useNonPassiveWheel } from "../../../utils/useNonPassiveWheel";
import type { TimeFormatContext, TimeUnit, TimeUnitChoice } from "./timeFormat.js";
import { TIME_UNITS, TIME_UNIT_CHOICES, formatCursorTime } from "./timeFormat.js";
import type { GridSize } from "../../../features/session/sessionTypes.ts";
import type { ScaleLike } from "../../../utils/musicalScales.ts";
import { SCALE_LABELS } from "../../../utils/musicalScales.ts";
import type { CustomScalePreset } from "../../../utils/customScales.ts";
import type { TempoMap } from "../../../utils/tempoMap.ts";
import {
    computeTempoFloatingLabelState,
    effectiveScaleAtSec,
    effectiveTimeSignatureAt,
    formatTempoBpm,
    formatTimeSignature,
    pointIndexAtSec,
    removeTempoPoint,
    tempoPointHitTest,
} from "../../../utils/tempoMap.ts";
import {
    TempoMapRulerRow,
    TEMPO_ROW_HEIGHT_PX,
    type TempoPointEditRequest,
} from "./TempoMapRulerRow.tsx";
import { RULER_BASE_HEIGHT_PX, timeRulerHeightPx } from "./rulerHeight.ts";
import type { TimelineTick } from "./runtime/buildTimelineTicks.js";
import {
    readDevicePixelRatio,
    verticalHairlineGeometry,
    wholeDevicePxLength,
} from "../../../utils/devicePixelLine.ts";
import { playheadLineLeftViewportPx } from "../renderKernel/timelineAxis.ts";
import { clampAxisPosition } from "../../appTooltipPosition";

function unitLabelKey(unit: TimeUnit): string {
    switch (unit) {
        case "barBeats":
            return "time_unit_bar_beats";
        case "barDivisions":
            return "time_unit_bar_divisions";
        case "seconds":
            return "time_unit_seconds";
        case "clock":
            return "time_unit_clock";
    }
}

function ContextMenuItem({
    active,
    label,
    onSelect,
}: {
    active: boolean;
    label: string;
    onSelect: () => void;
}) {
    return (
        <button
            type="button"
            className="px-3 py-1.5 text-left w-full text-[12px] transition-colors flex items-center justify-between gap-3 hover:bg-qt-button-hover"
            onPointerDown={(e) => e.stopPropagation()}
            onClick={(e) => {
                e.stopPropagation();
                onSelect();
            }}
        >
            <span>{label}</span>
            {active ? <span className="text-[10px] opacity-50 shrink-0">✓</span> : null}
        </button>
    );
}

const ContextDivider: React.FC = () => <div className="my-1 border-t border-qt-border" />;

/**
 * 标尺刻度。
 *
 * 只渲染统一刻度源中标记了 `showLabel` 的刻度与小节起点，其余刻度由背景网格
 * 绘制——两者消费同一份数据，因此标尺刻度必然落在网格线上。
 */
const TimeRulerMarks = React.memo(function TimeRulerMarks({
    ticks,
    scrollLeft,
    viewportWidth,
}: {
    ticks: readonly TimelineTick[];
    scrollLeft: number;
    viewportWidth?: number;
}) {
    const visibleTicks = React.useMemo(() => {
        // 只渲染带标签的刻度：网格负责画出全部刻度（含无标签的小节线），
        // 标尺若连无标签的一起渲染，缩小时会留下一堆没有文字的竖线。
        const labeled = ticks.filter((tick) => tick.showLabel);
        if (!Number.isFinite(viewportWidth) || viewportWidth == null || viewportWidth <= 0) {
            return labeled;
        }
        const bufferPx = Math.max(320, viewportWidth * 0.5);
        const leftPx = Math.max(0, scrollLeft - bufferPx);
        const rightPx = scrollLeft + viewportWidth + bufferPx;
        // 按内容坐标二分：坐标已由 axis 投影好，Tempo Map 下也无需再换算。
        const lowerBound = (target: number) => {
            let lo = 0;
            let hi = labeled.length;
            while (lo < hi) {
                const mid = (lo + hi) >> 1;
                if (labeled[mid].contentPx < target) lo = mid + 1;
                else hi = mid;
            }
            return lo;
        };
        const start = Math.max(0, lowerBound(leftPx) - 1);
        const end = Math.min(labeled.length, lowerBound(rightPx) + 1);
        return labeled.slice(start, end);
    }, [ticks, scrollLeft, viewportWidth]);

    return (
        <>
            {visibleTicks.map((tick, tickIndex) => {
                // 设备像素比在每次渲染时现读（放在 map 内而不是组件体：组件体里的
                // 非纯调用会让 React Compiler 无法保留下面那个 useMemo 的记忆化）。
                const dpr = readDevicePixelRatio();
                // 竖线的位置与线宽都按**设备像素**取整（见 `devicePixelLine`）。
                //
                // 【为什么必须这样】内容层虽已把平移量吸附到设备像素（见
                // `rulerLayerTranslatePx`），但刻度自身的 `contentPx` 是小数：
                // 层原点 + 小数刻度 ⇒ 竖线跨在两个物理像素上被抗锯齿，且覆盖度随每条
                // 刻度的小数部分变化 —— 系统缩放率 > 1 时表现为**同一排竖线粗细不一**。
                // 内核的网格线不会这样：它按设备像素绘制。这里采用同一份吸附，两层
                // 因此逐设备像素对齐。
                const { left, width: lineWidth } = verticalHairlineGeometry(
                    tick.contentPx,
                    tick.isBarStart ? 2 : 1,
                    dpr,
                );
                // 版式完全由生成器决定（见 `TimelineTick.labelMaxWidth`）：
                // 渲染期不再做"与可见切片里的下一条比较"——那个判据会随滚动位置
                // 改变，正是"标尺文字时有时无"的来源。这里只消费结果。
                const labelMaxWidth = tick.labelMaxWidth;
                return (
                    // key 用**位置**而不是音乐身份：刻度全是无状态的展示节点，位置
                    // key 让 React 永远复用同一批 DOM，只更新 left/文本。用身份 key
                    // 时，一旦网格步长在缩放/BPM 阈值处整档变化，全部 key 同时改名 ⇒
                    // 整棵刻度子树卸载重建 —— 连续手势下这就是一次可见的闪烁。
                    <div key={tickIndex} className="absolute top-0 bottom-0" style={{ left }}>
                        <div
                            className="absolute top-0 bottom-0"
                            style={{
                                // 与下方网格保持一致：小节线 2 物理像素、弱线 1 物理像素；
                                // 都以刻度位置为中心（居中量按设备像素宽度算）。
                                left: 0,
                                width: lineWidth,
                                backgroundColor: "var(--qt-border)",
                                opacity: tick.isBarStart ? 1 : 0.6,
                            }}
                        />
                        <div
                            className="flex flex-col justify-center h-full pl-2 pr-1 select-none"
                            style={{
                                maxWidth: labelMaxWidth ?? undefined,
                                overflow: "hidden",
                            }}
                        >
                            <div
                                className={
                                    tick.isBarStart
                                        ? "text-[13px] leading-tight font-semibold text-qt-text tabular-nums whitespace-nowrap"
                                        : "text-[13px] leading-tight text-qt-text tabular-nums whitespace-nowrap"
                                }
                            >
                                {tick.primaryLabel}
                            </div>
                            {tick.secondaryLabel != null ? (
                                <>
                                    <div className="w-5 border-t border-qt-border/20 my-[3px]" />
                                    <div className="text-[10px] leading-tight text-qt-text-muted/45 tabular-nums whitespace-nowrap">
                                        {tick.secondaryLabel}
                                    </div>
                                </>
                            ) : null}
                        </div>
                    </div>
                );
            })}
        </>
    );
});

const TimeRulerPlayhead = React.memo(function TimeRulerPlayhead({
    playheadSec,
    pxPerSec,
    scrollLeft,
    lineRef,
    headRef,
    positionFromProps,
}: {
    playheadSec: number;
    pxPerSec: number;
    /** 绘制坐标下的水平滚动量（线在视口坐标里定位，因此需要它）。 */
    scrollLeft: number;
    lineRef?: React.Ref<HTMLDivElement>;
    headRef?: React.Ref<HTMLDivElement>;
    /**
     * 位置是否由 React 渲染。
     *
     * 【为什么允许调用方关掉它】播放头的 DOM 线在调用方那里是**逐帧命令式写入**
     * 的（60Hz 插值位置），而 React 只持有 30Hz 轮询提交的**已提交位置**。两者
     * 同时写同一个 `style.left` 就会互相覆盖：React 那次提交晚于同帧的 rAF 写入
     * 时，标尺上的线会停在旧位置，而画布/GL 上的线已经在新位置 —— 表现为两条
     * 播放头线"分离"，播放中尤其明显（缩放越大，一个 33ms 采样周期对应的像素差
     * 越大）。传 `false` 即表示"位置由我负责"，React 只提供首帧的初始值。
     */
    positionFromProps?: boolean;
}) {
    // 播放头竖线设备像素对齐（与 56238d45 的网格线修复同根同法）：
    // 分数 DPR（125%/150%）下 1px CSS 线覆盖 1.25/1.5 物理像素，落点相位
    // 随播放/拖动连续变化，取整后粗细在 1↔2 物理像素间跳动。位置吸附到
    // 设备像素边界、线宽取整物理像素后粗细恒定；本渲染值是初始值，播放/
    // 滚动期间的逐帧写入（useVisualPlayhead onFrame / syncScrollLeft）使用
    // 同一套吸附，双方逐设备像素一致。
    const dpr = readDevicePixelRatio();
    // 视口坐标（不再是内容坐标）：与 GL 主体播放头同一个换算函数，两条线逐设备
    // 像素一致；也避免被内容层的小数平移带偏。
    const playheadLeft = playheadLineLeftViewportPx({
        sec: playheadSec,
        pxPerSec,
        scrollLeftPx: scrollLeft,
        dpr,
    });
    // 首帧仍给出正确位置（避免挂载瞬间闪到 0），此后不再由 React 改写。
    const lineStyle: React.CSSProperties =
        positionFromProps === false
            ? { width: wholeDevicePxLength(1, dpr) }
            : { left: playheadLeft, width: wholeDevicePxLength(1, dpr) };
    return (
        <>
            <div
                ref={lineRef}
                className="absolute top-0 bottom-0 bg-qt-playhead z-20 pointer-events-none"
                style={lineStyle}
            />
            <div
                ref={headRef}
                className="absolute top-0 z-30 pointer-events-none"
                style={{
                    ...(positionFromProps === false ? null : { left: playheadLeft }),
                    transform: "translateX(-6px)",
                }}
            >
                <div className="w-0 h-0 border-l-[6px] border-l-transparent border-r-[6px] border-r-transparent border-t-[8px] border-t-qt-playhead" />
            </div>
        </>
    );
});

/**
 * 视口左侧的“悬浮标签”：当管辖画面最左侧的 Tempo Map 变化点旗帜
 * （蓝色标签）滚出画面左侧后，在最左侧浮一个同款标签展示该段参数。
 *
 * - 外观与变化点旗帜一致，一眼可识别为“悬浮”的旗帜标签；
 * - 内容切换使用 key 重挂载 + 淡入动画（旧文本立即移除、新文本淡入，绝不重叠）；
 * - 任何旗帜与悬浮标签区域重叠时整体淡出，避免互相遮挡；
 * - 与固定标签提供完全相同的交互：双击进入输入编辑状态、
 *   右键直接弹出“速度映射变化点”编辑窗口。
 */
const TempoMapFloatingLabel = React.memo(function TempoMapFloatingLabel({
    tempoMap,
    scrollLeft,
    pxPerSec,
    tooltip,
    onInlineEdit,
    onOpenDialog,
    hidden,
}: {
    tempoMap: TempoMap;
    scrollLeft: number;
    pxPerSec: number;
    /** 自定义悬浮提示（与变化点旗帜一致的“位置/BPM/拍号/音阶”内容）。 */
    tooltip: string;
    /** 双击：进入管辖变化点的输入编辑状态。 */
    onInlineEdit: () => void;
    /** 右键：弹出管辖变化点的编辑窗口。 */
    onOpenDialog: () => void;
    /** 输入编辑进行中：隐藏自身，避免遮挡显示在视口左侧的内联输入框。 */
    hidden?: boolean;
}) {
    const { label, governingOffscreen, blocked } = computeTempoFloatingLabelState({
        tempoMap,
        scrollLeft,
        pxPerSec,
    });
    const visible = governingOffscreen && !blocked && !hidden;

    return (
        <>
            <style>{`
                @keyframes hs-tempo-float-in {
                    from { opacity: 0; transform: translateY(1px); }
                    to { opacity: 1; transform: translateY(0); }
                }
                .hs-tempo-float-label {
                    animation: hs-tempo-float-in 140ms ease-out;
                }
            `}</style>
            <div
                className="absolute select-none pointer-events-none"
                style={{
                    top: RULER_BASE_HEIGHT_PX + 4,
                    left: 2,
                    height: TEMPO_ROW_HEIGHT_PX,
                    display: "flex",
                    alignItems: "center",
                    opacity: visible ? 1 : 0,
                    transition: "opacity 150ms ease",
                    zIndex: 25,
                }}
            >
                {/* 与变化点旗帜同款尺寸（9px / 行高 11px / px-1），仅加投影以示“悬浮”。 */}
                {/* 注意：仅在可见时接收指针事件 —— 隐藏（opacity: 0）时若仍可点击， */}
                {/* 会挡住其下方的初始变化点旗帜（双击无法进入编辑模式）。 */}
                <div
                    className="px-1 rounded-[2px] text-[9px] leading-[11px] whitespace-nowrap font-medium shadow-md"
                    style={{
                        backgroundColor: "var(--qt-panel)",
                        color: "var(--qt-text)",
                        boxShadow:
                            "inset 0 0 0 1px color-mix(in srgb, var(--qt-border) 70%, transparent), 0 2px 8px var(--qt-overlay)",
                        pointerEvents: visible ? "auto" : "none",
                        cursor: "pointer",
                    }}
                    data-tooltip={tooltip}
                    onDoubleClick={(e) => {
                        e.preventDefault();
                        e.stopPropagation();
                        if (visible) onInlineEdit();
                    }}
                    onContextMenu={(e) => {
                        e.preventDefault();
                        e.stopPropagation();
                        if (visible) onOpenDialog();
                    }}
                >
                    <span key={label} className="hs-tempo-float-label">
                        {label}
                    </span>
                </div>
            </div>
        </>
    );
});

function TimeRulerContextMenu({
    x,
    y,
    primaryUnit,
    secondaryUnit,
    tempoMap,
    clickedSec,
    pxPerSec,
    t,
    onSelectPrimary,
    onSelectSecondary,
    onCopyPlayheadTime,
    onOpenSettings,
    onAddTempoPointAt,
    onEditTempoPoint,
    onDeleteTempoPoint,
    onClearTempoMap,
    onClose,
}: {
    x: number;
    y: number;
    primaryUnit: TimeUnit;
    secondaryUnit: TimeUnitChoice;
    tempoMap: TempoMap | null;
    clickedSec: number;
    pxPerSec: number;
    t: (key: string) => string;
    onSelectPrimary: (unit: TimeUnit) => void;
    onSelectSecondary: (unit: TimeUnitChoice) => void;
    onCopyPlayheadTime?: () => void;
    onOpenSettings?: () => void;
    onAddTempoPointAt: (sec: number, focus: "tempo" | "timeSignature" | "scale" | null) => void;
    onEditTempoPoint: (id: string) => void;
    onDeleteTempoPoint: (id: string) => void;
    onClearTempoMap: () => void;
    onClose: () => void;
}) {
    const ref = useRef<HTMLDivElement | null>(null);
    useLayoutEffect(() => {
        const el = ref.current;
        if (!el) return;
        const rect = el.getBoundingClientRect();
        const vw = window.innerWidth;
        const vh = window.innerHeight;
        if (rect.right > vw) {
            el.style.left = `${Math.max(0, vw - rect.width)}px`;
        }
        if (rect.bottom > vh) {
            el.style.top = `${Math.max(0, vh - rect.height)}px`;
        }
    }, [x, y]);

    // 找到点击位置命中的变化点（旗帜可视范围：点的位置向右延伸整个旗帜文本宽度）。
    const nearPoint = React.useMemo(() => {
        if (!tempoMap) return null;
        const index = tempoPointHitTest(tempoMap, clickedSec, pxPerSec);
        if (index == null) return null;
        return { point: tempoMap.points[index], isFirst: index === 0 };
    }, [tempoMap, clickedSec, pxPerSec]);
    const hasMap = tempoMap != null && tempoMap.points.length > 0;

    return createPortal(
        <div
            ref={ref}
            data-time-ruler-context-menu
            data-hs-context-menu="1"
            data-hs-floating-menu="1"
            className="fixed z-[999] min-w-[140px] rounded border border-qt-border bg-qt-window text-qt-text shadow-lg py-1"
            style={{ left: x, top: y }}
            onPointerDown={(e) => e.stopPropagation()}
            onMouseDown={(e) => {
                e.preventDefault();
                e.stopPropagation();
            }}
            onContextMenu={(e) => {
                e.preventDefault();
                e.stopPropagation();
            }}
        >
            {/* Tempo Map（位于“主时间单位”等选项之前） */}
            <div className="px-3 py-1 text-[11px] text-qt-text/50 select-none">
                {t("tempo_map")}
            </div>
            <ContextMenuItem
                active={false}
                label={t("tempo_map_add_point")}
                onSelect={() => {
                    onAddTempoPointAt(clickedSec, null);
                    onClose();
                }}
            />
            {nearPoint ? (
                <ContextMenuItem
                    active={false}
                    label={t("tempo_map_edit_point")}
                    onSelect={() => {
                        onEditTempoPoint(nearPoint.point.id);
                        onClose();
                    }}
                />
            ) : null}
            {nearPoint && !nearPoint.isFirst ? (
                <ContextMenuItem
                    active={false}
                    label={t("tempo_map_delete_point")}
                    onSelect={() => {
                        onDeleteTempoPoint(nearPoint.point.id);
                        onClose();
                    }}
                />
            ) : null}
            {hasMap ? (
                <ContextMenuItem
                    active={false}
                    label={t("tempo_map_clear_all")}
                    onSelect={() => {
                        onClearTempoMap();
                        onClose();
                    }}
                />
            ) : null}
            <ContextDivider />

            <div className="px-3 py-1 text-[11px] text-qt-text/50 select-none">
                {t("time_unit_primary")}
            </div>
            {TIME_UNITS.map((unit) => (
                <ContextMenuItem
                    key={unit}
                    active={primaryUnit === unit}
                    label={t(unitLabelKey(unit))}
                    onSelect={() => {
                        onSelectPrimary(unit);
                        onClose();
                    }}
                />
            ))}
            <ContextDivider />
            <div className="px-3 py-1 text-[11px] text-qt-text/50 select-none">
                {t("time_unit_secondary")}
            </div>
            {TIME_UNIT_CHOICES.map((unit) => (
                <ContextMenuItem
                    key={unit}
                    active={secondaryUnit === unit}
                    label={
                        unit === "none" ? t("time_unit_none") : t(unitLabelKey(unit as TimeUnit))
                    }
                    onSelect={() => {
                        onSelectSecondary(unit);
                        onClose();
                    }}
                />
            ))}
            <ContextDivider />
            {onCopyPlayheadTime ? (
                <ContextMenuItem
                    active={false}
                    label={t("copy_playhead_time")}
                    onSelect={() => {
                        onCopyPlayheadTime();
                        onClose();
                    }}
                />
            ) : null}
            {onOpenSettings ? (
                <ContextMenuItem
                    active={false}
                    label={t("timeline_display_settings")}
                    onSelect={() => {
                        onOpenSettings();
                        onClose();
                    }}
                />
            ) : null}
        </div>,
        document.body,
    );
}

const TimeRulerInner: React.FC<{
    /**
     * **真实**水平滚动位置（绘制坐标）。
     *
     * 用于交互换算（悬停时间、Tempo Map 行的可见段、播放头定位）——这些地方需要
     * 尽可能接近用户看到的滚动位置。**不**用于刻度切片：切片用
     * `tickWindowAnchorPx`（量化锚点），两者相差不超过一个量化步长，切片缓冲足以
     * 覆盖。曾经一个字段身兼两职，量化值会把悬停时间算偏最多一个步长。
     */
    scrollLeft: number;
    /**
     * 刻度切片用的量化锚点（`createTickAxis` 的产物）。
     *
     * 缺省取 `scrollLeft`。时间轴与参数编辑器都传量化值，以换取"滚动时不重算
     * 刻度"；它必须与生成刻度时使用的锚点**是同一个值**，否则切片窗口与生成窗口
     * 不一致（参数编辑器曾经就是这样漏掉宽度补偿的）。
     */
    tickWindowAnchorPx?: number;
    ticks: readonly TimelineTick[];
    pxPerSec: number;
    viewportWidth?: number;
    playheadSec: number;
    /**
     * 标尺上的滚轮（由面板决定语义：默认水平滚动，按住"滚动条滚轮缩放"修饰键时水平缩放）。
     *
     * 【为什么必须交给面板】标尺是内核容器**之外**的 DOM 条，它的滚轮进不了容器监听；
     * 而滚动/缩放的落点换算、上下限都在各自面板里。标尺只负责"阻止默认滚动 + 转交"。
     */
    onRulerWheel?: (event: React.WheelEvent<HTMLDivElement>) => void;
    /** 见 `TimeRulerPlayhead.positionFromProps`。 */
    positionPlayheadFromProps?: boolean;
    playheadLineRef?: React.Ref<HTMLDivElement>;
    playheadHeadRef?: React.Ref<HTMLDivElement>;
    onMouseDown: (e: React.MouseEvent<HTMLDivElement>) => void;
    onMouseDownAtSec?: (sec: number, e: React.MouseEvent<HTMLDivElement>) => void;
    contentRef?: React.Ref<HTMLDivElement>;
    timeContext: TimeFormatContext;
    primaryUnit: TimeUnit;
    secondaryUnit: TimeUnitChoice;
    onPrimaryUnitChange: (unit: TimeUnit) => void;
    onSecondaryUnitChange: (unit: TimeUnitChoice) => void;
    onOpenSettings?: () => void;
    onCopyPlayheadTime?: () => void;
    t?: (key: string) => string;
    /** ── Tempo Map ── */
    tempoMap?: TempoMap | null;
    tempoMapVisible?: boolean;
    projectSec?: number;
    grid?: GridSize;
    snapEnabled?: boolean;
    timelineSnap?: import("../../../features/session/sessionTypes").TimelineSnapSettings;
    projectScale?: ScaleLike | null;
    projectScaleName?: string;
    /** 工程基准拍号分母（无 Tempo Map 时新建首点 / 初始点拍号物化用）。 */
    fallbackDenominator?: number;
    customScalePresets?: readonly CustomScalePreset[];
    /** 本地即时更新（拖动中，仅 Redux）。 */
    onTempoMapChange?: (next: TempoMap | null) => void;
    /** 离散提交（对话框/菜单/拖拽结束），同步后端。 */
    onTempoMapCommit?: (next: TempoMap | null) => void;
    /**
     * 本面板的视口总线订阅，透传给 Tempo Map 行（拖拽期间的视口重放用）。
     * 见 `TempoMapRulerRow.subscribeViewport`。
     */
    subscribeViewport?: (listener: (scrollLeft: number, pxPerSec: number) => void) => () => void;
}> = ({
    scrollLeft,
    tickWindowAnchorPx,
    ticks,
    pxPerSec,
    viewportWidth,
    playheadSec,
    positionPlayheadFromProps,
    onRulerWheel,
    playheadLineRef,
    playheadHeadRef,
    onMouseDown,
    onMouseDownAtSec,
    contentRef,
    timeContext,
    primaryUnit,
    secondaryUnit,
    onPrimaryUnitChange,
    onSecondaryUnitChange,
    onOpenSettings,
    onCopyPlayheadTime,
    t,
    tempoMap = null,
    tempoMapVisible = true,
    projectSec = 0,
    grid = "1/4",
    snapEnabled = true,
    timelineSnap,
    projectScale = null,
    projectScaleName,
    fallbackDenominator,
    customScalePresets = [],
    onTempoMapChange,
    onTempoMapCommit,
    subscribeViewport,
}) => {
    const tAny = useMemo(() => t ?? ((key: string) => key), [t]);
    const useManualTransform = contentRef != null;
    const [ctxMenu, setCtxMenu] = useState<{ x: number; y: number; sec: number } | null>(null);
    const [hover, setHover] = useState<{ x: number; y: number; sec: number } | null>(null);
    const [tempoEditRequest, setTempoEditRequest] = useState<TempoPointEditRequest | null>(null);
    /** Tempo Map 编辑对话框打开时抑制标尺悬浮时间提示。 */
    const [tempoDialogOpen, setTempoDialogOpen] = useState(false);
    /** 标签内联输入编辑进行中：隐藏悬浮标签，避免遮挡视口左侧的输入框。 */
    const [tempoInlineEditing, setTempoInlineEditing] = useState(false);
    /**
     * 变化点正在被悬停或拖拽：此时已有一个内容更丰富的变化点提示，标尺自己的
     * 悬浮时间提示必须让位（两个气泡叠在一起既看不清也互相遮挡）。
     */
    const [tempoInteracting, setTempoInteracting] = useState(false);
    const rulerRef = useRef<HTMLDivElement | null>(null);

    /**
     * 标尺滚轮：**非被动**监听 + 转交面板。
     *
     * 曾经的 `onWheel` 只调 `e.preventDefault()` 就结束（注释写着"防止标尺成为第二个
     * 滚动源"）—— 而 React 的 `onWheel` 是 passive 的，那行 `preventDefault` 本身是空
     * 操作：既没阻止默认滚动，也没做任何滚动/缩放。用户报告"标尺上滚轮没反应"。
     */
    const attachRulerWheel = useNonPassiveWheel<HTMLDivElement>((event) => {
        event.preventDefault();
        onRulerWheel?.(event);
    });

    /** 标尺根：既供内部量测（`rulerRef`），也挂非被动滚轮监听。 */
    const attachRulerRoot = useCallback(
        (element: HTMLDivElement | null) => {
            rulerRef.current = element;
            attachRulerWheel(element);
        },
        [attachRulerWheel],
    );

    const showTempoRow = Boolean(tempoMap && tempoMap.points.length > 0 && tempoMapVisible);
    const rulerHeight = timeRulerHeightPx(showTempoRow);

    const handleAddTempoPointAt = useCallback(
        (sec: number, focus: "tempo" | "timeSignature" | "scale" | null) => {
            setTempoEditRequest({ pointId: null, positionSec: Math.max(0, sec), focus });
        },
        [],
    );
    const handleTempoDialogOpenChange = useCallback((open: boolean) => {
        setTempoDialogOpen(open);
        if (open) setHover(null);
    }, []);
    const handleTempoInteractionChange = useCallback((active: boolean) => {
        setTempoInteracting(active);
        if (active) setHover(null);
    }, []);
    const handleEditTempoPoint = useCallback((id: string) => {
        setTempoEditRequest({ pointId: id, positionSec: null, focus: null, mode: "dialog" });
    }, []);

    /** 悬浮标签管辖的变化点 id（画面最左侧生效的变化点）。 */
    const floatingGoverningPointId = useMemo(() => {
        if (!tempoMap || tempoMap.points.length === 0) return null;
        const leftSec = Math.max(0, scrollLeft / Math.max(1e-9, pxPerSec));
        return tempoMap.points[pointIndexAtSec(tempoMap, leftSec)].id;
    }, [tempoMap, scrollLeft, pxPerSec]);

    /** 悬浮标签双击：进入管辖变化点的输入编辑状态。 */
    const handleFloatingInlineEdit = useCallback(() => {
        if (floatingGoverningPointId) {
            setTempoEditRequest({
                pointId: floatingGoverningPointId,
                positionSec: null,
                focus: null,
                mode: "inline",
            });
        }
    }, [floatingGoverningPointId]);

    /** 悬浮标签右键：弹出管辖变化点的编辑窗口。 */
    const handleFloatingOpenDialog = useCallback(() => {
        if (floatingGoverningPointId) {
            setTempoEditRequest({
                pointId: floatingGoverningPointId,
                positionSec: null,
                focus: null,
                mode: "dialog",
            });
        }
    }, [floatingGoverningPointId]);
    const handleDeleteTempoPoint = useCallback(
        (id: string) => {
            if (tempoMap && onTempoMapCommit) {
                onTempoMapCommit(removeTempoPoint(tempoMap, id));
            }
        },
        [tempoMap, onTempoMapCommit],
    );
    const handleClearTempoMap = useCallback(() => {
        onTempoMapCommit?.(null);
    }, [onTempoMapCommit]);

    /** 视口左侧悬浮标签的自定义提示：管辖点的“位置/BPM/拍号/音阶”（跟随值解析为实际值）。 */
    const floatingTooltip = useMemo(() => {
        if (!tempoMap || tempoMap.points.length === 0) return "";
        const leftSec = Math.max(0, scrollLeft / Math.max(1e-9, pxPerSec));
        const idx = pointIndexAtSec(tempoMap, leftSec);
        const point = tempoMap.points[idx];
        const cursor = formatCursorTime(primaryUnit, secondaryUnit, point.positionSec, timeContext);
        const positionLine = cursor.secondaryLabel
            ? `${cursor.primaryLabel} / ${cursor.secondaryLabel}`
            : cursor.primaryLabel;
        const sig = effectiveTimeSignatureAt(tempoMap, idx);
        const effScale = effectiveScaleAtSec(
            tempoMap,
            point.positionSec,
            projectScale ?? undefined,
        );
        let effScaleLabel = "—";
        if (typeof effScale === "string") {
            effScaleLabel = SCALE_LABELS[effScale as keyof typeof SCALE_LABELS] ?? effScale;
        } else if (Array.isArray(effScale)) {
            effScaleLabel = projectScaleName || "…";
        }
        return [
            `${tAny("tempo_map_tooltip_position")}${positionLine}`,
            `${tAny("tempo_map_tooltip_bpm")}${formatTempoBpm(point.bpm)}`,
            `${tAny("tempo_map_tooltip_time_signature")}${formatTimeSignature(sig)}`,
            `${tAny("tempo_map_tooltip_scale")}${effScaleLabel}`,
        ].join("\n");
    }, [
        tempoMap,
        scrollLeft,
        pxPerSec,
        primaryUnit,
        secondaryUnit,
        timeContext,
        projectScale,
        projectScaleName,
        tAny,
    ]);

    useEffect(() => {
        if (!ctxMenu) return;
        const close = (e: PointerEvent) => {
            const target = e.target as HTMLElement | null;
            if (target?.closest?.("[data-time-ruler-context-menu]")) return;
            setCtxMenu(null);
        };
        const onKey = (e: KeyboardEvent) => {
            if (e.key === "Escape") setCtxMenu(null);
        };
        window.addEventListener("pointerdown", close, true);
        window.addEventListener("keydown", onKey, true);
        return () => {
            window.removeEventListener("pointerdown", close, true);
            window.removeEventListener("keydown", onKey, true);
        };
    }, [ctxMenu]);

    const handleMouseMove = useCallback(
        (e: React.MouseEvent<HTMLDivElement>) => {
            // 右键菜单 / Tempo Map 编辑对话框打开 / 变化点交互中：不显示标尺悬浮时间。
            // 最后一项覆盖"悬停或拖拽变化点"——那时已有变化点自己的提示。
            if (ctxMenu || tempoDialogOpen || tempoInteracting) {
                setHover(null);
                return;
            }
            const target = e.target as HTMLElement | null;
            if (target?.closest?.("[data-time-ruler-context-menu]")) {
                setHover(null);
                return;
            }
            // 门户内容（如弹出对话框）上的移动不触发标尺悬浮提示。
            const box = e.currentTarget as HTMLElement | null;
            if (!box || !target || !box.contains(target)) {
                setHover(null);
                return;
            }
            const bounds = e.currentTarget.getBoundingClientRect();
            const sec = Math.max(
                0,
                screenXToWorldSec(e.clientX - bounds.left, {
                    pxPerSec,
                    rowHeight: 1,
                    scrollLeftPx: scrollLeft,
                    scrollTopPx: 0,
                }),
            );
            setHover({ x: e.clientX - bounds.left, y: e.clientY - bounds.top, sec });
        },
        [ctxMenu, tempoDialogOpen, tempoInteracting, pxPerSec, scrollLeft],
    );

    const hoverTime = hover
        ? formatCursorTime(primaryUnit, secondaryUnit, hover.sec, timeContext)
        : null;
    // 悬停时间气泡的水平定位：挂载后按实测宽度夹紧（clampAxisPosition）。
    // 旧实现预留固定 260px（viewportWidth - 260），把标尺右缘 260px 内的
    // 悬停气泡整段甩离光标 —— 与 AppTooltip 已修复的同类问题同根。
    const hoverBubbleRef = useRef<HTMLDivElement | null>(null);
    useLayoutEffect(() => {
        const el = hoverBubbleRef.current;
        if (!el || !hover) return;
        const rulerWidth = rulerRef.current?.clientWidth ?? viewportWidth ?? 0;
        el.style.left = `${clampAxisPosition(hover.x, el.offsetWidth, rulerWidth, 10, 4)}px`;
        // 只在输入变化时重定位：无依赖数组会在滚动 / 缩放热路径的每次渲染
        // 都强制回流（读 offsetWidth / clientWidth）。
    }, [hover, viewportWidth]);

    /**
     * 事件是否来自标尺自身 DOM 子树之外（如 Radix Dialog 门户到 body 的
     * 弹窗内容）。React 门户事件会沿 React 树冒泡到本 Box，但这类点击
     * 完全不应触发标尺行为（播放头定位 / 右键菜单）。
     */
    const isForeignEvent = (e: React.SyntheticEvent) => {
        const target = e.target as Node | null;
        if (!target) return true;
        const box = e.currentTarget as HTMLElement | null;
        if (!box) return true;
        return !box.contains(target);
    };

    return (
        <Box
            ref={attachRulerRoot}
            className="bg-qt-window border-b border-qt-border relative overflow-hidden shrink-0 select-none"
            style={{ height: rulerHeight }}
            onMouseDown={(e) => {
                // 编辑对话框等门户内容上的点击不影响标尺（播放头定位）。
                if (isForeignEvent(e)) return;
                if (e.button === 1) {
                    e.preventDefault();
                    return;
                }
                if (e.button !== 0) return;
                const bounds = e.currentTarget.getBoundingClientRect();
                onMouseDownAtSec?.(
                    screenXToWorldSec(e.clientX - bounds.left, {
                        pxPerSec,
                        rowHeight: 1,
                        scrollLeftPx: scrollLeft,
                        scrollTopPx: 0,
                    }),
                    e,
                );
                onMouseDown(e);
            }}
            onAuxClick={(e) => {
                if (isForeignEvent(e)) return;
                if (e.button === 1) e.preventDefault();
            }}
            onContextMenu={(e) => {
                if (isForeignEvent(e)) return;
                e.preventDefault();
                e.stopPropagation();
                const bounds = e.currentTarget.getBoundingClientRect();
                const sec = Math.max(
                    0,
                    screenXToWorldSec(e.clientX - bounds.left, {
                        pxPerSec,
                        rowHeight: 1,
                        scrollLeftPx: scrollLeft,
                        scrollTopPx: 0,
                    }),
                );
                setHover(null);
                setCtxMenu({ x: e.clientX, y: e.clientY, sec });
            }}
            onMouseMove={handleMouseMove}
            onMouseLeave={() => setHover(null)}
        >
            <div
                ref={contentRef}
                className="absolute inset-0 will-change-transform"
                style={
                    useManualTransform ? undefined : { transform: `translateX(${-scrollLeft}px)` }
                }
            >
                <div className="absolute inset-x-0 top-0" style={{ height: RULER_BASE_HEIGHT_PX }}>
                    <TimeRulerMarks
                        ticks={ticks}
                        scrollLeft={tickWindowAnchorPx ?? scrollLeft}
                        viewportWidth={viewportWidth}
                    />
                </div>
                {/* Tempo Map 行：无论当前是否有数据都保持挂载 ——
                    右键菜单发出的“添加第一个变化点”请求由该组件内部的
                    editRequest effect 处理；无数据时组件内部渲染 null（行隐藏）。 */}
                <TempoMapRulerRow
                    tempoMap={tempoMap}
                    visible={tempoMapVisible}
                    pxPerSec={pxPerSec}
                    scrollLeft={scrollLeft}
                    viewportWidth={viewportWidth ?? 0}
                    projectSec={projectSec}
                    grid={grid}
                    snapEnabled={snapEnabled}
                    snapSettings={timelineSnap}
                    fallbackBpm={timeContext.bpm}
                    fallbackBeatsPerBar={timeContext.beatsPerBar}
                    fallbackDenominator={fallbackDenominator}
                    projectScale={projectScale}
                    projectScaleName={projectScaleName}
                    customScalePresets={customScalePresets}
                    primaryUnit={primaryUnit}
                    secondaryUnit={secondaryUnit}
                    timeContext={timeContext}
                    t={tAny}
                    onChange={onTempoMapChange ?? (() => undefined)}
                    onCommit={onTempoMapCommit ?? onTempoMapChange ?? (() => undefined)}
                    editRequest={tempoEditRequest}
                    onEditRequestHandled={() => setTempoEditRequest(null)}
                    onDialogOpenChange={handleTempoDialogOpenChange}
                    onTempoInteractionChange={handleTempoInteractionChange}
                    onFloatingInlineEditChange={setTempoInlineEditing}
                    subscribeViewport={subscribeViewport}
                />
            </div>

            {/*
              播放头竖线刻意放在**内容平移层之外**，按视口坐标定位。
              内容层带 `will-change: transform` 且被 `translateX(-小数)` 平移：合成层
              会以自己的原点栅格化子元素，于是 1 物理像素的竖线落在半个设备像素上被
              抗锯齿 —— 表现为"宽度忽粗忽细、颜色发淡"，与 GL 直接按设备像素绘制的
              主体线看上去不像同一条线。移出图层后，左缘即 GL 用的同一函数结果，
              两条线逐设备像素重合。
            */}
            <TimeRulerPlayhead
                playheadSec={playheadSec}
                positionFromProps={positionPlayheadFromProps}
                pxPerSec={pxPerSec}
                scrollLeft={scrollLeft}
                lineRef={playheadLineRef}
                headRef={playheadHeadRef}
            />

            {/* 时间标尺与 Tempo Map 行之间的分隔横线：固定在标尺盒内（视口宽度），
                不随内容平移/缩放伸缩 —— 与标尺底部边框等其他横线一致。 */}
            {showTempoRow ? (
                <div
                    className="absolute left-0 right-0 pointer-events-none"
                    style={{
                        top: RULER_BASE_HEIGHT_PX,
                        height: 1,
                        backgroundColor: "var(--qt-border)",
                        opacity: 0.6,
                    }}
                />
            ) : null}

            {/* 视口左侧悬浮标签（在 contentRef 之外，跟随视口而非内容滚动）。 */}
            {showTempoRow && tempoMap ? (
                <TempoMapFloatingLabel
                    tempoMap={tempoMap}
                    scrollLeft={scrollLeft}
                    pxPerSec={pxPerSec}
                    tooltip={floatingTooltip}
                    onInlineEdit={handleFloatingInlineEdit}
                    onOpenDialog={handleFloatingOpenDialog}
                    hidden={tempoInlineEditing}
                />
            ) : null}

            {hover && hoverTime ? (
                <div
                    ref={hoverBubbleRef}
                    className="absolute top-1 z-40 pointer-events-none rounded border border-qt-border bg-qt-panel px-2 py-1 shadow-lg"
                    style={{ left: hover.x + 10 }}
                >
                    <div className="text-[12px] leading-tight text-qt-text tabular-nums whitespace-nowrap">
                        {hoverTime.primaryLabel}
                    </div>
                    {hoverTime.secondaryLabel ? (
                        <div className="text-[10px] leading-tight text-qt-text-muted/60 tabular-nums whitespace-nowrap">
                            {hoverTime.secondaryLabel}
                        </div>
                    ) : null}
                </div>
            ) : null}

            {ctxMenu ? (
                <TimeRulerContextMenu
                    x={ctxMenu.x}
                    y={ctxMenu.y}
                    primaryUnit={primaryUnit}
                    secondaryUnit={secondaryUnit}
                    tempoMap={tempoMap}
                    clickedSec={ctxMenu.sec}
                    pxPerSec={pxPerSec}
                    t={tAny}
                    onSelectPrimary={onPrimaryUnitChange}
                    onSelectSecondary={onSecondaryUnitChange}
                    onCopyPlayheadTime={onCopyPlayheadTime}
                    onOpenSettings={onOpenSettings}
                    onAddTempoPointAt={handleAddTempoPointAt}
                    onEditTempoPoint={handleEditTempoPoint}
                    onDeleteTempoPoint={handleDeleteTempoPoint}
                    onClearTempoMap={handleClearTempoMap}
                    onClose={() => setCtxMenu(null)}
                />
            ) : null}
        </Box>
    );
};

/**
 * 标尺整体做记忆化。
 *
 * 滚动期间它的 props 现在都是稳定的：`scrollLeft` 走量化值（见
 * `TICK_WINDOW_STEP_PX` 的注释）、`ticks` 也只在量化锚点跨越时才变。此前它
 * 是裸 `React.FC`，父组件每帧重渲染都会把整棵标尺子树（刻度、拍号、Tempo
 * Map 行）重新协调一遍——而它内部真正需要重画的只有 `TimeRulerMarks`。
 *
 * 说明：个别 prop（如 `t`）若每次渲染都是新引用，memo 会退化为"每次都重渲
 * 染"，行为与改前一致、无副作用；但只要绝大多数 prop 稳定，就能跳过。
 */
export const TimeRuler = React.memo(TimeRulerInner);
