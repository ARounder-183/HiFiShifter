import React, { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { Box } from "@radix-ui/themes";
import { screenXToWorldSec } from "./runtime/timelineWorld.js";
import type { RulerTick, TimeFormatContext, TimeUnit, TimeUnitChoice } from "./timeFormat.js";
import { TIME_UNITS, TIME_UNIT_CHOICES, formatCursorTime } from "./timeFormat.js";
import type { GridSize } from "../../../features/session/sessionTypes.ts";
import type { ScaleLike } from "../../../utils/musicalScales.ts";
import type { CustomScalePreset } from "../../../utils/customScales.ts";
import type { TempoMap } from "../../../utils/tempoMap.ts";
import { removeTempoPoint } from "../../../utils/tempoMap.ts";
import {
    TempoMapRulerRow,
    type TempoPointEditRequest,
} from "./TempoMapRulerRow.tsx";
import { RULER_BASE_HEIGHT_PX, timeRulerHeightPx } from "./rulerHeight.ts";

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

const TimeRulerMarks = React.memo(function TimeRulerMarks({
    ticks,
    secPerBeat: _secPerBeat,
    pxPerSec,
    boundaryLeft,
    scrollLeft,
    viewportWidth,
}: {
    ticks: RulerTick[];
    secPerBeat: number;
    pxPerSec: number;
    boundaryLeft: number;
    scrollLeft: number;
    viewportWidth?: number;
}) {
    void _secPerBeat;
    const visibleTicks = React.useMemo(() => {
        if (!Number.isFinite(viewportWidth) || viewportWidth == null || viewportWidth <= 0) {
            return ticks;
        }
        const bufferPx = Math.max(320, viewportWidth * 0.5);
        const leftPx = Math.max(0, scrollLeft - bufferPx);
        const rightPx = scrollLeft + viewportWidth + bufferPx;
        const leftSec = leftPx / Math.max(1e-9, pxPerSec);
        const rightSec = rightPx / Math.max(1e-9, pxPerSec);
        // 按 sec 二分（Tempo Map 下 beat 与 px 不再线性，必须用 sec 判断可见性）。
        const lowerBound = (target: number) => {
            let lo = 0;
            let hi = ticks.length;
            while (lo < hi) {
                const mid = (lo + hi) >> 1;
                if (ticks[mid].sec < target) lo = mid + 1;
                else hi = mid;
            }
            return lo;
        };
        const start = Math.max(0, lowerBound(leftSec) - 1);
        const end = Math.min(ticks.length, lowerBound(rightSec + 1e-6) + 1);
        return ticks.slice(start, end);
    }, [ticks, pxPerSec, scrollLeft, viewportWidth]);

    return (
        <>
            {visibleTicks.map((tick) => {
                const left = tick.sec * pxPerSec;
                return (
                    <div
                        key={tick.beat}
                        className="absolute top-0 bottom-0"
                        style={{ left }}
                    >
                        <div
                            className="absolute top-0 bottom-0"
                            style={{
                                // 与下方网格保持一致：小节线 2px、弱网格线 1px；
                                // 2px 线以刻度位置为中心，避免左右偏移半个像素。
                                left: tick.isBarStart ? -1 : 0,
                                width: tick.isBarStart ? 2 : 1,
                                backgroundColor: "var(--qt-border)",
                                opacity: tick.isBarStart ? 1 : 0.6,
                            }}
                        />
                        <div className="flex flex-col justify-center h-full pl-2 pr-1 select-none">
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

            {Number.isFinite(boundaryLeft) && boundaryLeft >= -2 ? (
                <div
                    className="absolute top-0 bottom-0 w-px z-20"
                    style={{
                        left: boundaryLeft,
                        backgroundColor: "var(--qt-highlight)",
                        opacity: 0.9,
                    }}
                />
            ) : null}
        </>
    );
});

const TimeRulerPlayhead = React.memo(function TimeRulerPlayhead({
    playheadSec,
    pxPerSec,
    lineRef,
    headRef,
}: {
    playheadSec: number;
    pxPerSec: number;
    lineRef?: React.Ref<HTMLDivElement>;
    headRef?: React.Ref<HTMLDivElement>;
}) {
    const playheadLeft = playheadSec * pxPerSec;
    return (
        <>
            <div
                ref={lineRef}
                className="absolute top-0 bottom-0 w-px bg-qt-playhead z-20"
                style={{ left: playheadLeft }}
            />
            <div
                ref={headRef}
                className="absolute top-0 z-30"
                style={{
                    left: playheadLeft,
                    transform: "translateX(-6px)",
                }}
            >
                <div className="w-0 h-0 border-l-[6px] border-l-transparent border-r-[6px] border-r-transparent border-t-[8px] border-t-qt-playhead" />
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

    // 找到点击位置附近的变化点（8px 内）。
    const nearPoint = React.useMemo(() => {
        if (!tempoMap) return null;
        const nearPx = 8;
        let best: { point: (typeof tempoMap.points)[number]; isFirst: boolean } | null = null;
        let bestDist = nearPx;
        for (let i = 0; i < tempoMap.points.length; i += 1) {
            const dist = Math.abs(tempoMap.points[i].positionSec - clickedSec) * Math.max(1e-9, pxPerSec);
            if (dist <= bestDist) {
                bestDist = dist;
                best = { point: tempoMap.points[i], isFirst: i === 0 };
            }
        }
        return best;
    }, [tempoMap, clickedSec, pxPerSec]);
    const hasMap = tempoMap != null && tempoMap.points.length > 0;

    return (
        <div
            ref={ref}
            data-time-ruler-context-menu
            data-hs-context-menu="1"
            className="fixed z-50 min-w-[140px] rounded border border-qt-border bg-qt-window text-qt-text shadow-lg py-1"
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
        </div>
    );
}

export const TimeRuler: React.FC<{
    contentWidth: number;
    scrollLeft: number;
    ticks: RulerTick[];
    pxPerBeat: number;
    pxPerSec: number;
    secPerBeat: number;
    viewportWidth?: number;
    playheadSec: number;
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
    gridSnapEnabled?: boolean;
    projectScale?: ScaleLike | null;
    projectScaleName?: string;
    customScalePresets?: readonly CustomScalePreset[];
    /** 本地即时更新（拖动中，仅 Redux）。 */
    onTempoMapChange?: (next: TempoMap | null) => void;
    /** 离散提交（对话框/菜单/拖拽结束），同步后端。 */
    onTempoMapCommit?: (next: TempoMap | null) => void;
}> = ({
    contentWidth,
    scrollLeft,
    ticks,
    pxPerBeat: _pxPerBeat,
    pxPerSec,
    secPerBeat,
    viewportWidth,
    playheadSec,
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
    gridSnapEnabled = true,
    projectScale = null,
    projectScaleName,
    customScalePresets = [],
    onTempoMapChange,
    onTempoMapCommit,
}) => {
    void _pxPerBeat;
    const tAny = t ?? ((key: string) => key);
    const boundaryLeft = contentWidth - 1;
    const useManualTransform = contentRef != null;
    const [ctxMenu, setCtxMenu] = useState<{ x: number; y: number; sec: number } | null>(null);
    const [hover, setHover] = useState<{ x: number; y: number; sec: number } | null>(null);
    const [tempoEditRequest, setTempoEditRequest] = useState<TempoPointEditRequest | null>(null);
    const rulerRef = useRef<HTMLDivElement | null>(null);

    const showTempoRow = Boolean(tempoMap && tempoMap.points.length > 0 && tempoMapVisible);
    const rulerHeight = timeRulerHeightPx(showTempoRow);

    const handleAddTempoPointAt = useCallback(
        (sec: number, focus: "tempo" | "timeSignature" | "scale" | null) => {
            setTempoEditRequest({ pointId: null, positionSec: Math.max(0, sec), focus });
        },
        [],
    );
    const handleEditTempoPoint = useCallback((id: string) => {
        setTempoEditRequest({ pointId: id, positionSec: null, focus: null });
    }, []);
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
            // 右键菜单打开期间不显示标尺悬浮时间；菜单内部的事件也不触发。
            if (ctxMenu) {
                setHover(null);
                return;
            }
            const target = e.target as HTMLElement | null;
            if (target?.closest?.("[data-time-ruler-context-menu]")) {
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
        [ctxMenu, pxPerSec, scrollLeft],
    );

    const hoverTime = hover
        ? formatCursorTime(primaryUnit, secondaryUnit, hover.sec, timeContext)
        : null;
    const hoverTooltipLeft =
        hover != null && viewportWidth != null
            ? Math.min(Math.max(4, hover.x + 10), Math.max(4, viewportWidth - 260))
            : 4;

    return (
        <Box
            ref={rulerRef}
            className="bg-qt-window border-b border-qt-border relative overflow-hidden shrink-0 select-none"
            style={{ height: rulerHeight }}
            onMouseDown={(e) => {
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
                if (e.button === 1) e.preventDefault();
            }}
            onContextMenu={(e) => {
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
            onWheel={(e) => {
                // Prevent the ruler from becoming a separate scroll source.
                e.preventDefault();
            }}
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
                        secPerBeat={secPerBeat}
                        pxPerSec={pxPerSec}
                        boundaryLeft={boundaryLeft}
                        scrollLeft={scrollLeft}
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
                    gridSnapEnabled={gridSnapEnabled}
                    fallbackBpm={timeContext.bpm}
                    fallbackBeatsPerBar={timeContext.beatsPerBar}
                    projectScale={projectScale}
                    projectScaleName={projectScaleName}
                    customScalePresets={customScalePresets}
                    t={tAny}
                    onChange={onTempoMapChange ?? (() => undefined)}
                    onCommit={onTempoMapCommit ?? onTempoMapChange ?? (() => undefined)}
                    editRequest={tempoEditRequest}
                    onEditRequestHandled={() => setTempoEditRequest(null)}
                />
                <TimeRulerPlayhead
                    playheadSec={playheadSec}
                    pxPerSec={pxPerSec}
                    lineRef={playheadLineRef}
                    headRef={playheadHeadRef}
                />
            </div>

            {hover && hoverTime ? (
                <div
                    className="absolute top-1 z-40 pointer-events-none rounded border border-qt-border bg-qt-panel px-2 py-1 shadow-lg"
                    style={{ left: hoverTooltipLeft }}
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
