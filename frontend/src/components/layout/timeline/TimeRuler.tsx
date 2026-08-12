import React, { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { Box } from "@radix-ui/themes";
import { screenXToWorldSec } from "./runtime/timelineWorld.js";
import type { RulerTick, TimeFormatContext, TimeUnit, TimeUnitChoice } from "./timeFormat.js";
import { TIME_UNITS, TIME_UNIT_CHOICES, formatCursorTime } from "./timeFormat.js";

const RULER_HEIGHT_PX = 48;

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
    secPerBeat,
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
    const visibleTicks = React.useMemo(() => {
        if (!Number.isFinite(viewportWidth) || viewportWidth == null || viewportWidth <= 0) {
            return ticks;
        }
        const beatPx = Math.max(1e-9, secPerBeat * pxPerSec);
        const bufferPx = Math.max(320, viewportWidth * 0.5);
        const leftPx = Math.max(0, scrollLeft - bufferPx);
        const rightPx = scrollLeft + viewportWidth + bufferPx;
        const leftBeat = leftPx / beatPx;
        const rightBeat = rightPx / beatPx;
        const lowerBound = (target: number) => {
            let lo = 0;
            let hi = ticks.length;
            while (lo < hi) {
                const mid = (lo + hi) >> 1;
                if (ticks[mid].beat < target) lo = mid + 1;
                else hi = mid;
            }
            return lo;
        };
        const start = Math.max(0, lowerBound(leftBeat) - 1);
        const end = Math.min(ticks.length, lowerBound(rightBeat + 1) + 1);
        return ticks.slice(start, end);
    }, [ticks, secPerBeat, pxPerSec, scrollLeft, viewportWidth]);

    return (
        <>
            {visibleTicks.map((tick) => {
                const left = tick.beat * secPerBeat * pxPerSec;
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
    t,
    onSelectPrimary,
    onSelectSecondary,
    onCopyPlayheadTime,
    onOpenSettings,
    onClose,
}: {
    x: number;
    y: number;
    primaryUnit: TimeUnit;
    secondaryUnit: TimeUnitChoice;
    t: (key: string) => string;
    onSelectPrimary: (unit: TimeUnit) => void;
    onSelectSecondary: (unit: TimeUnitChoice) => void;
    onCopyPlayheadTime?: () => void;
    onOpenSettings?: () => void;
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
}) => {
    void _pxPerBeat;
    const tAny = t ?? ((key: string) => key);
    const boundaryLeft = contentWidth - 1;
    const useManualTransform = contentRef != null;
    const [ctxMenu, setCtxMenu] = useState<{ x: number; y: number } | null>(null);
    const [hover, setHover] = useState<{ x: number; y: number; sec: number } | null>(null);
    const rulerRef = useRef<HTMLDivElement | null>(null);

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
        [pxPerSec, scrollLeft],
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
            style={{ height: RULER_HEIGHT_PX }}
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
                setCtxMenu({ x: e.clientX, y: e.clientY });
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
                <TimeRulerMarks
                    ticks={ticks}
                    secPerBeat={secPerBeat}
                    pxPerSec={pxPerSec}
                    boundaryLeft={boundaryLeft}
                    scrollLeft={scrollLeft}
                    viewportWidth={viewportWidth}
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
                    t={tAny}
                    onSelectPrimary={onPrimaryUnitChange}
                    onSelectSecondary={onSecondaryUnitChange}
                    onCopyPlayheadTime={onCopyPlayheadTime}
                    onOpenSettings={onOpenSettings}
                    onClose={() => setCtxMenu(null)}
                />
            ) : null}
        </Box>
    );
};
