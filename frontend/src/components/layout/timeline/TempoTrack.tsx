import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
    Box,
    Popover,
    Text,
    TextField,
    Flex,
    IconButton,
} from "@radix-ui/themes";
import { Cross2Icon } from "@radix-ui/react-icons";
import { useAppDispatch, useAppSelector } from "../../../app/hooks";
import type { RootState } from "../../../app/store";
import { useI18n } from "../../../i18n/I18nProvider";
import {
    addTempoPoint,
    updateTempoPoint,
    removeTempoPoint,
    syncTempoMap,
} from "../../../features/session/sessionSlice";
import type { TempoPoint } from "../../../utils/tempoMap";
import {
    ticksToSeconds,
    secondsToTicks,
    createTempoPointId,
    getTempoAtTicks,
    snapSecToGrid,
} from "../../../utils/tempoMap";
import { gridStepBeats } from "./grid";

const TRACK_HEIGHT = 26;

interface TempoTrackProps {
    pxPerSec: number;
    scrollLeft: number;
    viewportWidth: number;
    contentWidth: number;
    contentRef?: React.Ref<HTMLDivElement>;
}

const TempoFlag: React.FC<{
    point: TempoPoint;
    leftPx: number;
    isFirst: boolean;
    onDelete: (id: string) => void;
    onDragStart: (id: string, e: React.PointerEvent) => void;
    isDragging: boolean;
}> = React.memo(({ point, leftPx, isFirst, onDelete, onDragStart, isDragging }) => {
    const [open, setOpen] = useState(false);
    const [bpmText, setBpmText] = useState(String(Math.round(point.bpm)));
    const [numText, setNumText] = useState(String(point.numerator));
    const [denText, setDenText] = useState(String(point.denominator));

    const dispatch = useAppDispatch();
    const { t } = useI18n();

    const commitEdit = useCallback(() => {
        const bpm = Number(bpmText);
        const num = Number(numText);
        const den = Number(denText);
        if (Number.isFinite(bpm) && bpm >= 10 && bpm <= 300) {
            dispatch(updateTempoPoint({ id: point.id, bpm }));
        }
        if (Number.isFinite(num) && num >= 1 && num <= 32) {
            dispatch(updateTempoPoint({ id: point.id, numerator: num }));
        }
        if ([1, 2, 4, 8, 16, 32].includes(den)) {
            dispatch(updateTempoPoint({ id: point.id, denominator: den }));
        }
        // Sync to backend
        void dispatch(syncTempoMap());
        setOpen(false);
    }, [bpmText, numText, denText, dispatch, point.id]);

    // Sync local state when popover opens
    const handleOpenChange = (nextOpen: boolean) => {
        if (nextOpen) {
            setBpmText(String(Math.round(point.bpm)));
            setNumText(String(point.numerator));
            setDenText(String(point.denominator));
        }
        setOpen(nextOpen);
    };

    return (
        <Popover.Root open={open} onOpenChange={handleOpenChange}>
            <Popover.Trigger>
                <div
                    className="absolute top-0 flex items-center gap-0.5 select-none group"
                    style={{
                        left: leftPx,
                        height: TRACK_HEIGHT,
                        cursor: isFirst ? "pointer" : isDragging ? "grabbing" : "grab",
                    }}
                    onDoubleClick={(e) => {
                        e.stopPropagation();
                        setOpen(true);
                    }}
                    onPointerDown={(e) => {
                        // Only start drag for non-first points on left click, not double click
                        if (!isFirst && e.button === 0 && !open) {
                            onDragStart(point.id, e);
                        }
                    }}
                >
                    {/* Flag marker */}
                    <div
                        className="w-px h-full"
                        style={{ backgroundColor: "var(--qt-highlight)" }}
                    />
                    <div
                        className="px-1 rounded-sm text-[10px] leading-tight whitespace-nowrap"
                        style={{
                            backgroundColor: "var(--qt-highlight)",
                            color: "var(--qt-window)",
                            opacity: isDragging ? 1 : 0.9,
                        }}
                    >
                        <span className="font-semibold">
                            {Math.round(point.bpm)}
                        </span>
                        <span className="ml-0.5 opacity-80">
                            {point.numerator}/{point.denominator}
                        </span>
                    </div>
                    {/* Delete button (hidden for first point) */}
                    {!isFirst && (
                        <IconButton
                            size="1"
                            variant="ghost"
                            className="opacity-0 group-hover:opacity-100 transition-opacity !w-3 !h-3"
                            onClick={(e) => {
                                e.stopPropagation();
                                onDelete(point.id);
                            }}
                        >
                            <Cross2Icon width={10} height={10} />
                        </IconButton>
                    )}
                </div>
            </Popover.Trigger>

            <Popover.Content
                side="bottom"
                align="start"
                sideOffset={4}
                style={{ zIndex: 50 }}
                onPointerDownOutside={() => commitEdit()}
            >
                <Flex direction="column" gap="2" style={{ minWidth: 160 }}>
                    <Flex align="center" gap="2">
                        <Text size="1" style={{ width: 36 }}>
                            BPM
                        </Text>
                        <TextField.Root
                            size="1"
                            value={bpmText}
                            onChange={(
                                e: React.ChangeEvent<HTMLInputElement>,
                            ) => setBpmText(e.target.value)}
                            onKeyDown={(e: React.KeyboardEvent) => {
                                if (e.key === "Enter") commitEdit();
                            }}
                            style={{ width: 60 }}
                        />
                    </Flex>
                    <Flex align="center" gap="2">
                        <Text size="1" style={{ width: 36 }}>
                            {t("tempo_time_signature" as any)}
                        </Text>
                        <TextField.Root
                            size="1"
                            value={numText}
                            onChange={(
                                e: React.ChangeEvent<HTMLInputElement>,
                            ) => setNumText(e.target.value)}
                            onKeyDown={(e: React.KeyboardEvent) => {
                                if (e.key === "Enter") commitEdit();
                            }}
                            style={{ width: 30 }}
                        />
                        <Text size="1">/</Text>
                        <TextField.Root
                            size="1"
                            value={denText}
                            onChange={(
                                e: React.ChangeEvent<HTMLInputElement>,
                            ) => setDenText(e.target.value)}
                            onKeyDown={(e: React.KeyboardEvent) => {
                                if (e.key === "Enter") commitEdit();
                            }}
                            style={{ width: 30 }}
                        />
                    </Flex>
                </Flex>
            </Popover.Content>
        </Popover.Root>
    );
});
TempoFlag.displayName = "TempoFlag";

export const TempoTrack: React.FC<TempoTrackProps> = ({
    pxPerSec,
    scrollLeft,
    viewportWidth,
    contentWidth: _contentWidth,
    contentRef,
}) => {
    const dispatch = useAppDispatch();
    const tempoMap = useAppSelector(
        (state: RootState) => state.session.tempoMap,
    );
    const grid = useAppSelector(
        (state: RootState) => state.session.grid,
    );
    const gridSnapEnabled = useAppSelector(
        (state: RootState) => state.session.gridSnapEnabled,
    );

    const useManualTransform = contentRef != null;

    // ── Drag state ──────────────────────────────────────────────
    const [draggingId, setDraggingId] = useState<string | null>(null);
    const dragStateRef = useRef<{
        id: string;
        startClientX: number;
        startPx: number;
        containerRect: DOMRect;
    } | null>(null);
    const containerRef = useRef<HTMLDivElement | null>(null);

    const handleDragStart = useCallback(
        (id: string, e: React.PointerEvent) => {
            const container = containerRef.current;
            if (!container) return;
            e.preventDefault();
            e.stopPropagation();
            (e.target as HTMLElement).setPointerCapture(e.pointerId);
            const rect = container.getBoundingClientRect();
            const pt = tempoMap.points.find((p) => p.id === id);
            if (!pt) return;
            const sec = ticksToSeconds(pt.positionTicks, tempoMap);
            dragStateRef.current = {
                id,
                startClientX: e.clientX,
                startPx: sec * pxPerSec,
                containerRect: rect,
            };
            setDraggingId(id);
        },
        [tempoMap, pxPerSec],
    );

    useEffect(() => {
        if (!draggingId) return;

        const handlePointerMove = (e: PointerEvent) => {
            const ds = dragStateRef.current;
            if (!ds) return;
            const dx = e.clientX - ds.startClientX;
            let newPx = Math.max(0, ds.startPx + dx);
            let newSec = newPx / pxPerSec;

            // Snap to grid if enabled
            if (gridSnapEnabled) {
                const stepBeats = gridStepBeats(grid);
                newSec = snapSecToGrid(newSec, tempoMap, stepBeats);
                newPx = newSec * pxPerSec;
            }

            const newTicks = Math.round(secondsToTicks(newSec, tempoMap));

            // Don't allow moving to position 0 (that's the first point)
            if (newTicks <= 0) return;

            // Don't allow overlapping with other points
            const minTickGap = tempoMap.ticksPerBeat / 4;
            for (const pt of tempoMap.points) {
                if (pt.id === ds.id) continue;
                if (Math.abs(pt.positionTicks - newTicks) < minTickGap) return;
            }

            dispatch(updateTempoPoint({ id: ds.id, positionTicks: newTicks }));
        };

        const handlePointerUp = () => {
            dragStateRef.current = null;
            setDraggingId(null);
            // Sync to backend after drag ends
            void dispatch(syncTempoMap());
        };

        window.addEventListener("pointermove", handlePointerMove);
        window.addEventListener("pointerup", handlePointerUp);
        return () => {
            window.removeEventListener("pointermove", handlePointerMove);
            window.removeEventListener("pointerup", handlePointerUp);
        };
    }, [draggingId, dispatch, pxPerSec, tempoMap, grid, gridSnapEnabled]);

    // Compute visible flags
    const visibleFlags = useMemo(() => {
        const result: Array<{
            point: TempoPoint;
            leftPx: number;
            isFirst: boolean;
        }> = [];
        const bufferPx = 200;
        const leftBound = scrollLeft - bufferPx;
        const rightBound = scrollLeft + viewportWidth + bufferPx;

        for (let i = 0; i < tempoMap.points.length; i++) {
            const pt = tempoMap.points[i];
            const sec = ticksToSeconds(pt.positionTicks, tempoMap);
            const px = sec * pxPerSec;
            if (px >= leftBound && px <= rightBound) {
                result.push({ point: pt, leftPx: px, isFirst: i === 0 });
            }
        }
        return result;
    }, [tempoMap, pxPerSec, scrollLeft, viewportWidth]);

    const handleDoubleClick = useCallback(
        (e: React.MouseEvent<HTMLDivElement>) => {
            // Don't add point if we just finished dragging
            if (draggingId) return;

            const rect = e.currentTarget.getBoundingClientRect();
            const x = e.clientX - rect.left + scrollLeft;
            let sec = Math.max(0, x / pxPerSec);

            // Snap new point to grid if enabled
            if (gridSnapEnabled) {
                const stepBeats = gridStepBeats(grid);
                sec = snapSecToGrid(sec, tempoMap, stepBeats);
            }

            const ticks = secondsToTicks(sec, tempoMap);

            // Don't add if very close to existing point
            const minTickGap = tempoMap.ticksPerBeat / 4;
            for (const pt of tempoMap.points) {
                if (Math.abs(pt.positionTicks - ticks) < minTickGap) return;
            }

            // Inherit BPM/time-sig from the point at this position
            const prevPt = getTempoAtTicks(ticks, tempoMap);

            dispatch(
                addTempoPoint({
                    id: createTempoPointId(),
                    positionTicks: Math.round(ticks),
                    bpm: prevPt.bpm,
                    numerator: prevPt.numerator,
                    denominator: prevPt.denominator,
                }),
            );
            // Sync to backend
            void dispatch(syncTempoMap());
        },
        [dispatch, pxPerSec, scrollLeft, tempoMap, draggingId, grid, gridSnapEnabled],
    );

    const handleDelete = useCallback(
        (id: string) => {
            dispatch(removeTempoPoint(id));
            // Sync to backend
            void dispatch(syncTempoMap());
        },
        [dispatch],
    );

    return (
        <Box
            ref={containerRef}
            className="relative overflow-hidden shrink-0 select-none border-b border-qt-border"
            style={{
                height: TRACK_HEIGHT,
                backgroundColor: "var(--qt-window)",
            }}
            onDoubleClick={handleDoubleClick}
        >
            <div
                ref={contentRef}
                className="absolute inset-0 will-change-transform"
                style={
                    useManualTransform
                        ? undefined
                        : { transform: `translateX(${-scrollLeft}px)` }
                }
            >
                {visibleFlags.map(({ point, leftPx, isFirst }) => (
                    <TempoFlag
                        key={point.id}
                        point={point}
                        leftPx={leftPx}
                        isFirst={isFirst}
                        onDelete={handleDelete}
                        onDragStart={handleDragStart}
                        isDragging={draggingId === point.id}
                    />
                ))}
            </div>
        </Box>
    );
};
