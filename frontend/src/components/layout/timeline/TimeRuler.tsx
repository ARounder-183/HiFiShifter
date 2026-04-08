import React from "react";
import { Box } from "@radix-ui/themes";

type TimeRulerBar = { beat: number; label: string };

const TimeRulerMarks = React.memo(function TimeRulerMarks({
    bars,
    secPerBeat,
    pxPerSec,
    boundaryLeft,
    scrollLeft,
    viewportWidth,
}: {
    bars: TimeRulerBar[];
    secPerBeat: number;
    pxPerSec: number;
    boundaryLeft: number;
    scrollLeft: number;
    viewportWidth?: number;
}) {
    const visibleBars = React.useMemo(() => {
        if (!Number.isFinite(viewportWidth) || viewportWidth == null || viewportWidth <= 0) {
            return bars;
        }

        const beatPx = Math.max(1e-9, secPerBeat * pxPerSec);
        const bufferPx = Math.max(240, viewportWidth * 0.5);
        const leftPx = Math.max(0, scrollLeft - bufferPx);
        const rightPx = scrollLeft + viewportWidth + bufferPx;

        const leftBeat = leftPx / beatPx;
        const rightBeat = rightPx / beatPx;

        // bars 已按 beat 升序，使用二分裁剪可视区，避免每次全量过滤。
        const lowerBound = (target: number) => {
            let lo = 0;
            let hi = bars.length;
            while (lo < hi) {
                const mid = (lo + hi) >> 1;
                if (bars[mid].beat < target) lo = mid + 1;
                else hi = mid;
            }
            return lo;
        };

        const start = Math.max(0, lowerBound(leftBeat) - 1);
        const end = Math.min(bars.length, lowerBound(rightBeat + 1) + 1);
        return bars.slice(start, end);
    }, [bars, secPerBeat, pxPerSec, scrollLeft, viewportWidth]);

    return (
        <>
            {visibleBars.map((m) => (
                <div
                    key={m.beat}
                    className="absolute top-0 bottom-0 text-[10px] text-qt-text-muted pt-1"
                    style={{ left: m.beat * secPerBeat * pxPerSec }}
                >
                    <div className="pl-1 border-l border-qt-border h-2">{m.label}</div>
                </div>
            ))}

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

export const TimeRuler: React.FC<{
    contentWidth: number;
    scrollLeft: number;
    bars: TimeRulerBar[];
    pxPerBeat: number;
    pxPerSec: number;
    secPerBeat: number;
    viewportWidth?: number;
    playheadSec: number;
    playheadLineRef?: React.Ref<HTMLDivElement>;
    playheadHeadRef?: React.Ref<HTMLDivElement>;
    onMouseDown: (e: React.MouseEvent<HTMLDivElement>) => void;
    contentRef?: React.Ref<HTMLDivElement>;
}> = ({
    contentWidth,
    scrollLeft,
    bars,
    pxPerBeat: _pxPerBeat,
    pxPerSec,
    secPerBeat,
    viewportWidth,
    playheadSec,
    playheadLineRef,
    playheadHeadRef,
    onMouseDown,
    contentRef,
}) => {
    // 统一用 sec 坐标系：beat 位置 = beat * secPerBeat * pxPerSec
    void _pxPerBeat;
    const boundaryLeft = contentWidth - 1;

    // If the parent passes a ref, it may be doing imperative scroll syncing
    // (e.g. updating transform every scroll event). In that case, avoid
    // re-applying a potentially stale transform during React renders.
    const useManualTransform = contentRef != null;

    return (
        <Box
            className="h-6 bg-qt-window border-b border-qt-border relative overflow-hidden shrink-0 select-none"
            onMouseDown={(e) => {
                if (e.button === 1) {
                    e.preventDefault();
                    return;
                }
                onMouseDown(e);
            }}
            onAuxClick={(e) => {
                if (e.button === 1) e.preventDefault();
            }}
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
                    bars={bars}
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
        </Box>
    );
};
