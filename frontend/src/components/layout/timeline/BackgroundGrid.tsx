import React from "react";
import { gridStepBeats } from "./grid";

function positiveMod(value: number, mod: number): number {
    if (!Number.isFinite(value) || !Number.isFinite(mod) || mod <= 0) return 0;
    const r = value % mod;
    return (r + mod) % mod;
}

export const BackgroundGrid: React.FC<{
    contentWidth: number;
    contentHeight: number;
    pxPerBeat: number;
    grid: string;
    beatsPerBar: number;
    viewportWidth?: number;
    scrollLeft?: number;
    layerRef?: React.Ref<HTMLDivElement>;
    boundaryRef?: React.Ref<HTMLDivElement>;
    lineOpacity?: number;
    showBoundary?: boolean;
}> = ({
    contentWidth,
    contentHeight,
    pxPerBeat,
    grid,
    beatsPerBar,
    viewportWidth,
    scrollLeft,
    layerRef,
    boundaryRef,
    lineOpacity = 0.9,
    showBoundary = true,
}) => {
    const useViewport =
        viewportWidth != null &&
        Number.isFinite(viewportWidth) &&
        scrollLeft != null &&
        Number.isFinite(scrollLeft);

    const weakStepPx = Math.max(1e-6, pxPerBeat * gridStepBeats(grid));
    const barStepPx = Math.max(1e-6, pxPerBeat * beatsPerBar);

    const width = useViewport ? Math.max(1, Math.floor(viewportWidth)) : contentWidth;
    const height = contentHeight;

    const weakOffsetPx = useViewport ? -positiveMod(scrollLeft as number, weakStepPx) : 0;
    const barOffsetPx = useViewport ? -positiveMod(scrollLeft as number, barStepPx) : 0;

    // If the parent provides refs in viewport mode, it may be doing imperative
    // syncing (e.g. in a scroll handler). Avoid overriding those styles with
    // potentially throttled/stale React props.
    const manualViewportSync = useViewport && (layerRef != null || boundaryRef != null);

    const boundaryLeft = useViewport ? contentWidth - 1 - (scrollLeft as number) : contentWidth - 1;

    const boundaryVisible =
        Number.isFinite(boundaryLeft) && boundaryLeft >= -2 && boundaryLeft <= width + 2;

    return (
        <>
            <div
                ref={layerRef}
                className="absolute left-0 top-0 pointer-events-none"
                style={{
                    width,
                    height,
                    backgroundImage: [
                        "linear-gradient(to right, var(--qt-graph-grid-weak) 1px, transparent 1px)",
                        "linear-gradient(to right, var(--qt-graph-grid-strong) 3px, transparent 3px)",
                    ].join(", "),
                    backgroundSize: [`${weakStepPx}px 100%`, `${barStepPx}px 100%`].join(", "),
                    backgroundPosition: useViewport
                        ? manualViewportSync
                            ? undefined
                            : [`${weakOffsetPx}px 0px`, `${barOffsetPx}px 0px`].join(", ")
                        : undefined,
                    opacity: lineOpacity,
                }}
            />

            <div
                ref={boundaryRef}
                className="absolute top-0 bottom-0 w-px z-20"
                style={{
                    left: manualViewportSync ? 0 : boundaryLeft,
                    backgroundColor: "var(--qt-highlight)",
                    opacity:
                        manualViewportSync || !boundaryVisible ? 0 : showBoundary ? lineOpacity : 0,
                }}
            />
        </>
    );
};
