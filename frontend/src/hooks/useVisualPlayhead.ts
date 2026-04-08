import { useEffect, useRef, type MutableRefObject } from "react";

export interface UseVisualPlayheadArgs {
    syncedPlayheadSec: number;
    isTransportAdvancing: boolean;
    onFrame?: (playheadSec: number) => void;
}

export function useVisualPlayhead({
    syncedPlayheadSec,
    isTransportAdvancing,
    onFrame,
}: UseVisualPlayheadArgs): MutableRefObject<number> {
    const visualPlayheadSecRef = useRef(syncedPlayheadSec);
    const syncAnchorRef = useRef({
        playheadSec: syncedPlayheadSec,
        timestampMs: 0,
    });
    const onFrameRef = useRef(onFrame);

    useEffect(() => {
        onFrameRef.current = onFrame;
    }, [onFrame]);

    useEffect(() => {
        const now = performance.now();
        syncAnchorRef.current = {
            playheadSec: syncedPlayheadSec,
            timestampMs: now,
        };
        visualPlayheadSecRef.current = syncedPlayheadSec;
        onFrameRef.current?.(syncedPlayheadSec);
    }, [syncedPlayheadSec]);

    useEffect(() => {
        if (!isTransportAdvancing) return;

        let rafId = 0;

        const tick = (timestampMs: number) => {
            const elapsedSec = (timestampMs - syncAnchorRef.current.timestampMs) / 1000;
            const nextPlayheadSec = Math.max(
                syncAnchorRef.current.playheadSec,
                syncAnchorRef.current.playheadSec + elapsedSec,
            );
            visualPlayheadSecRef.current = nextPlayheadSec;
            onFrameRef.current?.(nextPlayheadSec);
            rafId = requestAnimationFrame(tick);
        };

        rafId = requestAnimationFrame(tick);
        return () => cancelAnimationFrame(rafId);
    }, [isTransportAdvancing]);

    return visualPlayheadSecRef;
}
