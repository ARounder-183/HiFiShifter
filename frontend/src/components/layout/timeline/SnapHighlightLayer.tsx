import React, { useMemo } from "react";
import { useSyncExternalStore } from "react";
import { useAppSelector } from "../../../app/hooks";
import { useAppTheme } from "../../../theme/AppThemeProvider";
import { type RootState } from "../../../app/store";
import { getSnapHighlightSnapshot, subscribeSnapHighlight } from "../../../utils/snapHighlight";
import { useDevicePixelRatio } from "../../../hooks/useDevicePixelRatio";
import { SnapGuideOverlay } from "./SnapGuideOverlay";
import { NEW_TRACK_SENTINEL } from "./constants";

/**
 * Timeline adapter for the shared snap-guide renderer.
 *
 * It owns only timeline concerns: pxPerSec projection and track-row accents.
 * Entries published for the piano roll are filtered out before rendering, so
 * the same parameter-block guide cannot appear twice.
 */
export const SnapHighlightLayer: React.FC<{
    pxPerSec: number;
    rowHeight: number;
    tracks: ReadonlyArray<{ id: string }>;
    contentHeight: number;
    viewportHeightPx?: number;
    layerWidth?: number;
}> = ({ pxPerSec, rowHeight, tracks, contentHeight, viewportHeightPx = 0, layerWidth = 0 }) => {
    const snapshot = useSyncExternalStore(subscribeSnapHighlight, getSnapHighlightSnapshot);
    const snapHighlightEnabled = useAppSelector(
        (state: RootState) => state.session.timelineSnap.snapHighlightEnabled !== false,
    );
    const rowIndexById = useMemo(() => {
        const map = new Map<string, number>();
        tracks.forEach((track, index) => map.set(track.id, index));
        map.set(NEW_TRACK_SENTINEL, tracks.length);
        return map;
    }, [tracks]);
    const dpr = useDevicePixelRatio();
    const { mode: themeMode } = useAppTheme();

    if (!snapHighlightEnabled || snapshot.entries.length === 0) return null;

    const entries = snapshot.entries.filter(
        (entry) => entry.surface === "timeline" || entry.surface === "all",
    );
    if (entries.length === 0) return null;

    const safeRowHeight = Math.max(1, rowHeight);
    const safePxPerSec = Math.max(1e-9, pxPerSec);
    const layerHeightPx = Math.max(contentHeight, viewportHeightPx);

    return (
        <div
            className="absolute top-0 left-0 pointer-events-none z-13 overflow-hidden"
            style={{ width: "100%", height: layerHeightPx }}
        >
            <SnapGuideOverlay
                entries={entries}
                projectTimeX={(sec) => sec * safePxPerSec}
                layerWidth={Math.max(1, layerWidth)}
                layerHeight={layerHeightPx}
                rowLayout={{
                    rowHeight: safeRowHeight,
                    rowIndexById,
                    contentHeight,
                }}
                darkMode={themeMode === "dark"}
                dpr={dpr}
            />
        </div>
    );
};
