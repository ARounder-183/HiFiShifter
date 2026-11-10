import React from "react";
import { useSyncExternalStore } from "react";
import { useAppSelector } from "../../../../app/hooks";
import { useAppTheme } from "../../../../theme/AppThemeProvider";
import { type RootState } from "../../../../app/store";
import { getSnapHighlightSnapshot, subscribeSnapHighlight } from "../../../../utils/snapHighlight";
import { useDevicePixelRatio } from "../../../../hooks/useDevicePixelRatio";
import { SnapGuideOverlay } from "../../timeline/SnapGuideOverlay";
import { usePianoRollSnapGuideViewport } from "./usePianoRollSnapGuideViewport";
import type { ParamName } from "../types";

/**
 * Piano-roll adapter for the shared snap-guide renderer.
 *
 * Time guides are full-height vertical lines at the live viewport projection;
 * value guides are full-width horizontal lines. There are no track-row accents
 * here, and timeline-only entries are filtered out.
 */
export const PianoRollSnapGuideLayer: React.FC<{
    readonly param: ParamName;
    readonly viewWidth: number;
    readonly viewHeight: number;
    readonly valueToY: (param: ParamName, value: number, heightPx: number) => number;
}> = ({ param, viewWidth, viewHeight, valueToY }) => {
    const snapshot = useSyncExternalStore(subscribeSnapHighlight, getSnapHighlightSnapshot);
    const enabled = useAppSelector(
        (state: RootState) => state.session.timelineSnap.snapHighlightEnabled !== false,
    );
    const viewport = usePianoRollSnapGuideViewport();
    const dpr = useDevicePixelRatio();
    const { mode: themeMode } = useAppTheme();

    if (!enabled || snapshot.entries.length === 0 || viewWidth <= 0 || viewHeight <= 0) {
        return null;
    }
    const entries = snapshot.entries.filter(
        (entry) => entry.surface === "pianoRoll" || entry.surface === "all",
    );
    if (entries.length === 0) return null;

    return (
        <div className="absolute inset-0 pointer-events-none z-10" aria-hidden>
            <SnapGuideOverlay
                entries={entries}
                projectTimeX={(sec) => sec * viewport.pxPerSec - viewport.scrollLeftPx}
                projectValueY={(value) => valueToY(param, value, viewHeight)}
                layerWidth={viewWidth}
                layerHeight={viewHeight}
                darkMode={themeMode === "dark"}
                dpr={dpr}
            />
        </div>
    );
};
