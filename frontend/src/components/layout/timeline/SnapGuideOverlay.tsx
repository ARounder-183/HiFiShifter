import React from "react";
import type { SnapGuideEntry, SnapHighlightKind } from "../../../utils/snapHighlight";
import { verticalHairlineGeometry } from "../../../utils/devicePixelLine";

const SNAP_GUIDE_KIND_COLOR: Record<SnapHighlightKind, string> = {
    grid: "var(--qt-snap-grid)",
    clipStart: "var(--qt-snap-clip)",
    clipEnd: "var(--qt-snap-clip)",
    snapOffset: "var(--qt-snap-offset)",
    sourceStart: "var(--qt-snap-source)",
    sourceEnd: "var(--qt-snap-source)",
    selection: "var(--qt-snap-selection)",
    cursor: "var(--qt-snap-cursor)",
    sampleRate: "var(--qt-snap-sample)",
    loopBoundary: "var(--qt-snap-loop)",
    origin: "var(--qt-snap-origin)",
};

export interface SnapGuideRowLayout {
    readonly rowHeight: number;
    readonly rowIndexById: ReadonlyMap<string, number>;
    readonly contentHeight: number;
}

export interface SnapGuideOverlayProps {
    readonly entries: readonly SnapGuideEntry[];
    readonly projectTimeX: (sec: number) => number;
    readonly layerWidth: number;
    readonly layerHeight: number;
    readonly rowLayout?: SnapGuideRowLayout;
    readonly darkMode: boolean;
    readonly dpr: number;
}

interface GuideAccent {
    readonly key: string;
    readonly left: number;
    readonly top: number;
    readonly height: number;
}

function finite(value: number): boolean {
    return Number.isFinite(value);
}

/**
 * Pure snap-guide renderer. It knows nothing about buses, timeline rows or
 * piano-roll axes; adapters provide the time projection and optional row
 * layout. This is the single place where line widths and colours live.
 */
export const SnapGuideOverlay: React.FC<SnapGuideOverlayProps> = ({
    entries,
    projectTimeX,
    layerWidth,
    layerHeight,
    rowLayout,
    darkMode,
    dpr,
}) => {
    if (entries.length === 0 || layerWidth <= 0 || layerHeight <= 0) return null;

    return (
        <div
            className="absolute top-0 left-0 pointer-events-none overflow-hidden"
            style={{ width: layerWidth, height: layerHeight }}
            aria-hidden
        >
            {entries.map((entry) => (
                <SnapEntry
                    key={entry.id}
                    entry={entry}
                    projectTimeX={projectTimeX}
                    layerHeight={layerHeight}
                    rowLayout={rowLayout}
                    darkMode={darkMode}
                    dpr={dpr}
                />
            ))}
        </div>
    );
};

const SnapEntry: React.FC<{
    entry: SnapGuideEntry;
    projectTimeX: (sec: number) => number;
    layerHeight: number;
    rowLayout?: SnapGuideRowLayout;
    darkMode: boolean;
    dpr: number;
}> = ({
    entry,
    projectTimeX,
    layerHeight,
    rowLayout,
    darkMode,
    dpr,
}) => {
    if (entry.markers.length === 0) return null;
    const color = SNAP_GUIDE_KIND_COLOR[entry.kind];
    const timeMarkers = entry.markers.filter((marker) => marker.axis === "time");

    let rangeTop = 0;
    let rangeHeight = layerHeight;
    if (rowLayout !== undefined && timeMarkers.length > 0) {
        let minTop = Number.POSITIVE_INFINITY;
        let maxBottom = Number.NEGATIVE_INFINITY;
        let rowScoped = true;
        for (const marker of timeMarkers) {
            if (marker.trackId == null) {
                rowScoped = false;
                break;
            }
            const idx = rowLayout.rowIndexById.get(marker.trackId);
            if (idx == null) {
                rowScoped = false;
                break;
            }
            minTop = Math.min(minTop, idx * rowLayout.rowHeight);
            maxBottom = Math.max(maxBottom, (idx + 1) * rowLayout.rowHeight);
        }
        if (rowScoped) {
            rangeTop = Math.max(0, minTop);
            rangeHeight = Math.max(0, Math.min(rowLayout.contentHeight, maxBottom) - rangeTop);
        }
    }

    const lineXs: number[] = [];
    const lineSeen = new Set<number>();
    for (const marker of timeMarkers) {
        if (!finite(marker.position)) continue;
        const x = projectTimeX(marker.position);
        if (!finite(x)) continue;
        // De-duplicate on the physical pixel grid: a 0.5 CSS px threshold
        // collapses distinct targets once the editor is zoomed out.
        const key = Math.round(x * dpr);
        if (lineSeen.has(key)) continue;
        lineSeen.add(key);
        lineXs.push(x);
    }

    const accents: GuideAccent[] = [];
    const accentSeen = new Set<string>();
    if (rowLayout !== undefined) {
        for (const marker of timeMarkers) {
            if (marker.trackId == null || !finite(marker.position)) continue;
            const idx = rowLayout.rowIndexById.get(marker.trackId);
            if (idx == null) continue;
            const x = projectTimeX(marker.position);
            const key = Math.round(x * dpr) + ":" + marker.trackId;
            if (accentSeen.has(key)) continue;
            accentSeen.add(key);
            accents.push({
                key,
                left: x,
                top: idx * rowLayout.rowHeight + 2,
                height: Math.max(4, rowLayout.rowHeight - 4),
            });
        }
    }

    return (
        <>
            {rangeHeight > 0
                ? lineXs.map((x, index) => {
                      const line = verticalHairlineGeometry(x, 2, dpr);
                      return (
                          <div
                              key={"line-" + index}
                              className="absolute"
                              style={{
                                  left: line.left,
                                  top: rangeTop,
                                  width: line.width,
                                  height: rangeHeight,
                                  backgroundColor: color,
                                  opacity: darkMode ? 0.85 : 0.7,
                                  boxShadow: darkMode
                                      ? "0 0 6px 1px color-mix(in oklab, " +
                                        color +
                                        " 60%, transparent)"
                                      : "none",
                              }}
                          />
                      );
                  })
                : null}
            {accents.map((accent) => {
                const edge = verticalHairlineGeometry(accent.left, 3, dpr);
                return (
                    <div
                        key={accent.key}
                        className="absolute rounded-[1px]"
                        style={{
                            left: edge.left,
                            top: accent.top,
                            width: edge.width,
                            height: accent.height,
                            backgroundColor: darkMode
                                ? "color-mix(in oklab, " + color + " 70%, white 30%)"
                                : color,
                            boxShadow: darkMode
                                ? "0 0 8px 2px color-mix(in oklab, " + color + " 75%, transparent)"
                                : "none",
                        }}
                    />
                );
            })}
        </>
    );
};
