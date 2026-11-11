import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";

import { TrackDropTargetRings } from "./TrackDropTargetLayer";
import type { TrackDropTargetSnapshot } from "./trackDropTargetHighlight";

const TRACKS = [{ id: "track-a" }, { id: "track-b" }, { id: "track-c" }];

function snapshot(
    trackIds: readonly string[],
    tone: TrackDropTargetSnapshot["tone"] = "move",
): TrackDropTargetSnapshot {
    return { rev: 1, trackIds, tone };
}

describe("TrackDropTargetRings", () => {
    it("projects track ids to row rects using rowHeight", () => {
        const html = renderToStaticMarkup(
            <TrackDropTargetRings
                snapshot={snapshot(["track-b"])}
                tracks={TRACKS}
                rowHeight={80}
                contentWidth={1200}
                contentHeight={400}
            />,
        );
        expect(html).toContain('data-track-drop-target="track-b"');
        expect(html).toContain("top:82px");
        expect(html).toContain("height:76px");
        expect(html).toContain("width:1200px");
        expect(html).toContain("height:400px");
        expect(html).toContain("var(--qt-track-drop-target)");
    });

    it("uses the copy and snap tokens for their tones", () => {
        const copy = renderToStaticMarkup(
            <TrackDropTargetRings
                snapshot={snapshot(["track-a"], "copy")}
                tracks={TRACKS}
                rowHeight={50}
                contentWidth={500}
                contentHeight={200}
            />,
        );
        expect(copy).toContain("var(--qt-track-drop-copy)");
        const snap = renderToStaticMarkup(
            <TrackDropTargetRings
                snapshot={snapshot(["track-a"], "snap")}
                tracks={TRACKS}
                rowHeight={50}
                contentWidth={500}
                contentHeight={200}
            />,
        );
        expect(snap).toContain("var(--qt-snap-origin)");
        expect(snap).toContain("opacity:0.98");
    });

    it("renders nothing for an empty snapshot, unknown ids or zero geometry", () => {
        const empty = renderToStaticMarkup(
            <TrackDropTargetRings
                snapshot={snapshot([])}
                tracks={TRACKS}
                rowHeight={80}
                contentWidth={1200}
                contentHeight={400}
            />,
        );
        expect(empty).toBe("");
        const unknown = renderToStaticMarkup(
            <TrackDropTargetRings
                snapshot={snapshot(["missing"])}
                tracks={TRACKS}
                rowHeight={80}
                contentWidth={1200}
                contentHeight={400}
            />,
        );
        expect(unknown).toBe("");
        const zero = renderToStaticMarkup(
            <TrackDropTargetRings
                snapshot={snapshot(["track-a"])}
                tracks={TRACKS}
                rowHeight={80}
                contentWidth={0}
                contentHeight={400}
            />,
        );
        expect(zero).toBe("");
    });
});
