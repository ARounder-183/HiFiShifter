import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { SnapGuideOverlay } from "./SnapGuideOverlay";
import type { SnapGuideEntry } from "../../../utils/snapHighlight";

const timeEntry: SnapGuideEntry = {
    id: "time",
    kind: "grid",
    surface: "pianoRoll",
    markers: [
        {
            role: "target",
            axis: "time",
            position: 2,
            trackId: null,
            clipId: null,
        },
    ],
};

const valueEntry: SnapGuideEntry = {
    id: "value",
    kind: "grid",
    surface: "pianoRoll",
    markers: [
        {
            role: "target",
            axis: "value",
            position: 60,
            trackId: null,
            clipId: null,
        },
    ],
};

describe("SnapGuideOverlay", () => {
    it("projects time markers to full-height vertical lines", () => {
        const html = renderToStaticMarkup(
            <SnapGuideOverlay
                entries={[timeEntry]}
                projectTimeX={(sec) => sec * 100}
                layerWidth={400}
                layerHeight={200}
                darkMode={false}
                dpr={1}
            />,
        );
        expect(html).toContain("left:199px");
        expect(html).toContain("height:200px");
    });

    it("projects value markers to full-width horizontal lines", () => {
        const html = renderToStaticMarkup(
            <SnapGuideOverlay
                entries={[valueEntry]}
                projectTimeX={(sec) => sec * 100}
                projectValueY={(value) => 200 - value}
                layerWidth={400}
                layerHeight={200}
                darkMode={false}
                dpr={1}
            />,
        );
        expect(html).toContain("top:139px");
        expect(html).toContain("width:400px");
    });

    it("projects row markers to track-boundary horizontal lines", () => {
        const rowEntry: SnapGuideEntry = {
            id: "track",
            kind: "track",
            surface: "timeline",
            markers: [
                {
                    role: "target",
                    axis: "row",
                    position: 2,
                    trackId: "track-2",
                    clipId: null,
                },
            ],
        };
        const html = renderToStaticMarkup(
            <SnapGuideOverlay
                entries={[rowEntry]}
                projectTimeX={(sec) => sec * 100}
                layerWidth={400}
                layerHeight={320}
                rowLayout={{
                    rowHeight: 80,
                    rowIndexById: new Map([["track-2", 2]]),
                    contentHeight: 240,
                    trackCount: 3,
                }}
                darkMode={false}
                dpr={1}
            />,
        );
        expect(html).toContain("top:159px");
        expect(html).toContain("width:400px");
        expect(html).toContain("var(--qt-snap-track)");
    });
});
