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

    it("renders a full-height origin guide with a target-row accent", () => {
        const originEntry: SnapGuideEntry = {
            id: "origin",
            kind: "origin",
            surface: "timeline",
            markers: [
                {
                    role: "target",
                    axis: "time",
                    position: 2,
                    trackId: null,
                    clipId: null,
                },
                {
                    role: "source",
                    axis: "time",
                    position: 2,
                    trackId: "track-2",
                    clipId: "clip-2",
                },
            ],
        };
        const html = renderToStaticMarkup(
            <SnapGuideOverlay
                entries={[originEntry]}
                projectTimeX={(sec) => sec * 100}
                layerWidth={400}
                layerHeight={320}
                rowLayout={{
                    rowHeight: 80,
                    rowIndexById: new Map([["track-2", 2]]),
                    contentHeight: 240,
                }}
                darkMode={false}
                dpr={1}
            />,
        );
        expect(html).toContain("left:199px");
        expect(html).toContain("height:320px");
        expect(html).toContain("top:162px");
        expect(html).toContain("var(--qt-snap-origin)");
    });
});
