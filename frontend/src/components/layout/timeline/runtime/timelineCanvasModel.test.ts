import { test } from "vitest";

import { buildSparseClipRenderModel } from "./timelineCanvasModel.js";

test("components/layout/timeline/runtime/timelineCanvasModel.test.ts scripted checks", async () => {
    function assertEqual(actual: unknown, expected: unknown, label: string): void {
        const actualJson = JSON.stringify(actual);
        const expectedJson = JSON.stringify(expected);
        if (actualJson !== expectedJson) {
            throw new Error(`${label}: expected ${expectedJson}, received ${actualJson}`);
        }
    }

    const model = buildSparseClipRenderModel({
        visibleTracks: [
            { id: "track-a", color: "#ff7a00" },
            { id: "track-b", color: "#00a3ff" },
        ],
        visibleTrackClipsById: {
            "track-a": [
                {
                    id: "clip-a",
                    trackId: "track-a",
                    name: "Verse",
                    startSec: 2,
                    lengthSec: 3,
                    gain: 1,
                    playbackRate: 1.25,
                    muted: false,
                    fadeInSec: 0.25,
                    fadeOutSec: 0.5,
                    fadeInShape: 5,
                    fadeInDir: 0,
                    fadeOutShape: 5,
                    fadeOutDir: 0,
                },
                {
                    id: "clip-b",
                    trackId: "track-a",
                    name: "Fill",
                    startSec: 8,
                    lengthSec: 1,
                    gain: 0.5,
                    playbackRate: 0.8,
                    muted: true,
                    fadeInSec: 0,
                    fadeOutSec: 0,
                    fadeInShape: 0,
                    fadeInDir: 0,
                    fadeOutShape: 0,
                    fadeOutDir: 0,
                },
            ],
            "track-b": [
                {
                    id: "clip-c",
                    trackId: "track-b",
                    name: "Hook",
                    startSec: 4,
                    lengthSec: 2,
                    gain: 1,
                    playbackRate: 2,
                    muted: false,
                    fadeInSec: 0,
                    fadeOutSec: 0,
                    fadeInShape: 5,
                    fadeInDir: 0.25,
                    fadeOutShape: 2,
                    fadeOutDir: -0.25,
                },
            ],
        },
        pxPerSec: 100,
        rowHeight: 48,
        selectedClipId: "clip-b",
        multiSelectedClipIds: ["clip-c", "clip-b"],
        renamingClipId: "clip-a",
    });

    assertEqual(
        model.drawClips.map((clip) => ({
            id: clip.id,
            leftPx: clip.leftPx,
            topPx: clip.topPx,
            widthPx: clip.widthPx,
            fadeInPx: clip.fadeInPx,
            fadeOutPx: clip.fadeOutPx,
            trackColor: clip.trackColor,
            playbackRate: clip.playbackRate,
            selected: clip.selected,
            muted: clip.muted,
            isRenaming: clip.isRenaming,
        })),
        [
            {
                id: "clip-a",
                leftPx: 200,
                topPx: 0,
                widthPx: 300,
                fadeInPx: 25,
                fadeOutPx: 50,
                trackColor: "#ff7a00",
                playbackRate: 1.25,
                selected: false,
                muted: false,
                isRenaming: true,
            },
            {
                id: "clip-b",
                leftPx: 800,
                topPx: 0,
                widthPx: 100,
                fadeInPx: 0,
                fadeOutPx: 0,
                trackColor: "#ff7a00",
                playbackRate: 0.8,
                selected: true,
                muted: true,
                isRenaming: false,
            },
            {
                id: "clip-c",
                leftPx: 400,
                topPx: 48,
                widthPx: 200,
                fadeInPx: 0,
                fadeOutPx: 0,
                trackColor: "#00a3ff",
                playbackRate: 2,
                selected: true,
                muted: false,
                isRenaming: false,
            },
        ],
        "canvas keeps drawing overlay clips so visuals stay unified",
    );

    assertEqual(
        model.overlayClipIdsByTrackId,
        {
            "track-a": ["clip-a", "clip-b"],
            "track-b": ["clip-c"],
        },
        "sparse overlay ids",
    );
});