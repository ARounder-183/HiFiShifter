import { test } from "vitest";

import type { ClipInfo, ClipTakeInfo } from "../../../features/session/sessionTypes";
import type { WaveformSceneClip } from "../../../waveform/sceneBuilder";
import { expandClipToTakeSceneClips } from "./takeLanes.ts";

test("timeline/takeLanes.test.ts scripted checks", async () => {
    function assertEqual(actual: unknown, expected: unknown, label: string): void {
        const actualJson = JSON.stringify(actual);
        const expectedJson = JSON.stringify(expected);
        if (actualJson !== expectedJson) {
            throw new Error(`${label}: expected ${expectedJson}, received ${actualJson}`);
        }
    }

    function requireExpanded(
        value: WaveformSceneClip[] | null,
        label: string,
    ): WaveformSceneClip[] {
        if (value == null) throw new Error(`${label}: expected expanded lanes, received null`);
        return value;
    }

    function take(overrides: Partial<ClipTakeInfo> = {}): ClipTakeInfo {
        return {
            id: "take",
            name: "Take",
            gain: 1,
            sourceStartSec: 0,
            sourceEndSec: 10,
            playbackRate: 1,
            reversed: false,
            loopEnabled: false,
            ...overrides,
        };
    }

    function clip(overrides: Partial<ClipInfo> = {}): ClipInfo {
        return {
            id: "clip",
            trackId: "track",
            name: "Clip",
            startSec: 0,
            lengthSec: 5,
            color: "blue",
            gain: 1,
            muted: false,
            sourceStartSec: 0,
            sourceEndSec: 10,
            playbackRate: 1,
            reversed: false,
            loopEnabled: false,
            snapOffsetSec: 0,
            fadeInSec: 0,
            fadeOutSec: 0,
            fadeInShape: 0,
            fadeOutShape: 0,
            fadeInDir: 0,
            fadeOutDir: 0,
            ...overrides,
        };
    }

    const twoTakes = [
        take({ id: "t1", name: "A", sourcePath: "/a.wav" }),
        take({ id: "t2", name: "B", sourcePath: "/b.wav", sourceStartSec: 2 }),
    ];

    assertEqual(
        expandClipToTakeSceneClips(clip({ takes: twoTakes, activeTakeId: "t1" }), false, 40),
        null,
        "disabled option yields no lanes",
    );
    assertEqual(
        expandClipToTakeSceneClips(clip({ takes: [twoTakes[0]], activeTakeId: "t1" }), true, 40),
        null,
        "single audio take yields no lanes",
    );
    assertEqual(
        expandClipToTakeSceneClips(clip({ takes: twoTakes, activeTakeId: "t1" }), true, 27),
        null,
        "insufficient body height yields no lanes",
    );

    const expanded = requireExpanded(
        expandClipToTakeSceneClips(clip({ takes: twoTakes, activeTakeId: "t1" }), true, 40),
        "two-take clip",
    );
    assertEqual(
        expanded.map((entry) => entry.id),
        ["clip::take::t1", "clip::take::t2"],
        "expanded scene clips use take-scoped ids",
    );
    assertEqual(
        expanded.map((entry) => [entry.laneTopPx, entry.laneHeightPx]),
        [
            [0, 20],
            [20, 20],
        ],
        "lane bands tile the clip body",
    );
    assertEqual(
        expanded.map((entry) => entry.inactive),
        [false, true],
        "non-active takes are flagged inactive",
    );
    assertEqual(
        expanded.map((entry) => entry.sourcePath),
        ["/a.wav", "/b.wav"],
        "each lane renders its own take source",
    );

    const remainder = requireExpanded(
        expandClipToTakeSceneClips(
            clip({
                takes: [
                    take({ id: "t1", sourcePath: "/a.wav" }),
                    take({ id: "t2", sourcePath: "/b.wav" }),
                    take({ id: "t3", sourcePath: "/c.wav" }),
                ],
                activeTakeId: "t1",
            }),
            true,
            50,
        ),
        "three-take clip",
    );
    assertEqual(
        remainder.map((entry) => [entry.laneTopPx, entry.laneHeightPx]),
        [
            [0, 16],
            [16, 16],
            [32, 18],
        ],
        "the last lane absorbs the body height remainder",
    );

    const slipped = requireExpanded(
        expandClipToTakeSceneClips(
            clip({
                takes: twoTakes,
                activeTakeId: "t1",
                sourceStartSec: 3,
                sourceEndSec: 13,
            }),
            true,
            40,
        ),
        "slipped clip",
    );
    assertEqual(
        [slipped[0].sourceStartSec, slipped[0].sourceEndSec],
        [3, 13],
        "the active lane consumes the clip-level slip projection",
    );
    assertEqual(
        [slipped[1].sourceStartSec, slipped[1].sourceEndSec],
        [2, 10],
        "inactive lanes keep their own take windows",
    );

    const stretched = requireExpanded(
        expandClipToTakeSceneClips(
            clip({
                takes: [
                    take({ id: "t1", sourcePath: "/a.wav", playbackRate: 1.5 }),
                    take({ id: "t2", sourcePath: "/b.wav", playbackRate: 4 }),
                ],
                activeTakeId: "t1",
                clipPlaybackRate: 2,
            }),
            true,
            40,
        ),
        "stretched clip",
    );
    assertEqual(
        stretched.map((entry) => entry.playbackRate),
        [3, 8],
        "lane playback rates compose clip and take rates",
    );

    const clipped = requireExpanded(
        expandClipToTakeSceneClips(
            clip({
                takes: [
                    take({ id: "t1", sourcePath: "/a.wav", playbackRate: 9 }),
                    take({ id: "t2", sourcePath: "/b.wav", playbackRate: 0.01 }),
                ],
                activeTakeId: "t1",
                clipPlaybackRate: 5,
            }),
            true,
            40,
        ),
        "clamped clip",
    );
    assertEqual(
        clipped.map((entry) => entry.playbackRate),
        [10, 1],
        "lane playback rates clamp to the sane range",
    );

    const midiActive = requireExpanded(
        expandClipToTakeSceneClips(
            clip({
                takes: [
                    take({ id: "midi", name: "MIDI take" }),
                    take({ id: "t2", sourcePath: "/b.wav" }),
                    take({ id: "t3", sourcePath: "/c.wav" }),
                ],
                activeTakeId: "midi",
            }),
            true,
            40,
        ),
        "midi-active clip",
    );
    assertEqual(
        midiActive.map((entry) => [entry.sourcePath, entry.inactive]),
        [
            ["/b.wav", true],
            ["/c.wav", true],
        ],
        "audio takes still expand when the active take is MIDI",
    );

    const faded = requireExpanded(
        expandClipToTakeSceneClips(
            clip({ takes: twoTakes, activeTakeId: "t1", fadeInSec: 0.5, muted: true }),
            true,
            40,
        ),
        "faded clip",
    );
    assertEqual(
        faded.map((entry) => [entry.fadeInSec, entry.muted, entry.lengthSec]),
        [
            [0.5, true, 5],
            [0.5, true, 5],
        ],
        "clip-level fades, mute and length carry over to every lane",
    );
});
