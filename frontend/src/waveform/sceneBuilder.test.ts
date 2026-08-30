import { test } from "vitest";

import { buildWaveformScene, type WaveformSceneClip } from "./sceneBuilder.ts";
import { createTimelineAxis } from "../components/layout/timeline/runtime/timelineAxis.ts";

test("waveform/sceneBuilder.test.ts scripted checks", async () => {
    function assertEqual(actual: unknown, expected: unknown, label: string): void {
        const actualJson = JSON.stringify(actual);
        const expectedJson = JSON.stringify(expected);
        if (actualJson !== expectedJson) {
            throw new Error(`${label}: expected ${expectedJson}, received ${actualJson}`);
        }
    }

    function clip(overrides: Partial<WaveformSceneClip> = {}): WaveformSceneClip {
        return {
            id: "clip",
            sourcePath: "/audio.wav",
            startSec: 0,
            lengthSec: 5,
            sourceStartSec: 0,
            sourceEndSec: 5,
            durationSec: 10,
            sourceSampleRate: 44100,
            playbackRate: 1,
            reversed: false,
            loopEnabled: false,
            gain: 1,
            muted: false,
            fadeInSec: 0,
            fadeOutSec: 0,
            fadeInShape: 0,
            fadeInDir: 0,
            fadeOutShape: 0,
            fadeOutDir: 0,
            ...overrides,
        };
    }

    {
        const scene = buildWaveformScene({
            axis: createTimelineAxis({
                pxPerSec: 100,
                scrollLeftPx: 1000,
                viewportWidthPx: 1000,
            }),
            widthPx: 1000,
            rows: [
                {
                    topPx: 0,
                    waveformTopPx: 18,
                    waveformHeightPx: 72,
                    clips: [clip({ startSec: 8, lengthSec: 5, sourceStartSec: 2 })],
                },
            ],
        });

        assertEqual(
            scene.segments.map((segment) => segment.screenRect),
            [{ x: 0, y: 18, width: 300, height: 72 }],
            "visible clips use viewport-local coordinates",
        );
        assertEqual(
            scene.segments.map((segment) => [segment.sourceStartSec, segment.sourceEndSec]),
            [[4, 7]],
            "viewport clipping advances the source interval",
        );
    }

    {
        const scene = buildWaveformScene({
            axis: createTimelineAxis({
                pxPerSec: 100,
                scrollLeftPx: 0,
                viewportWidthPx: 1000,
            }),
            widthPx: 1000,
            rows: [
                {
                    topPx: 0,
                    waveformTopPx: 0,
                    waveformHeightPx: 80,
                    clips: [
                        clip({
                            lengthSec: 3,
                            sourceStartSec: 99,
                            sourceEndSec: 9,
                            playbackRate: 2,
                            reversed: true,
                        }),
                    ],
                },
            ],
        });

        assertEqual(
            scene.segments.map((segment) => [segment.sourceStartSec, segment.sourceEndSec]),
            [[3, 9]],
            "reversed clips anchor their source window at sourceEndSec",
        );
        assertEqual(scene.segments[0]?.reversed, true, "reversed direction is retained");
    }

    {
        const scene = buildWaveformScene({
            axis: createTimelineAxis({
                pxPerSec: 10,
                scrollLeftPx: 0,
                viewportWidthPx: 150,
            }),
            widthPx: 150,
            rows: [
                {
                    topPx: 0,
                    waveformTopPx: 0,
                    waveformHeightPx: 80,
                    clips: [
                        clip({
                            lengthSec: 15,
                            sourceStartSec: 8,
                            sourceEndSec: 10,
                            loopEnabled: true,
                        }),
                    ],
                },
            ],
        });

        assertEqual(
            scene.segments.map((segment) => [segment.sourceStartSec, segment.sourceEndSec]),
            [
                [8, 10],
                [0, 10],
                [0, 3],
            ],
            "looped clips render the media tail followed by whole-media periods",
        );
        assertEqual(
            scene.markers.map((marker) => marker.timelineSec),
            [2, 12],
            "loop markers use the same period boundaries as waveform segments",
        );
    }

    {
        const scene = buildWaveformScene({
            axis: createTimelineAxis({
                pxPerSec: 100,
                scrollLeftPx: 0,
                viewportWidthPx: 500,
            }),
            widthPx: 500,
            rows: [
                {
                    topPx: 0,
                    waveformTopPx: 0,
                    waveformHeightPx: 80,
                    leadingOverlapSecByClipId: { clip: 2 },
                    clips: [
                        clip({
                            autoFadeInSec: 1,
                            fadeInSec: 4,
                            muted: true,
                        }),
                    ],
                },
            ],
        });

        assertEqual(
            scene.segments.map((segment) => [segment.screenRect.x, segment.screenRect.width]),
            [
                [0, 200],
                [200, 300],
            ],
            "leading overlap splits the scene at the exact timeline boundary",
        );
        assertEqual(
            scene.segments.map((segment) => segment.alpha),
            [0.2, 0.4],
            "muted overlap alpha is applied once per split segment",
        );
        assertEqual(scene.segments[0]?.fadeInSec, 1, "automatic fade overrides manual fade");
    }

    {
        const scene = buildWaveformScene({
            axis: createTimelineAxis({
                pxPerSec: 100,
                scrollLeftPx: 0,
                viewportWidthPx: 1000,
            }),
            widthPx: 1000,
            rows: [
                {
                    topPx: 40,
                    waveformTopPx: 18,
                    waveformHeightPx: 72,
                    clips: [
                        clip({
                            laneTopPx: 20,
                            laneHeightPx: 10,
                            inactive: true,
                        }),
                    ],
                },
            ],
        });

        assertEqual(
            scene.segments.map((segment) => segment.screenRect),
            [{ x: 0, y: 78, width: 500, height: 10 }],
            "take lanes override the row waveform band relative to the body top",
        );
        assertEqual(
            scene.segments.map((segment) => [segment.alpha, segment.inactive]),
            [[0.78, true]],
            "inactive lanes dim segment alpha and carry the flag to geometry",
        );
    }

    {
        const scene = buildWaveformScene({
            axis: createTimelineAxis({
                pxPerSec: 10,
                scrollLeftPx: 0,
                viewportWidthPx: 150,
            }),
            widthPx: 150,
            rows: [
                {
                    topPx: 0,
                    waveformTopPx: 18,
                    waveformHeightPx: 72,
                    clips: [
                        clip({
                            lengthSec: 15,
                            sourceStartSec: 8,
                            sourceEndSec: 10,
                            loopEnabled: true,
                            muted: true,
                            laneTopPx: 6,
                            laneHeightPx: 24,
                            inactive: true,
                        }),
                    ],
                },
            ],
        });

        assertEqual(
            scene.markers.map((marker) => [marker.yPx, marker.heightPx, marker.inactive]),
            [
                [24, 24, true],
                [24, 24, true],
            ],
            "loop markers follow the take lane band and carry the dim flag",
        );
        assertEqual(
            scene.segments.map((segment) => segment.alpha),
            [0.4 * 0.78, 0.4 * 0.78, 0.4 * 0.78],
            "muted and inactive dim factors compose multiplicatively",
        );
    }
});
