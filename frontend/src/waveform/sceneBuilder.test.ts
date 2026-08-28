import { test } from "vitest";

import { buildWaveformScene, type WaveformSceneClip } from "./sceneBuilder.ts";

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
            viewportStartSec: 10,
            viewportEndSec: 20,
            pxPerSec: 100,
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
            viewportStartSec: 0,
            viewportEndSec: 10,
            pxPerSec: 100,
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
            viewportStartSec: 0,
            viewportEndSec: 15,
            pxPerSec: 10,
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
            viewportStartSec: 0,
            viewportEndSec: 5,
            pxPerSec: 100,
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
});
