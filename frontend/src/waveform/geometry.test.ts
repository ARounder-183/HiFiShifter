import { buildWaveformGeometry, parseWaveformColor } from "./geometry.ts";
import type { WaveformScene } from "./sceneBuilder.ts";

function assertEqual(actual: unknown, expected: unknown, label: string): void {
    const actualJson = JSON.stringify(actual);
    const expectedJson = JSON.stringify(expected);
    if (actualJson !== expectedJson) {
        throw new Error(`${label}: expected ${expectedJson}, received ${actualJson}`);
    }
}

assertEqual(
    parseWaveformColor("rgba(246,250,255,0.92)"),
    [246 / 255, 250 / 255, 1, 0.92],
    "rgba colors become normalized GPU channels",
);
assertEqual(
    parseWaveformColor("#804020"),
    [128 / 255, 64 / 255, 32 / 255, 1],
    "hex colors become normalized GPU channels",
);

const scene: WaveformScene = {
    segments: [
        {
            clipId: "clip",
            sourcePath: "/tone.wav",
            sourceSampleRate: 4,
            sourceStartSec: 0,
            sourceEndSec: 1,
            clipLocalStartSec: 0,
            clipLocalEndSec: 1,
            clipTotalDurationSec: 1,
            screenRect: { x: 0, y: 0, width: 4, height: 100 },
            reversed: false,
            gain: 1,
            fadeInSec: 0,
            fadeOutSec: 0,
            fadeInCurve: "linear",
            fadeOutCurve: "linear",
            alpha: 1,
        },
    ],
    markers: [],
};

const result = buildWaveformGeometry({
    scene,
    color: "rgba(255,255,255,0.5)",
    getPeaks: () => ({
        min: new Float32Array([-1, -0.5, -0.25, 0]),
        max: new Float32Array([1, 0.5, 0.25, 0]),
        dataStartSec: 0,
        dataDurationSec: 1,
    }),
});

assertEqual(result.complete, true, "geometry is complete when every segment has peaks");
assertEqual(result.lineCount, 4, "one GPU line is emitted per visible pixel column");
assertEqual(
    Array.from(result.vertices.slice(0, 12)),
    [0.5, 0, 1, 1, 1, 0.5, 0.5, 100, 1, 1, 1, 0.5],
    "first peak maps to the full waveform height with normalized color",
);

const missing = buildWaveformGeometry({
    scene,
    color: "#ffffff",
    getPeaks: () => null,
});
assertEqual(missing.complete, false, "missing peaks defer frame presentation");
assertEqual(missing.vertices.length, 0, "missing data renders no replacement geometry");

const partialScene: WaveformScene = {
    segments: [
        scene.segments[0],
        {
            ...scene.segments[0],
            clipId: "missing",
            sourcePath: "/missing.wav",
            screenRect: { x: 4, y: 0, width: 4, height: 100 },
        },
    ],
    markers: [],
};

const partial = buildWaveformGeometry({
    scene: partialScene,
    color: "#ffffff",
    getPeaks: (sourcePath: string) =>
        sourcePath === "/tone.wav"
          ? {
                min: new Float32Array([-1, 1]),
                max: new Float32Array([1, -1]),
                dataStartSec: 0,
                dataDurationSec: 1,
            }
          : null,
});
assertEqual(partial.complete, false, "partial availability is reported as incomplete");
assertEqual(partial.lineCount, 4, "available segments still render while missing data loads");

console.log("waveform geometry checks passed");
