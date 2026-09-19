import { test } from "vitest";

import { buildWaveformGeometry, parseWaveformColor } from "./geometry.ts";
import type { WaveformScene } from "./sceneBuilder.ts";

test("waveform/geometry.test.ts scripted checks", async () => {
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
                fadeInShape: 0,
                fadeInDir: 0,
                fadeOutShape: 0,
                fadeOutDir: 0,
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

    const dimmed = buildWaveformGeometry({
        scene: { segments: [{ ...scene.segments[0], inactive: true }], markers: [] },
        color: "#ffffff",
        getPeaks: () => ({
            min: new Float32Array([-1]),
            max: new Float32Array([1]),
            dataStartSec: 0,
            dataDurationSec: 1,
        }),
    });
    assertEqual(
        Array.from(dimmed.vertices.slice(2, 6)),
        [Math.fround(0.42), Math.fround(0.42), Math.fround(0.42), Math.fround(0.78)],
        "inactive take lanes darken rgb and color alpha",
    );

    const dimmedMarker = buildWaveformGeometry({
        scene: {
            segments: [],
            markers: [
                {
                    clipId: "clip",
                    timelineSec: 1,
                    xPx: 10,
                    yPx: 20,
                    heightPx: 100,
                    kind: "loop",
                    inactive: true,
                },
            ],
        },
        color: "#ffffff",
        getPeaks: () => null,
    });
    assertEqual(
        dimmedMarker.vertices.length,
        84,
        "inactive marker still emits its scanline-filled triangle",
    );
    assertEqual(
        Array.from(dimmedMarker.vertices.slice(2, 6)),
        [Math.fround(0.42), Math.fround(0.42), Math.fround(0.42), Math.fround(0.78)],
        "inactive markers darken too",
    );

    // 音量增益 > 1 时包络按 gain 放大，必须被钳制在波形矩形内（削顶显示），
    // 不能溢出 clip 上下边界。
    const boosted = buildWaveformGeometry({
        scene: { segments: [{ ...scene.segments[0], gain: 3 }], markers: [] },
        color: "#ffffff",
        getPeaks: () => ({
            min: new Float32Array([-1]),
            max: new Float32Array([1]),
            dataStartSec: 0,
            dataDurationSec: 1,
        }),
    });
    // 高度 100，中心 50，±1 峰值 × gain 3 → 名义 ±150，钳制后恰好 0 / 100。
    assertEqual(
        [boosted.vertices[1], boosted.vertices[7]],
        [0, 100],
        "gain-boosted envelope clamps to the waveform rect (flat-top display)",
    );

    // gain = 1 时钳制不影响正常包络（回归保护）。峰值取 ±0.5 保证精确：
    // 中心 50 ± 0.5×50 → 25 / 75。
    const unity = buildWaveformGeometry({
        scene,
        color: "#ffffff",
        getPeaks: () => ({
            min: new Float32Array([-0.5]),
            max: new Float32Array([0.5]),
            dataStartSec: 0,
            dataDurationSec: 1,
        }),
    });
    assertEqual(
        [unity.vertices[1], unity.vertices[7]],
        [25, 75],
        "unity gain envelope is untouched by the clamp",
    );
});

/**
 * 逐时间幅度映射（动态面板）。
 *
 * 【为什么必须单测这一层】"编辑动态时波形不重绘"的根因是映射拿不到时间：
 * 它只能施加一个全局系数，画多少曲线波形都不变。这里钉住"同一峰值在不同
 * 时间可以映射出不同高度"，以及像素列 → 时间轴绝对时间的换算正确。
 */
test("amplitude map receives pixel-column timeline time", async () => {
    const scene: WaveformScene = {
        segments: [
            {
                clipId: "clip",
                sourcePath: "/tone.wav",
                sourceSampleRate: 4,
                sourceStartSec: 0,
                sourceEndSec: 1,
                // Clip 从时间轴 10s 开始，本段覆盖它的第 0..1s。
                clipStartSec: 10,
                clipLocalStartSec: 0,
                clipLocalEndSec: 1,
                clipTotalDurationSec: 1,
                screenRect: { x: 0, y: 0, width: 4, height: 100 },
                reversed: false,
                gain: 1,
                fadeInSec: 0,
                fadeOutSec: 0,
                fadeInShape: 0,
                fadeInDir: 0,
                fadeOutShape: 0,
                fadeOutDir: 0,
                alpha: 1,
            },
        ],
        markers: [],
    };

    const seen: number[] = [];
    const result = buildWaveformGeometry({
        scene,
        color: "#ffffff",
        getPeaks: () => ({
            min: new Float32Array([-1, -1, -1, -1]),
            max: new Float32Array([1, 1, 1, 1]),
            dataStartSec: 0,
            dataDurationSec: 1,
        }),
        amplitudeMap: (value, gain, timeSec) => {
            if (timeSec !== null) seen.push(timeSec);
            return value * gain;
        },
    });

    // 4 列 × 2 次调用（min / max），每次拿到同一列的时间。
    if (seen.length !== 8) {
        throw new Error(`expected 8 samples, received ${seen.length}`);
    }
    // 列中心分别对应本段内的 0.125 / 0.375 / 0.625 / 0.875 秒，加 Clip 起点 10s。
    const expected = [10.125, 10.375, 10.625, 10.875];
    for (let i = 0; i < 4; i += 1) {
        const got = seen[i * 2];
        if (Math.abs(got - expected[i]) > 1e-6) {
            throw new Error(`column ${i}: expected ${expected[i]}, received ${got}`);
        }
        // min 与 max 必须用同一个时间（否则包络上下沿会被不同增益缩放，形状失真）。
        if (seen[i * 2] !== seen[i * 2 + 1]) {
            throw new Error(`column ${i}: min/max disagree on time`);
        }
    }
    if (result.lineCount !== 4) {
        throw new Error(`expected 4 lines, received ${result.lineCount}`);
    }
});

test("amplitude map can vary per pixel column (dynamic gain)", async () => {
    const scene: WaveformScene = {
        segments: [
            {
                clipId: "clip",
                sourcePath: "/tone.wav",
                sourceSampleRate: 4,
                sourceStartSec: 0,
                sourceEndSec: 1,
                clipStartSec: 0,
                clipLocalStartSec: 0,
                clipLocalEndSec: 1,
                clipTotalDurationSec: 1,
                screenRect: { x: 0, y: 0, width: 4, height: 100 },
                reversed: false,
                gain: 1,
                fadeInSec: 0,
                fadeOutSec: 0,
                fadeInShape: 0,
                fadeInDir: 0,
                fadeOutShape: 0,
                fadeOutDir: 0,
                alpha: 1,
            },
        ],
        markers: [],
    };

    // 模拟动态增益：前半段 ×2，后半段 ×0.5。
    const result = buildWaveformGeometry({
        scene,
        color: "#ffffff",
        getPeaks: () => ({
            min: new Float32Array([-0.5, -0.5, -0.5, -0.5]),
            max: new Float32Array([0.5, 0.5, 0.5, 0.5]),
            dataStartSec: 0,
            dataDurationSec: 1,
        }),
        amplitudeMap: (value, gain, timeSec) =>
            value * gain * (timeSec !== null && timeSec < 0.5 ? 2 : 0.5),
    });

    // 中心 50、半高 50：×2 → 峰值 0.5×2=1 → top 0；×0.5 → 0.25 → top 37.5。
    const topOf = (col: number) => result.vertices[col * 12 + 1];
    if (topOf(0) !== 0) {
        throw new Error(`column 0 should be boosted to the top, received ${topOf(0)}`);
    }
    if (Math.abs(topOf(3) - 37.5) > 1e-6) {
        throw new Error(`column 3 should be attenuated, received ${topOf(3)}`);
    }
    // 逐列不同 —— 这正是"波形按动态值重绘"的可观测判据。
    if (topOf(0) === topOf(3)) {
        throw new Error("columns must differ when the dynamic gain varies over time");
    }
});
