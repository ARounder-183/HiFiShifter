/**
 * 参数编辑器波形的**每帧重建成本**回归基准（真实调用形态）。
 *
 * 【为什么单独一份】几何层按**列内增益切片**调用幅度映射 —— 一次典型重建
 * （1 行 × 2624px 窗口 × L0 峰值密度）会产生数万次询问。任何在询问里做
 * "本可缓存"的工作（字符串解析、对象分配、重复曲线取样）都会被放大成
 * 每帧数十毫秒的卡顿（用户报告的「编辑音量/动态时卡顿」）。本基准把三条
 * 路径钉在同一场景里对比，防止这类回归重新引入。
 *
 * 【两个场景：细缩放与粗缩放】缩放等级改变的不只是峰值密度：
 *
 * - **细缩放**（150 px/s，L0 密度）列最多、每列窗口最窄，是"列数 × 切片数"
 *   主导的情形（2112 列 × 16 切片）；
 * - **粗缩放**（6 px/s）列少、每列窗口宽，**上界钳制**的逐帧枚举窗口随之变宽
 *   （`levelCeilingOverWindow` 的窗口 = 列宽 / 16），是另一条成本曲线 ——
 *   查表实现若只对细缩放友好，这里会立刻暴露。
 *
 * 【每帧几次重建不在此处钉】"一帧内多次请求只重建一次"由绘制调度保证
 * （`PianoRollPanel` 的 `waveformRepaintLoop` → `renderKernel/renderLoop`），
 * 其不变量由 `renderKernel/renderLoop.test.ts`（"同一帧内多次 invalidate 只
 * 调度一次、只绘制一次"）与 `loudnessLutEquivalence.test.ts`（查表不随查询
 * 次数增长）覆盖。本基准只负责**单次重建**的成本。
 *
 * 【运行】`npx vitest bench src/components/layout/pianoRoll/loudnessPerf.bench.ts`
 */

import { bench, describe } from "vitest";

import { makeLoudnessAmplitudeMap } from "./PianoRollWaveformSurface";
import { createLiveOverrideReader } from "./liveLoudnessOverride";
import { buildWaveformGeometry, type WaveformAmplitudeMap } from "../../../waveform/geometry";
import { buildWaveformScene, type WaveformSceneRow } from "../../../waveform/sceneBuilder";
import { createTimelineAxis } from "../renderKernel/timelineAxis";

const DPR = 1;
const VIEW_W = 1600;
const VIEW_H = 600;
const MARGIN = 512;
const SAMPLE_RATE = 44100;
const DIV = 16;
const FRAME_PERIOD_MS = 5;
const SOURCE_SEC = 60;
const PROJECT_SEC = 180;
const PEAK_COUNT = Math.floor(SOURCE_SEC * (SAMPLE_RATE / DIV));

const peaksMin = new Float32Array(PEAK_COUNT);
const peaksMax = new Float32Array(PEAK_COUNT);
for (let i = 0; i < PEAK_COUNT; i += 1) {
    const envelope = 0.2 + 0.75 * Math.abs(Math.sin(i * 0.011));
    peaksMax[i] = envelope * 0.9;
    peaksMin[i] = -envelope * 0.8;
}

const rows: WaveformSceneRow[] = [
    {
        topPx: 0,
        waveformTopPx: 0,
        waveformHeightPx: VIEW_H,
        clips: [
            {
                id: "clip-0",
                sourcePath: "/media/a.wav",
                startSec: 0,
                lengthSec: PROJECT_SEC,
                sourceStartSec: 0,
                sourceEndSec: SOURCE_SEC,
                durationSec: SOURCE_SEC,
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
            },
        ],
    },
];

function getPeaks(_path: string, _rate: number, startSec: number, durationSec: number) {
    const startIdx = Math.max(0, Math.floor((startSec * SAMPLE_RATE) / DIV));
    const endIdx = Math.min(PEAK_COUNT, Math.ceil(((startSec + durationSec) * SAMPLE_RATE) / DIV));
    if (endIdx <= startIdx) return null;
    return {
        min: peaksMin.subarray(startIdx, endIdx),
        max: peaksMax.subarray(startIdx, endIdx),
        dataStartSec: (startIdx * DIV) / SAMPLE_RATE,
        dataDurationSec: ((endIdx - startIdx) * DIV) / SAMPLE_RATE,
    };
}

const FRAME_COUNT = Math.floor((PROJECT_SEC * 1000) / FRAME_PERIOD_MS);
const volume: number[] = new Array(FRAME_COUNT);
const dynTarget: number[] = new Array(FRAME_COUNT);
const dynBaseline: number[] = new Array(FRAME_COUNT);
for (let i = 0; i < FRAME_COUNT; i += 1) {
    volume[i] = 0.5 + 0.5 * Math.sin(i * 0.0007);
    dynBaseline[i] = 0.3 + 0.25 * Math.abs(Math.sin(i * 0.0003));
    dynTarget[i] = dynBaseline[i] * (0.7 + 0.3 * Math.sin(i * 0.00011));
}
const snapshot = {
    startFrame: 0,
    stride: 1,
    framePeriodMs: FRAME_PERIOD_MS,
    volume,
    dynTarget,
    dynBaseline,
};

/** 拖动期间的 live 覆盖（**同一个对象**：edit 数组原地更新）。 */
const LIVE = { key: `v2|track-1|volume|0|${FRAME_COUNT}|1`, edit: volume };
const reader = createLiveOverrideReader();

const loudnessLive = makeLoudnessAmplitudeMap(
    snapshot,
    {
        volume: () => reader.read("volume", LIVE),
        dyn: () => reader.read("dyn", LIVE),
    },
    () => 0,
) as WaveformAmplitudeMap;

const loudnessSnapshot = makeLoudnessAmplitudeMap(
    snapshot,
    { volume: () => null, dyn: () => null },
    () => 0,
) as WaveformAmplitudeMap;

const linear: WaveformAmplitudeMap = (value, gain) => value * gain;

const sink = { buffer: new Float32Array(1 << 24) };

/** 按给定水平缩放构建窗口场景（同一份峰值与曲线，只改投影）。 */
function makeScene(pxPerSec: number): {
    windowW: number;
    scene: ReturnType<typeof buildWaveformScene>;
} {
    const windowW = VIEW_W + MARGIN * 2;
    return {
        windowW,
        scene: buildWaveformScene({
            axis: createTimelineAxis({
                pxPerSec,
                scrollLeftPx: -MARGIN,
                viewportWidthPx: windowW,
                dpr: DPR,
            }),
            widthPx: windowW,
            viewportTopPx: 0,
            rows,
        }),
    };
}

const fine = makeScene(150);
const coarse = makeScene(6);

function build(
    target: { windowW: number; scene: ReturnType<typeof buildWaveformScene> },
    map: WaveformAmplitudeMap | undefined,
) {
    return buildWaveformGeometry({
        scene: target.scene,
        color: "#8fa3bf",
        getPeaks,
        amplitudeMap: map,
        sink,
        dpr: DPR,
    });
}

const warm = build(fine, linear);
console.log(
    `[参数编辑器波形重建] segments=${fine.scene.segments.length} columns=${warm.lineCount} ` +
        `slices/column=${Math.min(
            Math.max(
                1,
                Math.round(
                    ((fine.scene.segments[0]?.sourceEndSec ?? 0) / (warm.lineCount || 1)) *
                        (SAMPLE_RATE / DIV),
                ),
            ),
            16,
        )}`,
);

describe("参数编辑器波形重建（1 行 / 2624px 窗口 / L0 / 150 px/s）", () => {
    bench("线性映射（无响度自动化）", () => {
        build(fine, linear);
    });

    bench("响度映射 · 无 live 覆盖", () => {
        build(fine, loudnessSnapshot);
    });

    bench("响度映射 · 拖动音量（live 覆盖）", () => {
        build(fine, loudnessLive);
    });
});

describe("参数编辑器波形重建（粗缩放 6 px/s：列少、列内窗口宽）", () => {
    bench("线性映射（无响度自动化）", () => {
        build(coarse, linear);
    });

    bench("响度映射 · 无 live 覆盖", () => {
        build(coarse, loudnessSnapshot);
    });

    bench("响度映射 · 拖动音量（live 覆盖）", () => {
        build(coarse, loudnessLive);
    });
});
