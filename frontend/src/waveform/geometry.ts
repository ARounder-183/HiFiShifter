import { fadeGainIn, fadeGainOut } from "../components/layout/timeline/paths.ts";
import {
    INACTIVE_TAKE_COLOR_ALPHA,
    INACTIVE_TAKE_RGB_SCALE,
    type WaveformScene,
} from "./sceneBuilder.ts";

export interface WaveformPeakView {
    min: Float32Array;
    max: Float32Array;
    dataStartSec: number;
    dataDurationSec: number;
}

export interface WaveformGeometry {
    vertices: Float32Array;
    lineCount: number;
    complete: boolean;
}

export type WaveformPeakResolver = (
    sourcePath: string,
    sourceSampleRate: number,
    sourceStartSec: number,
    sourceDurationSec: number,
) => WaveformPeakView | null;

/**
 * 幅度映射钩子：把「线性峰值 × 淡变增益」映射为绘制用的单位幅度。
 *
 * 缺省实现是**线性直投**（`value * gain`），即时间线与绝大多数参数面板的
 * 既有行为。参数编辑器在「动态（DYN）」面板下改传一个**逐帧**版本：它按像素
 * 所在的**时间轴绝对时间**取出该时刻的动态增益，把源波形画成"应用动态之后的
 * 结果电平" —— 于是波形、原声基线与用户曲线共用同一坐标系，用户画一笔就能
 * 立刻看到波形随之起伏。
 *
 * 【为什么必须带时间】动态是逐帧参数，而波形是逐像素列画的。没有时间输入时
 * 映射只能施加一个全局系数，画多少曲线波形都不变 —— 这正是"编辑动态时波形
 * 不按动态值重绘"的根因。
 *
 * 【对称性】钩子对 min / max 各调用一次，且两者拿到的是**同一个** `timeSec`
 * （同一像素列），因此包络的上下沿被同一个增益缩放，波形形状不会被扭曲。
 *
 * @param value 源文件峰值（线性，±1 满量程）。
 * @param gain clip 增益 × 淡变（线性）。
 * @param timeSec 该像素列对应的**时间轴绝对时间**（秒）。未知时为 null。
 * @returns 单位幅度（0 附近的相对量；具体量纲由调用方的映射定义）。
 */
export type WaveformAmplitudeMap = (
    value: number,
    gain: number,
    timeSec: number | null,
) => number;

/**
 * 幅度映射内部数据的**修订号读者**。
 *
 * 【为什么需要】动态面板的映射用延迟取值（读 ref 里的 live 覆盖），函数引用
 * 保持不变也能产出不同结果。几何缓存若只按引用比较会误判"没变"，于是绘制中的
 * 曲线画不上去。
 *
 * 约定：映射可以挂一个 `revision()` 返回单调递增的计数；几何缓存据此判定需要
 * 重建。用**函数**而非数字，是为了让计数变化不必经过 React 状态（参数面板拖动
 * 时每帧 setState 会让整块面板重渲染，代价过高）。
 */
export interface WaveformAmplitudeMapWithRevision {
    revision(): number;
}

/**
 * 读取幅度映射的修订号；未挂载时返回 0（视为恒定，保持既有缓存行为）。
 */
export function readAmplitudeRevision(map: WaveformAmplitudeMap | undefined): number {
    const fn = (map as WaveformAmplitudeMap & Partial<WaveformAmplitudeMapWithRevision>)
        ?.revision;
    return typeof fn === "function" ? fn() : 0;
}

/** 缺省幅度映射：线性直投（保持所有既有调用方的行为不变）。 */
export const linearAmplitudeMap: WaveformAmplitudeMap = (value, gain) => value * gain;

function clamp01(value: number): number {
    return Math.min(1, Math.max(0, value));
}

export function parseWaveformColor(value: string): [number, number, number, number] {
    const input = value.trim();
    const hex = /^#([\da-f]{6}|[\da-f]{8})$/i.exec(input)?.[1];
    if (hex) {
        return [
            Number.parseInt(hex.slice(0, 2), 16) / 255,
            Number.parseInt(hex.slice(2, 4), 16) / 255,
            Number.parseInt(hex.slice(4, 6), 16) / 255,
            hex.length === 8 ? Number.parseInt(hex.slice(6, 8), 16) / 255 : 1,
        ];
    }

    const rgba =
        /^rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)(?:\s*,\s*([\d.]+))?\s*\)$/i.exec(input);
    if (rgba) {
        return [
            clamp01(Number(rgba[1]) / 255),
            clamp01(Number(rgba[2]) / 255),
            clamp01(Number(rgba[3]) / 255),
            clamp01(rgba[4] == null ? 1 : Number(rgba[4])),
        ];
    }
    return [1, 1, 1, 1];
}

function gainAtClipTime(
    clipTimeSec: number,
    totalDurationSec: number,
    fadeInSec: number,
    fadeOutSec: number,
    fadeInShape: number,
    fadeInDir: number,
    fadeOutShape: number,
    fadeOutDir: number,
): number {
    let gain = 1;
    if (fadeInSec > 0 && clipTimeSec < fadeInSec) {
        gain *= fadeGainIn(fadeInShape, fadeInDir, clamp01(clipTimeSec / fadeInSec));
    }
    const fadeOutStart = totalDurationSec - fadeOutSec;
    if (fadeOutSec > 0 && clipTimeSec > fadeOutStart) {
        gain *= fadeGainOut(
            fadeOutShape,
            fadeOutDir,
            clamp01((clipTimeSec - fadeOutStart) / fadeOutSec),
        );
    }
    return gain;
}

/**
 * 顶点缓冲槽。
 *
 * 由调用方持有并跨帧复用：`buffer` 容量不足时会被替换为更大的新数组，
 * 调用方下次传入新的即可。这样**稳态零分配**——否则每帧末尾都要把顶点
 * `slice()` 成独立副本（实测 703 KB/帧），纯属浪费：顶点一旦上传 GPU /
 * 提交给 Canvas2D，CPU 侧就不再需要它了。
 */
export interface WaveformVertexSink {
    buffer: Float32Array;
}

/**
 * 未显式提供 sink 时的模块级兜底槽。
 *
 * 保留它有多个原因：基准（`waveformPerf.bench.ts`）与单元测试不传 sink，
 * 行为应与改动前一致（跨调用复用同一块缓冲、稳态零分配）；生产侧则由
 * `WaveformSurface` 各自持有一个槽，避免两个波形面互相踩缓冲。
 */
const fallbackSink: WaveformVertexSink = { buffer: new Float32Array(0) };

/** 写入游标与当前缓冲打包在一起，避免模块级可变状态。 */
interface VertexSinkState {
    buffer: Float32Array;
    length: number;
}

/** 初始容量（元素个数）：与改动前 `vertexScratch` 的初值一致。 */
const INITIAL_VERTEX_CAPACITY = 8192;

function createVertexSink(
    state: VertexSinkState,
): (x: number, y: number, r: number, g: number, b: number, a: number) => void {
    return (x, y, r, g, b, a) => {
        if (state.length + 6 > state.buffer.length) {
            const next = new Float32Array(
                Math.max(state.buffer.length * 2 || INITIAL_VERTEX_CAPACITY, state.length + 6),
            );
            next.set(state.buffer.subarray(0, state.length));
            state.buffer = next;
        }
        const i = state.length;
        state.buffer[i] = x;
        state.buffer[i + 1] = y;
        state.buffer[i + 2] = r;
        state.buffer[i + 3] = g;
        state.buffer[i + 4] = b;
        state.buffer[i + 5] = a;
        state.length = i + 6;
    };
}

/**
 * 由波形场景构建逐线段顶点几何。
 *
 * 流程：遍历场景的段与标记，按像素列采样 mipmap、算包络与淡变增益，把
 * 每条线段写成 2 个顶点（每顶点 `x, y, r, g, b, a` 共 6 个 float）。
 *
 * @param args.scene 绘制场景（`buildWaveformScene` 的产物）。
 * @param args.color 波形描边色（hex / rgb(a)）。
 * @param args.getPeaks 峰值解析器。
 * @param args.amplitudeMap 幅度映射（缺省线性直投）。见 {@link WaveformAmplitudeMap}。
 * @param args.sink 顶点写入目标；**跨帧复用可做到稳态零分配**。容量不足时
 *   内部倍增并把新缓冲写回 `sink.buffer`。省略时使用模块级兜底槽。
 * @returns 顶点视图（**借用 `sink.buffer`，调用方不得长期持有**——下一次
 *   构建会覆写它；生产侧只在同一次 `render()` 内使用，安全）、线段数与
 *   数据完整性标志。
 */
export function buildWaveformGeometry(args: {
    scene: WaveformScene;
    color: string;
    getPeaks: WaveformPeakResolver;
    amplitudeMap?: WaveformAmplitudeMap;
    sink?: WaveformVertexSink;
}): WaveformGeometry {
    const [red, green, blue, colorAlpha] = parseWaveformColor(args.color);
    const sink = args.sink ?? fallbackSink;
    const state: VertexSinkState = { buffer: sink.buffer, length: 0 };
    const push = createVertexSink(state);
    const amplitudeMap = args.amplitudeMap ?? linearAmplitudeMap;
    let complete = true;

    for (const segment of args.scene.segments) {
        const sourceDurationSec = segment.sourceEndSec - segment.sourceStartSec;
        if (!(sourceDurationSec > 1e-9) || segment.screenRect.width <= 0) continue;
        // inactive take lane：颜色整体压暗（rgb × RGB_SCALE），与场景层已乘入
        // segment.alpha 的 LANE_ALPHA 叠加，复刻旧 Canvas 多 Take 的观感。
        const inactive = Boolean(segment.inactive);
        const segmentRed = inactive ? red * INACTIVE_TAKE_RGB_SCALE : red;
        const segmentGreen = inactive ? green * INACTIVE_TAKE_RGB_SCALE : green;
        const segmentBlue = inactive ? blue * INACTIVE_TAKE_RGB_SCALE : blue;
        const peaks = args.getPeaks(
            segment.sourcePath,
            segment.sourceSampleRate,
            segment.sourceStartSec,
            sourceDurationSec,
        );
        if (!peaks || peaks.min.length === 0 || peaks.max.length === 0) {
            complete = false;
            continue;
        }

        const sampleCount = Math.min(peaks.min.length, peaks.max.length);
        const dataDurationSec = Math.max(1e-12, peaks.dataDurationSec);
        const dataEndSec = peaks.dataStartSec + dataDurationSec;
        const firstX = Math.max(0, Math.ceil(segment.screenRect.x));
        const lastX = Math.max(firstX, Math.ceil(segment.screenRect.x + segment.screenRect.width));
        const halfHeight = segment.screenRect.height / 2;
        const centerY = segment.screenRect.y + halfHeight;
        const sourceSecondsPerPixel = sourceDurationSec / segment.screenRect.width;

        for (let x = firstX; x < lastX; x += 1) {
            const t = clamp01((x + 0.5 - segment.screenRect.x) / segment.screenRect.width);
            const sourceCenterSec = segment.reversed
                ? segment.sourceEndSec - t * sourceDurationSec
                : segment.sourceStartSec + t * sourceDurationSec;
            const sourceLoSec = Math.max(
                peaks.dataStartSec,
                sourceCenterSec - sourceSecondsPerPixel / 2,
            );
            const sourceHiSec = Math.min(dataEndSec, sourceCenterSec + sourceSecondsPerPixel / 2);
            const indexStart = Math.max(
                0,
                Math.floor(((sourceLoSec - peaks.dataStartSec) / dataDurationSec) * sampleCount),
            );
            const indexEnd = Math.min(
                sampleCount - 1,
                Math.max(
                    indexStart,
                    Math.ceil(
                        ((sourceHiSec - peaks.dataStartSec) / dataDurationSec) * sampleCount,
                    ) - 1,
                ),
            );
            let peakMin = Number.POSITIVE_INFINITY;
            let peakMax = Number.NEGATIVE_INFINITY;
            for (let index = indexStart; index <= indexEnd; index += 1) {
                peakMin = Math.min(peakMin, peaks.min[index] ?? 0);
                peakMax = Math.max(peakMax, peaks.max[index] ?? 0);
            }
            if (!Number.isFinite(peakMin) || !Number.isFinite(peakMax)) continue;

            const clipLocalStartSec = segment.clipLocalStartSec;
            const localSpanSec = segment.clipLocalEndSec - segment.clipLocalStartSec;
            const clipTimeSec = clipLocalStartSec + t * localSpanSec;
            const gain =
                segment.gain *
                gainAtClipTime(
                    clipTimeSec,
                    segment.clipTotalDurationSec,
                    segment.fadeInSec,
                    segment.fadeOutSec,
                    segment.fadeInShape,
                    segment.fadeInDir,
                    segment.fadeOutShape,
                    segment.fadeOutDir,
                );
            // 音量增益（gain > 1）会把包络放大到波形矩形之外 —— 表现为波形
            // "溢出" clip 上下边界（DAW 通用 bug）。与 REAPER 一致，增益放大
            // 的显示按矩形削顶（flat-top），既保留"已削波"的视觉暗示又不越界。
            //
            // 幅度经 `amplitudeMap` 归一化：缺省是线性直投（与旧行为逐像素一致）；
            // 动态面板传逐帧映射，按该像素列的**时间轴绝对时间**取动态增益。
            //
            // 【像素 → 时间】`t` 是该列在本段内的归一化位置（0..1，与上面取峰值
            // 用的是同一个 `t`），乘上本段的本地时间跨度再加上 Clip 的时间轴
            // 起点即为绝对时间。倒放时 `t` 已经是"屏幕位置"的推进方向，因此
            // 这里直接用 `t`、不跟随 `reversed` 翻转 —— 与峰值取样的方向无关，
            // 我们要的是"屏幕这一列对应时间轴的哪一刻"。
            const clipStartSec = segment.clipStartSec ?? 0;
            const timeSec = clipStartSec + clipLocalStartSec + t * localSpanSec;

            const rectTop = segment.screenRect.y;
            const rectBottom = rectTop + segment.screenRect.height;
            const mappedTop = amplitudeMap(peakMax, gain, timeSec);
            const mappedBottom = amplitudeMap(peakMin, gain, timeSec);
            const yTop = Math.min(rectBottom, Math.max(rectTop, centerY - mappedTop * halfHeight));
            const yBottom = Math.min(
                rectBottom,
                Math.max(rectTop, centerY - mappedBottom * halfHeight),
            );
            const alpha = colorAlpha * segment.alpha * (inactive ? INACTIVE_TAKE_COLOR_ALPHA : 1);

            push(x + 0.5, yTop, segmentRed, segmentGreen, segmentBlue, alpha);
            push(x + 0.5, yBottom, segmentRed, segmentGreen, segmentBlue, alpha);
        }
    }

    for (const marker of args.scene.markers) {
        const size = Math.min(7, Math.max(4.5, marker.heightPx * 0.16));
        const halfWidth = size * 0.62;
        const inactive = Boolean(marker.inactive);
        const markerRed = inactive ? red * INACTIVE_TAKE_RGB_SCALE : red;
        const markerGreen = inactive ? green * INACTIVE_TAKE_RGB_SCALE : green;
        const markerBlue = inactive ? blue * INACTIVE_TAKE_RGB_SCALE : blue;
        const alpha = colorAlpha * (inactive ? INACTIVE_TAKE_COLOR_ALPHA : 1);
        // 实心 ▽：按整像素扫描线逐行填充。旧版是 1px 空心折线，WebGL 线元
        // 无抗锯齿，两条斜边锯齿非常明显；横线落在 x.5 上天然锐利，小尺寸
        // 下实心标记也更易读。
        const x = Math.round(marker.xPx) + 0.5;
        const yTop = Math.round(marker.yPx) + 0.5;
        const steps = Math.max(2, Math.round(size));
        for (let i = 0; i < steps; i += 1) {
            const hw = halfWidth * (1 - i / steps);
            if (hw < 0.5) break;
            const y = yTop + i;
            push(x - hw, y, markerRed, markerGreen, markerBlue, alpha);
            push(x + hw, y, markerRed, markerGreen, markerBlue, alpha);
        }
    }

    // 直接返回缓冲视图，不再 slice 出副本：顶点在 `render()` 内即被上传
    // GPU / 提交给 Canvas2D，此后 CPU 侧不再需要。这省掉每帧一次约 703 KB
    // 的拷贝（400 clip / 全览实测值）。代价是调用方不能跨帧持有返回值。
    const used = state.length;
    sink.buffer = state.buffer;
    return {
        vertices: state.buffer.subarray(0, used),
        lineCount: used / 12,
        complete,
    };
}
