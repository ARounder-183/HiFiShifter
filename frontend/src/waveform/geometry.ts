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

// ── 顶点写入的模块级复用缓冲 ─────────────────────────────────────────
// 滚动热路径上每帧构建几何：先写入可增长的 Float32Array（跨帧复用，
// 稳态零分配），最后 slice 出独立副本返回，避免 number[] 装箱推送与
// 逐帧新建大数组的 GC 压力。
let vertexScratch = new Float32Array(8192);
let vertexScratchLength = 0;

function createVertexSink(): (
    x: number,
    y: number,
    r: number,
    g: number,
    b: number,
    a: number,
) => void {
    return (x, y, r, g, b, a) => {
        if (vertexScratchLength + 6 > vertexScratch.length) {
            const next = new Float32Array(
                Math.max(vertexScratch.length * 2, vertexScratchLength + 6),
            );
            next.set(vertexScratch.subarray(0, vertexScratchLength));
            vertexScratch = next;
        }
        let i = vertexScratchLength;
        vertexScratch[i] = x;
        vertexScratch[i + 1] = y;
        vertexScratch[i + 2] = r;
        vertexScratch[i + 3] = g;
        vertexScratch[i + 4] = b;
        vertexScratch[i + 5] = a;
        vertexScratchLength = i + 6;
    };
}

export function buildWaveformGeometry(args: {
    scene: WaveformScene;
    color: string;
    getPeaks: WaveformPeakResolver;
}): WaveformGeometry {
    const [red, green, blue, colorAlpha] = parseWaveformColor(args.color);
    const push = createVertexSink();
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

            const clipTimeSec =
                segment.clipLocalStartSec +
                t * (segment.clipLocalEndSec - segment.clipLocalStartSec);
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
            const rectTop = segment.screenRect.y;
            const rectBottom = rectTop + segment.screenRect.height;
            const yTop = Math.min(
                rectBottom,
                Math.max(rectTop, centerY - peakMax * gain * halfHeight),
            );
            const yBottom = Math.min(
                rectBottom,
                Math.max(rectTop, centerY - peakMin * gain * halfHeight),
            );
            const alpha =
                colorAlpha * segment.alpha * (inactive ? INACTIVE_TAKE_COLOR_ALPHA : 1);

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

    // 先取长度再清零：slice 出独立副本供 GPU/2D 渲染器安全持有。
    const used = vertexScratchLength;
    vertexScratchLength = 0;
    return {
        vertices: vertexScratch.slice(0, used),
        lineCount: used / 12,
        complete,
    };
}
