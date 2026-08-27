import { fadeGainIn, fadeGainOut } from "../components/layout/timeline/paths.ts";
import type { WaveformScene } from "./sceneBuilder.ts";

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

export function buildWaveformGeometry(args: {
    scene: WaveformScene;
    color: string;
    getPeaks: WaveformPeakResolver;
}): WaveformGeometry {
    const [red, green, blue, colorAlpha] = parseWaveformColor(args.color);
    const values: number[] = [];
    let complete = true;

    for (const segment of args.scene.segments) {
        const sourceDurationSec = segment.sourceEndSec - segment.sourceStartSec;
        if (!(sourceDurationSec > 1e-9) || segment.screenRect.width <= 0) continue;
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
            const yTop = centerY - peakMax * gain * halfHeight;
            const yBottom = centerY - peakMin * gain * halfHeight;
            const alpha = colorAlpha * segment.alpha;

            values.push(x + 0.5, yTop, red, green, blue, alpha);
            values.push(x + 0.5, yBottom, red, green, blue, alpha);
        }
    }

    for (const marker of args.scene.markers) {
        const size = Math.min(7, Math.max(4.5, marker.heightPx * 0.16));
        const halfWidth = size * 0.62;
        const alpha = colorAlpha;
        const x = marker.xPx;
        const y = marker.yPx + 0.5;
        values.push(x - halfWidth, y, red, green, blue, alpha);
        values.push(x + halfWidth, y, red, green, blue, alpha);
        values.push(x - halfWidth, y, red, green, blue, alpha);
        values.push(x, y + size, red, green, blue, alpha);
        values.push(x + halfWidth, y, red, green, blue, alpha);
        values.push(x, y + size, red, green, blue, alpha);
    }

    return {
        vertices: new Float32Array(values),
        lineCount: values.length / 12,
        complete,
    };
}
