/**
 * 波形性能基准的合成数据源与场景行构造器。
 *
 * 【主要内容】
 * - `createSyntheticPeakSource()`：生成确定性的 min/max 峰值数据，并实现
 *   `WaveformPeakResolver` 接口，供 `buildWaveformGeometry` 直接消费；
 * - `buildSceneRows()`：把 `buildTimelinePerfScenario` 产出的纯数据 clip
 *   转换成 `WaveformSceneRow[]`。
 *
 * 【作用】让波形渲染链路（scene → geometry → quads）能在 Node 下**脱离
 * React / DOM / WebGL** 单独计时。三个核心函数都是纯函数，只要喂入形状正确
 * 的数据即可，无需真实音频与 mipmap 缓存。
 *
 * 【与其他模块的关系】
 * - 输入：`runtime/timelinePerfScenario.ts` 的 tracks / clips（纯数据）。
 * - 输出：`waveform/sceneBuilder.ts` 的 `WaveformSceneRow[]` 与
 *   `waveform/geometry.ts` 的 `WaveformPeakResolver`。
 * - 仅供基准与测试使用，**严禁**被生产代码引用。
 *
 * 【数据真实性约定】
 * 峰值密度与选级必须复刻生产行为：默认按 L2（divisionFactor = 4096）生成，
 * 44.1 kHz 下约 10.8 peaks/s——这正是"全览 / 小缩放"档位
 * `selectLevel()` 会选中的级别。切片索引与 `dataStartSec` / `dataDurationSec`
 * 的计算与 `waveformMipmapStore.getBestSliceView()` 逐字一致，
 * 否则基准测到的扫描量会失真。
 */

import type { WaveformPeakResolver, WaveformPeakView } from "./geometry.js";
import type { WaveformSceneClip, WaveformSceneRow } from "./sceneBuilder.js";

/** 合成数据的采样率，与主流音频素材一致。 */
const DEFAULT_SAMPLE_RATE = 44100;

/**
 * 默认 mipmap 除数因子：L2（最粗档）。
 *
 * 与 `waveformMipmapStore` 的 `DIV_FACTORS[2]` 同值。小缩放下
 * `selectLevel(samplesPerPixel)` 对 spp > 1024 一律返回 2，故基准用这一档。
 */
const DEFAULT_DIVISION_FACTOR = 4096;

export interface SyntheticPeakSourceOptions {
    sampleRate?: number;
    divisionFactor?: number;
    /** 每个源文件的媒体时长（秒）。默认 60。 */
    mediaDurationSec?: number;
}

export interface SyntheticPeakSource {
    /** 可直接传给 `buildWaveformGeometry` 的解析器。 */
    getPeaks: WaveformPeakResolver;
    /**
     * 已被请求的次数。
     *
     * 用于验证「per-clip 固定开销」假设：`buildWaveformGeometry` 对每个
     * segment 恰好调用一次，因此该计数应等于可见 segment 数。
     */
    callCount(): number;
    resetCount(): void;
}

/**
 * 线性同余伪随机数发生器。
 *
 * 基准必须**可复现**：跨运行、跨机器得到同一份峰值数据，否则耗时的抖动会
 * 淹没优化效果。故不用 `Math.random()`。
 */
function createRng(seed: number): () => number {
    let state = seed >>> 0;
    return () => {
        state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
        return state / 0x1_0000_0000;
    };
}

/** 由字符串派生 32 位种子（FNV-1a 变体），保证同一路径得到同一份数据。 */
function seedFromPath(sourcePath: string): number {
    let hash = 0x811c9dc5;
    for (let i = 0; i < sourcePath.length; i += 1) {
        hash ^= sourcePath.charCodeAt(i);
        hash = Math.imul(hash, 0x01000193) >>> 0;
    }
    return hash >>> 0;
}

/**
 * 创建合成峰值源。
 *
 * 流程：按 `sourcePath` 惰性生成一份 min/max 数组并缓存（每个源文件只生成
 * 一次，模拟 mipmap 缓存命中后的稳态）；每次 `getPeaks` 按请求的时间区间
 * 切出 subarray 视图返回。
 *
 * 切片公式与 `waveformMipmapStore.getBestSliceView()` 保持一致：
 *   startIdx = floor(startSec * sampleRate / divisionFactor)
 *   endIdx   = min(length, ceil((startSec + durationSec) * sampleRate / divisionFactor))
 *   dataStartSec    = startIdx * divisionFactor / sampleRate
 *   dataDurationSec = (endIdx - startIdx) * divisionFactor / sampleRate
 */
export function createSyntheticPeakSource(
    options: SyntheticPeakSourceOptions = {},
): SyntheticPeakSource {
    const sampleRate = options.sampleRate ?? DEFAULT_SAMPLE_RATE;
    const divisionFactor = options.divisionFactor ?? DEFAULT_DIVISION_FACTOR;
    const mediaDurationSec = options.mediaDurationSec ?? 60;

    const peakCount = Math.max(
        1,
        Math.ceil((mediaDurationSec * sampleRate) / divisionFactor),
    );
    const cache = new Map<string, { min: Float32Array; max: Float32Array }>();
    let calls = 0;

    /**
     * 生成一条带包络的伪音频峰值。
     *
     * 纯均匀随机会让每列的 min/max 都接近满量程，导致 `geometry.ts` 的聚合
     * 结果与真实音频不同（真实音频大部分区域远低于峰值）。这里叠一层缓慢
     * 变化的包络，使峰值分布更接近实际素材。
     */
    const create = (sourcePath: string) => {
        const rng = createRng(seedFromPath(sourcePath));
        const min = new Float32Array(peakCount);
        const max = new Float32Array(peakCount);
        for (let i = 0; i < peakCount; i += 1) {
            const envelope = 0.2 + 0.75 * Math.abs(Math.sin(i * 0.017));
            const peak = envelope * (0.35 + 0.65 * rng());
            max[i] = peak;
            min[i] = -peak * (0.55 + 0.45 * rng());
        }
        return { min, max };
    };

    const getPeaks: WaveformPeakResolver = (
        sourcePath,
        _sourceSampleRate,
        sourceStartSec,
        sourceDurationSec,
    ): WaveformPeakView | null => {
        calls += 1;
        let peaks = cache.get(sourcePath);
        if (peaks === undefined) {
            peaks = create(sourcePath);
            cache.set(sourcePath, peaks);
        }

        const startIdx = Math.max(0, Math.floor((sourceStartSec * sampleRate) / divisionFactor));
        const endIdx = Math.min(
            peaks.min.length,
            Math.ceil(((sourceStartSec + sourceDurationSec) * sampleRate) / divisionFactor),
        );
        if (endIdx <= startIdx) return null;

        return {
            min: peaks.min.subarray(startIdx, endIdx),
            max: peaks.max.subarray(startIdx, endIdx),
            dataStartSec: (startIdx * divisionFactor) / sampleRate,
            dataDurationSec: ((endIdx - startIdx) * divisionFactor) / sampleRate,
        };
    };

    return {
        getPeaks,
        callCount: () => calls,
        resetCount: () => {
            calls = 0;
        },
    };
}

/** 场景行构造所需的最小 clip 信息。 */
export interface SceneRowClip {
    id: string;
    trackId: string;
    startSec: number;
    lengthSec: number;
}

export interface SceneRowBuildArgs {
    tracks: ReadonlyArray<{ id: string }>;
    clips: ReadonlyArray<SceneRowClip>;
    /** 轨道行高（CSS 像素）。 */
    rowHeight: number;
    /** 波形带相对行顶的偏移，对应 `CLIP_HEADER_HEIGHT`。 */
    waveformTopPx?: number;
    /** 波形带高度；未指定时按 `rowHeight - 顶部偏移 - 底部留白` 计算。 */
    waveformHeightPx?: number;
    /** clip 的源路径；默认每个 clip 一个独立源文件。 */
    sourcePathOf?: (clip: { id: string; trackId: string }) => string;
}

/**
 * 把纯数据 clip 组装成 `WaveformSceneRow[]`。
 *
 * 与 `TimelineWaveformSurface` 的运行时构造保持一致：一个轨道一行，
 * `topPx` 为内容绝对坐标，波形带位于 header 之下。
 *
 * 说明：这里**不**走 `clipToSceneClip()`，因为那需要完整的 `ClipInfo`
 * （含 Redux 里的淡变、增益、loop 等字段）；基准只需最小可用投影，
 * 其余字段取生产默认值即可。
 */
export function buildSceneRows(args: SceneRowBuildArgs): WaveformSceneRow[] {
    const waveformTopPx = args.waveformTopPx ?? 18;
    const waveformHeightPx =
        args.waveformHeightPx ?? Math.max(1, args.rowHeight - waveformTopPx - 2);
    const sourcePathOf = args.sourcePathOf ?? ((clip) => `/media/${clip.id}.wav`);

    const byTrack = new Map<string, SceneRowClip[]>();
    for (const clip of args.clips) {
        const bucket = byTrack.get(clip.trackId);
        if (bucket) bucket.push(clip);
        else byTrack.set(clip.trackId, [clip]);
    }

    return args.tracks.map((track, trackIndex) => {
        const clips = byTrack.get(track.id) ?? [];
        const sceneClips: WaveformSceneClip[] = clips.map((clip) => ({
            id: clip.id,
            sourcePath: sourcePathOf(clip),
            startSec: clip.startSec,
            lengthSec: clip.lengthSec,
            sourceStartSec: 0,
            sourceEndSec: clip.lengthSec,
            durationSec: clip.lengthSec,
            sourceSampleRate: DEFAULT_SAMPLE_RATE,
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
        }));

        return {
            topPx: trackIndex * args.rowHeight,
            waveformTopPx,
            waveformHeightPx,
            clips: sceneClips,
        };
    });
}
