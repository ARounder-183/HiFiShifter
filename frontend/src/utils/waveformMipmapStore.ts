/**
 * 波形 Mipmap 缓存管理器（整文件级）
 *
 * 每个音频文件缓存三级 Float32Array 数据：
 * - L0 (div=64):   精细级，近距离对轨，spp ≤ 256
 * - L1 (div=512):  中间级，日常编辑，256 < spp ≤ 2048
 * - L2 (div=4096): 全局级，预览/导航，spp > 2048
 *
 * 状态管理策略：
 * - 波形二进制数据存在外部 Map（不放 Redux，避免序列化开销）
 * - 文件加载状态可通过回调通知 UI
 */

import { waveformApi } from "../services/api/waveform";
import {
    decodeWaveformFromBase64,
    type WaveformMipmapBinary,
} from "./waveformBinaryCodec";

// ============== 常量 ==============

/** 三级 mipmap 的除数因子 */
const DIV_FACTORS = [64, 512, 4096] as const;

/** 级别选择的 spp 阈值 */
const SPP_THRESHOLDS = [256, 2048] as const;
const SPP_HYSTERESIS_ENTER_SCALE = 1.25;
const SPP_HYSTERESIS_EXIT_SCALE = 0.75;

/** mipmap 级别数量 */
const LEVEL_COUNT = 3;

// ============== 类型 ==============

/** 单级 peaks 数据 */
export interface LevelPeaks {
    /** 最小值数组 */
    min: Float32Array;
    /** 最大值数组 */
    max: Float32Array;
    /** 该级别的除数因子 */
    divisionFactor: number;
    /** 采样率 */
    sampleRate: number;
}

export type WaveformMipmapLevel = 0 | 1 | 2;

/** 文件级缓存条目 */
interface FileMipmapCache {
    /** 采样率 */
    sampleRate: number;
    /** 三级 peaks 数据（null = 尚未加载） */
    levels: [LevelPeaks | null, LevelPeaks | null, LevelPeaks | null];
    /** 正在加载中的级别 */
    loadingLevels: Set<number>;
}

/** 加载状态回调 */
export type LoadCallback = (
    sourcePath: string,
    status: "loading" | "done" | "error",
    error?: string,
) => void;

// ============== 核心实现 ==============

class WaveformMipmapStoreImpl {
    /** sourcePath → FileMipmapCache */
    private cache = new Map<string, FileMipmapCache>();

    /** 加载状态监听器 */
    private listeners = new Set<LoadCallback>();

    /** 正在进行的加载 Promise（用于 preload 等待已发起的加载） */
    private loadingPromises = new Map<string, Promise<void>>();

    /**
     * interleaved 缓冲区复用池。
     * 每次 getInterleavedSlice 会优先从池中取出同等大小的 Float32Array 进行复用，
     * 避免快速缩放时每帧 new Float32Array 产生的 GC 压力。
     */
    private interleavedPool: Float32Array[] = [];
    /** 池的最大容量（条目数） */
    private static readonly POOL_MAX = 8;

    /**
     * 从池中获取一个 length === exactLen 的 Float32Array，或新建一个
     */
    private acquireInterleaved(exactLen: number): Float32Array {
        for (let i = 0; i < this.interleavedPool.length; i++) {
            if (this.interleavedPool[i].length === exactLen) {
                const buf = this.interleavedPool[i];
                this.interleavedPool.splice(i, 1);
                return buf;
            }
        }
        return new Float32Array(exactLen);
    }

    /**
     * 归还 buffer 到池（供下一帧复用）
     */
    releaseInterleaved(buf: Float32Array): void {
        if (
            buf.length > 0 &&
            this.interleavedPool.length < WaveformMipmapStoreImpl.POOL_MAX
        ) {
            this.interleavedPool.push(buf);
        }
    }

    // ---------- 公共 API ----------

    /**
     * 根据 samples_per_pixel 自动选择最佳 mipmap 级别
     */
    selectLevel(samplesPerPixel: number): 0 | 1 | 2 {
        if (samplesPerPixel <= SPP_THRESHOLDS[0]) return 0;
        if (samplesPerPixel <= SPP_THRESHOLDS[1]) return 1;
        return 2;
    }

    selectLevelStable(
        samplesPerPixel: number,
        previousLevel?: WaveformMipmapLevel | null,
    ): WaveformMipmapLevel {
        let newLevel: WaveformMipmapLevel;

        if (previousLevel == null) {
            newLevel = this.selectLevel(samplesPerPixel);
        } else {
            const enterL1 = SPP_THRESHOLDS[0] * SPP_HYSTERESIS_ENTER_SCALE;
            const exitL1 = SPP_THRESHOLDS[0] * SPP_HYSTERESIS_EXIT_SCALE;
            const enterL2 = SPP_THRESHOLDS[1] * SPP_HYSTERESIS_ENTER_SCALE;
            const exitL2 = SPP_THRESHOLDS[1] * SPP_HYSTERESIS_EXIT_SCALE;

            if (previousLevel === 0) {
                if (samplesPerPixel > enterL2) newLevel = 2;
                else if (samplesPerPixel > enterL1) newLevel = 1;
                else newLevel = 0;
            } else if (previousLevel === 1) {
                if (samplesPerPixel <= exitL1) newLevel = 0;
                else if (samplesPerPixel > enterL2) newLevel = 2;
                else newLevel = 1;
            } else {
                if (samplesPerPixel <= exitL1) newLevel = 0;
                else if (samplesPerPixel <= exitL2) newLevel = 1;
                else newLevel = 2;
            }
        }

        return newLevel;
    }

    /**
     * 获取指定文件指定级别的 peaks 数据
     *
     * 如果尚未加载，会自动发起请求并返回 null。
     * 数据加载完成后通过 listener 通知。
     */
    getPeaks(
        sourcePath: string,
        level: 0 | 1 | 2,
    ): LevelPeaks | null {
        const entry = this.cache.get(sourcePath);
        if (!entry) {
            // 首次请求，发起加载
            this.loadLevel(sourcePath, level);
            return null;
        }

        const data = entry.levels[level];
        if (data) return data;

        // 该级别尚未加载
        if (!entry.loadingLevels.has(level)) {
            this.loadLevel(sourcePath, level);
        }
        return null;
    }

    /**
     * 获取指定文件在指定时间范围内的 peaks 切片
     *
     * 使用 Float32Array.subarray（零拷贝）返回切片。
     *
     * @param sourcePath 音频文件路径
     * @param level mipmap 级别
     * @param startSec 开始时间（秒）
     * @param durationSec 持续时间（秒）
     * @returns peaks 切片，或 null（数据未加载时）
     */
    getSlice(
        sourcePath: string,
        level: 0 | 1 | 2,
        startSec: number,
        durationSec: number,
    ): { min: Float32Array; max: Float32Array } | null {
        const peaks = this.getPeaks(sourcePath, level);
        if (!peaks) return null;

        const { sampleRate, divisionFactor, min, max } = peaks;
        if (sampleRate <= 0 || divisionFactor <= 0) return null;

        // 计算索引范围
        const startIdx = Math.max(
            0,
            Math.floor((startSec * sampleRate) / divisionFactor),
        );
        const endIdx = Math.min(
            min.length,
            Math.ceil(((startSec + durationSec) * sampleRate) / divisionFactor),
        );

        if (endIdx <= startIdx) {
            return {
                min: new Float32Array(0),
                max: new Float32Array(0),
            };
        }

        // subarray 是零拷贝视图
        return {
            min: min.subarray(startIdx, endIdx),
            max: max.subarray(startIdx, endIdx),
        };
    }

    /**
     * 获取指定文件在指定时间范围内的 peaks 切片，并 resample 到目标像素宽度
     *
     * 返回 interleaved Float32Array [min0, max0, min1, max1, ...]，
     * 与 renderWaveform / applyGainsToPeaks 兼容。
     *
     * @param sourcePath 音频文件路径
     * @param spp samples_per_pixel（用于自动选级）
     * @param startSec 开始时间（秒，源文件坐标系）
     * @param durationSec 持续时间（秒）
     * @param targetWidth 目标像素宽度
     * @returns interleaved Float32Array 或 null（数据未加载时）
     */
    getResampledSlice(
        sourcePath: string,
        spp: number,
        startSec: number,
        durationSec: number,
        targetWidth: number,
        preferredLevel?: WaveformMipmapLevel,
    ): {
        interleaved: Float32Array;
        dataStartSec: number;
        dataDurationSec: number;
    } | null {
        const resolvedLevel = preferredLevel ?? this.selectLevel(spp);
        let peaks = this.getPeaks(sourcePath, resolvedLevel);
        if (!peaks) {
            peaks = this.getNearestLoadedLevel(sourcePath, resolvedLevel);
        }
        if (!peaks) return null;

        const slice = this.getSliceFromPeaks(peaks, startSec, durationSec);
        if (!slice) return null;

        const srcLen = slice.min.length;
        const w = Math.max(1, targetWidth);

        // 计算实际的数据时间范围（用于 renderWaveform 的 dataStartSec/dataDurationSec）
        let dataStartSec = startSec;
        let dataDurationSec = durationSec;
        if (peaks) {
            const { sampleRate, divisionFactor } = peaks;
            const startIdx = Math.max(
                0,
                Math.floor((startSec * sampleRate) / divisionFactor),
            );
            const endIdx = Math.min(
                peaks.min.length,
                Math.ceil(((startSec + durationSec) * sampleRate) / divisionFactor),
            );
            dataStartSec = (startIdx * divisionFactor) / sampleRate;
            dataDurationSec = ((endIdx - startIdx) * divisionFactor) / sampleRate;
        }

        if (srcLen === 0) {
            return {
                interleaved: new Float32Array(0),
                dataStartSec,
                dataDurationSec,
            };
        }

        // 直接映射：源数据已是固定 division factor 的 peaks
        const interleaved = new Float32Array(w * 2);

        if (w >= srcLen) {
            // 上采样：线性插值
            for (let i = 0; i < w; i++) {
                const srcPos = srcLen > 1 ? (i / (w - 1)) * (srcLen - 1) : 0;
                const idx = Math.floor(srcPos);
                const frac = srcPos - idx;

                if (idx >= srcLen - 1) {
                    interleaved[i * 2] = slice.min[srcLen - 1];
                    interleaved[i * 2 + 1] = slice.max[srcLen - 1];
                } else {
                    interleaved[i * 2] =
                        slice.min[idx] * (1 - frac) + slice.min[idx + 1] * frac;
                    interleaved[i * 2 + 1] =
                        slice.max[idx] * (1 - frac) + slice.max[idx + 1] * frac;
                }
            }
        } else {
            // 降采样：每像素取 min/max 聚合
            for (let i = 0; i < w; i++) {
                const srcStart = (i / w) * srcLen;
                const srcEnd = ((i + 1) / w) * srcLen;
                const iStart = Math.max(0, Math.floor(srcStart));
                const iEnd = Math.min(srcLen - 1, Math.ceil(srcEnd));

                let pMin = Infinity;
                let pMax = -Infinity;
                for (let j = iStart; j <= iEnd; j++) {
                    if (slice.min[j] < pMin) pMin = slice.min[j];
                    if (slice.max[j] > pMax) pMax = slice.max[j];
                }

                interleaved[i * 2] = pMin === Infinity ? 0 : pMin;
                interleaved[i * 2 + 1] = pMax === -Infinity ? 0 : pMax;
            }
        }

        return { interleaved, dataStartSec, dataDurationSec };
    }

    getBestSlice(
        sourcePath: string,
        preferredLevel: WaveformMipmapLevel,
        startSec: number,
        durationSec: number,
    ): { min: Float32Array; max: Float32Array } | null {
        let peaks = this.getPeaks(sourcePath, preferredLevel);
        if (!peaks) {
            peaks = this.getNearestLoadedLevel(sourcePath, preferredLevel);
        }
        if (!peaks) return null;
        return this.getSliceFromPeaks(peaks, startSec, durationSec);
    }

    getInterleavedSlice(
        sourcePath: string,
        preferredLevel: WaveformMipmapLevel,
        startSec: number,
        durationSec: number,
    ): {
        interleaved: Float32Array;
        dataStartSec: number;
        dataDurationSec: number;
    } | null {
        let peaks = this.getPeaks(sourcePath, preferredLevel);
        if (!peaks) {
            peaks = this.getNearestLoadedLevel(sourcePath, preferredLevel);
        }
        if (!peaks) return null;

        const slice = this.getSliceFromPeaks(peaks, startSec, durationSec);
        if (!slice) return null;

        const { sampleRate, divisionFactor } = peaks;
        const startIdx = Math.max(
            0,
            Math.floor((startSec * sampleRate) / divisionFactor),
        );
        const endIdx = Math.min(
            peaks.min.length,
            Math.ceil(((startSec + durationSec) * sampleRate) / divisionFactor),
        );
        const dataStartSec = (startIdx * divisionFactor) / sampleRate;
        const dataDurationSec = Math.max(
            0,
            ((endIdx - startIdx) * divisionFactor) / sampleRate,
        );

        const len = slice.min.length;
        const interleaved = this.acquireInterleaved(len * 2);
        for (let i = 0; i < len; i++) {
            interleaved[i * 2] = slice.min[i] ?? 0;
            interleaved[i * 2 + 1] = slice.max[i] ?? 0;
        }

        return {
            interleaved,
            dataStartSec,
            dataDurationSec,
        };
    }

    /**
     * 预加载文件的所有三级 mipmap 数据
     *
     * 音频导入/项目打开时调用。
     */
    async preload(sourcePath: string): Promise<void> {
        // 先通知后端预计算（触发磁盘缓存）
        try {
            await waveformApi.preloadWaveformMipmap(sourcePath);
        } catch {
            // 预加载失败不影响后续按需加载
        }

        // 并行加载所有三级
        const promises: Promise<void>[] = [];
        for (let level = 0; level < LEVEL_COUNT; level++) {
            promises.push(this.loadLevel(sourcePath, level as 0 | 1 | 2));
        }
        await Promise.allSettled(promises);
    }

    /**
     * 批量预加载多个文件的所有三级 mipmap 数据
     *
     * 将 N 个文件 × (1 preload + 3 loadLevel) = 4N 次 IPC 合并为 1 次。
     * 项目打开/重新打开时调用，大幅减少 IPC 往返开销。
     *
     * @param sourcePaths 需要预加载的音频文件路径数组
     */
    async batchPreload(sourcePaths: string[]): Promise<void> {
        if (sourcePaths.length === 0) return;

        // 过滤掉已完全缓存的文件（3 级都已加载）
        const needed = sourcePaths.filter((sp) => {
            const entry = this.cache.get(sp);
            if (!entry) return true;
            return entry.levels.some((l) => l == null);
        });

        if (needed.length === 0) return;

        // 通知所有需要加载的文件进入 loading 状态
        for (const sp of needed) {
            this.notify(sp, "loading");
        }

        try {
            // 单次 IPC 批量获取所有文件的 3 级 mipmap 数据
            const batchResult =
                await waveformApi.batchGetWaveformMipmap(needed);

            // 遍历结果，解码并写入缓存
            for (const [sourcePath, levels] of Object.entries(batchResult)) {
                let hasError = false;
                for (let level = 0; level < LEVEL_COUNT; level++) {
                    const base64 = levels[level];
                    if (!base64) {
                        hasError = true;
                        continue;
                    }
                    const decoded = decodeWaveformFromBase64(base64);
                    if (decoded) {
                        this.applyDecoded(
                            sourcePath,
                            level,
                            decoded,
                        );
                    } else {
                        hasError = true;
                    }
                }
                this.notify(
                    sourcePath,
                    hasError ? "error" : "done",
                    hasError ? "batch decode partial failure" : undefined,
                );
            }

        } catch (err) {
            // 批量 IPC 失败，回退到逐个 preload
            console.warn(
                "[WaveformMipmapStore] batchPreload failed, falling back to individual preload:",
                err,
            );
            const promises = needed.map((sp) => this.preload(sp));
            await Promise.allSettled(promises);
        }
    }

    /**
     * 检查指定文件的指定级别是否已缓存
     */
    hasLevel(sourcePath: string, level: 0 | 1 | 2): boolean {
        const entry = this.cache.get(sourcePath);
        return entry?.levels[level] != null;
    }

    /**
     * 清除指定文件缓存
     */
    invalidate(sourcePath: string): void {
        this.cache.delete(sourcePath);
    }

    /**
     * 清除所有缓存
     */
    clear(): void {
        this.cache.clear();
    }

    /**
     * 添加加载状态监听器
     */
    addListener(cb: LoadCallback): () => void {
        this.listeners.add(cb);
        return () => this.listeners.delete(cb);
    }

    /**
     * 获取当前缓存的文件数量
     */
    get size(): number {
        return this.cache.size;
    }

    // ---------- 内部方法 ----------

    /**
     * 加载指定文件的指定级别（异步，去重）
     *
     * 返回的 Promise 可被多次 await，确保 preload 能等待正在进行的加载。
     */
    private loadLevel(
        sourcePath: string,
        level: 0 | 1 | 2,
    ): Promise<void> {
        // 确保缓存条目存在
        let entry = this.cache.get(sourcePath);
        if (!entry) {
            entry = {
                sampleRate: 0,
                levels: [null, null, null],
                loadingLevels: new Set(),
            };
            this.cache.set(sourcePath, entry);
        }

        // 已加载 → 立即返回
        if (entry.levels[level]) return Promise.resolve();

        // 正在加载 → 返回已有 Promise（等待完成）
        const promiseKey = `${sourcePath}|${level}`;
        const existing = this.loadingPromises.get(promiseKey);
        if (existing) return existing;

        entry.loadingLevels.add(level);
        this.notify(sourcePath, "loading");

        const promise = (async () => {
            try {
                const raw = await waveformApi.getWaveformMipmapBinary(
                    sourcePath,
                    level,
                );
                const decoded = decodeWaveformFromBase64(raw);

                if (decoded) {
                    this.applyDecoded(sourcePath, level, decoded);
                    this.notify(sourcePath, "done");
                } else {
                    this.notify(sourcePath, "error", "decode failed");
                }
            } catch (err) {
                const msg = err instanceof Error ? err.message : String(err);
                this.notify(sourcePath, "error", msg);
            } finally {
                entry!.loadingLevels.delete(level);
                this.loadingPromises.delete(promiseKey);
            }
        })();

        this.loadingPromises.set(promiseKey, promise);
        return promise;
    }

    /**
     * 将解码后的二进制数据写入缓存
     */
    private applyDecoded(
        sourcePath: string,
        level: number,
        decoded: WaveformMipmapBinary,
    ): void {
        let entry = this.cache.get(sourcePath);
        if (!entry) {
            entry = {
                sampleRate: decoded.sampleRate,
                levels: [null, null, null],
                loadingLevels: new Set(),
            };
            this.cache.set(sourcePath, entry);
        }

        entry.sampleRate = decoded.sampleRate;
        const clampedLevel = Math.min(level, 2) as 0 | 1 | 2;
        entry.levels[clampedLevel] = {
            min: decoded.min,
            max: decoded.max,
            divisionFactor: decoded.divisionFactor,
            sampleRate: decoded.sampleRate,
        };

    }

    private getSliceFromPeaks(
        peaks: LevelPeaks,
        startSec: number,
        durationSec: number,
    ): { min: Float32Array; max: Float32Array } | null {
        const { sampleRate, divisionFactor, min, max } = peaks;
        if (sampleRate <= 0 || divisionFactor <= 0) return null;

        const startIdx = Math.max(
            0,
            Math.floor((startSec * sampleRate) / divisionFactor),
        );
        const endIdx = Math.min(
            min.length,
            Math.ceil(((startSec + durationSec) * sampleRate) / divisionFactor),
        );

        if (endIdx <= startIdx) {
            return {
                min: new Float32Array(0),
                max: new Float32Array(0),
            };
        }

        return {
            min: min.subarray(startIdx, endIdx),
            max: max.subarray(startIdx, endIdx),
        };
    }

    private getNearestLoadedLevel(
        sourcePath: string,
        preferredLevel: 0 | 1 | 2,
    ): LevelPeaks | null {
        const entry = this.cache.get(sourcePath);
        if (!entry) return null;

        const offsets = [0, -1, 1, -2, 2] as const;
        for (const offset of offsets) {
            const candidate = preferredLevel + offset;
            if (candidate < 0 || candidate >= LEVEL_COUNT) continue;
            const peaks = entry.levels[candidate as 0 | 1 | 2];
            if (peaks) return peaks;
        }

        return null;
    }

    /**
     * 通知所有监听器
     */
    private notify(
        sourcePath: string,
        status: "loading" | "done" | "error",
        error?: string,
    ): void {
        for (const cb of this.listeners) {
            try {
                cb(sourcePath, status, error);
            } catch {
                // 忽略监听器错误
            }
        }
    }
}

/** 全局单例 */
export const waveformMipmapStore = new WaveformMipmapStoreImpl();

/**
 * 获取三级 mipmap 的除数因子表
 */
export function getDivisionFactors(): readonly [number, number, number] {
    return DIV_FACTORS;
}

/**
 * 获取 spp 阈值表
 */
export function getSppThresholds(): readonly [number, number] {
    return SPP_THRESHOLDS;
}
