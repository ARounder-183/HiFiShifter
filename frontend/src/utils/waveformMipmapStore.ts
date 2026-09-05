/**
 * 波形 Mipmap 缓存管理器（整文件级）
 *
 * 每个音频文件缓存三级 Float32Array 数据：
 * - L0 (div=16):   精细级，近距离对轨，spp ≤ 512
 * - L1 (div=512):  中间级，日常编辑，512 < spp ≤ 1024
 * - L2 (div=4096): 全局级，预览/导航，spp > 1024
 *
 * 状态管理策略：
 * - 波形二进制数据存在外部 Map（不放 Redux，避免序列化开销）
 * - 文件加载状态可通过回调通知 UI
 *
 * 内存安全（2026-06-30 加固）：
 * - 文件级 cache 限定最多 MAX_FILE_CACHE_SIZE 个条目（一首歌的 3 级 mipmap
 *   通常占数 MB），通过 LRU 淘汰最久未访问的条目，防止长时间使用后
 *   内存无界增长导致的前端卡顿/泄漏。
 * - 缓存读写统一走 cacheGet/cacheSet/touchLru 三个 helper, 集中维护 LRU
 *   顺序与容量。Map 自身的"插入顺序 = LRU 顺序"是该实现的基础。
 */

import { waveformApi } from "../services/api/waveform";
import { decodeWaveformFromBase64, type WaveformMipmapBinary } from "./waveformBinaryCodec";
import {
    wfDiag_poolAcquire,
    wfDiag_poolRelease,
    wfDiag_poolRegister,
    wfDiag_setMipmapSizeFn,
} from "./waveformDebug";

// ============== 常量 ==============

/** 三级 mipmap 的除数因子 */
const DIV_FACTORS = [16, 512, 4096] as const;

/**
 * "尚未就绪"重试冷却（毫秒）。
 *
 * 后端在波形分析/首次计算完成前会返回**空串**（而非显式错误）——这对
 * 打开工程瞬间的抢先请求是瞬时条件，不是永久失败。冷却期内不重复请求，
 * 冷却后由下一次 preload/draw 或 `refresh()`（waveform_analysis_progress
 * done/cached 事件）自然重试；同时限制对真正缺失文件的请求频率。
 */
const RETRY_NOT_READY_COOLDOWN_MS = 3000;

/** 级别选择的 spp 阈值 */
const SPP_THRESHOLDS = [512, 1024] as const;
const SPP_HYSTERESIS_ENTER_SCALE = 1.25;
const SPP_HYSTERESIS_EXIT_SCALE = 0.75;

/** mipmap 级别数量 */
const LEVEL_COUNT = 3;

/**
 * 文件级 mipmap 缓存的最大 backing-store 字节数。
 *
 * 每个 entry 包含三级 Float32Array，单首 5 分钟立体声歌曲约占数 MB。
 * 该上限在"避免内存累积"与"频繁切换音频不需要重新解码"之间取折中。
 */
const MAX_CACHE_BYTES = 192 * 1024 * 1024;

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
    /** 已确认加载失败、在 invalidate() 前不再自动重试的级别 */
    failedLevels: Set<number>;
    /** 当前三级 Float32Array backing store 的总字节数 */
    bytes: number;
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
    private cacheBytes = 0;

    /** 加载状态监听器 */
    private listeners = new Set<LoadCallback>();

    /** 正在进行的加载 Promise（用于 preload 等待已发起的加载） */
    private loadingPromises = new Map<string, Promise<void>>();

    /**
     * "尚未就绪"重试冷却表：key = `${sourcePath}|${level}`，value = 冷却
     * 截止时间戳（Date.now() 语义）。后端返回空串（暂未就绪）时写入；
     * 冷却期内 loadLevel/batchPreload 不再发起请求。
     */
    private retryCooldownUntil = new Map<string, number>();

    /**
     * interleaved 缓冲区复用池。
     * 每次 getInterleavedSlice 会优先从池中取出同等大小的 Float32Array 进行复用，
     * 避免快速缩放时每帧 new Float32Array 产生的 GC 压力。
     */
    private interleavedPool: Float32Array[] = [];
    /** 池的最大容量（条目数） */
    private static readonly POOL_MAX = 32;

    private acquireInterleaved(minLen: number): Float32Array {
        for (let i = 0; i < this.interleavedPool.length; i++) {
            if (this.interleavedPool[i].buffer.byteLength / 4 >= minLen) {
                const buf = this.interleavedPool[i];
                this.interleavedPool.splice(i, 1);
                wfDiag_poolAcquire("interleaved", true);
                return new Float32Array(buf.buffer, 0, minLen);
            }
        }
        wfDiag_poolAcquire("interleaved", false);
        return new Float32Array(minLen);
    }

    releaseInterleaved(buf: Float32Array): void {
        const accepted =
            buf.length > 0 && this.interleavedPool.length < WaveformMipmapStoreImpl.POOL_MAX;
        if (accepted) {
            this.interleavedPool.push(new Float32Array(buf.buffer));
        }
        wfDiag_poolRelease("interleaved", accepted);
    }

    // ---------- LRU 缓存 helper ----------
    //
    // 设计说明：
    // - 利用 JS Map 自身的"插入顺序 = 迭代顺序"特性来记录 LRU 顺序，
    //   最旧的条目即为 keys().next().value。
    // - cacheGet: 读取并将命中的 key 移到末尾（视为最近访问）。
    // - cacheSet: 写入并保证不超过 MAX_FILE_CACHE_SIZE, 超出则淘汰最旧的。
    // - touchLru: 仅刷新顺序而不修改 entry, 适用于命中后无需重写值的场景。

    /**
     * 读取缓存条目；若命中则把该 key 提升到 LRU 末尾。
     */
    private cacheGet(sourcePath: string): FileMipmapCache | undefined {
        const entry = this.cache.get(sourcePath);
        if (entry !== undefined) {
            // 移到末尾以更新 LRU 顺序
            this.cache.delete(sourcePath);
            this.cache.set(sourcePath, entry);
        }
        return entry;
    }

    /**
     * 写入或覆盖缓存条目，并按 LRU 上限淘汰最旧条目。
     */
    private cacheSet(sourcePath: string, entry: FileMipmapCache): void {
        const previous = this.cache.get(sourcePath);
        if (previous) {
            // 删除旧位置，确保重新插入到末尾
            this.cache.delete(sourcePath);
            this.cacheBytes -= previous.bytes;
        }
        this.cache.set(sourcePath, entry);
        this.cacheBytes += entry.bytes;
        this.evictIfNeeded();
    }

    /**
     * 仅把命中的 key 提升为最近访问，不修改 entry 本身。
     */
    private touchLru(sourcePath: string): void {
        const entry = this.cache.get(sourcePath);
        if (entry === undefined) return;
        this.cache.delete(sourcePath);
        this.cache.set(sourcePath, entry);
    }

    /**
     * 当数据超过字节预算时，按 LRU 顺序淘汰最旧的**有字节贡献**的条目。
     * 被淘汰条目同步 notify "done" 状态以便 UI 释放任何关联视图缓存
     * （几何缓存持有指向被淘汰 buffer 的 subarray 视图，不通知的话视图
     * 会把 buffer 钉在内存里，cacheBytes 与真实占用背离）。
     *
     * 0 字节条目（缺失/损坏文件的负缓存标记）对预算无贡献：淘汰它们
     * 不会释放任何内存，只会丢掉失败状态，让已损坏文件重新打后端 ——
     * 因此跳过，让负缓存标记与条目同生共死。
     */
    private evictIfNeeded(): void {
        while (this.cacheBytes > MAX_CACHE_BYTES && this.cache.size > 1) {
            let oldestKey: string | undefined;
            for (const [key, entry] of this.cache) {
                if (entry.bytes > 0) {
                    oldestKey = key;
                    break;
                }
            }
            if (!oldestKey) break;
            const oldest = this.cache.get(oldestKey);
            this.cache.delete(oldestKey);
            this.cacheBytes -= oldest?.bytes ?? 0;
            this.notify(oldestKey, "done");
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
    getPeaks(sourcePath: string, level: 0 | 1 | 2): LevelPeaks | null {
        const entry = this.cacheGet(sourcePath);
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
        const startIdx = Math.max(0, Math.floor((startSec * sampleRate) / divisionFactor));
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
            const startIdx = Math.max(0, Math.floor((startSec * sampleRate) / divisionFactor));
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

        // 从复用池获取 Buffer
        const interleaved = this.acquireInterleaved(w * 2);

        if (w >= srcLen) {
            // 上采样：线性插值
            // 提取除法常数，消除循环内反复计算的除法与乘法开销
            const invWM1 = w > 1 ? 1 / (w - 1) : 0;
            const scale = (srcLen - 1) * invWM1;

            for (let i = 0; i < w; i++) {
                const srcPos = srcLen > 1 ? i * scale : 0;
                const idx = Math.floor(srcPos);
                const frac = srcPos - idx;

                if (idx >= srcLen - 1) {
                    interleaved[i * 2] = slice.min[srcLen - 1];
                    interleaved[i * 2 + 1] = slice.max[srcLen - 1];
                } else {
                    interleaved[i * 2] = slice.min[idx] * (1 - frac) + slice.min[idx + 1] * frac;
                    interleaved[i * 2 + 1] =
                        slice.max[idx] * (1 - frac) + slice.max[idx + 1] * frac;
                }
            }
        } else {
            // 每像素取 min/max 聚合
            // 提取线性步长常量
            const srcStep = srcLen / w;

            for (let i = 0; i < w; i++) {
                // 使用乘法和加法替代原本的 4 次浮点乘除运算
                const srcStart = i * srcStep;
                const srcEnd = srcStart + srcStep;

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
        const startIdx = Math.max(0, Math.floor((startSec * sampleRate) / divisionFactor));
        const endIdx = Math.min(
            peaks.min.length,
            Math.ceil(((startSec + durationSec) * sampleRate) / divisionFactor),
        );
        const dataStartSec = (startIdx * divisionFactor) / sampleRate;
        const dataDurationSec = Math.max(0, ((endIdx - startIdx) * divisionFactor) / sampleRate);

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
     * 获取零拷贝 min/max 视图及其实际时间边界。
     *
     * 共享 WebGL/Canvas surface 直接消费这两个 subarray，避免每帧构造
     * interleaved、gain 和 downsample 中间数组。
     */
    getBestSliceView(
        sourcePath: string,
        preferredLevel: WaveformMipmapLevel,
        startSec: number,
        durationSec: number,
    ): {
        min: Float32Array;
        max: Float32Array;
        dataStartSec: number;
        dataDurationSec: number;
    } | null {
        let peaks = this.getPeaks(sourcePath, preferredLevel);
        if (!peaks) peaks = this.getNearestLoadedLevel(sourcePath, preferredLevel);
        if (!peaks) return null;

        const { sampleRate, divisionFactor } = peaks;
        const startIdx = Math.max(0, Math.floor((startSec * sampleRate) / divisionFactor));
        const endIdx = Math.min(
            peaks.min.length,
            Math.ceil(((startSec + durationSec) * sampleRate) / divisionFactor),
        );
        if (endIdx <= startIdx) return null;

        return {
            min: peaks.min.subarray(startIdx, endIdx),
            max: peaks.max.subarray(startIdx, endIdx),
            dataStartSec: (startIdx * divisionFactor) / sampleRate,
            dataDurationSec: ((endIdx - startIdx) * divisionFactor) / sampleRate,
        };
    }

    /**
     * 预加载文件的所有三级 mipmap 数据
     *
     * 音频导入/项目打开时调用。
     */
    async preload(sourcePath: string): Promise<void> {
        // 所有级别都已知失败时不再触发后端计算（例如源文件已缺失）。
        const existing = this.cache.get(sourcePath);
        if (existing && existing.failedLevels.size >= LEVEL_COUNT) return;

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
     * 批量预加载多个文件的 mipmap 数据（仅 L2 轻量级，L0/L1 按需加载）
     *
     * L2 数据量约为 L0 的 1/250，500 文件仅需 ~13MB。
     * 用户缩放时 getPeaks 按需触发 L0/L1 的 loadLevel。
     * 加载期间通过 getNearestLoadedLevel 回落已有数据，避免闪烁。
     *
     * @param sourcePaths 需要预加载的音频文件路径数组
     */
    async batchPreload(sourcePaths: string[]): Promise<void> {
        if (sourcePaths.length === 0) return;

        const needed = sourcePaths.filter((sp) => {
            const entry = this.cache.get(sp);
            if (!entry) return true;
            // L2 已确认失败（解码损坏/文件缺失）时不再自动重试。
            if (entry.failedLevels.has(2)) return false;
            // 至少 L2 未加载则需要
            if (entry.levels[2] != null) return false;
            // 已有单发 loadLevel(2) 在途 → 交给它，避免同一文件双请求。
            if (this.loadingPromises.has(`${sp}|2`)) return false;
            // 冷却期（上次拿到空结果 = 后端暂未就绪）内不重复请求。
            const until = this.retryCooldownUntil.get(`${sp}|2`);
            return until == null || Date.now() >= until;
        });

        if (needed.length === 0) return;

        // 通知进入 loading 状态（仅标记 L2），并把本批次注册进共享去重表：
        // 否则批预载在途时并发的 loadLevel(2) 找不到已有 Promise，会对
        // 同一批文件再发一轮相同请求。
        const registered: string[] = [];
        for (const sp of needed) {
            if (this.loadingPromises.has(`${sp}|2`)) continue;
            let entry = this.cache.get(sp);
            if (!entry) {
                entry = {
                    sampleRate: 0,
                    levels: [null, null, null],
                    loadingLevels: new Set(),
                    failedLevels: new Set(),
                    bytes: 0,
                };
                this.cacheSet(sp, entry);
            } else {
                this.touchLru(sp);
            }
            entry.loadingLevels.add(2);
            this.notify(sp, "loading");
            registered.push(sp);
        }

        if (registered.length === 0) return;

        const batchPromise = (async () => {
            try {
                const batchResult = await waveformApi.batchGetWaveformMipmap(registered);

                for (const [sourcePath, levels] of Object.entries(batchResult)) {
                    // 仅解码 L2（索引 2），L0/L1 丢弃
                    const l2Base64 = levels[2];
                    if (l2Base64) {
                        const decoded = decodeWaveformFromBase64(l2Base64);
                        if (decoded) {
                            this.applyDecoded(sourcePath, 2, decoded);
                            this.notify(sourcePath, "done");
                        } else {
                            const entry = this.cache.get(sourcePath);
                            if (entry) entry.failedLevels.add(2);
                            this.notify(sourcePath, "error", "batch decode L2 failure");
                        }
                    } else {
                        // 空串 = 后端尚未就绪（波形分析/首次计算未完成），是
                        // **瞬时**条件：绝不能写 failedLevels——否则打开工程
                        // 瞬间的抢先批量请求会把 L2 永久毒化，波形要等用户
                        // 滚动/缩放才出现。进入重试冷却，等待 refresh()
                        // （waveform_analysis_progress done/cached 事件）。
                        this.retryCooldownUntil.set(
                            `${sourcePath}|2`,
                            Date.now() + RETRY_NOT_READY_COOLDOWN_MS,
                        );
                        this.notify(sourcePath, "loading");
                    }
                }
            } catch (err) {
                console.warn(
                    "[WaveformMipmapStore] batchPreload failed, falling back to individual preload:",
                    err,
                );
                // 回退前先注销本批次的共享 Promise：loadLevel 以它做去重，
                // 保留键会让回退路径的 loadLevel(2) 等待本批次自身 → 死锁。
                for (const sp of registered) {
                    this.loadingPromises.delete(`${sp}|2`);
                }
                const promises = registered.map((sp) => this.preload(sp));
                await Promise.allSettled(promises);
            } finally {
                for (const sp of registered) {
                    const entry = this.cache.get(sp);
                    if (entry) {
                        entry.loadingLevels.delete(2);
                    }
                    this.loadingPromises.delete(`${sp}|2`);
                }
            }
        })();

        for (const sp of registered) {
            this.loadingPromises.set(`${sp}|2`, batchPromise);
        }

        await batchPromise;
    }

    /**
     * 检查指定文件的指定级别是否已缓存
     */
    hasLevel(sourcePath: string, level: 0 | 1 | 2): boolean {
        const entry = this.cache.get(sourcePath);
        // 注意：此处仅做存在性检查, 不刷新 LRU 顺序; 真正消费 peaks 的 getPeaks /
        // getInterleavedSlice 等路径会通过 cacheGet 刷新顺序。
        return entry?.levels[level] != null;
    }

    /**
     * 清除指定文件缓存
     */
    invalidate(sourcePath: string): void {
        this.cacheBytes -= this.cache.get(sourcePath)?.bytes ?? 0;
        this.cache.delete(sourcePath);
        // 换源/重载时让重试冷却一并失效，新来源立即可以重新请求。
        this.retryCooldownUntil.delete(`${sourcePath}|0`);
        this.retryCooldownUntil.delete(`${sourcePath}|1`);
        this.retryCooldownUntil.delete(`${sourcePath}|2`);
    }

    /**
     * 将文件标记为当前不可用（缺失/无法读取）。
     *
     * 与加载失败的自然负缓存一致：所有级别在 invalidate() 前都不会再发起
     * 自动加载，避免缺失文件在渲染循环中被反复请求并持续触发后端进度事件。
     * 若后续通过重新指定文件等方式恢复，replaceClipSourceRemote 会调用
     * invalidate() 清除该标记。
     */
    markUnavailable(sourcePath: string): void {
        const normalized = sourcePath;
        if (!normalized) return;

        let entry = this.cache.get(normalized);
        if (!entry) {
            entry = {
                sampleRate: 0,
                levels: [null, null, null],
                loadingLevels: new Set(),
                failedLevels: new Set(),
                bytes: 0,
            };
            this.cacheSet(normalized, entry);
        } else {
            this.touchLru(normalized);
        }

        const wasAlreadyMarked = entry.failedLevels.size >= LEVEL_COUNT;
        for (let level = 0; level < LEVEL_COUNT; level++) {
            entry.failedLevels.add(level);
        }
        if (!wasAlreadyMarked) {
            this.notify(normalized, "error", "source unavailable");
        }
    }

    /**
     * 数据就绪信号（后端 `waveform_analysis_progress` 的 done/cached 事件，
     * 或调用方主动重试）：清除该文件的重试冷却与失败负缓存，并按需重载。
     *
     * 打开工程瞬间的首次预加载/绘制常早于后端分析完成，拿到的是空结果；
     * 若没有本入口，曾被误判失败的级别只能等 `invalidate()`（换源）才恢复，
     * 波形要等用户滚动/缩放才出现。数据就绪后由 listener（"done"）驱动
     * 各波形面重绘。
     */
    refresh(sourcePath: string): void {
        if (!sourcePath) return;
        this.retryCooldownUntil.delete(`${sourcePath}|0`);
        this.retryCooldownUntil.delete(`${sourcePath}|1`);
        this.retryCooldownUntil.delete(`${sourcePath}|2`);
        const entry = this.cacheGet(sourcePath);
        if (!entry) return;
        const poisoned = entry.failedLevels.size > 0;
        const nothingLoaded =
            entry.levels[0] == null && entry.levels[1] == null && entry.levels[2] == null;
        if (poisoned || nothingLoaded) {
            // 清掉陈旧条目后整体重载（含三级），后端此时应已就绪。
            this.invalidate(sourcePath);
            void this.batchPreload([sourcePath]);
        }
    }

    /**
     * 清除所有缓存
     */
    clear(): void {
        this.cache.clear();
        this.cacheBytes = 0;
        this.interleavedPool.length = 0;
        // 在途请求与重试冷却一并清空：否则 clear() 之后完成的在途加载会
        // 通过 applyDecoded 让"已清空"的缓存悄悄复活，冷却键也会残留。
        this.loadingPromises.clear();
        this.retryCooldownUntil.clear();
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
    private loadLevel(sourcePath: string, level: 0 | 1 | 2): Promise<void> {
        // 确保缓存条目存在
        let entry = this.cache.get(sourcePath);
        if (!entry) {
            entry = {
                sampleRate: 0,
                levels: [null, null, null],
                loadingLevels: new Set(),
                failedLevels: new Set(),
                bytes: 0,
            };
            this.cacheSet(sourcePath, entry);
        } else {
            this.touchLru(sourcePath);
        }

        // 已加载 → 立即返回
        if (entry.levels[level]) return Promise.resolve();

        // 已确认失败 → 不再自动重试，避免缺失文件在每次渲染时反复触发后端计算。
        if (entry.failedLevels.has(level)) return Promise.resolve();

        // 冷却期（上次拿到空结果 = 后端暂未就绪）内不重复请求，等待
        // refresh()（波形分析完成事件）或冷却结束后的下一次绘制重试。
        const retryKey = `${sourcePath}|${level}`;
        const cooldownUntil = this.retryCooldownUntil.get(retryKey);
        if (cooldownUntil != null && Date.now() < cooldownUntil) return Promise.resolve();

        // 正在加载 → 返回已有 Promise（等待完成）
        const promiseKey = `${sourcePath}|${level}`;
        const existing = this.loadingPromises.get(promiseKey);
        if (existing) return existing;

        entry.loadingLevels.add(level);
        this.notify(sourcePath, "loading");

        const promise = (async () => {
            try {
                const raw = await waveformApi.getWaveformMipmapBinary(sourcePath, level);
                const decoded = decodeWaveformFromBase64(raw);

                if (decoded) {
                    this.applyDecoded(sourcePath, level, decoded);
                    this.notify(sourcePath, "done");
                } else if (raw === "") {
                    // 空串 = 后端尚未就绪（波形分析/首次计算未完成），
                    // 是**瞬时**条件：绝不写 failedLevels（否则打开工程
                    // 瞬间的抢先请求会把该级别永久毒化，波形要等用户
                    // 滚动/缩放才出现）。进入重试冷却，等待 refresh()。
                    this.retryCooldownUntil.set(retryKey, Date.now() + RETRY_NOT_READY_COOLDOWN_MS);
                    this.notify(sourcePath, "loading");
                } else {
                    // 非空但解码失败 = 数据损坏，视为永久失败。
                    const current = this.cache.get(sourcePath);
                    if (current) current.failedLevels.add(level);
                    this.notify(sourcePath, "error", "decode failed");
                }
            } catch (err) {
                const msg = err instanceof Error ? err.message : String(err);
                const current = this.cache.get(sourcePath);
                if (current) current.failedLevels.add(level);
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
    private applyDecoded(sourcePath: string, level: number, decoded: WaveformMipmapBinary): void {
        let entry = this.cache.get(sourcePath);
        if (!entry) {
            entry = {
                sampleRate: decoded.sampleRate,
                levels: [null, null, null],
                loadingLevels: new Set(),
                failedLevels: new Set(),
                bytes: 0,
            };
            this.cacheSet(sourcePath, entry);
        } else {
            this.touchLru(sourcePath);
        }

        entry.sampleRate = decoded.sampleRate;
        const clampedLevel = Math.min(level, 2) as 0 | 1 | 2;
        entry.failedLevels.delete(clampedLevel);
        this.retryCooldownUntil.delete(`${sourcePath}|${clampedLevel}`);
        const previous = entry.levels[clampedLevel];
        const previousBytes = previous ? previous.min.byteLength + previous.max.byteLength : 0;
        const nextBytes = decoded.min.byteLength + decoded.max.byteLength;
        entry.levels[clampedLevel] = {
            min: decoded.min,
            max: decoded.max,
            divisionFactor: decoded.divisionFactor,
            sampleRate: decoded.sampleRate,
        };
        entry.bytes += nextBytes - previousBytes;
        this.cacheBytes += nextBytes - previousBytes;
        this.evictIfNeeded();
    }

    private getSliceFromPeaks(
        peaks: LevelPeaks,
        startSec: number,
        durationSec: number,
    ): { min: Float32Array; max: Float32Array } | null {
        const { sampleRate, divisionFactor, min, max } = peaks;
        if (sampleRate <= 0 || divisionFactor <= 0) return null;

        const startIdx = Math.max(0, Math.floor((startSec * sampleRate) / divisionFactor));
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
        const entry = this.cacheGet(sourcePath);
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
    private notify(sourcePath: string, status: "loading" | "done" | "error", error?: string): void {
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

// ── 诊断：注册 mipmap 文件缓存大小 ──
wfDiag_setMipmapSizeFn(() => waveformMipmapStore.size);

// ── 诊断：注册 interleaved 池 ──
wfDiag_poolRegister("interleaved", () => waveformMipmapStore["interleavedPool"].length);

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
