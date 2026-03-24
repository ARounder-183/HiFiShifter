/**
 * OffscreenCanvas 缓存管理器
 *
 * 为每个 Clip 维护一个离屏 Canvas，波形数据不变时不重绘。
 * 主 Canvas 通过 drawImage() 贴图，避免每帧全量重绘。
 *
 * 缓存失效条件：
 * - Clip 的 source 数据变化（sourceStartSec, lengthSec 等）
 * - 缩放级别变化（需要切换 mipmap level）
 * - Clip 的增益/淡入淡出参数变化
 *
 * 兼容性：优先使用 OffscreenCanvas，不支持时回退到普通 <canvas> 元素。
 */

/** 缓存条目 */
interface CacheEntry {
    /** 离屏画布（OffscreenCanvas 或普通 Canvas） */
    canvas: OffscreenCanvas | HTMLCanvasElement;
    /** 2D 上下文 */
    ctx: OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D;
    /** 缓存指纹（参数变化时 fingerprint 不同 → 需要重绘） */
    fingerprint: string;
    /** 缓存的像素宽度 */
    width: number;
    /** 缓存的像素高度 */
    height: number;
    /** 最后访问时间（用于 LRU 淘汰） */
    lastAccess: number;
}

/** 获取或创建缓存的返回结果 */
export interface CacheResult {
    /** 缓存条目（包含 canvas 和 ctx） */
    entry: CacheEntry;
    /** 是否需要重绘（fingerprint 不匹配或首次创建） */
    needsRedraw: boolean;
}

/**
 * 创建离屏画布（兼容 OffscreenCanvas 不支持的环境）
 */
function createOffscreenCanvas(
    w: number,
    h: number,
): OffscreenCanvas | HTMLCanvasElement {
    if (typeof OffscreenCanvas !== "undefined") {
        return new OffscreenCanvas(w, h);
    }
    // 回退到普通 Canvas
    const canvas = document.createElement("canvas");
    canvas.width = w;
    canvas.height = h;
    return canvas;
}

class OffscreenCanvasCacheImpl {
    private cache = new Map<string, CacheEntry>();
    /** 最大缓存条目数 */
    private maxEntries: number;

    constructor(maxEntries = 200) {
        this.maxEntries = maxEntries;
    }

    /**
     * 获取或创建 Clip 的离屏 Canvas
     *
     * @param clipId - Clip 的唯一标识
     * @param width - 所需像素宽度
     * @param height - 所需像素高度
     * @param fingerprint - 缓存指纹（参数哈希，变化时触发重绘）
     * @returns 缓存结果，包含 canvas/ctx 和是否需要重绘
     */
    getOrCreate(
        clipId: string,
        width: number,
        height: number,
        fingerprint: string,
    ): CacheResult {
        const existing = this.cache.get(clipId);

        if (existing) {
            existing.lastAccess = performance.now();

            // 指纹匹配且尺寸一致 → 直接复用
            if (
                existing.fingerprint === fingerprint &&
                existing.width === width &&
                existing.height === height
            ) {
                return { entry: existing, needsRedraw: false };
            }

            // 尺寸变化 → 需要调整画布大小
            if (existing.width !== width || existing.height !== height) {
                existing.canvas.width = width;
                existing.canvas.height = height;
                existing.width = width;
                existing.height = height;
            }

            existing.fingerprint = fingerprint;
            return { entry: existing, needsRedraw: true };
        }

        // 首次创建
        if (this.cache.size >= this.maxEntries) {
            this.evict();
        }

        const canvas = createOffscreenCanvas(width, height);
        const ctx = canvas.getContext("2d") as
            | OffscreenCanvasRenderingContext2D
            | CanvasRenderingContext2D;

        if (!ctx) {
            // 极端情况：ctx 创建失败，创建一个备用 Canvas
            const fallback = document.createElement("canvas");
            fallback.width = width;
            fallback.height = height;
            const fallbackCtx = fallback.getContext("2d")!;
            const entry: CacheEntry = {
                canvas: fallback,
                ctx: fallbackCtx,
                fingerprint,
                width,
                height,
                lastAccess: performance.now(),
            };
            this.cache.set(clipId, entry);
            return { entry, needsRedraw: true };
        }

        const entry: CacheEntry = {
            canvas,
            ctx,
            fingerprint,
            width,
            height,
            lastAccess: performance.now(),
        };
        this.cache.set(clipId, entry);
        return { entry, needsRedraw: true };
    }

    /**
     * 删除指定 Clip 的缓存
     */
    remove(clipId: string): void {
        this.cache.delete(clipId);
    }

    /**
     * LRU 淘汰：移除最久未访问的条目
     */
    private evict(): void {
        if (this.cache.size === 0) return;

        let oldestKey: string | null = null;
        let oldestTime = Infinity;

        for (const [key, entry] of this.cache) {
            if (entry.lastAccess < oldestTime) {
                oldestTime = entry.lastAccess;
                oldestKey = key;
            }
        }

        if (oldestKey) {
            this.cache.delete(oldestKey);
        }
    }

    /**
     * 清除所有缓存
     */
    clear(): void {
        this.cache.clear();
    }

    /**
     * 获取当前缓存条目数
     */
    get size(): number {
        return this.cache.size;
    }
}

/** 全局单例 */
export const offscreenCanvasCache = new OffscreenCanvasCacheImpl();
