/*
 * 图片 src 的解析与缓存。
 *
 * 正文里的 `src` 有四种形态，渲染前都要变成 WebView 能加载的 URL：
 *
 * | 正文里的 src | 解析方式 |
 * |---|---|
 * | `hifi-asset://<id>` | 后端 `readAsset` 取字节 → blob URL（CSP 允许 `blob:`） |
 * | `data:image/...` | 直接用（CSP 允许 `data:`） |
 * | `https://...` | 直接用（可被设置关掉，离线/隐私场景） |
 * | 相对 / 绝对路径 | 相对工程目录解析为绝对路径 → 后端读文件 → blob URL |
 *
 * 第四种让用户手写的 `![](./素材/截图.png)` 也能显示 —— 记事本首先是
 * Markdown，不能只认自己生成的那一种引用。
 *
 * 缓存按 assetId / 绝对路径做键，LRU 上限内的 blob URL 在淘汰时显式 revoke，
 * 否则滚动长文档会持续泄漏内存。
 */

import { notebookApi } from "../../../services/api/notebook";
import { assetIdFromSrc, isAssetRef } from "./assetRef";
import { baseName, decodePathFromMarkdown, dirName, isAbsolutePath, resolvePath } from "./notebookPaths";
import { base64ToBytes, toBlobPart } from "./notebookImagePipeline";

export interface ResolvedImage {
    /** 可放进 `<img src>` 的 URL；解析失败为 null。 */
    url: string | null;
    /** 资源确实不存在（区别于"还在加载"）。 */
    missing: boolean;
    /** 失败原因，用于提示文案。 */
    reason?: "not-found" | "read-failed" | "remote-blocked" | "unsupported";
}

const CACHE_LIMIT = 96;

interface CacheEntry {
    url: string;
    /** 命中顺序（LRU）。 */
    usedAt: number;
}

const cache = new Map<string, CacheEntry>();
const pending = new Map<string, Promise<ResolvedImage>>();
const listeners = new Set<() => void>();
let clock = 0;

function notify(): void {
    for (const listener of listeners) listener();
}

/** 订阅缓存变化（NodeView 用它触发重渲染）。 */
export function subscribeAssetCache(listener: () => void): () => void {
    listeners.add(listener);
    return () => listeners.delete(listener);
}

function cacheKey(src: string, projectDir: string | null): string {
    if (isAssetRef(src)) return `asset:${assetIdFromSrc(src)}`;
    if (src.startsWith("data:") || /^https?:/i.test(src)) return `direct:${src}`;
    return `file:${resolveForFile(src, projectDir)}`;
}

function resolveForFile(src: string, projectDir: string | null): string {
    const decoded = decodePathFromMarkdown(src);
    if (isAbsolutePath(decoded)) return decoded;
    if (!projectDir) return decoded;
    return resolvePath(projectDir, decoded);
}

function put(key: string, url: string): void {
    clock += 1;
    cache.set(key, { url, usedAt: clock });
    if (cache.size > CACHE_LIMIT) {
        // 淘汰最久未使用的条目，并 revoke 它的 blob URL。
        let oldestKey: string | null = null;
        let oldest = Number.POSITIVE_INFINITY;
        for (const [k, entry] of cache) {
            if (entry.usedAt < oldest) {
                oldest = entry.usedAt;
                oldestKey = k;
            }
        }
        if (oldestKey !== null && oldestKey !== key) {
            const evicted = cache.get(oldestKey);
            cache.delete(oldestKey);
            if (evicted?.url.startsWith("blob:")) URL.revokeObjectURL(evicted.url);
        }
    }
}

/** 同步取已缓存的 URL（未命中返回 null，用于避免闪烁）。 */
export function peekImageUrl(src: string, projectDir: string | null): string | null {
    const entry = cache.get(cacheKey(src, projectDir));
    if (!entry) return null;
    clock += 1;
    entry.usedAt = clock;
    return entry.url;
}

/** 清空全部缓存（切换工程时调用，避免跨工程的 id 冲突）。 */
export function clearImageCache(): void {
    for (const entry of cache.values()) {
        if (entry.url.startsWith("blob:")) URL.revokeObjectURL(entry.url);
    }
    cache.clear();
    pending.clear();
    notify();
}

/** 失效单条（图片被替换/删除后调用）。 */
export function invalidateImage(src: string, projectDir: string | null): void {
    const key = cacheKey(src, projectDir);
    const entry = cache.get(key);
    if (entry?.url.startsWith("blob:")) URL.revokeObjectURL(entry.url);
    cache.delete(key);
    notify();
}

function blobUrlFromBase64(base64: string, mime: string): string {
    return URL.createObjectURL(new Blob([toBlobPart(base64ToBytes(base64))], { type: mime }));
}

export interface ResolveImageOptions {
    projectDir: string | null;
    allowRemoteImages: boolean;
}

/**
 * 解析图片 src（异步，带并发去重与缓存）。
 *
 * 同一个 src 的并发请求共用一次后端读取 —— 长文档里同一张图出现多次是常态。
 */
export function resolveImage(src: string, options: ResolveImageOptions): Promise<ResolvedImage> {
    const key = cacheKey(src, options.projectDir);
    const cached = peekImageUrl(src, options.projectDir);
    if (cached) return Promise.resolve({ url: cached, missing: false });

    const inFlight = pending.get(key);
    if (inFlight) return inFlight;

    const task = (async (): Promise<ResolvedImage> => {
        try {
            if (src.startsWith("data:")) {
                put(key, src);
                return { url: src, missing: false };
            }
            if (/^https?:/i.test(src)) {
                if (!options.allowRemoteImages) {
                    return { url: null, missing: false, reason: "remote-blocked" };
                }
                put(key, src);
                return { url: src, missing: false };
            }
            if (isAssetRef(src)) {
                const assetId = assetIdFromSrc(src);
                if (!assetId) return { url: null, missing: true, reason: "unsupported" };
                const result = await notebookApi.readAsset(assetId);
                if (!result.ok || !result.base64) {
                    return {
                        url: null,
                        missing: true,
                        reason: result.missing ? "not-found" : "read-failed",
                    };
                }
                const url = blobUrlFromBase64(result.base64, result.mime ?? "image/png");
                put(key, url);
                return { url, missing: false };
            }
            // 相对 / 绝对路径：读原文件。
            const absolute = resolveForFile(src, options.projectDir);
            if (!absolute || !isAbsolutePath(absolute)) {
                return { url: null, missing: true, reason: "not-found" };
            }
            const file = await notebookApi.readFileBase64(absolute);
            if (!file.ok || !file.base64) {
                return { url: null, missing: true, reason: "not-found" };
            }
            const url = blobUrlFromBase64(file.base64, file.mime ?? "image/png");
            put(key, url);
            return { url, missing: false };
        } catch {
            return { url: null, missing: true, reason: "read-failed" };
        } finally {
            pending.delete(key);
            notify();
        }
    })();

    pending.set(key, task);
    return task;
}

/** 图片建议的文件名（"另存为"的默认值）。 */
export function suggestImageFileName(src: string, meta?: unknown): string {
    const fromMeta =
        meta && typeof meta === "object" && typeof (meta as { originalName?: unknown }).originalName === "string"
            ? (meta as { originalName: string }).originalName
            : null;
    if (fromMeta) return fromMeta;
    const id = assetIdFromSrc(src);
    if (id) return id;
    return baseName(decodePathFromMarkdown(src)) || "image.png";
}

/** 当前 src 所在目录（用于"插入相对路径"时计算基准）。 */
export function imageSourceDir(src: string): string | null {
    if (isAssetRef(src) || src.startsWith("data:") || /^https?:/i.test(src)) return null;
    const decoded = decodePathFromMarkdown(src);
    return dirName(decoded) || null;
}
