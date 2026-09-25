/*
 * 图片 src 的解析与缓存。
 *
 * 正文里的 `src` 有四种形态，渲染前都要变成 WebView 能加载的 URL：
 *
 * | 正文里的 src | 解析方式 |
 * |---|---|
 * | `hifi-asset://<id>` | 后端 `readAsset` 取字节 → blob URL（CSP 允许 `blob:`） |
 * | `data:image/...` | 直接用（CSP 允许 `data:`） |
 * | `https://...` | 直接用（CSP 只放行 `https:`，`http:` 一律被拦；可被设置关掉，离线/隐私场景） |
 * | 相对 / 绝对路径 | 相对工程目录解析为绝对路径 → 后端读文件 → blob URL |
 *
 * 第四种让用户手写的 `![](./素材/截图.png)` 也能显示 —— 记事本首先是
 * Markdown，不能只认自己生成的那一种引用。
 *
 * ## 通知模型（这里曾经导致整窗崩溃，改动前请读完）
 *
 * 缓存只在**被作废**时通知订阅者（整体清空 / 单条失效），**成功写入新条目时
 * 不通知**。原因：
 *
 * - 需要新条目的调用方本来就从 `resolveImage` 的 promise 里拿到 URL，不需要
 *   额外通知；
 * - 而"解析失败"的 src 不会进缓存，若在解析结束时通知订阅者，订阅者会立刻
 *   再解析一次 → 再失败 → 再通知 …… 形成**无界递归**。对同步返回失败的分支
 *   （远程图被禁用、未落盘工程里的相对路径）它是同步递归，直接从 `useEffect`
 *   里抛 `RangeError` 打掉整个 React 树；对要走 IPC 的分支（附件缺失）它变成
 *   无界的 IPC + setState 风暴。
 *
 * 一句话：**通知是"缓存作废"信号，不是"解析完成"信号**。
 */

import { notebookApi } from "../../../services/api/notebook";
import { assetIdFromSrc, isAssetRef } from "./assetRef";
import {
    baseName,
    decodePathFromMarkdown,
    dirName,
    isAbsolutePath,
    resolvePath,
} from "./notebookPaths";
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
/** 被淘汰但因可能仍在画面里而暂缓释放的 URL 上限。 */
const DEFERRED_REVOKE_LIMIT = 512;

interface CacheEntry {
    url: string;
    /** 命中顺序（LRU）。 */
    usedAt: number;
}

const cache = new Map<string, CacheEntry>();
const pending = new Map<string, Promise<ResolvedImage>>();
const invalidators = new Set<() => void>();
/**
 * 被 LRU 淘汰的 blob URL。
 *
 * 淘汰时**不立即 revoke**：那张图可能正显示在某个 NodeView 里，revoke 会让它
 * 变成裂图。改为挂在这里，等整体清空（切工程）时统一释放；只有超过上限的旧
 * 条目才真正 revoke（那时它早已不在画面上）。
 */
const deferredRevoke: string[] = [];
let clock = 0;

function notifyInvalidated(): void {
    for (const listener of invalidators) listener();
}

/** 订阅"缓存被作废"（整体清空 / 单条失效）。 */
export function subscribeAssetInvalidation(listener: () => void): () => void {
    invalidators.add(listener);
    return () => {
        invalidators.delete(listener);
    };
}

function revoke(url: string): void {
    if (!url.startsWith("blob:")) return;
    try {
        URL.revokeObjectURL(url);
    } catch {
        // 已释放过：忽略。
    }
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
    if (cache.size <= CACHE_LIMIT) return;

    let oldestKey: string | null = null;
    let oldest = Number.POSITIVE_INFINITY;
    for (const [k, entry] of cache) {
        if (entry.usedAt < oldest) {
            oldest = entry.usedAt;
            oldestKey = k;
        }
    }
    if (oldestKey === null || oldestKey === key) return;
    const evicted = cache.get(oldestKey);
    cache.delete(oldestKey);
    if (!evicted) return;
    deferredRevoke.push(evicted.url);
    while (deferredRevoke.length > DEFERRED_REVOKE_LIMIT) {
        const stale = deferredRevoke.shift();
        if (stale) revoke(stale);
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
    for (const entry of cache.values()) revoke(entry.url);
    cache.clear();
    pending.clear();
    while (deferredRevoke.length > 0) {
        const url = deferredRevoke.pop();
        if (url) revoke(url);
    }
    notifyInvalidated();
}

/** 失效单条（图片被替换/删除后调用）。 */
export function invalidateImage(src: string, projectDir: string | null): void {
    const key = cacheKey(src, projectDir);
    const entry = cache.get(key);
    if (entry) revoke(entry.url);
    cache.delete(key);
    notifyInvalidated();
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
 * 失败**不缓存也不通知**（见文件头注释）。
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
        }
    })();

    pending.set(key, task);
    return task;
}

/** 图片建议的文件名（"另存为"的默认值）。 */
export function suggestImageFileName(src: string, meta?: unknown): string {
    const fromMeta =
        meta &&
        typeof meta === "object" &&
        typeof (meta as { originalName?: unknown }).originalName === "string"
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
