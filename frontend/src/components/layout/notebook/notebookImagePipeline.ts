/*
 * 记事本图片流水线：解码 → 缩放 → 重编码 → 内容哈希。
 *
 * 【为什么全在前端做】后端要缩图就得引入图像编解码依赖（Rust 侧目前只有
 * 音频相关的 crate）。而浏览器本来就有解码器（`createImageBitmap`）、画布
 * （缩放 + 编码 WebP/JPEG/PNG）和摘要（`crypto.subtle.digest`）—— 全部零
 * 依赖、零 IPC 往返。后端只负责把最终字节写到磁盘。
 *
 * 内容哈希同时充当附件 id：同一张图重复拖入会命中同一个 id，天然去重。
 */

export type NotebookImageFormatSetting = "auto" | "webp" | "jpeg" | "png";

export interface PreparedImage {
    /** 内容哈希（前 16 位 hex），用作附件 id。 */
    assetId: string;
    ext: string;
    mime: string;
    /** base64（不含 data: 前缀）。 */
    base64: string;
    width: number;
    height: number;
    byteLen: number;
    /** 原始文件名（仅用于元数据与"另存为"的默认名）。 */
    originalName: string;
}

/**
 * 等比缩放到长边不超过 `maxDimension`；`maxDimension <= 0` 表示不缩放。
 *
 * 纯函数，单独测：尺寸算错一位就会得到半张图或撑爆内存。
 */
export function computeTargetSize(
    width: number,
    height: number,
    maxDimension: number,
): { width: number; height: number; scaled: boolean } {
    if (maxDimension <= 0 || width <= 0 || height <= 0) {
        return { width, height, scaled: false };
    }
    const longest = Math.max(width, height);
    if (longest <= maxDimension) return { width, height, scaled: false };
    const ratio = maxDimension / longest;
    return {
        width: Math.max(1, Math.round(width * ratio)),
        height: Math.max(1, Math.round(height * ratio)),
        scaled: true,
    };
}

export interface EncodeChoice {
    ext: string;
    mime: string;
    /** 1 表示无损；调用方据此决定"能否原样保留字节"。 */
    quality: number;
}

/**
 * 决定编码格式。
 *
 * `auto` 的取舍：**不需要缩放且原本就是无损格式时保持原样** —— 截图、线框图
 * 转成有损格式会出现肉眼可见的色块；其余情况转 WebP，照片类素材通常能省下
 * 60% 体积。
 */
export function chooseEncodeFormat(
    sourceMime: string,
    setting: NotebookImageFormatSetting,
    scaled: boolean,
): EncodeChoice {
    if (setting === "png") return { ext: "png", mime: "image/png", quality: 1 };
    if (setting === "jpeg") return { ext: "jpg", mime: "image/jpeg", quality: 0.85 };
    if (setting === "webp") return { ext: "webp", mime: "image/webp", quality: 0.85 };

    const normalized = (sourceMime || "image/png").toLowerCase();
    const losslessSource = /^image\/(png|gif|bmp|webp|avif)$/.test(normalized);
    if (losslessSource && !scaled) {
        const subtype = normalized.split("/")[1];
        const ext = subtype === "jpeg" ? "jpg" : subtype;
        return { ext, mime: normalized, quality: 1 };
    }
    return { ext: "webp", mime: "image/webp", quality: 0.85 };
}

/** 字节 → base64（分块避免 `String.fromCharCode(...)` 的参数长度上限）。 */
export function bytesToBase64(bytes: Uint8Array): string {
    const CHUNK = 0x8000;
    let binary = "";
    for (let i = 0; i < bytes.length; i += CHUNK) {
        binary += String.fromCharCode(...bytes.subarray(i, i + CHUNK));
    }
    return btoa(binary);
}

/** base64 → 字节。 */
export function base64ToBytes(base64: string): Uint8Array {
    const binary = atob(base64);
    const out = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i += 1) out[i] = binary.charCodeAt(i);
    return out;
}

/**
 * `Uint8Array` → `BlobPart`。
 *
 * TS 5.9 起 `Uint8Array<ArrayBufferLike>` 不再直接满足 `BlobPart`
 * （底层可能是 `SharedArrayBuffer`）。切出精确的 `ArrayBuffer` 既满足类型，
 * 也避免把整个底层缓冲（可能远大于视图）交给 Blob。
 */
export function toBlobPart(bytes: Uint8Array): ArrayBuffer {
    return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
}

/** 内容哈希（SHA-256 前 16 位 hex）。 */
export async function contentHash(bytes: Uint8Array): Promise<string> {
    const digest = await crypto.subtle.digest("SHA-256", toBlobPart(bytes));
    return Array.from(new Uint8Array(digest))
        .map((b) => b.toString(16).padStart(2, "0"))
        .join("")
        .slice(0, 16);
}

/**
 * 把 Windows 剪贴板的原始 DIB 包成 BMP 文件。
 *
 * DIB 就是"去掉 14 字节 BMP 文件头"的 BMP，补齐文件头即可交给浏览器解码。
 * 像素数据偏移靠"总长 − 像素数据长度"反推，从而无需解析压缩方式与调色板大小。
 */
export function dibToBmpBlob(
    dib: Uint8Array,
    width: number,
    height: number,
    bitsPerPixel: number,
): Blob {
    const FILE_HEADER = 14;
    const stride = Math.floor((width * bitsPerPixel + 31) / 32) * 4;
    const pixelBytes = stride * height;
    let pixelOffset = FILE_HEADER + dib.length - pixelBytes;
    if (pixelOffset < FILE_HEADER + 40) pixelOffset = FILE_HEADER + 40;

    const out = new Uint8Array(FILE_HEADER + dib.length);
    const view = new DataView(out.buffer);
    out[0] = 0x42; // 'B'
    out[1] = 0x4d; // 'M'
    view.setUint32(2, out.length, true);
    view.setUint32(6, 0, true);
    view.setUint32(10, pixelOffset, true);
    out.set(dib, FILE_HEADER);
    return new Blob([out], { type: "image/bmp" });
}

interface EncodeOptions {
    maxDimension: number;
    format: NotebookImageFormatSetting;
}

/**
 * 完整流水线：Blob → 附件字节。
 *
 * 失败返回 null，由调用方给提示 —— 拖进来一个损坏文件不该让面板崩掉。
 */
export async function prepareImage(
    blob: Blob,
    originalName: string,
    options: EncodeOptions,
): Promise<PreparedImage | null> {
    let bitmap: ImageBitmap;
    try {
        bitmap = await createImageBitmap(blob);
    } catch {
        return null;
    }

    try {
        const sourceMime = (blob.type || "image/png").toLowerCase();
        const target = computeTargetSize(bitmap.width, bitmap.height, options.maxDimension);
        const choice = chooseEncodeFormat(sourceMime, options.format, target.scaled);

        let bytes: Uint8Array;
        let mime = choice.mime;
        let ext = choice.ext;

        // 原样保留的前提：无需缩放、无损目标格式、且源类型与目标类型一致 ——
        // 否则会把 JPEG 字节挂上 .png 后缀。
        const keepOriginal = !target.scaled && choice.quality >= 1 && sourceMime === choice.mime;

        if (keepOriginal) {
            bytes = new Uint8Array(await blob.arrayBuffer());
        } else {
            const canvas = document.createElement("canvas");
            canvas.width = target.width;
            canvas.height = target.height;
            const ctx = canvas.getContext("2d");
            if (!ctx) return null;
            ctx.drawImage(bitmap, 0, 0, target.width, target.height);

            let encoded = await canvasToBlob(canvas, mime, choice.quality);
            if (!encoded) {
                // WebView 不支持目标格式（极少见）时退回 PNG，保证一定存得下来。
                encoded = await canvasToBlob(canvas, "image/png", 1);
                mime = "image/png";
                ext = "png";
            }
            if (!encoded) return null;
            bytes = new Uint8Array(await encoded.arrayBuffer());
        }

        return {
            assetId: await contentHash(bytes),
            ext,
            mime,
            base64: bytesToBase64(bytes),
            width: target.width,
            height: target.height,
            byteLen: bytes.length,
            originalName,
        };
    } finally {
        bitmap.close?.();
    }
}

function canvasToBlob(
    canvas: HTMLCanvasElement,
    mime: string,
    quality: number,
): Promise<Blob | null> {
    return new Promise<Blob | null>((resolve) =>
        canvas.toBlob((result) => resolve(result), mime, quality),
    );
}
