/*
 * 往记事本正文里插入内容：图片、时间码、Clip 引用、项目信息。
 *
 * 图片插入是这里的重点，三条来源（拖入文件 / 粘贴位图 / 工具栏选文件）都
 * 收敛到同一个流程：
 *
 * ```
 * Blob → prepareImage（解码/缩放/编码/哈希）
 *      → putAsset（字节内嵌进工程文件的附件表）
 *      → src = hifi-asset://<id>.<ext>
 *      → 插入 image 节点
 * ```
 *
 * 字节内嵌而不是写旁挂目录：工程自包含，拷走/分享/打包都不会丢图。代价是
 * 工程文件变大，抵消手段在写入前（长边缩放 + WebP/JPEG 重编码 + 内容哈希去重）。
 */

import type { Editor } from "@tiptap/core";

import { notebookApi } from "../../../services/api/notebook";
import { formatAssetRef } from "./assetRef";
import type { ResolvedNotebookSettings } from "./notebookSettings";
import {
    bytesToBase64,
    dibToBmpBlob,
    prepareImage,
    toBlobPart,
    type PreparedImage,
} from "./notebookImagePipeline";
import { clearImageCache } from "./notebookImageCache";
import { baseName } from "./notebookPaths";

/**
 * 工具栏/斜杠菜单共用的插入动作集合。
 *
 * 定义在这里而不是工具栏组件里：斜杠菜单（`useNotebookSlashMenu`）也要用同一
 * 个形状，两者放在同一个模块可以避免"工具栏 → 斜杠菜单 → 工具栏"的循环依赖。
 */
export interface ToolbarInsertHandlers {
    insertImage: () => void;
    insertTimecode: () => void;
    insertClipReference: () => void;
    insertProjectInfo: () => void;
    stageClipboard: () => void;
}

export interface InsertContext {
    editor: Editor;
    settings: ResolvedNotebookSettings;
    /** 工程文件所在目录（绝对路径）；工程未落盘时为 null。 */
    projectDir: string | null;
    /** 失败/提示回调（面板把它接到状态栏或轻提示上）。 */
    notify?: (message: string, kind?: "info" | "error") => void;
    /** 资产写入后刷新附件索引。 */
    onAssetsChanged?: () => void;
}

export interface InsertResult {
    ok: boolean;
    /** 插入的图片 src（便于调用方给出"改为链接 / 撤销"提示）。 */
    src?: string;
    assetId?: string;
    reason?: "unsupported" | "too-large" | "write-failed";
}

/** 插入一张图片（Blob 形态，来自拖放/粘贴/文件读取）。 */
export async function insertImageFromBlob(
    ctx: InsertContext,
    blob: Blob,
    originalName: string,
): Promise<InsertResult> {
    if (blob.size > ctx.settings.maxImageBytes) {
        ctx.notify?.(`图片过大（${formatBytes(blob.size)}），已超过上限`, "error");
        return { ok: false, reason: "too-large" };
    }

    const prepared = await prepareImage(blob, originalName, {
        maxDimension: ctx.settings.imageMaxDimensionPx,
        format: ctx.settings.imageFormat,
    });
    if (!prepared) {
        ctx.notify?.("无法解码该图片", "error");
        return { ok: false, reason: "unsupported" };
    }

    const src = await resolveImageSrc(ctx, prepared);
    if (!src) return { ok: false, reason: "write-failed" };

    insertImageNode(ctx.editor, src, originalName);
    ctx.onAssetsChanged?.();
    return {
        ok: true,
        src,
        assetId: src.startsWith("hifi-asset://") ? prepared.assetId : undefined,
    };
}

/** 插入磁盘上的图片文件（拖放得到的是路径）。 */
export async function insertImageFromPath(
    ctx: InsertContext,
    absolutePath: string,
): Promise<InsertResult> {
    const file = await notebookApi.readFileBase64(absolutePath, ctx.settings.maxImageBytes);
    if (!file.ok || !file.base64) {
        ctx.notify?.(
            file.error === "notebook_file_too_large" ? "图片过大，已超过上限" : "读取图片失败",
            "error",
        );
        return {
            ok: false,
            reason: file.error === "notebook_file_too_large" ? "too-large" : "unsupported",
        };
    }
    const blob = new Blob([toBlobPart(base64ToBlobPart(file.base64))], {
        type: file.mime ?? "image/png",
    });
    return insertImageFromBlob(ctx, blob, baseName(absolutePath));
}

/** 插入 Windows 剪贴板里的位图（截图粘贴的兜底路径）。 */
export async function insertImageFromClipboardBitmap(ctx: InsertContext): Promise<InsertResult> {
    const bitmap = await notebookApi.readClipboardImage();
    if (!bitmap.ok || !bitmap.available || !bitmap.base64 || !bitmap.width || !bitmap.height) {
        return { ok: false, reason: "unsupported" };
    }
    const dib = base64ToBlobPart(bitmap.base64);
    const bmp = dibToBmpBlob(dib, bitmap.width, bitmap.height, bitmap.bitsPerPixel ?? 32);
    return insertImageFromBlob(ctx, bmp, "clipboard.png");
}

/**
 * 决定图片的 src。
 *
 * 三种存储模式的区别只体现在这里，节点与渲染层不需要知道差别 ——
 * 渲染层按 src 前缀分派（见 `notebookImageCache`）。
 */
async function resolveImageSrc(
    ctx: InsertContext,
    prepared: PreparedImage,
): Promise<string | null> {
    const result = await notebookApi.putAsset({
        assetId: prepared.assetId,
        kind: "image",
        ext: prepared.ext,
        mime: prepared.mime,
        dataBase64: prepared.base64,
        meta: {
            width: prepared.width,
            height: prepared.height,
            originalName: prepared.originalName,
            byteLen: prepared.byteLen,
        },
    });
    if (!result.ok) {
        ctx.notify?.(result.error ?? "保存图片失败", "error");
        return null;
    }
    // 不写 `#w=`：插入时按原始尺寸显示，宽度由用户拖拽把手决定。
    // （缩放到长边上限是**存储**层面的优化，与显示宽度无关。）
    return formatAssetRef(prepared.assetId, prepared.ext);
}

/** 在当前选区插入一个 image 节点。 */
export function insertImageNode(editor: Editor, src: string, alt: string): void {
    editor
        .chain()
        .focus()
        .insertContent([
            { type: "image", attrs: { src, alt, width: null } },
            // 图片是块级节点，后面补一个空段落，否则光标无处可去（用户得先
            // 按一次回车才能继续打字）。
            { type: "paragraph" },
        ])
        .run();
}

/** 插入一段 Markdown 文本（时间码链接、Clip 引用、项目信息等）。 */
export function insertMarkdown(ctx: InsertContext, markdown: string): void {
    // tiptap-markdown 覆盖了 insertContentAt，接受 Markdown 字符串并按
    // 当前 schema 解析 —— 因此这里不需要先转 HTML。
    ctx.editor.chain().focus().insertContent(markdown).run();
}

function base64ToBlobPart(base64: string): Uint8Array {
    const binary = atob(base64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
    return bytes;
}

/** 字节数的人类可读形式（提示文案用）。 */
export function formatBytes(bytes: number): string {
    if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
    const units = ["B", "KB", "MB", "GB"];
    let value = bytes;
    let unit = 0;
    while (value >= 1024 && unit < units.length - 1) {
        value /= 1024;
        unit += 1;
    }
    return `${value >= 10 || unit === 0 ? Math.round(value) : value.toFixed(1)} ${units[unit]}`;
}

/** 图片字节的 base64 编码（导出/复制时用）。 */
export function encodeImageBytes(bytes: Uint8Array): string {
    return bytesToBase64(bytes);
}

/** 切换工程时清空图片缓存（不同工程的附件 id 空间可能重叠）。 */
export function resetImageCaches(): void {
    clearImageCache();
}

/** 插入一个 3×3 表格（带表头行）。 */
export function insertTable(editor: Editor): void {
    editor.chain().focus().insertTable({ rows: 3, cols: 3, withHeaderRow: true }).run();
}
