/*
 * 记事本后端接口。
 *
 * 附件（图片 / 剪贴板载荷）的字节都落在磁盘上，这里只做传输：
 * - 写：`putAsset` 把 base64 交给后端落盘并登记；
 * - 读：`readAsset` 取回 base64，由调用方转成 blob URL（CSP 允许 `blob:`）；
 * - 剪贴板：`readClipboardPayload` / `writeClipboardPayload` 走**原始字节**
 *   通道 —— 时间轴载荷是 MessagePack，`read_system_clipboard_object` 那条
 *   只认 UTF-8 的路径读不到它。
 */

import { invoke } from "../invoke";

/** 暂存块携带的载荷摘要（由后端从载荷本身解析，不是前端猜的）。 */
export interface ClipboardPayloadSummary {
    clipKind: string;
    clipCount: number;
    trackCount: number;
    sourceProject: string | null;
    durationSec: number;
    preview: Array<{
        trackId: string;
        trackName: string;
        name: string;
        startSec: number;
        lengthSec: number;
    }>;
    /** 参数线载荷专有：参数名 / 帧数 / 降采样曲线。 */
    param?: {
        param: string;
        framePeriodMs: number;
        frameCount: number;
        sparkline: number[];
    };
}

export interface ClipboardPayloadResult {
    ok: boolean;
    error?: string;
    available?: boolean;
    kind?: "clips" | "tracks" | "project" | "param";
    encoding?: "fragment" | "param";
    ext?: string;
    mime?: string;
    byteLen?: number;
    base64?: string;
    summary?: ClipboardPayloadSummary;
}

export interface NotebookAssetEntry {
    id: string;
    kind: "image" | "clip_payload";
    ext: string;
    mime: string;
    byteLen: number;
    createdAtMs: number;
    meta: unknown;
    /** 登记项里是否真的有字节（v6 旧工程迁移失败的条目为 false）。 */
    hasData: boolean;
}

export const notebookApi = {
    /** 写入一条附件：字节随工程文件内嵌保存。 */
    putAsset: (payload: {
        assetId: string;
        kind: "image" | "clip_payload";
        ext: string;
        mime?: string;
        dataBase64: string;
        meta?: unknown;
    }) =>
        invoke<{ ok: boolean; error?: string; assetId?: string; byteLen?: number }>(
            "notebook_put_asset",
            payload.assetId,
            payload.kind,
            payload.ext,
            payload.mime,
            payload.dataBase64,
            payload.meta,
        ),

    readAsset: (assetId: string) =>
        invoke<{ ok: boolean; error?: string; missing?: boolean; mime?: string; base64?: string }>(
            "notebook_read_asset",
            assetId,
        ),

    listAssets: () => invoke<{ ok: boolean; assets: NotebookAssetEntry[] }>("notebook_list_assets"),

    removeAsset: (assetId: string) =>
        invoke<{ ok: boolean; removed: boolean }>("notebook_remove_asset", assetId),

    pruneAssets: () => invoke<{ ok: boolean; removed: number }>("notebook_prune_assets"),

    readFileBase64: (path: string, maxBytes?: number) =>
        invoke<{
            ok: boolean;
            error?: string;
            mime?: string;
            ext?: string;
            byteLen?: number;
            base64?: string;
        }>("notebook_read_file_base64", path, maxBytes),

    readClipboardPayload: () => invoke<ClipboardPayloadResult>("notebook_read_clipboard_payload"),

    writeClipboardPayload: (payloadBase64: string, textSummary?: string) =>
        invoke<{ ok: boolean; error?: string }>(
            "notebook_write_clipboard_payload",
            payloadBase64,
            textSummary,
        ),

    readClipboardImage: () =>
        invoke<{
            ok: boolean;
            error?: string;
            available?: boolean;
            width?: number;
            height?: number;
            bitsPerPixel?: number;
            base64?: string;
        }>("notebook_read_clipboard_image"),

    /** 关闭记事本编辑的撤销合并窗口（切模式 / 失焦 / 关面板 / 保存前）。 */
    sealNotesHistory: () => invoke<{ ok: boolean }>("seal_project_notes_history"),

    /** 导出为自包含文档（图片以 data URI 内嵌）。 */
    exportDocument: (suggestedName: string, extension: "md" | "html", content: string) =>
        invoke<{
            ok: boolean;
            error?: string;
            canceled?: boolean;
            path?: string;
            missingAssets?: string[];
        }>("notebook_export_document", suggestedName, extension, content),

    saveAssetAs: (assetId: string, suggestedName?: string) =>
        invoke<{ ok: boolean; error?: string; canceled?: boolean; path?: string }>(
            "notebook_save_asset_as",
            assetId,
            suggestedName,
        ),
};
