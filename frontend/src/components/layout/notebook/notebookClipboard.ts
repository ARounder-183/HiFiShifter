/*
 * 记事本与系统剪贴板之间的双向搬运。
 *
 * 三件事：
 *
 * 1. **复制**（写出多 flavor）：`text/html` 给 Word/网页，`text/plain` 放
 *    Markdown 源码（粘回 Markdown 编辑器/聊天窗口都不丢结构），额外再写一个
 *    `text/markdown`（少数应用认它）。ProseMirror 自己会写 html 与 plain，
 *    这里只补 Markdown 那一份、并按设置裁剪不需要的 flavor。
 * 2. **粘贴**（读入并分流）：图片文件 → 插图片；HiFiShifter 载荷 → 暂存块；
 *    HTML → 消毒后转 Markdown；纯文本 → 按设置决定是否当 Markdown 解析。
 * 3. **暂存块与系统剪贴板的互转**：把载荷字节取出来存进工程附件，或把暂存的
 *    字节原样写回剪贴板，再复用既有的粘贴链路插进时间轴。
 */

import type { Editor } from "@tiptap/core";

import { getActiveSurface } from "../../../features/uiFocus/focusSurface";
import { resolvePasteRoute } from "../../../features/keybindings/focusRouting";
import { notebookApi, type ClipboardPayloadSummary } from "../../../services/api/notebook";
import { webApi } from "../../../services/webviewApi";
import { contentHash } from "./notebookImagePipeline";
import { markdownStorage } from "./markdownCodec";
import { htmlToMarkdown } from "./htmlToMarkdown";
import { insertImageFromBlob, insertImageFromClipboardBitmap, type InsertContext } from "./notebookInsert";
import {
    HIFI_CLIP_FENCE_LANG,
    defaultClipTitle,
    serializeHifiClipFenceBody,
    type HifiClipBlockAttrs,
    type HifiClipKind,
} from "./hifiClipBlock";

/** 复制内容的 flavor 裁剪结果。 */
export interface ClipboardFlavors {
    /** 是否保留 ProseMirror 写的 `text/html`。 */
    keepHtml: boolean;
    /** `text/plain` 的内容（Markdown 源码或渲染后纯文本）。 */
    plainText: string;
    /** 额外的 `text/markdown`；为空表示不写。 */
    markdown: string | null;
}

/** 取当前选区的 Markdown 源码。 */
export function selectionMarkdown(editor: Editor): string {
    const { state } = editor;
    const { from, to, empty } = state.selection;
    if (empty) return "";
    try {
        const storage = markdownStorage(editor);
        const slice = state.doc.cut(from, to);
        return storage.serializer.serialize(slice.content);
    } catch {
        try {
            return markdownStorage(editor).getMarkdown();
        } catch {
            // 编辑器正在重建：复制出去一个空串，总好过抛异常打断复制。
            return "";
        }
    }
}

/** 取当前选区的纯文本（去掉 Markdown 标记）。 */
export function selectionPlainText(editor: Editor): string {
    const { state } = editor;
    const { from, to, empty } = state.selection;
    if (empty) return "";
    return state.doc.textBetween(from, to, "\n", "\n");
}

/**
 * 计算复制时要写入的 flavor。
 *
 * `copyFormat` 的四种取值决定"给外部什么"：
 * - `markdown+html`（默认）：富文本应用拿 HTML、Markdown 应用拿源码；
 * - `markdown`：只要源码（粘进代码/聊天场景不会被塞一堆 HTML）；
 * - `html`：只要富文本（`text/plain` 退化成去掉标记的纯文本）；
 * - `text`：只要纯文本。
 */
export function computeClipboardFlavors(
    editor: Editor,
    settings: { copyFormat: string; copyPlainTextAs: string },
): ClipboardFlavors {
    const markdown = selectionMarkdown(editor);
    const plainFromDoc = selectionPlainText(editor);
    switch (settings.copyFormat) {
        case "markdown":
            return { keepHtml: false, plainText: markdown, markdown };
        case "html":
            return { keepHtml: true, plainText: plainFromDoc, markdown: null };
        case "text":
            return { keepHtml: false, plainText: plainFromDoc, markdown: null };
        default:
            return {
                keepHtml: true,
                plainText: settings.copyPlainTextAs === "text" ? plainFromDoc : markdown,
                markdown,
            };
    }
}

/**
 * 在编辑器的 copy/cut 事件上补写 Markdown flavor。
 *
 * 必须挂在编辑器 DOM 上、且在 ProseMirror 自己的监听之后执行：ProseMirror
 * 会写 `text/html` 与 `text/plain`，这里只做补充与裁剪。
 */
export function installClipboardFlavorWriter(
    element: HTMLElement,
    getSettings: () => { copyFormat: string; copyPlainTextAs: string },
    getEditor: () => Editor | null,
): () => void {
    const handler = (event: ClipboardEvent) => {
        const editor = getEditor();
        const data = event.clipboardData;
        if (!editor || !data) return;
        const flavors = computeClipboardFlavors(editor, getSettings());
        if (!flavors.keepHtml) data.setData("text/html", "");
        if (flavors.markdown) data.setData("text/markdown", flavors.markdown);
        data.setData("text/plain", flavors.plainText);
    };
    element.addEventListener("copy", handler);
    element.addEventListener("cut", handler);
    return () => {
        element.removeEventListener("copy", handler);
        element.removeEventListener("cut", handler);
    };
}

/** 粘贴处理的上下文。 */
export interface PasteContext extends InsertContext {
    /** 是否处于源码视图（源码视图下不接管粘贴，交给 textarea）。 */
    sourceMode: boolean;
}

/**
 * 处理一次粘贴。返回 `true` 表示已经接管（调用方需阻止默认行为）。
 *
 * 判定顺序即优先级：
 * 1. 剪贴板里有图片文件 → 插图片（这是"粘贴截图/图片文件"的主路径）；
 * 2. 剪贴板里是 HiFiShifter 载荷 → 插暂存块（**优先于**它的摘要文本，
 *    否则用户粘贴 clip 只会得到一句 "HiFiShifter: 3 clip(s) copied"）；
 * 3. 有 HTML → 消毒 + 转 Markdown；
 * 4. 有纯文本 → 按 `plainPasteMode` 决定当不当 Markdown；
 * 5. 只有位图（截图工具）→ 走后端读 CF_DIB 的兜底路径。
 *
 * 全部为异步：`handlePaste` 必须同步返回 true 才能压住默认插入，因此这里
 * 先同步判定"要不要接管"，具体插入在 promise 里完成。
 */
export function handleNotebookPaste(ctx: PasteContext, event: ClipboardEvent): boolean {
    if (ctx.sourceMode) return false;
    const data = event.clipboardData;
    if (!data) return false;

    const files = Array.from(data.files ?? []).filter((file) => file.type.startsWith("image/"));
    const html = data.getData("text/html");
    const text = data.getData("text/plain");

    if (files.length > 0) {
        void (async () => {
            for (const file of files) {
                await insertImageFromBlob(ctx, file, file.name || "pasted.png");
            }
        })();
        return true;
    }

    const hasAnyPayload = Boolean(html) || Boolean(text);
    void (async () => {
        const staged = await stageClipboardPayload();
        if (staged.ok && staged.body) {
            ctx.editor.chain().focus().insertContent(blockNodeFromBody(staged.body)).run();
            ctx.onAssetsChanged?.();
            ctx.notify?.("已暂存剪贴板数据");
            return;
        }

        if (html && ctx.settings.smartPaste && ctx.settings.htmlPasteMode === "markdown") {
            const markdown = htmlToMarkdown(html);
            if (markdown) {
                ctx.editor.chain().focus().insertContent(markdown).run();
                return;
            }
        }
        if (html && ctx.settings.smartPaste && ctx.settings.htmlPasteMode === "html") {
            // 原样富文本：交给 ProseMirror 自己的 HTML 解析（更保真，但不是
            // 规范 Markdown —— 由用户显式选择这一档）。
            ctx.editor.chain().focus().insertContent(html).run();
            return;
        }

        if (text) {
            const asMarkdown =
                ctx.settings.smartPaste &&
                (ctx.settings.plainPasteMode === "markdown" ||
                    (ctx.settings.plainPasteMode === "auto" && looksLikeMarkdown(text)));
            ctx.editor
                .chain()
                .focus()
                .insertContent(asMarkdown ? text : escapeMarkdownText(text))
                .run();
            return;
        }

        if (!hasAnyPayload) {
            // 截图工具只放位图的情形。
            const inserted = await insertImageFromClipboardBitmap(ctx);
            if (!inserted.ok) ctx.notify?.("剪贴板里没有可粘贴的内容", "error");
        }
    })();

    return true;
}

/** 明显的 Markdown 特征（标题 / 列表 / 围栏 / 引用 / 表格分隔行）。 */
export function looksLikeMarkdown(text: string): boolean {
    const lines = text.split(/\r?\n/);
    let hits = 0;
    for (const line of lines) {
        const trimmed = line.trim();
        if (/^#{1,6}\s+\S/.test(trimmed)) hits += 1;
        else if (/^([-*+]|\d+\.)\s+\S/.test(trimmed)) hits += 1;
        else if (/^>\s?\S/.test(trimmed)) hits += 1;
        else if (/^```/.test(trimmed)) hits += 1;
        else if (/^\|.+\|$/.test(trimmed)) hits += 1;
        else if (/^\[.+\]\(.+\)$/.test(trimmed)) hits += 1;
        if (hits >= 2) return true;
    }
    return hits >= 1 && lines.length > 1;
}

/**
 * 转义纯文本里的 Markdown 元字符。
 *
 * `plainPasteMode = "text"` 时，用户明确要求"原样文本"，此时 `# `、`- `
 * 之类的行首标记不能被解释成标题/列表。
 */
export function escapeMarkdownText(text: string): string {
    return text
        .split(/\r?\n/)
        .map((line) => line.replace(/^(\s*)([#>|])/, "$1\\$2").replace(/^(\s*)([-*+]\s)/, "$1\\$2"))
        .join("\n");
}

// ─── 暂存块 ⇄ 系统剪贴板 ──────────────────────────────────────────────────────

function kindFromSummary(summary: ClipboardPayloadSummary | undefined, fallback: string): HifiClipKind {
    const raw = summary?.clipKind ?? fallback;
    if (raw === "clips" || raw === "tracks" || raw === "project" || raw === "param") return raw;
    return "clips";
}

export interface StagedPayload {
    ok: boolean;
    body?: string;
    assetId?: string;
    error?: string;
}

/**
 * 把系统剪贴板里的 HiFiShifter 载荷暂存为附件，并返回围栏正文。
 *
 * 载荷字节**原样**存下来（不做任何重新序列化）：恢复时写回的必须是同一份
 * 字节，重新编码会随版本漂移，跨进程/跨版本粘贴就可能失效。
 */
export async function stageClipboardPayload(): Promise<StagedPayload> {
    const payload = await notebookApi.readClipboardPayload();
    if (!payload.ok || !payload.available || !payload.base64 || !payload.kind) {
        return { ok: false };
    }

    const bytes = base64ToBytes(payload.base64);
    const assetId = await contentHash(bytes);
    const summary = payload.summary;
    const kind = kindFromSummary(summary, payload.kind);

    const attrs: HifiClipBlockAttrs = {
        id: assetId,
        kind,
        title: "",
        source: summary?.sourceProject ?? "",
        clipCount: summary?.clipCount ?? 0,
        trackCount: summary?.trackCount ?? 0,
        durationSec: summary?.durationSec ?? 0,
        captured: new Date().toISOString().slice(0, 19) + "Z",
        encoding: payload.encoding === "param" ? "param" : "fragment",
        param: summary?.param?.param,
        frameCount: summary?.param?.frameCount,
    };
    if (!attrs.title) attrs.title = defaultClipTitle(attrs);

    const put = await notebookApi.putAsset({
        assetId,
        kind: "clip_payload",
        ext: payload.ext ?? "hsf",
        mime: payload.mime,
        dataBase64: payload.base64,
        meta: {
            clipKind: kind,
            title: attrs.title,
            clipCount: attrs.clipCount,
            trackCount: attrs.trackCount,
            durationSec: attrs.durationSec,
            sourceProject: attrs.source,
            preview: summary?.preview ?? [],
            param: summary?.param ?? null,
            byteLen: payload.byteLen ?? bytes.length,
        },
    });
    if (!put.ok) return { ok: false, error: put.error };

    return { ok: true, body: serializeHifiClipFenceBody(attrs), assetId };
}

/** 由围栏正文构造一个 hifiClipBlock 节点（供插入用）。 */
export function blockNodeFromBody(body: string): {
    type: string;
    attrs: { body: string };
} {
    return { type: "hifiClipBlock", attrs: { body } };
}

/** 围栏语言标记（导出/复制时用）。 */
export { HIFI_CLIP_FENCE_LANG };

export interface RestoreResult {
    ok: boolean;
    kind?: string;
    error?: string;
}

/** 把暂存的载荷原样写回系统剪贴板。 */
export async function restoreClipPayload(assetId: string): Promise<RestoreResult> {
    const asset = await notebookApi.readAsset(assetId);
    if (!asset.ok || !asset.base64) {
        return { ok: false, error: asset.error ?? "notebook_asset_not_found" };
    }
    const written = await notebookApi.writeClipboardPayload(asset.base64, "HiFiShifter data restored.");
    if (!written.ok) return { ok: false, error: written.error };
    // 通知参数编辑器丢弃内部的参数线剪贴板缓存（槽位内容已变）。
    window.dispatchEvent(new CustomEvent("hifi:clipboardReplaced"));
    const probe = await webApi.clipboardKind();
    return { ok: true, kind: probe.kind ?? undefined };
}

/**
 * 把暂存的载荷插入工程：先写回系统剪贴板，再走既有的粘贴路由。
 *
 * 【为什么要绕一次剪贴板】app 的"单剪贴板纪律"把系统槽位当作唯一逻辑剪贴板，
 * 时间轴/参数编辑器的粘贴链路只认槽位内容。走同一条路，就自动获得了选中
 * 轨道、自动交叉淡化、REAPER 回退等全部既有行为 —— 不需要为"从记事本插入"
 * 再造一条旁路。
 */
export async function insertClipPayload(
    assetId: string,
    mode: "selected" | "newTracks",
): Promise<RestoreResult> {
    const restored = await restoreClipPayload(assetId);
    if (!restored.ok) return restored;
    const channel = resolvePasteRoute(restored.kind ?? null, getActiveSurface());
    if (!channel) return { ok: false, error: "notebook_no_paste_route" };
    const op = mode === "newTracks" && channel === "hifi:timelineEditOp" ? "pasteTracks" : "paste";
    window.dispatchEvent(new CustomEvent(channel, { detail: { op } }));
    return { ok: true, kind: restored.kind };
}

function base64ToBytes(base64: string): Uint8Array {
    const binary = atob(base64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
    return bytes;
}
