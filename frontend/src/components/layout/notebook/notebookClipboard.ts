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
import {
    insertImageFromBlob,
    insertImageFromClipboardBitmap,
    translateInsert,
    type InsertContext,
} from "./notebookInsert";
import type { ResolvedNotebookSettings } from "./notebookSettings";
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
 * 事件目标是否是普通文本输入框（input / textarea）。
 *
 * 面板里的查找条、图片 alt 编辑器、暂存块改名与链接浮层都是 input：它们
 * 在编辑器/面板容器 DOM **之内**，粘贴与复制事件会冒泡（或被捕获监听截获）。
 * 这些输入框的复制是"复制我正在编辑的文本"，不是"复制正文"，一旦被接管，
 * 用户会看到输入框内容原样落进文档。
 */
export function isPlainInputTarget(target: EventTarget | null): boolean {
    return (target as HTMLElement | null)?.closest?.("input, textarea") != null;
}

/**
 * 在编辑器的 copy/cut 事件上补写 Markdown flavor。
 *
 * 【分工】ProseMirror 自己会写 `text/html` 与 `text/plain`；这里挂在编辑器 DOM 上、
 * 在它**之后**执行，只做补充（`text/markdown`）与裁剪（按 `copyFormat` 决定留哪些）。
 * 必须排在它之后，否则我们写的东西会被它覆盖掉。
 *
 * 【剪切为什么要提前算】`cut` 与 `copy` 有一个致命差别：**ProseMirror 的 cut 处理器
 * 在同一次事件里就把选区删掉了**（它 dispatch 一条删除事务），而本写出器跑在冒泡
 * 阶段 —— 那时 `state.selection` 已经塌缩，`selectionMarkdown` 只能返回空串。实测
 * （全选后 Ctrl+X）：捕获阶段 `collapsed:false`，冒泡阶段 `collapsed:true`，
 * 写出去的 `text/plain` 长度为 0。
 *
 * 后果不是"少一个 flavor"，而是**剪贴板里只剩下 `text/html`**（空串的 `text/plain`
 * 会被浏览器直接丢掉）。于是：
 *   - `Ctrl+V` 仍然能用（它从 `text/html` 还原）；
 *   - **右键菜单的粘贴用不了** —— 它只能 `navigator.clipboard.readText()`，读到空串，
 *     于是"用快捷键剪切、用菜单粘贴"这条混搭路径什么都粘不出来。
 *
 * 修法：在**捕获阶段**（同一元素上，先于冒泡阶段执行，且此时 ProseMirror 还没动手、
 * 选区完好）把 flavor 快照下来，冒泡阶段直接套用。
 */
export function installClipboardFlavorWriter(
    element: HTMLElement,
    getSettings: () => { copyFormat: string; copyPlainTextAs: string },
    getEditor: () => Editor | null,
): () => void {
    /** 本次剪切在选区被删之前算好的 flavor（见上方说明）。 */
    let cutFlavors: ClipboardFlavors | null = null;

    const snapshotForCut = (event: ClipboardEvent) => {
        if (event.type !== "cut") return;
        if (isPlainInputTarget(event.target)) return;
        const editor = getEditor();
        if (!editor) return;
        cutFlavors = computeClipboardFlavors(editor, getSettings());
    };

    const handler = (event: ClipboardEvent) => {
        // NodeView 里的输入框（alt 编辑等）冒泡到编辑器 DOM：它们的复制必须
        // 走原生行为，重写 flavor 会把输入框草稿当成正文 Markdown 写上剪贴板。
        if (isPlainInputTarget(event.target)) return;
        const editor = getEditor();
        const data = event.clipboardData;
        if (!editor || !data) return;
        /*
         * 剪切用捕获阶段的快照（那时选区还在）；复制照常现算（选区没被动过）。
         *
         * 快照**在这里无条件消费掉**：万一这次剪切的冒泡处理器提前 return（例如
         * `data` 为空），留着它就会被**下一次复制**误用 —— 那会拿上一次剪切的内容
         * 去覆盖新的选区。
         */
        const stashed = cutFlavors;
        cutFlavors = null;
        const flavors =
            event.type === "cut" && stashed
                ? stashed
                : computeClipboardFlavors(editor, getSettings());
        if (!flavors.keepHtml) data.setData("text/html", "");
        if (flavors.markdown) data.setData("text/markdown", flavors.markdown);
        data.setData("text/plain", flavors.plainText);
    };

    // 捕获阶段只服务 `cut`（`copy` 不需要，冒泡阶段选区仍完好）。
    element.addEventListener("cut", snapshotForCut, true);
    element.addEventListener("copy", handler);
    element.addEventListener("cut", handler);
    return () => {
        element.removeEventListener("cut", snapshotForCut, true);
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
 * 一次粘贴的**载荷**（与事件解耦）。
 *
 * 【为什么要从 `ClipboardEvent` 里拆出来】右键菜单的"粘贴为纯文本 / 粘贴为
 * Markdown"拿不到真实事件：浏览器不允许脚本在用户手势之外读剪贴板，只能
 * `navigator.clipboard.readText()` 拿到文本。若把分流逻辑留在"吃事件"的函数里，
 * 菜单就得把整条优先级链抄第二遍 —— 而"HiFiShifter 载荷优先于它的摘要文本"
 * 这类顺序一旦分叉，用户会看到同一次粘贴在两个入口下结果不同。
 *
 * 拆开之后：事件路径（Ctrl+V）与菜单路径共用 `applyNotebookPastePayload`，
 * 差异只剩"载荷从哪来"。
 */
export interface NotebookPastePayload {
    /** 剪贴板里的图片文件（有则优先插图片）。 */
    files: File[];
    html: string;
    text: string;
}

/**
 * 单次粘贴的模式覆盖。
 *
 * 菜单的"粘贴为纯文本 / 粘贴为 Markdown"就是靠它把 `plainPasteMode` 顶掉一次，
 * 而**不改设置**：用户想这一次别解析 Markdown，不该顺手改掉他所有的粘贴行为。
 */
export interface NotebookPasteOverrides {
    plainPasteMode?: ResolvedNotebookSettings["plainPasteMode"];
    htmlPasteMode?: ResolvedNotebookSettings["htmlPasteMode"];
    smartPaste?: boolean;
}

/** 从一次真实的 `paste` 事件里取载荷。 */
export function clipboardEventPayload(event: ClipboardEvent): NotebookPastePayload {
    const data = event.clipboardData;
    return {
        files: Array.from(data?.files ?? []).filter((file) => file.type.startsWith("image/")),
        html: data?.getData("text/html") ?? "",
        text: data?.getData("text/plain") ?? "",
    };
}

/**
 * 把一份载荷按既定优先级插进文档。返回"是否真的插入了内容"。
 *
 * 判定顺序即优先级：
 * 1. 图片文件 → 插图片（这是"粘贴截图/图片文件"的主路径）；
 * 2. HiFiShifter 载荷 → 插暂存块（**优先于**它的摘要文本，否则用户粘贴 clip
 *    只会得到一句 "HiFiShifter: 3 clip(s) copied"）；
 * 3. 有 HTML → 消毒 + 转 Markdown（或按 `htmlPasteMode` 原样富文本）；
 * 4. 有纯文本 → 按 `plainPasteMode` 决定当不当 Markdown；
 * 5. 什么都没有 → 走后端读 CF_DIB 的兜底路径（截图工具只放位图）。
 *
 * 【为什么是 async 而 `handleNotebookPaste` 是 sync】`paste` 事件必须在处理器
 * 同步返回时就被 `preventDefault` 掉，否则默认插入已经发生。因此事件路径只
 * 做"要不要接管"的同步判定，插入本身在这里异步完成 —— 菜单路径没有这个约束，
 * 直接 await 即可。
 */
export async function applyNotebookPastePayload(
    ctx: InsertContext,
    payload: NotebookPastePayload,
    overrides?: NotebookPasteOverrides,
): Promise<boolean> {
    const settings: ResolvedNotebookSettings = overrides
        ? {
              ...ctx.settings,
              ...(overrides.plainPasteMode !== undefined
                  ? { plainPasteMode: overrides.plainPasteMode }
                  : {}),
              ...(overrides.htmlPasteMode !== undefined
                  ? { htmlPasteMode: overrides.htmlPasteMode }
                  : {}),
              ...(overrides.smartPaste !== undefined ? { smartPaste: overrides.smartPaste } : {}),
          }
        : ctx.settings;
    const scoped: InsertContext = { ...ctx, settings };

    if (payload.files.length > 0) {
        for (const file of payload.files) {
            await insertImageFromBlob(scoped, file, file.name || "pasted.png");
        }
        return true;
    }

    const hasAnyPayload = Boolean(payload.html) || Boolean(payload.text);

    const staged = await stageClipboardPayload();
    if (staged.ok && staged.body) {
        scoped.editor.chain().focus().insertContent(blockNodeFromBody(staged.body)).run();
        scoped.onAssetsChanged?.();
        scoped.notify?.(translateInsert(scoped, "notebook_clipboard_staged"));
        return true;
    }

    if (payload.html && settings.smartPaste && settings.htmlPasteMode === "markdown") {
        // `maxImageBytes` 同样约束粘贴内容里的内嵌 data URI（见 htmlToMarkdown）。
        const markdown = htmlToMarkdown(payload.html, {
            maxDataImageBytes: settings.maxImageBytes,
        });
        if (markdown) {
            scoped.editor.chain().focus().insertContent(markdown).run();
            return true;
        }
    }
    if (payload.html && settings.smartPaste && settings.htmlPasteMode === "html") {
        // 原样富文本：交给 ProseMirror 自己的 HTML 解析（更保真，但不是
        // 规范 Markdown —— 由用户显式选择这一档）。
        scoped.editor.chain().focus().insertContent(payload.html).run();
        return true;
    }

    if (payload.text) {
        const asMarkdown =
            settings.smartPaste &&
            (settings.plainPasteMode === "markdown" ||
                (settings.plainPasteMode === "auto" && looksLikeMarkdown(payload.text)));
        scoped.editor
            .chain()
            .focus()
            .insertContent(asMarkdown ? payload.text : escapeMarkdownText(payload.text))
            .run();
        return true;
    }

    if (!hasAnyPayload) {
        // 截图工具只放位图的情形。
        const inserted = await insertImageFromClipboardBitmap(scoped);
        if (!inserted.ok) {
            scoped.notify?.(translateInsert(scoped, "notebook_paste_nothing"), "error");
            return false;
        }
        return true;
    }

    return false;
}

/**
 * 处理一次粘贴事件。返回 `true` 表示已经接管（调用方需阻止默认行为）。
 *
 * 只负责三件同步的事：源码视图与面板内输入框放行、取载荷、判定"要不要接管"。
 * 真正的插入交给 `applyNotebookPastePayload`（与菜单路径同一份实现）。
 */
export function handleNotebookPaste(ctx: PasteContext, event: ClipboardEvent): boolean {
    if (ctx.sourceMode) return false;
    // 查找条等面板内输入框的粘贴必须走原生行为。本函数由调用方在**捕获阶段**
    // 监听（先于输入框自己的处理），这里不放行就没有任何后续 handler 能补救。
    if (isPlainInputTarget(event.target)) return false;
    if (!event.clipboardData) return false;

    void applyNotebookPastePayload(ctx, clipboardEventPayload(event));
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
        .map((line) =>
            line
                .replace(/^(\s*)([#>|])/, "$1\\$2")
                .replace(/^(\s*)([-*+]\s)/, "$1\\$2")
                // 有序列表标记：转义**分隔符**（`.` / `)`）而不是数字 —— 反斜杠
                // 只能转义 ASCII 标点，`\1` 会原样留下一个可见的反斜杠。
                .replace(/^(\s*)(\d+)([.)]\s)/, "$1$2\\$3"),
        )
        .join("\n");
}

// ─── 暂存块 ⇄ 系统剪贴板 ──────────────────────────────────────────────────────

function kindFromSummary(
    summary: ClipboardPayloadSummary | undefined,
    fallback: string,
): HifiClipKind {
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
    const written = await notebookApi.writeClipboardPayload(
        asset.base64,
        "HiFiShifter data restored.",
    );
    if (!written.ok) return { ok: false, error: written.error };
    // 槽位内容已变，通知各消费者**重新对齐剪贴板**：参数编辑器据此丢弃自己的
    // 内部缓存并回读槽位（恢复的是参数线载荷时，剪贴板预览要立刻显示它），
    // 而不是继续沿用"本面板上次复制了什么"。派发在写入成功之后，消费者回读
    // 时槽位里已经是新内容。
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
