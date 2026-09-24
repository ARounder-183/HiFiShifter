/*
 * 记事本的两个对话框：附件管理器与设置。
 *
 * 附件管理器回答"这个工程里存了哪些图片/剪贴板载荷、谁还在被引用、占多大"，
 * 并提供清理未引用、另存为、在文件管理器中显示 —— 附件是**只增不删**的
 * （撤销要能恢复），所以"清理"必须由用户显式触发。
 */

import { Button, Dialog, Flex, Select, Switch, Text } from "@radix-ui/themes";
import { useEffect, useMemo, useState } from "react";

import type { NotebookAssetSummary } from "../../../features/notebook/notebookSlice";
import { useI18n } from "../../../i18n/I18nProvider";
import { notebookApi } from "../../../services/api/notebook";
import { formatAssetRef } from "./assetRef";
import { resolveImage } from "./notebookImageCache";
import { formatBytes } from "./notebookInsert";
import type { ResolvedNotebookSettings } from "./notebookSettings";

// ─── 附件管理器 ──────────────────────────────────────────────────────────────

export interface NotebookAttachmentsDialogProps {
    onClose: () => void;
    /** 正文里仍被引用的附件 id。 */
    usedAssetIds: Set<string>;
    assetIndex: Record<string, NotebookAssetSummary>;
    onChanged: () => void;
    notify: (message: string) => void;
}

export function NotebookAttachmentsDialog({
    onClose,
    usedAssetIds,
    assetIndex,
    onChanged,
    notify,
}: NotebookAttachmentsDialogProps) {
    const { t } = useI18n();
    const entries = useMemo(() => Object.values(assetIndex), [assetIndex]);
    const unusedCount = entries.filter((entry) => !usedAssetIds.has(entry.id)).length;
    const totalBytes = entries.reduce((sum, entry) => sum + (entry.byteLen || 0), 0);

    return (
        <Dialog.Root open onOpenChange={(open) => !open && onClose()}>
            <Dialog.Content
                maxWidth="520px"
                onKeyDown={(event) => event.stopPropagation()}
                style={{ maxHeight: "70vh", display: "flex", flexDirection: "column" }}
            >
                <Dialog.Title>{t("notebook_attachments")}</Dialog.Title>
                <Dialog.Description size="1" color="gray">
                    {entries.length} · {formatBytes(totalBytes)}
                </Dialog.Description>

                <div className="mt-2 min-h-0 flex-1 overflow-auto">
                    {entries.length === 0 ? (
                        <Text size="1" color="gray">
                            {t("notebook_attachments_empty")}
                        </Text>
                    ) : (
                        entries.map((entry) => (
                            <AttachmentRow
                                key={entry.id}
                                entry={entry}
                                used={usedAssetIds.has(entry.id)}
                                onChanged={onChanged}
                                notify={notify}
                            />
                        ))
                    )}
                </div>

                <Flex justify="between" align="center" className="mt-3" gap="2">
                    <Button
                        variant="soft"
                        color="gray"
                        disabled={unusedCount === 0}
                        data-tooltip={t("notebook_attachments_clean_hint")}
                        onClick={() => {
                            void (async () => {
                                const result = await notebookApi.pruneAssets();
                                notify(t("notebook_attachments_cleaned") + ` (${result.removed})`);
                                onChanged();
                            })();
                        }}
                    >
                        {t("notebook_attachments_clean")} ({unusedCount})
                    </Button>
                    <Dialog.Close>
                        <Button variant="soft" color="gray">
                            {t("close")}
                        </Button>
                    </Dialog.Close>
                </Flex>
            </Dialog.Content>
        </Dialog.Root>
    );
}

function AttachmentRow({
    entry,
    used,
    onChanged,
    notify,
}: {
    entry: NotebookAssetSummary;
    used: boolean;
    onChanged: () => void;
    notify: (message: string) => void;
}) {
    const { t } = useI18n();
    return (
        <div className="hs-notebook-attachment-row">
            <AttachmentThumb entry={entry} />
            <span className="hs-notebook-attachment-name" data-tooltip={entry.id}>
                {describeEntry(entry, t as unknown as (key: string) => string)}
            </span>
            <span className={used ? undefined : "hs-notebook-attachment-unused"}>
                {used ? t("notebook_attachments_used") : t("notebook_attachments_unused")}
            </span>
            <span>{formatBytes(entry.byteLen)}</span>
            <Button
                variant="ghost"
                color="gray"
                size="1"
                data-tooltip={t("notebook_attachments_save_as")}
                disabled={!entry.hasData}
                onClick={() => void notebookApi.saveAssetAs(entry.id).catch(() => {})}
            >
                ⤓
            </Button>
            <Button
                variant="ghost"
                color="gray"
                size="1"
                data-tooltip={t("notebook_image_remove")}
                onClick={() => {
                    void (async () => {
                        await notebookApi.removeAsset(entry.id);
                        notify(t("notebook_attachments_removed"));
                        onChanged();
                    })();
                }}
            >
                ✕
            </Button>
        </div>
    );
}

function describeEntry(entry: NotebookAssetSummary, t: (key: string) => string): string {
    if (entry.kind === "clip_payload") {
        const meta = entry.meta as { title?: string } | null;
        return `${t("notebook_clip_kind_clips")} · ${meta?.title || entry.id}`;
    }
    const meta = entry.meta as { originalName?: string; width?: number; height?: number } | null;
    const name = meta?.originalName || `${entry.id}.${entry.ext}`;
    const size = meta?.width && meta?.height ? ` · ${meta.width}×${meta.height}` : "";
    return `${name}${size}`;
}

/** 附件缩略图（图片走统一解析缓存，载荷类显示占位符）。 */
function AttachmentThumb({ entry }: { entry: NotebookAssetSummary }) {
    const [url, setUrl] = useState<string | null>(null);
    const isImage = entry.kind === "image";

    useEffect(() => {
        if (!isImage) return;
        let cancelled = false;
        void resolveImage(formatAssetRef(entry.id, entry.ext), {
            projectDir: null,
            allowRemoteImages: false,
        }).then((result) => {
            if (!cancelled) setUrl(result.url);
        });
        return () => {
            cancelled = true;
        };
    }, [entry.ext, entry.id, isImage]);

    if (isImage && url) {
        return <img className="hs-notebook-attachment-thumb" src={url} alt="" />;
    }
    return (
        <span className="hs-notebook-attachment-thumb grid place-items-center text-qt-text-muted">
            {isImage ? "🖼" : "📋"}
        </span>
    );
}

// ─── 设置 ────────────────────────────────────────────────────────────────────

export interface NotebookSettingsDialogProps {
    settings: ResolvedNotebookSettings;
    onClose: () => void;
    onChange: (patch: Record<string, unknown>) => void;
    markdown: string;
    projectName: string;
    /** 取富文本视图的 HTML（HTML 导出用）；源码视图下可能为 null。 */
    getHtml?: () => string | null;
}

export function NotebookSettingsDialog({
    settings,
    onClose,
    onChange,
    markdown,
    projectName,
    getHtml,
}: NotebookSettingsDialogProps) {
    const { t } = useI18n();
    const tAny = t as unknown as (key: string) => string;
    const [exportNotice, setExportNotice] = useState<string | null>(null);

    /**
     * 导出并给出反馈。
     *
     * 导出会弹系统保存对话框，因此结果必须回显（成功/取消/失败），否则用户
     * 无法区分"没点着"和"选了路径但写失败"。
     */
    async function runExport(extension: "md" | "html", content: string) {
        setExportNotice(tAny("notebook_export_running"));
        try {
            const result = await notebookApi.exportDocument(
                projectName || "notes",
                extension,
                content,
            );
            if (result.canceled) {
                setExportNotice(null);
                return;
            }
            if (!result.ok) {
                setExportNotice(`${tAny("notebook_export_failed")}: ${result.error ?? ""}`);
                return;
            }
            const missing = result.missingAssets?.length ?? 0;
            setExportNotice(
                missing > 0
                    ? `${tAny("notebook_export_done")} (${tAny("notebook_export_missing")}: ${missing})`
                    : tAny("notebook_export_done"),
            );
        } catch {
            setExportNotice(tAny("notebook_export_failed"));
        }
    }

    return (
        <Dialog.Root open onOpenChange={(open) => !open && onClose()}>
            <Dialog.Content
                maxWidth="560px"
                onKeyDown={(event) => event.stopPropagation()}
                style={{ maxHeight: "76vh", display: "flex", flexDirection: "column" }}
            >
                <Dialog.Title>{t("notebook_settings")}</Dialog.Title>

                <div className="min-h-0 flex-1 overflow-auto pr-1">
                    <Section title={tAny("notebook_settings_group_view")}>
                        <SelectRow
                            label={tAny("notebook_setting_default_mode")}
                            value={settings.defaultMode}
                            options={[
                                { value: "rich", label: t("notebook_mode_rich") },
                                { value: "source", label: t("notebook_mode_source") },
                                { value: "split", label: t("notebook_mode_split") },
                            ]}
                            onChange={(value) => onChange({ defaultMode: value })}
                        />
                        <SwitchRow
                            label={tAny("notebook_setting_toolbar")}
                            checked={settings.showToolbar}
                            onChange={(value) => onChange({ showToolbar: value })}
                        />
                        <SwitchRow
                            label={tAny("notebook_setting_markdown_shortcuts")}
                            checked={settings.markdownShortcuts}
                            onChange={(value) => onChange({ markdownShortcuts: value })}
                        />
                        <SwitchRow
                            label={tAny("notebook_setting_slash")}
                            checked={settings.slashCommands}
                            onChange={(value) => onChange({ slashCommands: value })}
                        />
                        <SwitchRow
                            label={tAny("notebook_setting_word_wrap")}
                            checked={settings.sourceWordWrap}
                            onChange={(value) => onChange({ sourceWordWrap: value })}
                        />
                        <SwitchRow
                            label={tAny("notebook_setting_spellcheck")}
                            checked={settings.spellCheck}
                            onChange={(value) => onChange({ spellCheck: value })}
                        />
                        <SelectRow
                            label={tAny("notebook_setting_font_size")}
                            value={String(settings.sourceFontSize)}
                            options={[
                                { value: "11", label: "11" },
                                { value: "12", label: "12" },
                                { value: "13", label: "13" },
                                { value: "15", label: "15" },
                                { value: "17", label: "17" },
                            ]}
                            onChange={(value) => onChange({ sourceFontSize: Number(value) })}
                        />
                        <SelectRow
                            label={tAny("notebook_setting_history_split")}
                            value={String(settings.historySplitIdleMs)}
                            options={[
                                { value: "0", label: tAny("notebook_setting_history_split_off") },
                                { value: "2000", label: "2s" },
                                { value: "5000", label: "5s" },
                                { value: "15000", label: "15s" },
                            ]}
                            onChange={(value) => onChange({ historySplitIdleMs: Number(value) })}
                        />
                    </Section>

                    <Section title={tAny("notebook_settings_group_image")}>
                        <SelectRow
                            label={tAny("notebook_setting_image_max_dim")}
                            value={String(settings.imageMaxDimensionPx)}
                            options={[
                                {
                                    value: "0",
                                    label: tAny("notebook_setting_image_max_dim_original"),
                                },
                                { value: "1280", label: "1280" },
                                { value: "2048", label: "2048" },
                                { value: "2560", label: "2560" },
                                { value: "3840", label: "3840" },
                            ]}
                            onChange={(value) => onChange({ imageMaxDimensionPx: Number(value) })}
                        />
                        <SelectRow
                            label={tAny("notebook_setting_image_format")}
                            value={settings.imageFormat}
                            options={[
                                {
                                    value: "auto",
                                    label: tAny("notebook_setting_image_format_auto"),
                                },
                                { value: "webp", label: "WebP" },
                                { value: "jpeg", label: "JPEG" },
                                { value: "png", label: "PNG" },
                            ]}
                            onChange={(value) => onChange({ imageFormat: value })}
                        />
                        <SwitchRow
                            label={tAny("notebook_setting_remote_images")}
                            checked={settings.allowRemoteImages}
                            onChange={(value) => onChange({ allowRemoteImages: value })}
                        />
                    </Section>

                    <Section title={tAny("notebook_settings_group_clipboard")}>
                        <SwitchRow
                            label={tAny("notebook_setting_smart_paste")}
                            checked={settings.smartPaste}
                            onChange={(value) => onChange({ smartPaste: value })}
                        />
                        <SelectRow
                            label={tAny("notebook_setting_html_paste")}
                            value={settings.htmlPasteMode}
                            options={[
                                { value: "markdown", label: "Markdown" },
                                { value: "html", label: "HTML" },
                                { value: "text", label: tAny("notebook_setting_paste_text") },
                            ]}
                            onChange={(value) => onChange({ htmlPasteMode: value })}
                        />
                        <SelectRow
                            label={tAny("notebook_setting_plain_paste")}
                            value={settings.plainPasteMode}
                            options={[
                                { value: "auto", label: tAny("notebook_setting_plain_paste_auto") },
                                { value: "markdown", label: "Markdown" },
                                { value: "text", label: tAny("notebook_setting_paste_text") },
                            ]}
                            onChange={(value) => onChange({ plainPasteMode: value })}
                        />
                        <SelectRow
                            label={tAny("notebook_setting_copy_format")}
                            value={settings.copyFormat}
                            options={[
                                {
                                    value: "markdown+html",
                                    label: tAny("notebook_setting_copy_format_both"),
                                },
                                { value: "markdown", label: "Markdown" },
                                { value: "html", label: "HTML" },
                                { value: "text", label: tAny("notebook_setting_paste_text") },
                            ]}
                            onChange={(value) => onChange({ copyFormat: value })}
                        />
                        <SelectRow
                            label={tAny("notebook_setting_copy_plain")}
                            value={settings.copyPlainTextAs}
                            options={[
                                {
                                    value: "markdown",
                                    label: tAny("notebook_setting_copy_plain_markdown"),
                                },
                                { value: "text", label: tAny("notebook_setting_paste_text") },
                            ]}
                            onChange={(value) => onChange({ copyPlainTextAs: value })}
                        />
                    </Section>

                    <Section title={tAny("notebook_settings_group_clip_block")}>
                        <SelectRow
                            label={tAny("notebook_setting_clip_insert_mode")}
                            value={settings.clipInsertMode}
                            options={[
                                {
                                    value: "selected",
                                    label: tAny("notebook_setting_clip_insert_selected"),
                                },
                                {
                                    value: "newTracks",
                                    label: tAny("notebook_setting_clip_insert_new_tracks"),
                                },
                            ]}
                            onChange={(value) => onChange({ clipInsertMode: value })}
                        />
                        <SwitchRow
                            label={tAny("notebook_setting_keep_clip")}
                            checked={settings.keepClipAfterInsert}
                            onChange={(value) => onChange({ keepClipAfterInsert: value })}
                        />
                        <SwitchRow
                            label={tAny("notebook_setting_clip_preview")}
                            checked={settings.clipShowPreview}
                            onChange={(value) => onChange({ clipShowPreview: value })}
                        />
                    </Section>

                    <Section title={tAny("notebook_settings_group_export")}>
                        <Flex gap="2" wrap="wrap">
                            <Button
                                variant="soft"
                                color="gray"
                                onClick={() => {
                                    void runExport("md", markdown);
                                }}
                            >
                                {tAny("notebook_export_md")}
                            </Button>
                            <Button
                                variant="soft"
                                color="gray"
                                onClick={() => {
                                    void runExport(
                                        "html",
                                        buildExportHtml(markdown, projectName, getHtml?.() ?? null),
                                    );
                                }}
                            >
                                {tAny("notebook_export_html")}
                            </Button>
                        </Flex>
                    </Section>
                </div>

                <Flex justify="between" align="center" className="mt-3" gap="2">
                    <Text size="1" color="gray">
                        {exportNotice ?? ""}
                    </Text>
                    <Dialog.Close>
                        <Button variant="soft" color="gray">
                            {t("close")}
                        </Button>
                    </Dialog.Close>
                </Flex>
            </Dialog.Content>
        </Dialog.Root>
    );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
    return (
        <div className="mb-3">
            <div className="mb-1 text-xs font-medium text-qt-text-muted">{title}</div>
            {children}
        </div>
    );
}

/**
 * 设置行：左标签（定宽）+ 控件。
 *
 * 与 `TimelineDisplaySettingsDialog` / `RenderCacheDialog` 同一形态（标签定宽、
 * 左对齐、控件占满剩余宽度）—— 弹窗之间的一致性靠沿用同一套排布约定，
 * 而不是各自调参。
 */
const SETTING_LABEL_STYLE: React.CSSProperties = { minWidth: 132 };

function SwitchRow({
    label,
    checked,
    onChange,
}: {
    label: string;
    checked: boolean;
    onChange: (value: boolean) => void;
}) {
    return (
        <Flex align="center" gap="2">
            <Text size="2" style={SETTING_LABEL_STYLE}>
                {label}
            </Text>
            <Switch checked={checked} onCheckedChange={onChange} />
        </Flex>
    );
}

function SelectRow({
    label,
    value,
    options,
    hint,
    onChange,
}: {
    label: string;
    value: string;
    options: Array<{ value: string; label: string }>;
    hint?: string;
    onChange: (value: string) => void;
}) {
    return (
        <Flex align="center" gap="2">
            <Flex direction="column" style={SETTING_LABEL_STYLE}>
                <Text size="2">{label}</Text>
                {hint ? (
                    <Text size="1" color="gray">
                        {hint}
                    </Text>
                ) : null}
            </Flex>
            <Select.Root value={value} onValueChange={onChange}>
                <Select.Trigger style={{ flex: 1 }} />
                <Select.Content>
                    {options.map((option) => (
                        <Select.Item key={option.value} value={option.value}>
                            {option.label}
                        </Select.Item>
                    ))}
                </Select.Content>
            </Select.Root>
        </Flex>
    );
}

/**
 * HTML 导出：把富文本视图的 HTML 包进最小文档骨架 + 一段排版样式。
 *
 * `hifi-asset://` 引用交给后端改写成内嵌 data URI（导出物自包含，不生成
 * 任何旁挂目录），这里不重复实现。剪贴板暂存块的 `<div data-hifi-clip="...">` 转成
 * `<pre><code class="language-hifi-clip">` —— 与其他 Markdown 渲染器看到
 * 的形态一致（一个代码块），外部打开时不会是一团乱码。
 */
function buildExportHtml(markdown: string, title: string, editorHtml: string | null): string {
    const body = editorHtml
        ? transformClipBlocks(editorHtml)
        : `<pre>${escapeHtml(markdown)}</pre>`;
    return [
        "<!doctype html>",
        '<html><head><meta charset="utf-8">',
        `<title>${escapeHtml(title)}</title>`,
        "<style>",
        "body{font-family:system-ui,-apple-system,'Segoe UI',sans-serif;line-height:1.7;max-width:52rem;margin:2rem auto;padding:0 1rem}",
        "pre{background:#f5f5f5;padding:.75rem;border-radius:4px;overflow:auto}",
        "code{font-family:ui-monospace,Consolas,monospace;font-size:.92em}",
        "table{border-collapse:collapse;width:100%}th,td{border:1px solid #ddd;padding:.25rem .5rem}",
        "img{max-width:100%;height:auto}",
        "blockquote{margin-left:0;padding-left:.8em;border-left:2px solid #ddd;color:#555}",
        "</style></head><body>",
        body,
        "</body></html>",
    ].join("");
}

/** `div[data-hifi-clip]` → `<pre><code class="language-hifi-clip">`。 */
function transformClipBlocks(html: string): string {
    return html.replace(
        /<div data-hifi-clip="([^"]*)"><\/div>/g,
        (_match, body: string) =>
            `<pre><code class="language-hifi-clip">${escapeHtml(decodeEntities(body))}</code></pre>`,
    );
}

function decodeEntities(text: string): string {
    return text
        .replaceAll("&quot;", '"')
        .replaceAll("&lt;", "<")
        .replaceAll("&gt;", ">")
        .replaceAll("&#10;", "\n")
        .replaceAll("&amp;", "&");
}

function escapeHtml(text: string): string {
    return text.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
}
