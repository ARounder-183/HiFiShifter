/*
 * 记事本的两个对话框：附件管理器与设置。
 *
 * 附件管理器回答"这个工程里存了哪些图片/剪贴板载荷、谁还在被引用、占多大"，
 * 并提供清理未引用、另存为、在文件管理器中显示 —— 附件是**只增不删**的
 * （撤销要能恢复），所以"清理"必须由用户显式触发。
 */

import { useEffect, useMemo, useState } from "react";

import type { NotebookAssetSummary } from "../../../features/notebook/notebookSlice";
import { useI18n } from "../../../i18n/I18nProvider";
import { notebookApi } from "../../../services/api/notebook";
import { AppSelect } from "../../../ui";
import { ClipboardIcon, ImageIcon } from "@radix-ui/react-icons";

import { AppButton, AppNumberField } from "../../../ui";
import { AppDialog } from "../../../ui/Dialog";
import { AppField, AppForm, AppFormSection, AppSwitchRow } from "../../../ui/Field";
import { formatAssetRef } from "./assetRef";
import { clipKindLabelKey } from "./hifiClipBlock";
import { resolveImage } from "./notebookImageCache";
import { formatBytes, notebookErrorKey } from "./notebookInsert";
import type { ResolvedNotebookSettings } from "./notebookSettings";
import { NOTEBOOK_FONT_SIZE_MAX, NOTEBOOK_FONT_SIZE_MIN } from "./notebookSettings";

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
        <AppDialog
            open
            onOpenChange={(open) => !open && onClose()}
            title={t("notebook_attachments")}
            description={
                <>
                    {entries.length} · {formatBytes(totalBytes)}
                </>
            }
            size="md"
            actions={[
                {
                    id: "clean",
                    label: (
                        <span data-tooltip={t("notebook_attachments_clean_hint")}>
                            {t("notebook_attachments_clean")} ({unusedCount})
                        </span>
                    ),
                    align: "start",
                    disabled: unusedCount === 0,
                    onClick: () =>
                        (async () => {
                            const result = await notebookApi.pruneAssets();
                            notify(t("notebook_attachments_cleaned") + ` (${result.removed})`);
                            onChanged();
                        })(),
                },
                {
                    id: "close",
                    label: t("close"),
                    onClick: () => onClose(),
                },
            ]}
        >
            {/*
             * 附件列表是这个对话框唯一的滚动区：`flex-1 min-h-0` 需要 body 是
             * flex 列才生效，否则 `flex-1` 是空转（子元素高度仍由内容决定，
             * 结果是 body 在滚、这层 `overflow-auto` 永不触发）。这里把外层补上。
             */}
            <div className="flex h-full min-h-0 flex-col pt-2">
                <div className="hs-scroll-gutter min-h-0 flex-1 overflow-auto">
                    {entries.length === 0 ? (
                        <span className="hs-type-muted">{t("notebook_attachments_empty")}</span>
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
            </div>
        </AppDialog>
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
    const { t, tf } = useI18n();
    return (
        <div className="hs-notebook-attachment-row">
            <AttachmentThumb entry={entry} />
            <span className="hs-notebook-attachment-name" data-tooltip={entry.id}>
                {describeEntry(entry, tf)}
            </span>
            <span className={used ? undefined : "hs-notebook-attachment-unused"}>
                {used ? t("notebook_attachments_used") : t("notebook_attachments_unused")}
            </span>
            <span>{formatBytes(entry.byteLen)}</span>
            <AppButton
                size="sm"
                data-tooltip={t("notebook_attachments_save_as")}
                disabled={!entry.hasData}
                onClick={() => void notebookApi.saveAssetAs(entry.id).catch(() => {})}
            >
                ⤓
            </AppButton>
            <AppButton
                size="sm"
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
            </AppButton>
        </div>
    );
}

function describeEntry(entry: NotebookAssetSummary, t: (key: string) => string): string {
    if (entry.kind === "clip_payload") {
        const meta = entry.meta as { title?: string; clipKind?: string } | null;
        // 种类按载荷自报的 `clipKind` 取名：此前一律写"片段"，参数线载荷会被
        // 标成时间轴片段 —— 用户据此以为这条能插到时间轴上去。
        return `${t(clipKindLabelKey(meta?.clipKind))} · ${meta?.title || entry.id}`;
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
            {isImage ? <ImageIcon /> : <ClipboardIcon />}
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
    const { t, tf } = useI18n();
    const [exportNotice, setExportNotice] = useState<string | null>(null);

    /**
     * 导出并给出反馈。
     *
     * 导出会弹系统保存对话框，因此结果必须回显（成功/取消/失败），否则用户
     * 无法区分"没点着"和"选了路径但写失败"。
     */
    async function runExport(extension: "md" | "html", content: string) {
        setExportNotice(tf("notebook_export_running"));
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
                // 已知稳定错误码（见后端 commands/notebook.rs）给完整本地化
                // 句子；未知错误沿用"前缀 + 原始码"的回显，方便用户原样报障。
                const errorKey = notebookErrorKey(result.error);
                setExportNotice(
                    errorKey
                        ? tf(errorKey)
                        : `${tf("notebook_export_failed")}: ${result.error ?? ""}`,
                );
                return;
            }
            const missing = result.missingAssets?.length ?? 0;
            setExportNotice(
                missing > 0
                    ? `${tf("notebook_export_done")} (${tf("notebook_export_missing")}: ${missing})`
                    : tf("notebook_export_done"),
            );
        } catch {
            setExportNotice(tf("notebook_export_failed"));
        }
    }

    return (
        <AppDialog
            open
            onOpenChange={(open) => !open && onClose()}
            title={t("notebook_settings")}
            description={tf("notebook_settings_desc")}
            size="md"
            // 内容远高于视口：动作区需要一条分割线与长内容分开。
            footerDivider
            actions={[
                {
                    id: "close",
                    label: t("close"),
                    intent: "primary",
                    onClick: () => onClose(),
                },
            ]}
        >
            {/*
             * 混排表单：字段与开关共用标签列，因此显式声明 aligned（与
             * `DockLayoutSettingsDialog` 同一形态）。
             *
             * 分组走 `AppFormSection`（节标题 13px/600 + 留白分组），不用就地
             * 手写的标题行 —— 那是全应用之外的第二种节标题，且 11px 会**小于**
             * 它所统领的 12px 字段标签。
             *
             * 【标签列取 lg（132px）】本窗口最长的一条是「Markdown 输入快捷转换」
             * （实测自然宽 131.9px），`md`（112px）装不下 —— 标签 `shrink-0`，
             * 多出的 20px 会把那一行自己的开关往右顶，整列控件因此破口。132 正是
             * 这一组设置迁移到原语之前自己用的档位（`Field.tsx` 里记的
             * `SETTING_LABEL_STYLE = 132`），迁移时按默认 `md` 下发才丢的。
             */}
            <AppForm booleanRow="aligned" labelWidth="lg">
                <AppFormSection title={tf("notebook_settings_group_view")}>
                    <AppField label={tf("notebook_setting_default_mode")}>
                        <AppSelect
                            value={settings.defaultMode}
                            ariaLabel={tf("notebook_setting_default_mode")}
                            onValueChange={(value) => onChange({ defaultMode: value })}
                            options={[
                                { value: "rich", label: t("notebook_mode_rich") },
                                { value: "source", label: t("notebook_mode_source") },
                                { value: "split", label: t("notebook_mode_split") },
                            ]}
                        />
                    </AppField>
                    <AppSwitchRow
                        label={tf("notebook_setting_toolbar")}
                        checked={settings.showToolbar}
                        onCheckedChange={(value) => onChange({ showToolbar: value })}
                    />
                    <AppSwitchRow
                        label={tf("notebook_setting_markdown_shortcuts")}
                        checked={settings.markdownShortcuts}
                        onCheckedChange={(value) => onChange({ markdownShortcuts: value })}
                    />
                    <AppSwitchRow
                        label={tf("notebook_setting_slash")}
                        checked={settings.slashCommands}
                        onCheckedChange={(value) => onChange({ slashCommands: value })}
                    />
                    <AppSwitchRow
                        label={tf("notebook_setting_word_wrap")}
                        checked={settings.sourceWordWrap}
                        onCheckedChange={(value) => onChange({ sourceWordWrap: value })}
                    />
                    <AppSwitchRow
                        label={tf("notebook_setting_spellcheck")}
                        checked={settings.spellCheck}
                        onCheckedChange={(value) => onChange({ spellCheck: value })}
                    />
                    {/*
                     * 右键菜单的详略。
                     *
                     * 一个下拉而不是三个开关："菜单要多全"是一根轴，拆成
                     * `enabled` / `showFormatting` / `showInsert` 能表达 8 种组合，
                     * 其中至少 3 种没有意义（关了总开关，另外两个仍在生效）。
                     * `off` 这一档是必需的退路：应用**全局禁用**了 WebView 原生
                     * 右键菜单，接管右键就得让用户能还回去。
                     */}
                    <AppField label={tf("notebook_setting_context_menu")}>
                        <AppSelect
                            value={settings.contextMenu}
                            ariaLabel={tf("notebook_setting_context_menu")}
                            onValueChange={(value) => onChange({ contextMenu: value })}
                            options={[
                                {
                                    value: "full",
                                    label: t("notebook_setting_context_menu_full"),
                                },
                                {
                                    value: "compact",
                                    label: t("notebook_setting_context_menu_compact"),
                                },
                                { value: "off", label: t("notebook_setting_context_menu_off") },
                            ]}
                        />
                    </AppField>
                    {/*
                     * 字号、撤销分节、图片长边都是**连续量**，因此是输入框而不是
                     * 下拉：下拉只能给出几个预设档位（字号原本 11/12/13/15/17），
                     * 用户想要 14 就得改配置文件。上下界与 `notebookSettings.ts`
                     * 的归一化一致，所以"输入框里看到的"就是"存下去的"。
                     *
                     * 只接 `onCommit`（失焦 / Enter 提交），不接 `onChange`：这几个
                     * 值没有实时预览的消费者，逐字符写 Redux + 落盘是白费。
                     */}
                    <AppField label={tf("notebook_setting_font_size")}>
                        <AppNumberField
                            value={settings.sourceFontSize}
                            unit="pixels"
                            min={NOTEBOOK_FONT_SIZE_MIN}
                            max={NOTEBOOK_FONT_SIZE_MAX}
                            suffix="px"
                            ariaLabel={tf("notebook_setting_font_size")}
                            onCommit={(value) => onChange({ sourceFontSize: value })}
                        />
                    </AppField>
                    <AppField
                        label={tf("notebook_setting_history_split")}
                        // 0 是**语义值**（不做空闲分节），不是"未设置"：必须写在
                        // 提示里，否则用户看到 0 会以为是坏了。
                        hint={`0 = ${tf("notebook_setting_history_split_off")}`}
                    >
                        <AppNumberField
                            value={settings.historySplitIdleMs}
                            // 以秒计量的等待时长：滚一格 1s（按住精细调整键 100ms）。
                            unit="idleMilliseconds"
                            min={0}
                            max={600000}
                            suffix="ms"
                            ariaLabel={tf("notebook_setting_history_split")}
                            onCommit={(value) => onChange({ historySplitIdleMs: value })}
                        />
                    </AppField>
                </AppFormSection>

                <AppFormSection title={tf("notebook_settings_group_image")}>
                    <AppField
                        label={tf("notebook_setting_image_max_dim")}
                        // 同上：0 = 不缩放，保留原图尺寸。
                        hint={`0 = ${tf("notebook_setting_image_max_dim_original")}`}
                    >
                        <AppNumberField
                            value={settings.imageMaxDimensionPx}
                            // 图片尺寸量级：滚一格 128px（按住精细调整键 8px）。
                            unit="imagePixels"
                            min={0}
                            max={16384}
                            suffix="px"
                            ariaLabel={tf("notebook_setting_image_max_dim")}
                            onCommit={(value) => onChange({ imageMaxDimensionPx: value })}
                        />
                    </AppField>
                    <AppField label={tf("notebook_setting_image_format")}>
                        <AppSelect
                            value={settings.imageFormat}
                            ariaLabel={tf("notebook_setting_image_format")}
                            onValueChange={(value) => onChange({ imageFormat: value })}
                            options={[
                                {
                                    value: "auto",
                                    label: tf("notebook_setting_image_format_auto"),
                                },
                                { value: "webp", label: "WebP" },
                                { value: "jpeg", label: "JPEG" },
                                { value: "png", label: "PNG" },
                            ]}
                        />
                    </AppField>
                    <AppSwitchRow
                        label={tf("notebook_setting_remote_images")}
                        checked={settings.allowRemoteImages}
                        onCheckedChange={(value) => onChange({ allowRemoteImages: value })}
                    />
                </AppFormSection>

                <AppFormSection title={tf("notebook_settings_group_clipboard")}>
                    <AppSwitchRow
                        label={tf("notebook_setting_smart_paste")}
                        checked={settings.smartPaste}
                        onCheckedChange={(value) => onChange({ smartPaste: value })}
                    />
                    <AppField label={tf("notebook_setting_html_paste")}>
                        <AppSelect
                            value={settings.htmlPasteMode}
                            ariaLabel={tf("notebook_setting_html_paste")}
                            onValueChange={(value) => onChange({ htmlPasteMode: value })}
                            options={[
                                { value: "markdown", label: "Markdown" },
                                { value: "html", label: "HTML" },
                                { value: "text", label: tf("notebook_setting_paste_text") },
                            ]}
                        />
                    </AppField>
                    <AppField label={tf("notebook_setting_plain_paste")}>
                        <AppSelect
                            value={settings.plainPasteMode}
                            ariaLabel={tf("notebook_setting_plain_paste")}
                            onValueChange={(value) => onChange({ plainPasteMode: value })}
                            options={[
                                {
                                    value: "auto",
                                    label: tf("notebook_setting_plain_paste_auto"),
                                },
                                { value: "markdown", label: "Markdown" },
                                { value: "text", label: tf("notebook_setting_paste_text") },
                            ]}
                        />
                    </AppField>
                    <AppField label={tf("notebook_setting_copy_format")}>
                        <AppSelect
                            value={settings.copyFormat}
                            ariaLabel={tf("notebook_setting_copy_format")}
                            onValueChange={(value) => onChange({ copyFormat: value })}
                            options={[
                                {
                                    value: "markdown+html",
                                    label: tf("notebook_setting_copy_format_both"),
                                },
                                { value: "markdown", label: "Markdown" },
                                { value: "html", label: "HTML" },
                                { value: "text", label: tf("notebook_setting_paste_text") },
                            ]}
                        />
                    </AppField>
                    <AppField label={tf("notebook_setting_copy_plain")}>
                        <AppSelect
                            value={settings.copyPlainTextAs}
                            ariaLabel={tf("notebook_setting_copy_plain")}
                            onValueChange={(value) => onChange({ copyPlainTextAs: value })}
                            options={[
                                {
                                    value: "markdown",
                                    label: tf("notebook_setting_copy_plain_markdown"),
                                },
                                { value: "text", label: tf("notebook_setting_paste_text") },
                            ]}
                        />
                    </AppField>
                </AppFormSection>

                <AppFormSection title={tf("notebook_settings_group_clip_block")}>
                    <AppField label={tf("notebook_setting_clip_insert_mode")}>
                        <AppSelect
                            value={settings.clipInsertMode}
                            ariaLabel={tf("notebook_setting_clip_insert_mode")}
                            onValueChange={(value) => onChange({ clipInsertMode: value })}
                            options={[
                                {
                                    value: "selected",
                                    label: tf("notebook_setting_clip_insert_selected"),
                                },
                                {
                                    value: "newTracks",
                                    label: tf("notebook_setting_clip_insert_new_tracks"),
                                },
                            ]}
                        />
                    </AppField>
                    <AppSwitchRow
                        label={tf("notebook_setting_keep_clip")}
                        checked={settings.keepClipAfterInsert}
                        onCheckedChange={(value) => onChange({ keepClipAfterInsert: value })}
                    />
                    <AppSwitchRow
                        label={tf("notebook_setting_clip_preview")}
                        checked={settings.clipShowPreview}
                        onCheckedChange={(value) => onChange({ clipShowPreview: value })}
                    />
                </AppFormSection>

                {/*
                 * 两个导出按钮挂在节头的 `action` 槽位（分区级控件），而不是挤在
                 * 正文里另起一行：这一节只有按钮、没有字段，正文若再画一行，
                 * 它们要么贴左（与上面四节的控件列错位），要么为了对齐凭空缩进
                 * 一个 112px —— 两种都不如放进槽位。与 `AppearanceSettingsPanel`
                 * 的「已保存主题」节（导入/导出在节头）同一形态。
                 */}
                <AppFormSection
                    title={tf("notebook_settings_group_export")}
                    action={
                        <div className="flex items-center gap-2">
                            <AppButton
                                size="sm"
                                onClick={() => {
                                    void runExport("md", markdown);
                                }}
                            >
                                {tf("notebook_export_md")}
                            </AppButton>
                            <AppButton
                                size="sm"
                                onClick={() => {
                                    void runExport(
                                        "html",
                                        buildExportHtml(markdown, projectName, getHtml?.() ?? null),
                                    );
                                }}
                            >
                                {tf("notebook_export_html")}
                            </AppButton>
                        </div>
                    }
                >
                    {/*
                     * 导出反馈挂在本节正文（按钮正下方），而不是表单末尾：
                     * 此前它是表单最后一个子项，与触发它的按钮隔着一整个节间距，
                     * 读不出"这句话是刚才那次导出的结果"。
                     */}
                    {exportNotice ? <span className="hs-type-caption">{exportNotice}</span> : null}
                </AppFormSection>
            </AppForm>
        </AppDialog>
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
        /*
         * 【这里的颜色是**故意**写死的】这段 CSS 属于**导出物**（一份自包含的
         * 独立 HTML 文档，用户在浏览器里打开），不是应用内预览 —— 它不该跟随
         * 应用主题，浅色排版是对文档的正确选择。不要把它换成 --qt-* 令牌。
         */
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
