/*
 * 记事本设置：类型、默认值、归一化。
 *
 * 独立成文件而不是塞进 `services/api/settings.ts`：这一组设置项有 20 多个，
 * 且每个都需要"非法值退回默认"的收敛逻辑 —— 混在通用 UI 设置里会让两边都
 * 难以阅读。`UiSettings.notebook` 只是一个可缺省的嵌套对象，旧配置文件里
 * 没有它，归一化会补全默认值，因此新增字段永远不需要配置迁移。
 */

import type { NotebookImageFormatSetting } from "./notebookImagePipeline";

export interface NotebookSettings {
    /** 打开记事本时的默认视图模式。 */
    defaultMode?: "rich" | "source" | "split";
    /** 面板宽度（像素）。 */
    panelWidth?: number;
    /** 是否显示格式化工具栏。 */
    showToolbar?: boolean;
    /** Markdown 输入规则（`## ` / `- ` / `> ` 即时转换）。 */
    markdownShortcuts?: boolean;
    /** `/` 唤出插入菜单。 */
    slashCommands?: boolean;
    /** 源码模式软换行。 */
    sourceWordWrap?: boolean;
    /** 源码模式字号。 */
    sourceFontSize?: number;
    /** 拼写检查（项目笔记里罗马音/术语多，默认关）。 */
    spellCheck?: boolean;
    /** 图片存储模式。 */
    imageStorage?: "sidecar" | "embed" | "link";
    /** 图片长边上限（0 = 原图）。 */
    imageMaxDimensionPx?: number;
    /** 图片编码格式。 */
    imageFormat?: NotebookImageFormatSetting;
    /** 单图字节上限（超出拒绝插入）。 */
    maxImageBytes?: number;
    /** 是否允许加载 http(s) 外链图片。 */
    allowRemoteImages?: boolean;
    /** 智能粘贴总开关。 */
    smartPaste?: boolean;
    /** HTML 粘贴策略。 */
    htmlPasteMode?: "markdown" | "html" | "text";
    /** 纯文本粘贴策略。 */
    plainPasteMode?: "auto" | "markdown" | "text";
    /** 复制时写入的 flavor 组合。 */
    copyFormat?: "markdown+html" | "markdown" | "html" | "text";
    /** `text/plain` 里放 Markdown 源码还是渲染后纯文本。 */
    copyPlainTextAs?: "markdown" | "text";
    /** 暂存块插入时间轴时的轨道模式。 */
    clipInsertMode?: "selected" | "newTracks" | "ask";
    /** 插入后是否保留暂存块（中转站语义）。 */
    keepClipAfterInsert?: boolean;
    /** 暂存块内是否画示意预览。 */
    clipShowPreview?: boolean;
    /** 后端写入去抖（毫秒）。 */
    autosaveDebounceMs?: number;
    /** 编辑停顿多久后另起一个撤销步（0 = 沿用后端的结构性合并）。 */
    historySplitIdleMs?: number;
    /** 导出文档时的图片处理方式。 */
    exportImageMode?: "copyFolder" | "embed";
}

export type ResolvedNotebookSettings = Required<NotebookSettings>;

export const DEFAULT_NOTEBOOK_SETTINGS: ResolvedNotebookSettings = {
    defaultMode: "rich",
    panelWidth: 360,
    showToolbar: true,
    markdownShortcuts: true,
    slashCommands: true,
    sourceWordWrap: true,
    sourceFontSize: 13,
    spellCheck: false,
    imageStorage: "sidecar",
    imageMaxDimensionPx: 2048,
    imageFormat: "auto",
    maxImageBytes: 20 * 1024 * 1024,
    allowRemoteImages: true,
    smartPaste: true,
    htmlPasteMode: "markdown",
    plainPasteMode: "auto",
    copyFormat: "markdown+html",
    copyPlainTextAs: "markdown",
    clipInsertMode: "selected",
    keepClipAfterInsert: true,
    clipShowPreview: true,
    autosaveDebounceMs: 400,
    historySplitIdleMs: 0,
    exportImageMode: "copyFolder",
};

export const NOTEBOOK_PANEL_MIN_WIDTH = 260;
export const NOTEBOOK_PANEL_MAX_WIDTH = 760;

function pickEnum<T extends string>(value: unknown, allowed: readonly T[], fallback: T): T {
    return typeof value === "string" && (allowed as readonly string[]).includes(value)
        ? (value as T)
        : fallback;
}

function clampNumber(value: unknown, min: number, max: number, fallback: number): number {
    const parsed = typeof value === "number" ? value : Number(value);
    if (!Number.isFinite(parsed)) return fallback;
    return Math.min(max, Math.max(min, parsed));
}

/**
 * 收敛任意输入为一份合法的记事本设置。
 *
 * 所有枚举都用"白名单 + 回退默认"而不是直接采信：配置可能被手工编辑过，
 * 也可能来自未来版本（用户降级运行），任何未知取值都必须退化成合法值，
 * 否则会在渲染路径上炸掉。
 */
export function normalizeNotebookSettings(
    input?: NotebookSettings | null,
): ResolvedNotebookSettings {
    const raw = input ?? {};
    const d = DEFAULT_NOTEBOOK_SETTINGS;
    const bool = (value: unknown, fallback: boolean): boolean =>
        typeof value === "boolean" ? value : fallback;

    return {
        defaultMode: pickEnum(raw.defaultMode, ["rich", "source", "split"] as const, d.defaultMode),
        panelWidth: clampNumber(
            raw.panelWidth,
            NOTEBOOK_PANEL_MIN_WIDTH,
            NOTEBOOK_PANEL_MAX_WIDTH,
            d.panelWidth,
        ),
        showToolbar: bool(raw.showToolbar, d.showToolbar),
        markdownShortcuts: bool(raw.markdownShortcuts, d.markdownShortcuts),
        slashCommands: bool(raw.slashCommands, d.slashCommands),
        sourceWordWrap: bool(raw.sourceWordWrap, d.sourceWordWrap),
        sourceFontSize: clampNumber(raw.sourceFontSize, 9, 24, d.sourceFontSize),
        spellCheck: bool(raw.spellCheck, d.spellCheck),
        imageStorage: pickEnum(raw.imageStorage, ["sidecar", "embed", "link"] as const, d.imageStorage),
        // 0 是合法值（= 不缩放），因此下界是 0。
        imageMaxDimensionPx: clampNumber(raw.imageMaxDimensionPx, 0, 16384, d.imageMaxDimensionPx),
        imageFormat: pickEnum(
            raw.imageFormat,
            ["auto", "webp", "jpeg", "png"] as const,
            d.imageFormat,
        ),
        maxImageBytes: clampNumber(raw.maxImageBytes, 256 * 1024, 512 * 1024 * 1024, d.maxImageBytes),
        allowRemoteImages: bool(raw.allowRemoteImages, d.allowRemoteImages),
        smartPaste: bool(raw.smartPaste, d.smartPaste),
        htmlPasteMode: pickEnum(raw.htmlPasteMode, ["markdown", "html", "text"] as const, d.htmlPasteMode),
        plainPasteMode: pickEnum(
            raw.plainPasteMode,
            ["auto", "markdown", "text"] as const,
            d.plainPasteMode,
        ),
        copyFormat: pickEnum(
            raw.copyFormat,
            ["markdown+html", "markdown", "html", "text"] as const,
            d.copyFormat,
        ),
        copyPlainTextAs: pickEnum(raw.copyPlainTextAs, ["markdown", "text"] as const, d.copyPlainTextAs),
        clipInsertMode: pickEnum(
            raw.clipInsertMode,
            ["selected", "newTracks", "ask"] as const,
            d.clipInsertMode,
        ),
        keepClipAfterInsert: bool(raw.keepClipAfterInsert, d.keepClipAfterInsert),
        clipShowPreview: bool(raw.clipShowPreview, d.clipShowPreview),
        autosaveDebounceMs: clampNumber(raw.autosaveDebounceMs, 0, 5000, d.autosaveDebounceMs),
        historySplitIdleMs: clampNumber(raw.historySplitIdleMs, 0, 600000, d.historySplitIdleMs),
        exportImageMode: pickEnum(
            raw.exportImageMode,
            ["copyFolder", "embed"] as const,
            d.exportImageMode,
        ),
    };
}
