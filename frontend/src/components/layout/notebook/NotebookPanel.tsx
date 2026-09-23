/*
 * 记事本面板。
 *
 * 三视图（富文本 / Markdown 源码 / 分栏）+ 格式化工具栏 + 查找 + 附件管理 +
 * 导出，外加图片拖放/粘贴与剪贴板暂存块。
 *
 * 【与旧版的区别】旧版是"textarea 编辑 + 只读预览"，现在是**以 Markdown 为
 * 唯一真源的富文本编辑器**：默认打开就是可交互编辑的富文本视图，源码视图
 * 退居"专家模式"。
 *
 * 【状态边界】
 * - 正文只在 `session.project.notesMarkdown`（与后端同名字段，随工程保存）；
 * - 视图模式与设置在本面板的 `notebook` 切片里；
 * - 附件索引只存摘要（id/大小/元数据），字节永远留在后端磁盘上。
 */

import { EditorContent } from "@tiptap/react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { useAppDispatch, useAppSelector } from "../../../app/hooks";
import { store } from "../../../app/store";
import { closeFormById } from "../../../features/dock/dockApi";
import { PANEL_NOTEBOOK } from "../../dock/registerBuiltinPanels";
import {
    setNotebookAssetIndex,
    setNotebookMode,
    setNotebookSettings,
    type NotebookMode,
} from "../../../features/notebook/notebookSlice";
import {
    redoRemote,
    setProjectNotesMarkdown,
    setplayheadSec,
    undoRemote,
} from "../../../features/session/sessionSlice";
import { selectClipRemote } from "../../../features/session/thunks/timelineThunks";
import { seekPlayhead } from "../../../features/session/thunks/transportThunks";
import { useI18n } from "../../../i18n/I18nProvider";
import { notebookApi } from "../../../services/api/notebook";
import { settingsApi } from "../../../services/api/settings";
import { webApi } from "../../../services/webviewApi";
import { NotebookAttachmentsDialog, NotebookSettingsDialog } from "./NotebookDialogs";
import { NotebookFindBar } from "./NotebookFindBar";
import { NotebookReadonlyPreview } from "./NotebookReadonlyPreview";
import { NotebookStatusBar } from "./NotebookStatusBar";
import { NotebookToolbar } from "./NotebookToolbar";
import { handleNotebookPaste, installClipboardFlavorWriter, stageClipboardPayload } from "./notebookClipboard";
import { clearImageCache } from "./notebookImageCache";
import {
    insertImageFromBlob,
    insertImageFromPath,
    insertMarkdown,
    type InsertContext,
} from "./notebookInsert";
import { dirName } from "./notebookPaths";
import { buildClipLink, buildSeekLink, formatTimecode, parseInternalLink } from "./timecode";
import { useNotebookEditor } from "./useNotebookEditor";
import "./notebook.css";

export function NotebookPanel() {
    const dispatch = useAppDispatch();
    const { t } = useI18n();
    const mode = useAppSelector((state) => state.notebook.mode);
    const settings = useAppSelector((state) => state.notebook.settings);
    const assetIndex = useAppSelector((state) => state.notebook.assetIndex);
    const markdown = useAppSelector((state) => state.session.project.notesMarkdown);
    const projectPath = useAppSelector((state) => state.session.project.path);
    const projectName = useAppSelector((state) => state.session.project.name);
    const baseScale = useAppSelector((state) => state.session.project.baseScale);
    const dirty = useAppSelector((state) => state.session.project.dirty);
    const playheadSec = useAppSelector((state) => state.session.playheadSec);
    const selectedClipId = useAppSelector((state) => state.session.selectedClipId);
    const clips = useAppSelector((state) => state.session.clips);
    const bpm = useAppSelector((state) => state.session.bpm);

    const [findOpen, setFindOpen] = useState(false);
    const [attachmentsOpen, setAttachmentsOpen] = useState(false);
    const [settingsOpen, setSettingsOpen] = useState(false);
    const [notice, setNotice] = useState<string | null>(null);
    const [dropActive, setDropActive] = useState(false);

    const projectDir = useMemo(() => (projectPath ? dirName(projectPath) : null), [projectPath]);
    const containerRef = useRef<HTMLDivElement | null>(null);
    const richScrollRef = useRef<HTMLDivElement | null>(null);
    const sourceTextareaRef = useRef<HTMLTextAreaElement | null>(null);
    const fileInputRef = useRef<HTMLInputElement | null>(null);

    const notify = useCallback((message: string) => {
        setNotice(message);
        window.setTimeout(() => setNotice((current) => (current === message ? null : current)), 2600);
    }, []);

    const refreshAssets = useCallback(async () => {
        try {
            const result = await notebookApi.listAssets();
            if (result.ok && Array.isArray(result.assets)) {
                dispatch(setNotebookAssetIndex(result.assets));
            }
        } catch {
            // 附件索引只是"存在吗 / 多大"的缓存：拿不到时界面照常工作，只是
            // 暂存块会显示为"载荷缺失"。绝不能让它把面板带崩。
        }
    }, [dispatch]);

    // ── 设置加载（一次）────────────────────────────────────────────
    useEffect(() => {
        let cancelled = false;
        void settingsApi
            .getUiSettings()
            .then((ui) => {
                if (!cancelled) dispatch(setNotebookSettings(ui.notebook));
            })
            .catch(() => {
                // 读不到就用默认值，不阻塞记事本使用。
            });
        return () => {
            cancelled = true;
        };
    }, [dispatch]);

    // ── 附件索引：挂载时与切换工程时刷新 ───────────────────────────
    useEffect(() => {
        // 换工程意味着附件 id 空间换了，图片缓存必须整体作废。
        clearImageCache();
        void refreshAssets();
    }, [projectPath, refreshAssets]);

    // ── 编辑器 ─────────────────────────────────────────────────────
    const persist = useCallback((value: string) => {
        void webApi.setProjectNotes(value).catch(() => {
            // 后端写入失败由保存路径兜底（保存时会把当前正文一并带上）。
        });
    }, []);

    const onMarkdownChange = useCallback(
        (value: string) => {
            dispatch(setProjectNotesMarkdown(value));
        },
        [dispatch],
    );

    const bridge = useMemo(
        () => ({
            undo: () => void dispatch(undoRemote()),
            redo: () => void dispatch(redoRemote()),
        }),
        [dispatch],
    );

    const { editor, flush, seal } = useNotebookEditor({
        markdown,
        settings,
        placeholder: t("notebook_placeholder"),
        bridge,
        onMarkdownChange,
        persist,
        onIdleSplit: () => void notebookApi.sealNotesHistory().catch(() => {}),
    });

    const insertContext = useMemo<InsertContext | null>(
        () =>
            editor ? { editor, settings, projectDir, notify, onAssetsChanged: () => void refreshAssets() } : null,
        [editor, notify, projectDir, refreshAssets, settings],
    );

    // 编辑器挂载后：装剪贴板 flavor 写出器。
    //
    // 直接挂 `editor.view.dom`：它晚于 ProseMirror 自己的 copy 监听注册，
    // 因此一定在其写完 `text/html` / `text/plain` 之后补 flavor；用类名查
    // DOM 反而会因 `editorProps.attributes` 覆盖默认 class 而落空。
    useEffect(() => {
        if (!editor || editor.isDestroyed) return;
        return installClipboardFlavorWriter(
            editor.view.dom,
            () => ({ copyFormat: settings.copyFormat, copyPlainTextAs: settings.copyPlainTextAs }),
            () => editor,
        );
    }, [editor, settings.copyFormat, settings.copyPlainTextAs]);

    // 粘贴分流：捕获阶段先于 ProseMirror 自己的粘贴处理。
    useEffect(() => {
        const element = containerRef.current;
        if (!element || !editor || editor.isDestroyed || mode === "source") return;
        const handler = (event: Event) => {
            const clipboardEvent = event as ClipboardEvent;
            const ctx: InsertContext = {
                editor,
                settings,
                projectDir,
                notify,
                onAssetsChanged: () => void refreshAssets(),
            };
            if (handleNotebookPaste({ ...ctx, sourceMode: false }, clipboardEvent)) {
                clipboardEvent.preventDefault();
            }
        };
        element.addEventListener("paste", handler, true);
        return () => element.removeEventListener("paste", handler, true);
    }, [editor, mode, notify, projectDir, refreshAssets, settings]);

    // ── 内部链接点击（跳播放头 / 引用 Clip）────────────────────────
    useEffect(() => {
        const element = containerRef.current;
        if (!element) return;
        const handler = (event: MouseEvent) => {
            const anchor = (event.target as HTMLElement | null)?.closest?.("a[href]");
            if (!anchor) return;
            const link = parseInternalLink(anchor.getAttribute("href") ?? "");
            if (!link) return;
            event.preventDefault();
            event.stopPropagation();
            if (link.type === "seek") {
                dispatch(setplayheadSec(link.seconds));
                void dispatch(seekPlayhead(link.seconds));
                notify(`${t("notebook_seek_jump")} ${formatTimecode(link.seconds)}`);
                return;
            }
            const clip = clips.find((entry) => entry.id === link.clipId);
            if (clip) {
                dispatch(setplayheadSec(clip.startSec));
                void dispatch(seekPlayhead(clip.startSec));
            }
            void dispatch(selectClipRemote(link.clipId));
        };
        element.addEventListener("click", handler, true);
        return () => element.removeEventListener("click", handler, true);
    }, [clips, dispatch, notify, t]);

    /**
     * 窗口级拖放的落点判定与插入，放进 ref 供注册一次的监听调用。
     *
     * 之所以要 ref：注册是异步的（`onDragDropEvent` 返回 Promise），而
     * cleanup 可能在 await 落地之前就跑完（StrictMode 的挂载-卸载-再挂载）。
     * 若把 `unlisten` 存在闭包变量里，晚到的注册永远不会被注销 —— 每次开关
     * 面板都会多堆一个窗口级监听。用 ref 还有一个好处：注册只做一次，不必
     * 因 settings / projectDir 变化而重新注册。
     */
    const dropHandlerRef = useRef<
        (payload: {
            type?: string;
            paths?: string[];
            position?: { x?: number; y?: number };
        }) => void
    >(() => {});

    // ── 拖放（Tauri 原生）──────────────────────────────────────────
    useEffect(() => {
        let disposed = false;
        let cancelled = false;
        let unlisten: (() => void) | null = null;
        void (async () => {
            try {
                const mod = await import("@tauri-apps/api/window");
                const win = mod.getCurrentWindow();
                const off = await win.onDragDropEvent((event) => {
                    if (disposed) return;
                    const payload = ("payload" in event ? event.payload : event) as {
                        type?: string;
                        event?: string;
                        paths?: string[];
                        position?: { x?: number; y?: number };
                        pos?: { x?: number; y?: number };
                        cursorPosition?: { x?: number; y?: number };
                    };
                    dropHandlerRef.current({
                        type: String(payload?.type ?? payload?.event ?? ""),
                        paths: Array.isArray(payload?.paths) ? payload.paths : [],
                        // 位置字段名跨平台不一致（与时间轴同一套回退链）。
                        position: payload?.position ?? payload?.pos ?? payload?.cursorPosition,
                    });
                });
                if (cancelled) {
                    off();
                    return;
                }
                unlisten = off;
            } catch {
                // 非 Tauri 环境（浏览器里跑前端）时只保留 HTML5 拖放。
            }
        })();
        return () => {
            disposed = true;
            cancelled = true;
            unlisten?.();
            unlisten = null;
        };
    }, []);

    useEffect(() => {
        dropHandlerRef.current = (payload) => {
            if (!editor || editor.isDestroyed || mode === "source") return;
            const type = payload.type ?? "";
            const paths = payload.paths ?? [];
            const position = payload.position;
            const rect = containerRef.current?.getBoundingClientRect();
            const dpr = window.devicePixelRatio || 1;
            const clientX = typeof position?.x === "number" ? position.x / dpr : null;
            const clientY = typeof position?.y === "number" ? position.y / dpr : null;
            const inside =
                rect != null &&
                clientX != null &&
                clientY != null &&
                clientX >= rect.left &&
                clientX <= rect.right &&
                clientY >= rect.top &&
                clientY <= rect.bottom;

            if (type === "enter" || type === "over") {
                if (inside) setDropActive(true);
                return;
            }
            if (type === "leave") {
                setDropActive(false);
                return;
            }
            if (type !== "drop") return;
            setDropActive(false);
            if (!inside) return;
            // 时间轴那边同样监听窗口级拖放，但它按自己的矩形判定归属，且非媒体
            // 扩展名会被它的白名单拒绝 —— 图片不会被误导入。

            // 落点即插入点：拖到哪儿就插到哪儿。
            if (clientX != null && clientY != null) {
                const pos = editor.view.posAtCoords({ left: clientX, top: clientY });
                if (pos) editor.chain().focus().setTextSelection(pos.pos).run();
            }
            const ctx: InsertContext = {
                editor,
                settings,
                projectDir,
                notify,
                onAssetsChanged: () => void refreshAssets(),
            };
            void (async () => {
                try {
                    for (const path of paths) {
                        if (!looksLikeImagePath(path)) continue;
                        await insertImageFromPath(ctx, path);
                    }
                } catch {
                    notify(t("notebook_image_insert_failed"));
                }
            })();
        };
    }, [editor, mode, notify, projectDir, refreshAssets, settings, t]);

    const onHtml5Drop = useCallback(
        (event: React.DragEvent<HTMLDivElement>) => {
            setDropActive(false);
            if (!insertContext) return;
            const files = Array.from(event.dataTransfer?.files ?? []).filter((file) =>
                file.type.startsWith("image/"),
            );
            if (files.length === 0) return;
            event.preventDefault();
            void (async () => {
                for (const file of files) {
                    await insertImageFromBlob(insertContext, file, file.name);
                }
            })();
        },
        [insertContext],
    );

    // ── 插入动作 ───────────────────────────────────────────────────
    const handlers = useMemo(
        () => ({
            insertImage: () => fileInputRef.current?.click(),
            insertTimecode: () => {
                if (insertContext) insertMarkdown(insertContext, buildSeekLink(playheadSec) + " ");
            },
            insertClipReference: () => {
                if (!insertContext) return;
                const clip = clips.find((entry) => entry.id === selectedClipId);
                if (!clip) {
                    notify(t("notebook_clip_ref_none"));
                    return;
                }
                insertMarkdown(insertContext, buildClipLink(clip.id, clip.name) + " ");
            },
            insertProjectInfo: () => {
                if (!insertContext) return;
                const today = new Date().toISOString().slice(0, 10);
                const rows = [
                    `| ${t("notebook_project_info_item")} | ${t("notebook_project_info_value")} |`,
                    "| --- | --- |",
                    `| ${t("notebook_project_info_name")} | ${projectName} |`,
                    `| ${t("notebook_project_info_bpm")} | ${bpm} |`,
                    `| ${t("notebook_project_info_scale")} | ${baseScale} |`,
                    `| ${t("notebook_project_info_date")} | ${today} |`,
                ];
                insertMarkdown(insertContext, "\n\n" + rows.join("\n") + "\n\n");
            },
            stageClipboard: () => {
                if (!insertContext) return;
                void (async () => {
                    const staged = await stageClipboardPayload();
                    if (!staged.ok || !staged.body) {
                        notify(t("notebook_clipboard_unavailable"));
                        return;
                    }
                    editor
                        ?.chain()
                        .focus()
                        .insertContent({ type: "hifiClipBlock", attrs: { body: staged.body } })
                        .run();
                    await refreshAssets();
                    notify(t("notebook_clipboard_staged"));
                })();
            },
        }),
        [
            baseScale,
            bpm,
            clips,
            editor,
            insertContext,
            notify,
            playheadSec,
            projectName,
            refreshAssets,
            selectedClipId,
            t,
        ],
    );

    const onFilePicked = useCallback(
        (event: React.ChangeEvent<HTMLInputElement>) => {
            const files = Array.from(event.target.files ?? []);
            event.target.value = "";
            if (!insertContext) return;
            void (async () => {
                for (const file of files) {
                    await insertImageFromBlob(insertContext, file, file.name);
                }
            })();
        },
        [insertContext],
    );

    // ── 模式切换 / 关闭：收尾并分节 ────────────────────────────────
    const changeMode = useCallback(
        (next: NotebookMode) => {
            if (next === mode) return;
            seal();
            dispatch(setNotebookMode(next));
            void settingsApi
                .saveUiSettings({ notebook: { ...settings, defaultMode: next } })
                .catch(() => {});
        },
        [dispatch, mode, seal, settings],
    );

    const close = useCallback(() => {
        seal();
        // 显隐归停靠布局管：面板只请求"关掉我自己"。
        closeFormById(dispatch, store.getState, PANEL_NOTEBOOK);
    }, [dispatch, seal]);

    // Ctrl+F 面板内查找（全局 Ctrl+F 被 app 拦截，这里自己接）。
    useEffect(() => {
        const element = containerRef.current;
        if (!element) return;
        const handler = (event: KeyboardEvent) => {
            if (!(event.ctrlKey || event.metaKey) || event.key.toLowerCase() !== "f") return;
            event.preventDefault();
            event.stopPropagation();
            setFindOpen(true);
        };
        element.addEventListener("keydown", handler, true);
        return () => element.removeEventListener("keydown", handler, true);
    }, []);

    // 保存前收尾：工程保存会把当前正文一并带上，但待写内容也要落盘，
    // 否则"刚打完字就保存"会出现后端与工程文件不一致。
    useEffect(() => {
        return () => flush();
    }, [flush]);

    const usedAssetIds = useMemo(() => {
        const ids = new Set<string>();
        for (const match of markdown.matchAll(/hifi-asset:\/\/([A-Za-z0-9_-]+)/g)) ids.add(match[1]);
        for (const match of markdown.matchAll(/^id:\s*([A-Za-z0-9_-]+)\s*$/gm)) ids.add(match[1]);
        return ids;
    }, [markdown]);

    const imageCount = useMemo(
        () => Object.values(assetIndex).filter((entry) => entry.kind === "image").length,
        [assetIndex],
    );
    const clipCount = useMemo(
        () => Object.values(assetIndex).filter((entry) => entry.kind === "clip_payload").length,
        [assetIndex],
    );
    const totalBytes = useMemo(
        () => Object.values(assetIndex).reduce((sum, entry) => sum + (entry.byteLen || 0), 0),
        [assetIndex],
    );

    const setSetting = useCallback(
        (patch: Record<string, unknown>) => {
            const merged = { ...settings, ...patch };
            dispatch(setNotebookSettings(merged));
            void settingsApi.saveUiSettings({ notebook: merged }).catch(() => {});
        },
        [dispatch, settings],
    );

    return (
        <div
            ref={containerRef}
            className="flex h-full min-h-0 flex-col bg-qt-window"
            data-drop-active={dropActive ? "true" : "false"}
            // 字号同时作用于富文本与源码视图：两种视图的字号不一致会让模式
            // 切换时"跳一下"，也让人怀疑内容变了。
            style={
                {
                    "--hs-notebook-font-size": `${settings.sourceFontSize}px`,
                } as React.CSSProperties
            }
            onDragOver={(event) => {
                if (Array.from(event.dataTransfer?.items ?? []).some((item) => item.kind === "file")) {
                    event.preventDefault();
                    setDropActive(true);
                }
            }}
            onDragLeave={() => setDropActive(false)}
            onDrop={onHtml5Drop}
        >
            <div className="flex shrink-0 items-center justify-between gap-2 border-b border-qt-border px-2 py-1.5">
                <span className="truncate text-xs font-medium text-qt-text">{t("notebook")}</span>
                <div className="flex shrink-0 items-center gap-1">
                    <ModeButton
                        active={mode === "rich"}
                        label={t("notebook_mode_rich")}
                        onClick={() => changeMode("rich")}
                    />
                    <ModeButton
                        active={mode === "source"}
                        label={t("notebook_mode_source")}
                        onClick={() => changeMode("source")}
                    />
                    <ModeButton
                        active={mode === "split"}
                        label={t("notebook_mode_split")}
                        onClick={() => changeMode("split")}
                    />
                    <HeaderButton
                        label={settings.showToolbar ? "▾" : "▸"}
                        title={t("notebook_toggle_toolbar")}
                        onClick={() => setSetting({ showToolbar: !settings.showToolbar })}
                    />
                    <HeaderButton
                        label="🗂"
                        title={t("notebook_attachments")}
                        onClick={() => setAttachmentsOpen(true)}
                    />
                    <HeaderButton
                        label="⚙"
                        title={t("notebook_settings")}
                        onClick={() => setSettingsOpen(true)}
                    />
                    <HeaderButton label="✕" title={t("close")} onClick={close} />
                </div>
            </div>

            {findOpen ? (
                <NotebookFindBar
                    editor={editor}
                    sourceMode={mode === "source"}
                    sourceValue={markdown}
                    getSourceTextarea={() => sourceTextareaRef.current}
                    onClose={() => setFindOpen(false)}
                />
            ) : null}

            {mode === "rich" && settings.showToolbar && editor ? (
                <NotebookToolbar editor={editor} handlers={handlers} slashCommands={settings.slashCommands} />
            ) : null}

            <div className="relative flex min-h-0 flex-1">
                {mode === "source" ? (
                    <textarea
                        ref={sourceTextareaRef}
                        className="hs-notebook-source"
                        data-wrap={settings.sourceWordWrap ? "true" : "false"}
                        style={
                            {
                                "--hs-notebook-source-font-size": `${settings.sourceFontSize}px`,
                                whiteSpace: settings.sourceWordWrap ? "pre-wrap" : "pre",
                            } as React.CSSProperties
                        }
                        value={markdown}
                        spellCheck={settings.spellCheck}
                        placeholder={t("notebook_placeholder")}
                        onChange={(event) => {
                            dispatch(setProjectNotesMarkdown(event.target.value));
                            persist(event.target.value);
                        }}
                    />
                ) : null}

                {mode !== "source" ? (
                    <div ref={richScrollRef} className="min-w-0 flex-1 overflow-auto bg-qt-base px-3 py-3">
                        {editor ? <EditorContent editor={editor} /> : null}
                    </div>
                ) : null}

                {mode === "split" ? (
                    <NotebookReadonlyPreview
                        markdown={markdown}
                        placeholder={t("notebook_placeholder")}
                        scrollSyncSource={richScrollRef.current}
                    />
                ) : null}

                {dropActive ? (
                    <div className="hs-notebook-drop-active pointer-events-none absolute inset-0" />
                ) : null}
            </div>

            <NotebookStatusBar
                editor={editor}
                imageCount={imageCount}
                clipCount={clipCount}
                totalBytes={totalBytes}
                dirty={dirty}
                notice={notice}
            />

            <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                multiple
                className="hidden"
                onChange={onFilePicked}
            />

            {attachmentsOpen ? (
                <NotebookAttachmentsDialog
                    onClose={() => setAttachmentsOpen(false)}
                    usedAssetIds={usedAssetIds}
                    assetIndex={assetIndex}
                    onChanged={() => void refreshAssets()}
                    notify={notify}
                />
            ) : null}

            {settingsOpen ? (
                <NotebookSettingsDialog
                    settings={settings}
                    onClose={() => setSettingsOpen(false)}
                    onChange={setSetting}
                    markdown={markdown}
                    projectName={projectName}
                    getHtml={() => (editor ? editor.getHTML() : null)}
                />
            ) : null}
        </div>
    );
}

function ModeButton({ active, label, onClick }: { active: boolean; label: string; onClick: () => void }) {
    return (
        <button
            type="button"
            onClick={onClick}
            className="rounded px-1.5 py-0.5 text-xs"
            style={{
                background: active ? "var(--accent-3, rgba(79,142,247,0.25))" : "transparent",
                color: active ? "var(--accent-11, #4f8ef7)" : "var(--qt-text-muted)",
            }}
        >
            {label}
        </button>
    );
}

function HeaderButton({ label, title, onClick }: { label: string; title: string; onClick: () => void }) {
    return (
        <button
            type="button"
            title={title}
            aria-label={title}
            onClick={onClick}
            className="rounded px-1 text-xs text-qt-text-muted hover:bg-qt-hover hover:text-qt-text"
        >
            {label}
        </button>
    );
}

/** 扩展名判据（拖入文件时过滤掉非图片）。 */
function looksLikeImagePath(path: string): boolean {
    return /\.(png|jpe?g|gif|webp|bmp|avif|svg)$/i.test(path);
}
