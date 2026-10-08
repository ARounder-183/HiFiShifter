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
import type { Editor } from "@tiptap/core";
import { CardStackIcon, ChevronDownIcon, ChevronRightIcon, GearIcon } from "@radix-ui/react-icons";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { useAppDispatch, useAppSelector } from "../../../app/hooks";
import { isPluginMode } from "../../../services/hostCapabilities";
import {
    setNotebookAssetIndex,
    setNotebookMode,
    setNotebookSettings,
    patchNotebookSettings,
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
import {
    isModifierActive,
    isNoneBinding,
    selectKeybinding,
} from "../../../features/keybindings/keybindingsSlice";
import { AppFileInput } from "../../../ui/FileInput";
import type { AppMenuItemSpec } from "../../../ui";
import { useNonPassiveWheel } from "../../../utils/useNonPassiveWheel";
import { useDebouncedCallback } from "../../../utils/useDebouncedCallback";
import { readWheelPixels } from "../timeline/kernel/input/normalizeWheel";
import {
    createWheelZoomAccumulator,
    resolveWheelZoomStep,
} from "../timeline/kernel/input/wheelZoomIntent";
import { useI18n } from "../../../i18n/I18nProvider";
import { PanelToolbar, PanelToolbarButton, PanelToolbarTextButton } from "../shared/PanelToolbar";
import { notebookApi } from "../../../services/api/notebook";
import { settingsApi } from "../../../services/api/settings";
import { webApi } from "../../../services/webviewApi";
import { NotebookAttachmentsDialog, NotebookSettingsDialog } from "./NotebookDialogs";
import { NotebookContextMenu } from "./NotebookContextMenu";
import { NotebookFindBar } from "./NotebookFindBar";
import { NotebookLinkEditor } from "./NotebookLinkEditor";
import { applyNotebookLink, clearNotebookLink, currentLinkHref } from "./notebookLinkEdit";
import { normalizeLinkHref } from "./notebookLinkUrl";
import { NotebookReadonlyPreview } from "./NotebookReadonlyPreview";
import type { NotebookPreviewContextRequest } from "./NotebookReadonlyPreview";
import { NotebookStatusBar } from "./NotebookStatusBar";
import { NotebookToolbar } from "./NotebookToolbar";
import {
    applyNotebookPastePayload,
    handleNotebookPaste,
    installClipboardFlavorWriter,
    isPlainInputTarget,
    stageClipboardPayload,
    type NotebookPasteOverrides,
} from "./notebookClipboard";
import { writeNotebookSelection, type NotebookCopyFlavorOverride } from "./notebookClipboardWrite";
import {
    prepareNotebookContext,
    prepareNotebookContextAtCaret,
    type NotebookMenuTarget,
} from "./notebookContextTarget";
import {
    buildNotebookContextMenu,
    type NotebookMenuActions,
    type NotebookMenuDisabled,
} from "./notebookMenu";
import { clearImageCache } from "./notebookImageCache";
import {
    insertImageFromBlob,
    insertImageFromClipboardBitmap,
    insertImageFromPath,
    insertMarkdown,
    insertTable,
    type InsertContext,
} from "./notebookInsert";
import { dirName } from "./notebookPaths";
import { normalizeNotebookSettings } from "./notebookSettings";
import { nextFontSizeForZoomStep, NOTEBOOK_ZOOM_LINE_HEIGHT_PX } from "./notebookFontZoom";
import { referencedAssetIds } from "./assetRef";
import {
    buildClipLink,
    buildSeekLink,
    formatTimecode,
    parseInternalLink,
    type NotebookInternalLink,
} from "./timecode";
import { runNotebookRedo, runNotebookUndo, useNotebookEditor } from "./useNotebookEditor";
import { copyTextToClipboard } from "../../../utils/copyText";
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
    /**
     * 右键菜单：坐标 + **已算好的**菜单项。
     *
     * 项在打开那一刻一次性算完（快照），不在渲染期重算 —— 菜单是弹出表面，
     * 外部 pointerdown 即关闭，生命周期内编辑器不会再变。这样也免去了给菜单挂
     * `editor.on("transaction")` 订阅。
     */
    const [menu, setMenu] = useState<{
        x: number;
        y: number;
        items: AppMenuItemSpec[];
    } | null>(null);

    const projectDir = useMemo(() => (projectPath ? dirName(projectPath) : null), [projectPath]);
    const containerRef = useRef<HTMLDivElement | null>(null);
    const richScrollRef = useRef<HTMLDivElement | null>(null);
    const sourceTextareaRef = useRef<HTMLTextAreaElement | null>(null);
    const fileInputRef = useRef<HTMLInputElement | null>(null);

    const notify = useCallback((message: string) => {
        setNotice(message);
        window.setTimeout(
            () => setNotice((current) => (current === message ? null : current)),
            2600,
        );
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
                if (cancelled) return;
                // 归一化一次再分发：设置进 slice，defaultMode 直接决定面板
                // 打开时的视图 —— 否则设置对话框里选的"默认视图"是死配置。
                const notebook = normalizeNotebookSettings(ui.notebook);
                dispatch(setNotebookSettings(notebook));
                dispatch(setNotebookMode(notebook.defaultMode));
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

    // 源码视图的写入同样要去抖：每次按键都调 `webApi.setProjectNotes` 会在后端
    // 登记一条可撤销历史，撤销栈瞬间被按键塞满。富文本路径由
    // `useNotebookEditor` 按 `autosaveDebounceMs` 去抖，这里为 textarea 复制
    // 同一套节流，并在失焦 / 切模式 / 卸载时把最后值冲刷出去（不丢编辑）。
    const sourcePersistTimerRef = useRef<number | null>(null);
    const sourcePendingRef = useRef<string | null>(null);

    const flushSourcePersist = useCallback(() => {
        if (sourcePersistTimerRef.current !== null) {
            window.clearTimeout(sourcePersistTimerRef.current);
            sourcePersistTimerRef.current = null;
        }
        const pending = sourcePendingRef.current;
        sourcePendingRef.current = null;
        if (pending !== null) persist(pending);
    }, [persist]);

    const scheduleSourcePersist = useCallback(
        (value: string) => {
            sourcePendingRef.current = value;
            if (sourcePersistTimerRef.current !== null) {
                window.clearTimeout(sourcePersistTimerRef.current);
            }
            sourcePersistTimerRef.current = window.setTimeout(() => {
                sourcePersistTimerRef.current = null;
                const pending = sourcePendingRef.current;
                sourcePendingRef.current = null;
                if (pending !== null) persist(pending);
            }, settings.autosaveDebounceMs);
        },
        [persist, settings.autosaveDebounceMs],
    );

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

    /*
     * 链接编辑浮层。
     *
     * 【状态为什么在面板而不在工具栏】工具栏是可收起的（`showToolbar` 设置），
     * 而 Ctrl/⌘+K 这个入口不该跟着消失。浮层因此由面板渲染、定位在编辑区上沿
     * （见 `NotebookLinkEditor`）—— 工具栏可见时它落在工具栏下方，收起时落在
     * 正文上沿。
     *
     * `editorRef` 的存在只为打破"回调要用 editor、editor 又来自用到回调的 hook"
     * 这个循环：快捷键经 ref 取最新实例（编辑器本身由 `useNotebookEditor` 持有，
     * 这里只是镜像）。
     */
    const editorRef = useRef<Editor | null>(null);
    const [linkDraft, setLinkDraft] = useState<string | null>(null);

    /** 打开浮层（工具栏按钮 / Ctrl+⌘+K）。初值取当前选区已有的链接地址。 */
    const openLinkEditor = useCallback(() => {
        const instance = editorRef.current;
        if (!instance || instance.isDestroyed) return;
        setLinkDraft(currentLinkHref(instance));
    }, []);

    /*
     * 链接浮层的入口对象。**身份必须稳定**（`openLinkEditor` 的依赖为空，
     * 因此这里也稳定）：它进 `useEditor` 的扩展依赖，一变编辑器就整体重建。
     */
    const linkBridge = useMemo(() => ({ open: openLinkEditor }), [openLinkEditor]);

    const closeLinkEditor = useCallback(() => setLinkDraft(null), []);

    const applyLinkDraft = useCallback(() => {
        const instance = editorRef.current;
        setLinkDraft(null);
        if (!instance || instance.isDestroyed) return;
        applyNotebookLink(instance, linkDraft ?? "");
    }, [linkDraft]);

    const { editor, flush, seal } = useNotebookEditor({
        markdown,
        settings,
        bridge,
        onMarkdownChange,
        persist,
        onIdleSplit: () => void notebookApi.sealNotesHistory().catch(() => {}),
        linkBridge,
    });

    useEffect(() => {
        editorRef.current = editor;
    }, [editor]);

    // 选区变化即收起浮层：浮层编辑的是"当前选区"的链接，选区一旦移走，
    // 再确认就会把链接贴到错误的位置上。
    useEffect(() => {
        if (linkDraft === null || !editor || editor.isDestroyed) return;
        const close = () => setLinkDraft(null);
        editor.on("selectionUpdate", close);
        return () => {
            editor.off("selectionUpdate", close);
        };
    }, [editor, linkDraft]);

    const insertContext = useMemo<InsertContext | null>(
        () =>
            editor
                ? {
                      editor,
                      settings,
                      projectDir,
                      notify,
                      translate: t,
                      onAssetsChanged: () => void refreshAssets(),
                  }
                : null,
        [editor, notify, projectDir, refreshAssets, settings, t],
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
            // 查找条 / alt 编辑器 / 暂存块改名 / 链接浮层都在本容器内：它们的
            // 粘贴必须走原生行为。捕获监听先于输入框自己的 handler 触发，这里
            // 不放行，粘贴就会落到正文里。
            if (isPlainInputTarget(clipboardEvent.target)) return;
            const ctx: InsertContext = {
                editor,
                settings,
                projectDir,
                notify,
                translate: t,
                onAssetsChanged: () => void refreshAssets(),
            };
            if (handleNotebookPaste({ ...ctx, sourceMode: false }, clipboardEvent)) {
                clipboardEvent.preventDefault();
            }
        };
        element.addEventListener("paste", handler, true);
        return () => element.removeEventListener("paste", handler, true);
    }, [editor, mode, notify, projectDir, refreshAssets, settings, t]);

    // ── 内部链接（跳播放头 / 引用 Clip）────────────────────────────
    //
    // 抽成回调是因为它有**两个**入口：正文里的点击拦截（下面那个 effect）与
    // 右键菜单的「打开链接」。两处若各写一遍，"跳播放头时要不要顺带提示"
    // 这类细节必然分叉。
    const runInternalLink = useCallback(
        (link: NotebookInternalLink) => {
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
        },
        [clips, dispatch, notify, t],
    );

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
            runInternalLink(link);
        };
        element.addEventListener("click", handler, true);
        return () => element.removeEventListener("click", handler, true);
    }, [runInternalLink]);

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
                const mod = await (
                    await import("../../../services/hostWindow")
                ).loadStandaloneWindowApi();
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
                translate: t,
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
        (files: File[]) => {
            if (!insertContext) return;
            void (async () => {
                for (const file of files) {
                    await insertImageFromBlob(insertContext, file, file.name);
                }
            })();
        },
        [insertContext],
    );

    // ── 右键菜单 ───────────────────────────────────────────────────
    //
    // 结构：`openXxxMenu` 负责"落点 + 开关快照 → 菜单项"，`buildMenuActions`
    // 负责"这个落点能做什么"。内容规则全在 `notebookMenu.ts`（纯函数、可单测），
    // 这里只做接线。
    const copySelection = useCallback(
        async (instance: Editor | null, cut: boolean, override?: NotebookCopyFlavorOverride) => {
            if (!instance || instance.isDestroyed) return;
            const written = await writeNotebookSelection(instance, settings, override);
            if (!written) {
                // 写不进去就**不要删**：剪切失败还照删，等于把用户的内容吃掉。
                notify(t("notebook_ctx_copy_failed"));
                return;
            }
            if (cut) instance.chain().focus().deleteSelection().run();
        },
        [notify, settings, t],
    );

    /**
     * 从系统剪贴板粘贴。
     *
     * 【为什么只读文本】浏览器不允许脚本在用户手势之外读剪贴板，能拿到的只有
     * `readText()`；`text/html` 与图片读不到。图片另有一条路（"从剪贴板粘贴图片"
     * 走后端读 CF_DIB），因此这里把空文本原样交给同一条优先级链 —— 万一剪贴板
     * 里其实是一张位图，位图兜底仍会命中。
     */
    const pasteFromClipboard = useCallback(
        async (overrides?: NotebookPasteOverrides) => {
            if (!insertContext) return;
            let text = "";
            try {
                text = await navigator.clipboard.readText();
            } catch {
                // 读不到（无权限 / 非安全上下文）不在这里报错：交给下面的统一
                // 载荷链去决定（它可能命中位图，也可能给出"没有可粘贴的内容"）。
                text = "";
            }
            await applyNotebookPastePayload(
                insertContext,
                { files: [], html: "", text },
                overrides,
            );
        },
        [insertContext],
    );

    /**
     * 按落点组装动作。
     *
     * 【为什么要显式传编辑器实例】分栏预览栏有**自己的**编辑器实例（同一套扩展、
     * `editable: false`）。复制必须作用于用户实际右键的那一份 —— 用主编辑器去
     * 复制，拿到的是左栏的选区，不是他选中的东西。其余动作（粘贴 / 插入 /
     * 链接编辑）只在 `editable` 时才会被发出，而预览栏的 `editable` 是 false，
     * 因此那些绑定到主编辑器的动作不会在预览栏出现。
     *
     * 图片与暂存块走各自的 NodeView（它们 `stopPropagation`，面板收不到事件），
     * 因此这里的 `target` 实际只会是 text / empty / link / table / list。
     * 仍然按完整联合类型处理，免得将来放开 NodeView 时漏掉分支。
     */
    const buildMenuActions = useCallback(
        (
            target: NotebookMenuTarget,
            instance: Editor | null,
        ): { actions: NotebookMenuActions; disabled: NotebookMenuDisabled } => {
            if (!instance || instance.isDestroyed || !insertContext) {
                return { actions: {}, disabled: {} };
            }
            const chain = () => instance.chain().focus();
            const itemType = target.kind === "list" ? target.itemType : "listItem";
            const actions: NotebookMenuActions = {
                undo: () => runNotebookUndo(instance, bridge),
                redo: () => runNotebookRedo(instance, bridge),
                cut: () => void copySelection(instance, true),
                copy: () => void copySelection(instance, false),
                copyMarkdown: () => void copySelection(instance, false, "markdown"),
                copyPlain: () => void copySelection(instance, false, "text"),
                paste: () => void pasteFromClipboard(),
                pastePlain: () => void pasteFromClipboard({ plainPasteMode: "text" }),
                pasteMarkdown: () => void pasteFromClipboard({ plainPasteMode: "markdown" }),
                pasteImage: () => void insertImageFromClipboardBitmap(insertContext),
                stageClipboard: () => handlers.stageClipboard(),
                selectAll: () => {
                    chain().selectAll().run();
                },
                bold: () => {
                    chain().toggleBold().run();
                },
                italic: () => {
                    chain().toggleItalic().run();
                },
                strike: () => {
                    chain().toggleStrike().run();
                },
                code: () => {
                    chain().toggleCode().run();
                },
                clearFormatting: () => {
                    chain().unsetAllMarks().run();
                },
                heading1: () => {
                    chain().toggleHeading({ level: 1 }).run();
                },
                heading2: () => {
                    chain().toggleHeading({ level: 2 }).run();
                },
                heading3: () => {
                    chain().toggleHeading({ level: 3 }).run();
                },
                paragraph: () => {
                    chain().setParagraph().run();
                },
                indent: () => {
                    chain().sinkListItem(itemType).run();
                },
                outdent: () => {
                    chain().liftListItem(itemType).run();
                },
                tableRowAbove: () => {
                    chain().addRowBefore().run();
                },
                tableRowBelow: () => {
                    chain().addRowAfter().run();
                },
                tableColLeft: () => {
                    chain().addColumnBefore().run();
                },
                tableColRight: () => {
                    chain().addColumnAfter().run();
                },
                tableDeleteRow: () => {
                    chain().deleteRow().run();
                },
                tableDeleteCol: () => {
                    chain().deleteColumn().run();
                },
                tableDelete: () => {
                    chain().deleteTable().run();
                },
                tableToggleHeader: () => {
                    chain().toggleHeaderRow().run();
                },
                insertImage: () => handlers.insertImage(),
                insertTable: () => insertTable(instance),
                insertRule: () => {
                    chain().setHorizontalRule().run();
                },
                insertTimecode: () => handlers.insertTimecode(),
                insertClipReference: () => handlers.insertClipReference(),
                insertProjectInfo: () => handlers.insertProjectInfo(),
                find: () => setFindOpen(true),
            };

            if (target.kind === "link") {
                const href = target.href;
                actions.linkOpen = () => {
                    const internal = parseInternalLink(href);
                    if (internal) {
                        runInternalLink(internal);
                        return;
                    }
                    if (!/^https?:/i.test(href)) return;
                    void (async () => {
                        // 插件里没有 Tauri opener（而且原生的 `NavigationStarting`
                        // 会拦下一切外链导航），所以直接走复制 —— 与 `AboutDialog`
                        // 的插件分支同一条策略，不浪费一次注定失败的调用。
                        if (!isPluginMode()) {
                            try {
                                const { openUrl } = await import("@tauri-apps/plugin-opener");
                                await openUrl(href);
                                return;
                            } catch {
                                // 非 Tauri 环境（浏览器调试）没有 opener：退回到
                                // 复制地址，至少让用户能自己粘进浏览器。
                            }
                        }
                        await copyTextToClipboard(href);
                    })();
                };
                // 复制的是**归一化后**的地址：与正文里渲染出来的 href 一致。
                // 直接复制存储值会把 `www.bilibili.com` 这种缺协议的写法给出去。
                actions.linkCopy = () => void copyTextToClipboard(normalizeLinkHref(href));
                actions.linkEdit = () => openLinkEditor();
                actions.linkRemove = () => {
                    const current = editorRef.current;
                    if (current && !current.isDestroyed) clearNotebookLink(current);
                };
            }

            const disabled: NotebookMenuDisabled = {};
            // 没有选中音频块时，"插入 Clip 引用"点了只会弹一句提示；菜单里直接
            // 置灰并说明原因更省事（tooltip 就是为"为什么点不了"准备的）。
            const hasClip =
                selectedClipId !== null && clips.some((entry) => entry.id === selectedClipId);
            if (!hasClip) disabled.insertClipReference = t("notebook_clip_ref_none");

            return { actions, disabled };
        },
        [
            bridge,
            clips,
            copySelection,
            handlers,
            insertContext,
            openLinkEditor,
            pasteFromClipboard,
            runInternalLink,
            selectedClipId,
            t,
        ],
    );

    /** 富文本：落点由编辑器几何解析，并可能移动光标。 */
    const openEditorMenu = useCallback(
        (x: number, y: number) => {
            if (settings.contextMenu === "off") return;
            const instance = editorRef.current;
            if (!instance || instance.isDestroyed) return;
            const { target, flags } = prepareNotebookContext(instance, x, y);
            const { actions, disabled } = buildMenuActions(target, instance);
            setMenu({
                x,
                y,
                items: buildNotebookContextMenu({
                    surface: "rich",
                    scope: settings.contextMenu === "compact" ? "compact" : "full",
                    context: { target, flags },
                    translate: t,
                    actions,
                    disabled,
                }),
            });
        },
        [buildMenuActions, settings.contextMenu, t],
    );

    /**
     * 富文本的**键盘**入口：锚在光标处，落点直接用光标位置。
     *
     * 不做"坐标 → 位置"的反查（`prepareNotebookContextAtCaret` 的注释解释了为什么：
     * 文档末尾的坐标反查回来会命中最后一块，实测在表格结尾的笔记上给出表格菜单）。
     */
    const openEditorMenuAtCaret = useCallback(() => {
        if (settings.contextMenu === "off") return;
        const instance = editorRef.current;
        if (!instance || instance.isDestroyed) return;
        const coords = instance.view.coordsAtPos(instance.state.selection.from);
        const prepared = prepareNotebookContextAtCaret(instance, {
            x: coords.left,
            y: coords.bottom,
        });
        const { actions, disabled } = buildMenuActions(prepared.context.target, instance);
        setMenu({
            x: prepared.x,
            y: prepared.y,
            items: buildNotebookContextMenu({
                surface: "rich",
                scope: settings.contextMenu === "compact" ? "compact" : "full",
                context: prepared.context,
                translate: t,
                actions,
                disabled,
            }),
        });
    }, [buildMenuActions, settings.contextMenu, t]);

    /**
     * 分栏只读预览：落点由**预览栏自己**的编辑器解析。
     *
     * 预览栏是另一个编辑器实例，几何与开关都得问它要 —— 拿主编辑器的坐标去
     * 解析会得到错位的落点，而它的 `editable: true` 会让菜单错误地给出编辑项。
     * 因此由预览栏解析好再传上来（`onContextMenu` 的载荷）。
     */
    const openPreviewMenu = useCallback(
        (payload: NotebookPreviewContextRequest) => {
            if (settings.contextMenu === "off") return;
            const { actions, disabled } = buildMenuActions(payload.target, payload.editor);
            setMenu({
                x: payload.x,
                y: payload.y,
                items: buildNotebookContextMenu({
                    surface: "preview",
                    scope: settings.contextMenu === "compact" ? "compact" : "full",
                    context: { target: payload.target, flags: payload.flags },
                    translate: t,
                    actions,
                    disabled,
                }),
            });
        },
        [buildMenuActions, settings.contextMenu, t],
    );

    /**
     * 源码视图：`<textarea>` 没有 schema，只有"有没有选中"。
     *
     * 撤销 / 剪切 / 复制走 `document.execCommand`（textarea 的原生路径，与
     * Ctrl+Z / Ctrl+X 今天的行为完全一致）；粘贴因为浏览器不允许脚本读剪贴板，
     * 只能 `readText()` + `insertText` —— 后者会派发 `input` 事件，因此 React 的
     * onChange 照常触发，Redux 与原生撤销栈都不会被绕过。
     */
    const openSourceMenu = useCallback(
        (x: number, y: number) => {
            if (settings.contextMenu === "off") return;
            const textarea = sourceTextareaRef.current;
            const runNative = (command: "undo" | "redo" | "cut" | "copy") => {
                const element = sourceTextareaRef.current;
                if (!element) return;
                element.focus();
                document.execCommand(command);
            };
            const actions: NotebookMenuActions = {
                undo: () => runNative("undo"),
                redo: () => runNative("redo"),
                cut: () => runNative("cut"),
                copy: () => runNative("copy"),
                paste: () => {
                    const element = sourceTextareaRef.current;
                    if (!element) return;
                    element.focus();
                    void (async () => {
                        try {
                            const text = await navigator.clipboard.readText();
                            if (!text) return;
                            document.execCommand("insertText", false, text);
                        } catch {
                            notify(t("notebook_paste_nothing"));
                        }
                    })();
                },
                selectAll: () => {
                    textarea?.focus();
                    textarea?.select();
                },
                find: () => setFindOpen(true),
            };
            const context = {
                target: { kind: "empty" } as const,
                flags: {
                    selectionEmpty: !textarea || textarea.selectionStart === textarea.selectionEnd,
                    editable: true,
                    canUndo: true,
                    canRedo: true,
                    isBold: false,
                    isItalic: false,
                    isStrike: false,
                    isCode: false,
                    hasMarks: false,
                    headingLevel: null,
                    canIndent: false,
                },
            };
            setMenu({
                x,
                y,
                items: buildNotebookContextMenu({
                    surface: "source",
                    scope: settings.contextMenu === "compact" ? "compact" : "full",
                    context,
                    translate: t,
                    actions,
                }),
            });
        },
        [notify, settings.contextMenu, t],
    );

    /**
     * 键盘入口：`ContextMenu` 键 / `Shift+F10` 在光标处开菜单。
     *
     * `App.tsx` 全局屏蔽了这两个键的**默认行为**（原生菜单），但不阻止传播 ——
     * 因此这里还能收到。不给它做快捷键注册表项：它只在"焦点在记事本编辑器里"
     * 时成立，是表面局部行为，与 Ctrl+F 的处理方式一致。
     */
    useEffect(() => {
        const element = containerRef.current;
        if (!element) return;
        const handler = (event: KeyboardEvent) => {
            if (event.key !== "ContextMenu" && !(event.key === "F10" && event.shiftKey)) return;
            const target = event.target as HTMLElement | null;
            if (!target?.closest?.(".hs-notebook-rich, .hs-notebook-source")) return;
            if (settings.contextMenu === "off") return;
            event.preventDefault();
            event.stopPropagation();

            if (target.closest(".hs-notebook-source")) {
                const rect = sourceTextareaRef.current?.getBoundingClientRect();
                openSourceMenu(rect ? rect.left + 24 : 0, rect ? rect.top + 24 : 0);
                return;
            }
            openEditorMenuAtCaret();
        };
        element.addEventListener("keydown", handler);
        return () => element.removeEventListener("keydown", handler);
    }, [openEditorMenuAtCaret, openSourceMenu, settings.contextMenu]);

    const closeMenu = useCallback(() => setMenu(null), []);

    // ── 模式切换 / 关闭：收尾并分节 ────────────────────────────────
    const changeMode = useCallback(
        (next: NotebookMode) => {
            if (next === mode) return;
            seal();
            // 源码视图的待写内容也要在切模式前落盘（seal 只收编辑器那一侧）。
            flushSourcePersist();
            // 链接浮层编辑的是富文本的 mark，源码视图下没有意义：切走即收起，
            // 免得切回来时它凭空又出现（还带着上一次的草稿）。
            closeLinkEditor();
            // 菜单里的项是**切模式前**算出来的快照（例如源码视图的"粘贴"、
            // 富文本的"加粗"）：留着它，用户切完模式还能点到作用于旧表面的项。
            closeMenu();
            dispatch(setNotebookMode(next));
            void settingsApi
                .saveUiSettings({ notebook: { ...settings, defaultMode: next } })
                .catch(() => {});
        },
        [closeLinkEditor, closeMenu, dispatch, flushSourcePersist, mode, seal, settings],
    );

    /**
     * 面板自身的关闭入口已移除（关闭键属于窗框：停靠时在标签上、浮动时在浮动
     * 标题栏上）。这里保留 `seal` 供模式切换等需要"先落盘再改状态"的路径使用。
     */

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

    /*
     * Ctrl/⌘ + 滚轮 = 编辑区字号缩放。
     *
     * 【修饰键走键位绑定，不写死 ctrlKey】判定用 `isModifierActive`（与时间轴 /
     * 参数编辑器的滚轮修饰键同一套），它内部按平台取主修饰键：Windows / Linux 是
     * Ctrl，macOS 是 ⌘（见 `utils/platform.ts`）。这正是"macOS 风格"的落点 ——
     * macOS 上 Ctrl+滚轮是系统级缩放/辅助功能手势，属于应用不该抢的键。
     *
     * 【方向判定复用内核的那一套，不自己看 deltaY 符号】`resolveWheelZoomStep`
     * 用"主轴 + 死区累积"：`deltaY === 0` 的纯横向手势不会被判成缩小，precision
     * touchpad 的小幅变号增量也累积不到死区（否则表现为剧烈抖动）。增量先经
     * `readWheelPixels` 归一化 —— Firefox 的 `deltaMode = 1`（行）不换算的话，
     * 一格只有几个像素，永远跨不过死区。
     *
     * 【监听挂在面板根节点】面板的滚轮语义只属于记事本自己（富文本区、源码
     * textarea、状态栏都在这个根节点内），因此"焦点在记事本窗口内"就等于"滚轮
     * 落在这个节点上"。设置对话框是 portal 到 body 的，不在其内 —— 框里滚轮
     * 仍归输入框的精细调整，不会连带缩放字号。
     */
    const fontZoomKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.notebookFontZoom"),
    );
    const fontZoomAccumulatorRef = useRef(createWheelZoomAccumulator());

    /*
     * 落盘去抖：Redux 立即更新（廉价、界面即时响应），IPC 在停止滚动后下发一次。
     * 逐格 `saveUiSettings` 会把一次滑动变成几十次 IPC —— 与吸附设置对后端同步
     * 的处理一致。卸载时补发，因此"滚完就关窗"不会丢。
     */
    const persistNotebookSettings = useDebouncedCallback((next: typeof settings) => {
        void settingsApi.saveUiSettings({ notebook: next }).catch(() => {});
    }, 400);

    const onFontZoomWheel = (event: WheelEvent) => {
        if (isNoneBinding(fontZoomKb) || !isModifierActive(fontZoomKb, event)) return;
        /*
         * 命中手势后每一格都要拦，包括死区里那些不产生缩放的：不拦的话 WebView
         * 会自己缩放整个应用 —— 滚到字号上限后继续滚，界面整体变大而设置里的数
         * 没变。
         */
        event.preventDefault();
        const pixels = readWheelPixels(event, {
            lineHeightPx: NOTEBOOK_ZOOM_LINE_HEIGHT_PX,
            pageHeightPx: Math.max(1, containerRef.current?.clientHeight ?? 0),
        });
        const step = resolveWheelZoomStep({
            accumulator: fontZoomAccumulatorRef.current,
            deltaX: pixels.x,
            deltaY: pixels.y,
        });
        fontZoomAccumulatorRef.current = step.accumulator;

        const next = nextFontSizeForZoomStep(settings.sourceFontSize, step.direction);
        if (next === null) return;
        dispatch(patchNotebookSettings({ sourceFontSize: next }));
        persistNotebookSettings.call({ ...settings, sourceFontSize: next });
    };

    const attachFontZoomWheel = useNonPassiveWheel<HTMLDivElement>(onFontZoomWheel);

    /*
     * 面板根节点要同时被两处使用：`containerRef`（拖放、查找的容器查询）与
     * `useNonPassiveWheel` 的回调 ref。回调 ref 里转发给两者 —— 与本仓库
     * 既有的"回调 ref 写 ref.current"写法一致（见 `PianoRollPanel` 的
     * `attachRulerPlayheadLine`）。
     */
    const attachContainer = useCallback(
        (element: HTMLDivElement | null) => {
            containerRef.current = element;
            attachFontZoomWheel(element);
        },
        [attachFontZoomWheel],
    );

    // 保存前收尾：工程保存会把当前正文一并带上，但待写内容也要落盘，
    // 否则"刚打完字就保存"会出现后端与工程文件不一致。源码视图的待写值
    // 走自己的定时器，同样在这里冲刷。
    useEffect(() => {
        return () => {
            flush();
            flushSourcePersist();
        };
    }, [flush, flushSourcePersist]);

    const usedAssetIds = useMemo(() => referencedAssetIds(markdown), [markdown]);

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
            ref={attachContainer}
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
                if (
                    Array.from(event.dataTransfer?.items ?? []).some((item) => item.kind === "file")
                ) {
                    event.preventDefault();
                    setDropActive(true);
                }
            }}
            onDragLeave={() => setDropActive(false)}
            onDrop={onHtml5Drop}
        >
            {/* 工具条：只放本面板**独有**的功能按钮。标题与关闭属于窗框
                （停靠时是标签行、浮动时是浮动标题栏），面板内再画一遍就是重复。 */}
            <PanelToolbar
                leading={
                    <>
                        <ModeButton
                            active={mode === "rich"}
                            label={t("notebook_mode_rich")}
                            tooltip={t("notebook_mode_rich")}
                            onClick={() => changeMode("rich")}
                        />
                        <ModeButton
                            active={mode === "source"}
                            label={t("notebook_mode_source")}
                            tooltip={t("notebook_mode_source")}
                            onClick={() => changeMode("source")}
                        />
                        <ModeButton
                            active={mode === "split"}
                            label={t("notebook_mode_split")}
                            tooltip={t("notebook_mode_split")}
                            onClick={() => changeMode("split")}
                        />
                    </>
                }
                trailing={
                    <>
                        <PanelToolbarButton
                            icon={
                                settings.showToolbar ? (
                                    <ChevronDownIcon width={ICON} height={ICON} />
                                ) : (
                                    <ChevronRightIcon width={ICON} height={ICON} />
                                )
                            }
                            tooltip={t("notebook_toggle_toolbar")}
                            onClick={() => setSetting({ showToolbar: !settings.showToolbar })}
                        />
                        <PanelToolbarButton
                            icon={<CardStackIcon width={ICON} height={ICON} />}
                            tooltip={t("notebook_attachments")}
                            onClick={() => setAttachmentsOpen(true)}
                        />
                        <PanelToolbarButton
                            icon={<GearIcon width={ICON} height={ICON} />}
                            tooltip={t("notebook_settings")}
                            onClick={() => setSettingsOpen(true)}
                        />
                    </>
                }
            />

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
                <NotebookToolbar
                    editor={editor}
                    handlers={handlers}
                    slashCommands={settings.slashCommands}
                    onEditLink={openLinkEditor}
                    linkEditorOpen={linkDraft !== null}
                />
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
                        onChange={(event) => {
                            const value = event.target.value;
                            dispatch(setProjectNotesMarkdown(value));
                            scheduleSourcePersist(value);
                        }}
                        onBlur={flushSourcePersist}
                        onContextMenu={(event) => {
                            event.preventDefault();
                            openSourceMenu(event.clientX, event.clientY);
                        }}
                    />
                ) : null}

                {mode !== "source" ? (
                    /* 内边距归可编辑区自己（见 notebook.css 的 .hs-notebook-rich）：
                       容器留白会变成一圈点不动的死边。 */
                    <div
                        ref={richScrollRef}
                        className="hs-scroll-gutter min-w-0 flex-1 overflow-auto bg-qt-base"
                    >
                        {editor ? (
                            <EditorContent
                                editor={editor}
                                className="hs-notebook-rich"
                                // 挂在编辑器元素上（而不是外层滚动容器）：图片与
                                // 暂存块卡片会 `stopPropagation`，因此落在这两类
                                // 节点上的右键由各自的 NodeView 处理，不会双重弹窗。
                                onContextMenu={(event) => {
                                    event.preventDefault();
                                    openEditorMenu(event.clientX, event.clientY);
                                }}
                            />
                        ) : null}
                    </div>
                ) : null}

                {mode === "split" ? (
                    <NotebookReadonlyPreview
                        markdown={markdown}
                        scrollSyncSource={richScrollRef.current}
                        onContextMenu={openPreviewMenu}
                    />
                ) : null}

                {/* 源码视图没有链接标记，浮层只在富文本 / 分栏下有意义。 */}
                {linkDraft !== null && mode !== "source" ? (
                    <NotebookLinkEditor
                        value={linkDraft}
                        onChange={setLinkDraft}
                        onApply={applyLinkDraft}
                        onCancel={closeLinkEditor}
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

            {/* 菜单挂在 `document.body`（见 NotebookContextMenu），因此放在
                JSX 的哪一层都不影响定位 —— 这里放在面板根部只是为了"面板关掉
                菜单就跟着没"。 */}
            {menu ? (
                <NotebookContextMenu
                    x={menu.x}
                    y={menu.y}
                    items={menu.items}
                    ariaLabel={t("notebook_ctx_aria")}
                    onClose={closeMenu}
                />
            ) : null}

            <AppFileInput
                inputRef={fileInputRef}
                accept="image/*"
                multiple
                onFiles={onFilePicked}
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

/** 工具条图标尺寸（与 `PanelToolbar` 的约定一致）。 */
const ICON = 12;

/** 视图模式切换：文字按钮，套用与图标按钮同一套度量（见 `PanelToolbar`）。 */
function ModeButton({
    active,
    label,
    tooltip,
    onClick,
}: {
    active: boolean;
    label: string;
    tooltip: string;
    onClick: () => void;
}) {
    return (
        <PanelToolbarTextButton label={label} tooltip={tooltip} active={active} onClick={onClick} />
    );
}

/** 扩展名判据（拖入文件时过滤掉非图片）。 */
function looksLikeImagePath(path: string): boolean {
    return /\.(png|jpe?g|gif|webp|bmp|avif|svg)$/i.test(path);
}
