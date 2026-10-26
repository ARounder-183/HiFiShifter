// FileBrowser 的对话框、行内编辑、右键菜单与面板级键盘模型。拆自 FileBrowserPanel。
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { StarIcon } from "@radix-ui/react-icons";
import {
    buildFileBrowserContextMenu,
    type FileBrowserMenuActions,
} from "../../../features/fileBrowser/fileBrowserMenu";
import {
    emitExternalFileAction,
    emitImportMidiRequest,
    emitImportProjectPick,
    emitOpenProjectPath,
} from "../../../features/session/projectOpenEvents";
import { emitFolderImportRequest } from "../../../features/fileBrowser/folderImportEvents";
import {
    isAudioFile,
    isMidiFile,
    isProjectFile,
    isReaperFile,
    isVocalShifterFile,
} from "../../../features/fileBrowser/fileKinds";
import { rootIndexAtDrop } from "../../../features/session/trackUtils";
import {
    importAudioAtPosition,
    importMultipleAudioAtPosition,
} from "../../../features/session/thunks/importThunks";
import {
    persistUiSettings,
    setFileBrowserFavorites,
    setFileBrowserView,
} from "../../../features/session/sessionSlice";
import type { FileBrowserViewOptions } from "../../../features/fileBrowser/fileBrowserViewOptions";
import { locationLabel, parentDirOf } from "../../../features/fileBrowser/fileBrowserPaths";
import { fileBrowserApi, type FileEntry } from "../../../services/api/fileBrowser";
import { type AppMenuItemSpec } from "../../../ui";
import { copyTextToClipboard } from "../../../utils/copyText";
import { takePanelFocus } from "../fileBrowserPanelFocus";
import { TYPE_AHEAD_RESET_MS, nextTypeAhead } from "../fileBrowserKeyboardNav";
import type { FileBrowserListing } from "./useFileBrowserListing";
import type { FileBrowserSelection } from "./useFileBrowserSelection";

/** 面板内的行内编辑状态。 */
type EditingState =
    { kind: "rename"; path: string; initial: string } | { kind: "newFolder" } | null;

export function useFileBrowserActions(
    listing: FileBrowserListing,
    selection: FileBrowserSelection,
) {
    const {
        dispatch,
        t,
        tf,
        tVars,
        plural,
        fb,
        view,
        favorites,
        selectedTrackId,
        playheadSec,
        tracks,
        isSearchMode,
        isComputerLevel,
        displayEntries,
        navigateTo,
        handleOpenFolder,
        handleRefresh,
        handleParentDir,
        handleEnterDir,
        goBack,
        goForward,
        refreshListing,
        previewToggle,
        history,
    } = listing;
    const {
        selectedPaths,
        setSelectedPaths,
        selectionAnchorRef,
        activeIndex,
        setActiveIndex,
        rowRefs,
        listScrollRef,
        focusRow,
        maybePreviewOnNavigate,
        selectAll,
        selectedEntries,
    } = selection;
    const [propertiesEntry, setPropertiesEntry] = useState<FileEntry | null>(null);
    const [viewOptionsOpen, setViewOptionsOpen] = useState(false);
    const [deleteRequest, setDeleteRequest] = useState<string[] | null>(null);
    const [editing, setEditing] = useState<EditingState>(null);
    const [pathDraft, setPathDraft] = useState<string | null>(null);
    /** 写操作失败的一次性提示（重名 / 非法名 / 受保护路径）。 */
    const [transientError, setError] = useState<string | null>(null);
    const [menu, setMenu] = useState<{ x: number; y: number; entry: FileEntry | null } | null>(
        null,
    );
    const handlePanelPointerDown = useCallback(
        (event: React.PointerEvent<HTMLDivElement>) => {
            takePanelFocus(event, listScrollRef.current);
        },
        [listScrollRef],
    );
    const handlePanelMouseDown = useCallback(
        (event: React.MouseEvent<HTMLDivElement>) => {
            takePanelFocus(event, listScrollRef.current);
        },
        [listScrollRef],
    );

    // Clear selection when directory changes
    useEffect(() => {
        setSelectedPaths(new Set());
        selectionAnchorRef.current = -1;
        setActiveIndex(-1);
        setEditing(null);
        // 上一次目录里的失败提示（重名 / 非法名）不该跟着走进新目录。
        setError(null);
    }, [
        fb.currentPath,
        selectionAnchorRef,
        setActiveIndex,
        setSelectedPaths,
        setEditing,
        setError,
    ]);

    // ── 输入字母快速跳转（type-ahead，与资源管理器一致） ─────────────────────
    // 名单与 displayEntries 同步，供增量搜索逐键匹配。
    const entryNames = useMemo(() => displayEntries.map((e) => e.name), [displayEntries]);
    const typeAheadBufferRef = useRef("");
    const typeAheadLastKeyAtRef = useRef(0);
    const panelRootRef = useRef<HTMLDivElement>(null);

    const handlePanelKeyDown = useCallback(
        (event: React.KeyboardEvent<HTMLDivElement>) => {
            if (event.defaultPrevented) return;
            const target = event.target as HTMLElement | null;
            // 输入框 / 多行文本 / contentEditable：里面的按键属于文本编辑。
            // Radix 弹层挂在 portal 上、不在面板 DOM 内 —— 一并排除。
            if (
                !target ||
                !panelRootRef.current?.contains(target) ||
                target.tagName === "INPUT" ||
                target.tagName === "TEXTAREA" ||
                target.isContentEditable
            ) {
                return;
            }
            if (event.nativeEvent.isComposing) return;

            const mod = event.ctrlKey || event.metaKey;
            const activeEntry = displayEntries[activeIndex];

            // ── 导航类（Alt+方向键 / Backspace / F5）──────────────────────
            if (event.altKey && !mod) {
                if (event.key === "ArrowLeft") {
                    event.preventDefault();
                    goBack();
                    return;
                }
                if (event.key === "ArrowRight") {
                    event.preventDefault();
                    goForward();
                    return;
                }
                if (event.key === "ArrowUp") {
                    event.preventDefault();
                    handleParentDir();
                    return;
                }
            }
            if (!mod && !event.altKey && (event.key === "Backspace" || event.key === "F5")) {
                event.preventDefault();
                if (event.key === "Backspace") handleParentDir();
                else handleRefresh();
                return;
            }

            // ── 组合键 ────────────────────────────────────────────────────
            if (mod && !event.altKey) {
                const key = event.key.toLowerCase();
                if (key === "a") {
                    event.preventDefault();
                    selectAll();
                    return;
                }
                if (key === "c" && selectedPaths.size > 0) {
                    event.preventDefault();
                    void copyTextToClipboard(Array.from(selectedPaths).join("\n"));
                    return;
                }
                if (key === "n" && event.shiftKey && !isComputerLevel) {
                    event.preventDefault();
                    setEditing({ kind: "newFolder" });
                    return;
                }
                // Ctrl+L：把路径栏切成可编辑并聚焦（资源管理器 / 访达的"定位到"）。
                // 「计算机」虚拟层没有可编辑的真实路径，跳过。
                if (key === "l" && fb.currentPath && !isComputerLevel) {
                    event.preventDefault();
                    setPathDraft(fb.currentPath);
                    return;
                }
                return;
            }
            if (event.altKey || mod) return;

            // ── 单键 ──────────────────────────────────────────────────────
            if (event.key === "F2" && activeEntry && !isComputerLevel) {
                event.preventDefault();
                setEditing({ kind: "rename", path: activeEntry.path, initial: activeEntry.name });
                return;
            }
            if (event.key === "Delete" && selectedPaths.size > 0 && !isComputerLevel) {
                event.preventDefault();
                setDeleteRequest(Array.from(selectedPaths));
                return;
            }
            if (event.key === "Escape") {
                if (selectedPaths.size > 0) {
                    event.preventDefault();
                    setSelectedPaths(new Set());
                }
                return;
            }
            if (
                (event.key === "ContextMenu" || (event.key === "F10" && event.shiftKey)) &&
                activeEntry
            ) {
                event.preventDefault();
                const rect = rowRefs.current[activeIndex]?.getBoundingClientRect();
                setMenu({
                    x: rect ? rect.left + 12 : 40,
                    y: rect ? rect.bottom : 40,
                    entry: activeEntry,
                });
                return;
            }

            // ── 输入字母快速跳转 ──────────────────────────────────────────
            const key = event.key;
            // 只接可打印单字符：空格留给激活/滚动，多字符键（方向键、Enter、F 键）
            // 不属于 type-ahead。
            if (key.length !== 1 || key === " ") return;
            const now = Date.now();
            if (now - typeAheadLastKeyAtRef.current > TYPE_AHEAD_RESET_MS) {
                typeAheadBufferRef.current = "";
            }
            const result = nextTypeAhead(entryNames, typeAheadBufferRef.current, key, activeIndex);
            typeAheadBufferRef.current = result.buffer;
            typeAheadLastKeyAtRef.current = now;
            if (result.index == null) return;
            event.preventDefault();
            setActiveIndex(result.index);
            focusRow(result.index);
            maybePreviewOnNavigate(displayEntries[result.index]);
        },
        [
            displayEntries,
            activeIndex,
            entryNames,
            selectedPaths,
            isComputerLevel,
            // Ctrl+L 要读当前路径：漏了它，回调会捕获到过期的那一条。
            fb.currentPath,
            focusRow,
            goBack,
            goForward,
            handleParentDir,
            handleRefresh,
            maybePreviewOnNavigate,
            selectAll,
            setDeleteRequest,
            setEditing,
            setMenu,
            setPathDraft,
            rowRefs,
            setActiveIndex,
            setSelectedPaths,
        ],
    );

    // 列表内容变化（搜索、排序、过滤）后，活动行可能越界：收回为"无活动行"。
    useEffect(() => {
        setActiveIndex((current) => (current >= displayEntries.length ? -1 : current));
    }, [displayEntries.length, setActiveIndex]);

    const handleRowContextMenu = useCallback(
        (event: React.MouseEvent, entry: FileEntry) => {
            event.preventDefault();
            event.stopPropagation();
            // Explorer 语义：右键未选中项 → 先把它选中，菜单作用于它。
            setSelectedPaths((prev) => (prev.has(entry.path) ? prev : new Set([entry.path])));
            setActiveIndex(displayEntries.findIndex((candidate) => candidate.path === entry.path));
            setMenu({ x: event.clientX, y: event.clientY, entry });
        },
        [displayEntries, setActiveIndex, setSelectedPaths],
    );

    const handleBackgroundContextMenu = useCallback((event: React.MouseEvent) => {
        event.preventDefault();
        setMenu({ x: event.clientX, y: event.clientY, entry: null });
    }, []);

    // ── 常用位置（固定 + 最近访问） ─────────────────────────────────────────
    const locationsButtonRef = useRef<HTMLButtonElement | null>(null);
    const [locationsAt, setLocationsAt] = useState<{ x: number; y: number } | null>(null);
    const isCurrentPinned = favorites.includes(fb.currentPath);

    /** 最近访问：历史倒序、去掉当前目录与重复项，最多 10 条。 */
    const recentLocations = useMemo(() => {
        const seen = new Set<string>([fb.currentPath]);
        const out: string[] = [];
        for (let i = history.entries.length - 1; i >= 0 && out.length < 10; i--) {
            const path = history.entries[i];
            if (seen.has(path)) continue;
            seen.add(path);
            out.push(path);
        }
        return out;
    }, [history.entries, fb.currentPath]);

    const locationItems: AppMenuItemSpec[] = useMemo(() => {
        const items: AppMenuItemSpec[] = [];
        if (fb.currentPath && !isComputerLevel) {
            items.push({
                key: "pin-toggle",
                label: isCurrentPinned ? t("fb_unpin_current") : t("fb_pin_current"),
                icon: <StarIcon />,
                onSelect: () => {
                    const next = isCurrentPinned
                        ? favorites.filter((path) => path !== fb.currentPath)
                        : [...favorites, fb.currentPath];
                    dispatch(setFileBrowserFavorites(next));
                    void dispatch(persistUiSettings());
                },
            });
        }
        if (favorites.length > 0) {
            items.push({ key: "pinned-heading", label: t("fb_pinned_locations"), heading: true });
            for (const path of favorites) {
                items.push({
                    key: `pinned:${path}`,
                    label: locationLabel(path, tf("fb_computer")),
                    tooltip: path,
                    checked: path === fb.currentPath,
                    onSelect: () => navigateTo(path),
                });
            }
        }
        if (recentLocations.length > 0) {
            items.push({
                key: "recent-heading",
                label: t("fb_recent_locations"),
                heading: true,
                separatorBefore: favorites.length > 0,
            });
            for (const path of recentLocations) {
                items.push({
                    key: `recent:${path}`,
                    label: locationLabel(path, tf("fb_computer")),
                    tooltip: path,
                    onSelect: () => navigateTo(path),
                });
            }
        }
        return items;
    }, [
        dispatch,
        favorites,
        fb.currentPath,
        isComputerLevel,
        isCurrentPinned,
        navigateTo,
        recentLocations,
        t,
        tf,
    ]);

    const handleRenameCommit = useCallback(
        async (entry: FileEntry, newName: string) => {
            setEditing(null);
            const trimmed = newName.trim();
            if (!trimmed || trimmed === entry.name) return;
            try {
                const newPath = await fileBrowserApi.renamePath(entry.path, trimmed);
                setSelectedPaths(new Set([newPath]));
                await refreshListing();
            } catch {
                // 后端已把非法名 / 重名 / 受保护路径拒绝掉了，这里只需让用户看到结果。
                setError(tf("fb_rename_failed"));
            }
        },
        [refreshListing, tf, setSelectedPaths],
    );

    const handleNewFolderCommit = useCallback(
        async (name: string) => {
            setEditing(null);
            const trimmed = name.trim();
            if (!trimmed) return;
            try {
                const created = await fileBrowserApi.createDirectory(fb.currentPath, trimmed);
                setSelectedPaths(new Set([created]));
                await refreshListing();
            } catch {
                setError(tf("fb_create_folder_failed"));
            }
        },
        [refreshListing, fb.currentPath, tf, setSelectedPaths],
    );

    const handleDelete = useCallback(
        async (permanent: boolean) => {
            const paths = deleteRequest;
            setDeleteRequest(null);
            if (!paths || paths.length === 0) return;
            try {
                const result = await fileBrowserApi.deletePaths(paths, permanent);
                if (!result.ok) {
                    /*
                     * 把后端的**具体原因**带出来。后端逐条汇总失败原因
                     * （`protected path: …` / `not found: …` / 系统错误原文），
                     * 只显示"删除失败"会让用户无从判断是权限、占用还是路径不对 ——
                     * 而这三者的处理方式完全不同。
                     */
                    setError(
                        result.error
                            ? tVars("common_label_value", {
                                  label: tf("fb_delete_failed"),
                                  value: result.error,
                              })
                            : tf("fb_delete_failed"),
                    );
                }
            } catch {
                setError(tf("fb_delete_failed"));
            }
            setSelectedPaths(new Set());
            await refreshListing();
        },
        [deleteRequest, refreshListing, tf, tVars, setSelectedPaths],
    );

    /** 菜单动作集合：菜单只决定"显示什么"，这里决定"做什么"。 */
    const menuActions: FileBrowserMenuActions = useMemo(
        () => ({
            openEntry: (entry) => {
                if (entry.isDir) {
                    handleEnterDir(entry.path);
                } else if (isAudioFile(entry)) {
                    // 菜单里的「试听」是**显式命令**：用户点名要听，就不受
                    // `previewOnClick`（管的是"点一下行会不会顺手出声"）管辖。
                    previewToggle.toggle(entry.path);
                } else if (isMidiFile(entry)) {
                    emitImportMidiRequest({
                        path: entry.path,
                        startSec: playheadSec,
                        trackId: selectedTrackId,
                    });
                } else if (isReaperFile(entry)) {
                    emitExternalFileAction("importReaper", entry.path);
                } else if (isVocalShifterFile(entry)) {
                    emitExternalFileAction("importVocalShifter", entry.path);
                } else if (isProjectFile(entry)) {
                    emitOpenProjectPath(entry.path);
                }
            },
            insertAtPlayhead: (entries) => {
                const paths = entries.map((entry) => entry.path);
                if (paths.length === 0) return;
                if (paths.length === 1) {
                    void dispatch(
                        importAudioAtPosition({
                            audioPath: paths[0],
                            trackId: selectedTrackId,
                            startSec: playheadSec,
                        }),
                    );
                } else {
                    void dispatch(
                        importMultipleAudioAtPosition({
                            audioPaths: paths,
                            mode: "across-time",
                            trackId: selectedTrackId,
                            startSec: playheadSec,
                        }),
                    );
                }
            },
            insertOnNewTrack: (entries) => {
                const paths = entries.map((entry) => entry.path);
                if (paths.length === 0) return;
                if (paths.length === 1) {
                    // `trackId: null` 让 thunk 先建一条新轨道再导入。
                    void dispatch(
                        importAudioAtPosition({
                            audioPath: paths[0],
                            trackId: null,
                            startSec: playheadSec,
                        }),
                    );
                } else {
                    void dispatch(
                        importMultipleAudioAtPosition({
                            audioPaths: paths,
                            mode: "across-tracks",
                            trackId: null,
                            startSec: playheadSec,
                        }),
                    );
                }
            },
            insertMultiple: (entries, mode) => {
                const paths = entries.map((entry) => entry.path);
                if (paths.length === 0) return;
                void dispatch(
                    importMultipleAudioAtPosition({
                        audioPaths: paths,
                        mode,
                        trackId: selectedTrackId,
                        startSec: playheadSec,
                    }),
                );
            },
            togglePreview: (entry) => previewToggle.toggle(entry.path),
            importFolder: (entries) => {
                const dirs = entries.filter((entry) => entry.isDir).map((entry) => entry.path);
                if (dirs.length === 0) return;
                // 右键入口是用户**显式**发起的：目录里没有媒体文件时也弹窗给出解释，
                // 而不是静默什么都不做（`fromExplicitRequest` 的唯一作用）。
                emitFolderImportRequest({
                    dirs,
                    looseFiles: [],
                    trackId: selectedTrackId,
                    startSec: playheadSec,
                    insertIndex: rootIndexAtDrop(tracks, selectedTrackId),
                    fromExplicitRequest: true,
                });
            },
            reveal: (paths) => {
                void fileBrowserApi.revealPaths(paths);
            },
            openWithDefaultApp: (path) => {
                void fileBrowserApi.openPathWithDefaultApp(path);
            },
            copyPaths: (paths) => {
                void copyTextToClipboard(paths.join("\n"));
            },
            copyName: (name) => {
                void copyTextToClipboard(name);
            },
            openProject: (path) => emitOpenProjectPath(path),
            importProject: (path) => emitImportProjectPick(path),
            importMidi: (path) =>
                emitImportMidiRequest({
                    path,
                    startSec: playheadSec,
                    trackId: selectedTrackId,
                }),
            openContainingFolder: (entry) => {
                const parent = parentDirOf(entry.path);
                if (parent) navigateTo(parent);
            },
            rename: (entry) =>
                setEditing({ kind: "rename", path: entry.path, initial: entry.name }),
            remove: (entries) => setDeleteRequest(entries.map((entry) => entry.path)),
            showProperties: (entry) => setPropertiesEntry(entry),
            newFolder: () => setEditing({ kind: "newFolder" }),
            refresh: handleRefresh,
            openFolderDialog: () => void handleOpenFolder(),
            selectAll,
            clearSelection: () => setSelectedPaths(new Set()),
            setSortMode: (mode) => {
                dispatch(setFileBrowserView({ sortMode: mode }));
                void dispatch(persistUiSettings());
            },
            setSortDescending: (descending) => {
                dispatch(setFileBrowserView({ sortDescending: descending }));
                void dispatch(persistUiSettings());
            },
            patchView: (patch: Partial<FileBrowserViewOptions>) => {
                dispatch(setFileBrowserView(patch));
                void dispatch(persistUiSettings());
            },
            openViewOptions: () => setViewOptionsOpen(true),
        }),
        [
            dispatch,
            handleEnterDir,
            handleOpenFolder,
            handleRefresh,
            navigateTo,
            playheadSec,
            previewToggle,
            selectAll,
            selectedTrackId,
            tracks,
            setSelectedPaths,
        ],
    );

    const menuItems = useMemo(
        () =>
            menu
                ? buildFileBrowserContextMenu(menu.entry, {
                      t,
                      plural,
                      view,
                      isComputerLevel,
                      isSearchMode,
                      previewingPath: fb.previewingFile,
                      selected: selectedEntries,
                      currentPath: fb.currentPath,
                      actions: menuActions,
                  })
                : [],
        [
            menu,
            t,
            plural,
            view,
            isComputerLevel,
            isSearchMode,
            fb.previewingFile,
            fb.currentPath,
            selectedEntries,
            menuActions,
        ],
    );

    // ── 拖拽（自定义 pointer 事件，替代 HTML5 drag API）────────────────────
    return {
        propertiesEntry,
        setPropertiesEntry,
        viewOptionsOpen,
        setViewOptionsOpen,
        deleteRequest,
        setDeleteRequest,
        editing,
        setEditing,
        pathDraft,
        setPathDraft,
        transientError,
        setError,
        menu,
        setMenu,
        panelRootRef,
        handlePanelPointerDown,
        handlePanelMouseDown,
        handlePanelKeyDown,
        handleRowContextMenu,
        handleBackgroundContextMenu,
        locationsButtonRef,
        locationsAt,
        setLocationsAt,
        locationItems,
        handleRenameCommit,
        handleNewFolderCommit,
        handleDelete,
        menuActions,
        menuItems,
    };
}

export type FileBrowserActions = ReturnType<typeof useFileBrowserActions>;
