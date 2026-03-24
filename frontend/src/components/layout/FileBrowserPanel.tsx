import React, {
    useCallback,
    useEffect,
    useMemo,
    useRef,
    useState,
} from "react";
import {
    Flex,
    Text,
    IconButton,
    Select,
    Slider,
    TextField,
    ScrollArea,
} from "@radix-ui/themes";
import {
    Cross2Icon,
    FileIcon,
    MagnifyingGlassIcon,
    ReloadIcon,
    ChevronUpIcon,
    SpeakerLoudIcon,
    PlayIcon,
    StopIcon,
} from "@radix-ui/react-icons";
import { useAppDispatch, useAppSelector } from "../../app/hooks";
import type { RootState } from "../../app/store";
import { useI18n } from "../../i18n/I18nProvider";
import {
    loadDirectory,
    setPreviewVolume,
    setPreviewingFile,
    setSearchQuery,
    setVisible,
    searchFilesRecursive,
    toggleRegex,
    setSortMode,
    toggleAudioOnly,
    type SortMode,
} from "../../features/fileBrowser/fileBrowserSlice";
import { audioPreview } from "../../features/fileBrowser/audioPreview";
import type { FileEntry } from "../../services/api/fileBrowser";

/** 支持的音频扩展名 */
const AUDIO_EXTENSIONS = new Set([
    "wav",
    "mp3",
    "flac",
    "ogg",
    "aac",
    "aif",
    "aiff",
    "m4a",
]);

function isAudioFile(entry: FileEntry): boolean {
    return (
        !entry.isDir &&
        !!entry.extension &&
        AUDIO_EXTENSIONS.has(entry.extension)
    );
}

/** 格式化文件大小 */
function formatSize(bytes: number | null): string {
    if (bytes == null) return "";
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/** 文件夹图标 SVG */
function FolderIcon({ className }: { className?: string }) {
    return (
        <svg
            width="14"
            height="14"
            viewBox="0 0 15 15"
            fill="none"
            className={className}
        >
            <path
                d="M1 3.5C1 3.22386 1.22386 3 1.5 3H5.29289L6.64645 4.35355C6.74021 4.44732 6.86739 4.5 7 4.5H13.5C13.7761 4.5 14 4.72386 14 5V12.5C14 12.7761 13.7761 13 13.5 13H1.5C1.22386 13 1 12.7761 1 12.5V3.5Z"
                fill="currentColor"
            />
        </svg>
    );
}

/** 音频文件图标 SVG */
function AudioIcon({ className }: { className?: string }) {
    return (
        <svg
            width="14"
            height="14"
            viewBox="0 0 15 15"
            fill="none"
            className={className}
        >
            <path
                d="M7.5 0.75L7.5 14.25M10.5 3L10.5 12M4.5 3L4.5 12M13.5 5.5L13.5 9.5M1.5 5.5L1.5 9.5"
                stroke="currentColor"
                strokeWidth="1.2"
                strokeLinecap="round"
            />
        </svg>
    );
}

export const FileBrowserPanel: React.FC = () => {
    const dispatch = useAppDispatch();
    const { t } = useI18n();
    const tAny = t as (key: string) => string;
    const fb = useAppSelector((state: RootState) => state.fileBrowser);
    const searchInputRef = useRef<HTMLInputElement>(null);
    const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

    // 清除 debounce
    useEffect(
        () => () => {
            if (debounceRef.current) clearTimeout(debounceRef.current);
        },
        [],
    );

    // 预览音量同步
    useEffect(() => {
        audioPreview.setVolume(fb.previewVolume);
    }, [fb.previewVolume]);

    // 组件挂载时，如果有上次的路径，自动加载
    useEffect(() => {
        if (fb.currentPath && fb.entries.length === 0 && !fb.loading) {
            void dispatch(loadDirectory(fb.currentPath));
        }
    }, []); // eslint-disable-line react-hooks/exhaustive-deps

    // 根据搜索模式决定展示配表
    const isSearchMode = fb.searchQuery.trim().length > 0;
    const rawEntries = isSearchMode ? (fb.searchResults ?? []) : fb.entries;
    const trimmedSearchQuery = fb.searchQuery.trim();

    const hasRegexError = useMemo(() => {
        if (!isSearchMode || !fb.regexEnabled || !trimmedSearchQuery) {
            return false;
        }
        try {
            // Validate regex and let UI display an explicit error.
            void new RegExp(trimmedSearchQuery, "i");
            return false;
        } catch {
            return true;
        }
    }, [isSearchMode, fb.regexEnabled, trimmedSearchQuery]);

    // 客户端正则过滤（仅在搜索模式且 regexEnabled 时）
    const regexFilteredEntries = useMemo(() => {
        if (!isSearchMode || !fb.regexEnabled || !trimmedSearchQuery) {
            return rawEntries;
        }
        try {
            const re = new RegExp(trimmedSearchQuery, "i");
            return rawEntries.filter((e) => re.test(e.name));
        } catch {
            return [];
        }
    }, [rawEntries, fb.regexEnabled, trimmedSearchQuery, isSearchMode]);

    // 音频过滤
    const audioFilteredEntries = useMemo(() => {
        if (!fb.audioOnly) return regexFilteredEntries;
        return regexFilteredEntries.filter(
            (e) => e.isDir || isAudioFile(e),
        );
    }, [regexFilteredEntries, fb.audioOnly]);

    // 排序
    const displayEntries = useMemo(() => {
        const sorted = [...audioFilteredEntries];
        switch (fb.sortMode) {
            case "name":
                sorted.sort((a, b) => a.name.localeCompare(b.name));
                break;
            case "date":
                sorted.sort(
                    (a, b) => (b.modifiedTime ?? 0) - (a.modifiedTime ?? 0),
                );
                break;
            case "size":
                sorted.sort((a, b) => (b.size ?? 0) - (a.size ?? 0));
                break;
        }
        // 目录始终排在前面
        sorted.sort((a, b) => (a.isDir === b.isDir ? 0 : a.isDir ? -1 : 1));
        return sorted;
    }, [audioFilteredEntries, fb.sortMode]);

    // 计算展示相对路径（搜索模式下显示文件所在目录）
    function getRelativeDirHint(fullPath: string): string {
        const normalFull = fullPath.replace(/\\/g, "/");
        const normalBase = fb.currentPath
            .replace(/\\/g, "/")
            .replace(/\/$/, "");
        if (
            normalFull.toLowerCase().startsWith(normalBase.toLowerCase() + "/")
        ) {
            const rel = normalFull.slice(normalBase.length + 1);
            const lastSlash = rel.lastIndexOf("/");
            return lastSlash >= 0 ? rel.slice(0, lastSlash) : "";
        }
        return "";
    }

    // 选择文件夹（通过后端 rfd dialog）
    const handleOpenFolder = useCallback(async () => {
        try {
            const { fileBrowserApi } =
                await import("../../services/api/fileBrowser");
            const result = await fileBrowserApi.pickDirectory();
            if (result.ok && !result.canceled && result.path) {
                void dispatch(loadDirectory(result.path));
            }
        } catch {
            // 忽略错误
        }
    }, [dispatch]);

    // 刷新当前目录
    const handleRefresh = useCallback(() => {
        if (fb.currentPath) {
            void dispatch(loadDirectory(fb.currentPath));
        }
    }, [dispatch, fb.currentPath]);

    // 返回上级目录
    const handleParentDir = useCallback(() => {
        if (!fb.currentPath) return;
        // 处理 Windows 和 Unix 路径
        const normalized = fb.currentPath.replace(/\\/g, "/");
        const parts = normalized.split("/").filter(Boolean);
        if (parts.length <= 1) return; // 已经是根目录
        parts.pop();
        // Windows 路径恢复
        let parentPath = parts.join("/");
        if (/^[A-Za-z]:$/.test(parts[0])) {
            parentPath = parts[0] + "/" + parts.slice(1).join("/");
        }
        if (fb.currentPath.includes("\\")) {
            parentPath = parentPath.replace(/\//g, "\\");
        }
        void dispatch(loadDirectory(parentPath));
    }, [dispatch, fb.currentPath]);

    // 进入子目录
    const handleEnterDir = useCallback(
        (dirPath: string) => {
            if (debounceRef.current) clearTimeout(debounceRef.current);
            dispatch(setSearchQuery(""));
            void dispatch(loadDirectory(dirPath));
        },
        [dispatch],
    );

    // ── 多选状态 ───────────────────────────────────────────────────────────
    const [selectedPaths, setSelectedPaths] = useState<Set<string>>(new Set());
    const lastClickedIndexRef = useRef<number>(-1);

    // 获取仅音频的列表用于 shift-range 选择
    const audioEntries = useMemo(
        () => displayEntries.filter(isAudioFile),
        [displayEntries],
    );

    const handleClickAudio = useCallback(
        (entry: FileEntry, ev?: React.MouseEvent) => {
            const idx = audioEntries.findIndex((e) => e.path === entry.path);

            if (ev?.ctrlKey || ev?.metaKey) {
                // Ctrl+click: toggle selection
                setSelectedPaths((prev) => {
                    const next = new Set(prev);
                    if (next.has(entry.path)) next.delete(entry.path);
                    else next.add(entry.path);
                    return next;
                });
                lastClickedIndexRef.current = idx;
                return;
            }

            if (ev?.shiftKey && lastClickedIndexRef.current >= 0) {
                // Shift+click: range selection
                const start = Math.min(lastClickedIndexRef.current, idx);
                const end = Math.max(lastClickedIndexRef.current, idx);
                setSelectedPaths((prev) => {
                    const next = new Set(prev);
                    for (let i = start; i <= end; i++) {
                        next.add(audioEntries[i].path);
                    }
                    return next;
                });
                return;
            }

            // Normal click: clear selection, preview
            setSelectedPaths(new Set());
            lastClickedIndexRef.current = idx;
            dispatch(setPreviewingFile(entry.path));
            void audioPreview.play(entry.path, () => {
                dispatch(setPreviewingFile(null));
            });
        },
        [dispatch, audioEntries],
    );

    // Clear selection when directory changes
    useEffect(() => {
        setSelectedPaths(new Set());
        lastClickedIndexRef.current = -1;
    }, [fb.currentPath]);

    // 拖拽开始 — 使用自定义 pointer 事件实现，替代 HTML5 drag API
    const [dragState, setDragState] = useState<{
        filePath: string;
        fileName: string;
        allFilePaths: string[];
        startX: number;
        startY: number;
        active: boolean; // 超过阈值后才真正激活拖拽
        isRightDrag: boolean; // 右键拖拽标记
    } | null>(null);
    const dragStateRef = useRef(dragState);
    dragStateRef.current = dragState;

    // ghost 元素跟随鼠标
    const ghostRef = useRef<HTMLDivElement | null>(null);

    const DRAG_THRESHOLD = 5; // 像素阈值，防止误触

    const handlePointerDownForDrag = useCallback(
        (e: React.PointerEvent<HTMLDivElement>, entry: FileEntry) => {
            // 允许左键(0)和右键(2)拖拽
            if (e.button !== 0 && e.button !== 2) return;
            // Collect all selected paths (include current entry)
            const paths = selectedPaths.size > 0 && selectedPaths.has(entry.path)
                ? Array.from(selectedPaths)
                : [entry.path];
            // 不拦截 pointer，让 click 事件仍能触发预览
            setDragState({
                filePath: entry.path,
                fileName: entry.name,
                allFilePaths: paths,
                startX: e.clientX,
                startY: e.clientY,
                active: false,
                isRightDrag: e.button === 2,
            });
        },
        [selectedPaths],
    );

    useEffect(() => {
        if (!dragState) return;

        function onPointerMove(e: PointerEvent) {
            const ds = dragStateRef.current;
            if (!ds) return;

            if (!ds.active) {
                const dx = e.clientX - ds.startX;
                const dy = e.clientY - ds.startY;
                if (Math.sqrt(dx * dx + dy * dy) < DRAG_THRESHOLD) return;
                // 激活拖拽
                dragStateRef.current = { ...ds, active: true };
                setDragState(dragStateRef.current);
                // 发送拖拽开始事件
                window.dispatchEvent(
                    new CustomEvent("hifi-file-drag", {
                        detail: {
                            type: "start",
                            filePath: ds.filePath,
                            fileName: ds.fileName,
                            filePaths: ds.allFilePaths,
                            clientX: e.clientX,
                            clientY: e.clientY,
                            isRightDrag: ds.isRightDrag,
                        },
                    }),
                );
                // 异步获取音频时长，获取后通知 TimelinePanel 更新 ghost 宽度
                import("../../services/api/fileBrowser").then(
                    ({ fileBrowserApi }) => {
                        fileBrowserApi
                            .getAudioFileInfo(ds.filePath)
                            .then((info) => {
                                if (
                                    info &&
                                    dragStateRef.current?.filePath ===
                                        ds.filePath
                                ) {
                                    window.dispatchEvent(
                                        new CustomEvent("hifi-file-drag", {
                                            detail: {
                                                type: "duration",
                                                filePath: ds.filePath,
                                                durationSec: info.durationSec,
                                            },
                                        }),
                                    );
                                }
                            })
                            .catch(() => {
                                /* 获取失败则保持默认宽度 */
                            });
                    },
                );
            }

            // 更新 ghost 位置（clamp 到窗口可视范围内，鼠标超出界面时 ghost 停在边缘）
            if (ghostRef.current) {
                const clampedX = Math.max(
                    0,
                    Math.min(e.clientX + 12, window.innerWidth - 100),
                );
                const clampedY = Math.max(
                    0,
                    Math.min(e.clientY + 12, window.innerHeight - 30),
                );
                ghostRef.current.style.left = `${clampedX}px`;
                ghostRef.current.style.top = `${clampedY}px`;
            }

            // 发送拖拽移动事件（TimelinePanel 监听）
            window.dispatchEvent(
                new CustomEvent("hifi-file-drag", {
                    detail: {
                        type: "move",
                        filePath: dragStateRef.current!.filePath,
                        fileName: dragStateRef.current!.fileName,
                        filePaths: dragStateRef.current!.allFilePaths,
                        clientX: e.clientX,
                        clientY: e.clientY,
                        isRightDrag: dragStateRef.current!.isRightDrag,
                    },
                }),
            );
        }

        function onPointerUp(e: PointerEvent) {
            const ds = dragStateRef.current;
            if (ds?.active) {
                // 发送拖拽结束（drop）事件
                window.dispatchEvent(
                    new CustomEvent("hifi-file-drag", {
                        detail: {
                            type: "drop",
                            filePath: ds.filePath,
                            fileName: ds.fileName,
                            filePaths: ds.allFilePaths,
                            clientX: e.clientX,
                            clientY: e.clientY,
                            isRightDrag: ds.isRightDrag,
                        },
                    }),
                );
            }
            setDragState(null);
        }

        // 右键拖拽时抑制浏览器原生右键菜单
        function onContextMenu(e: MouseEvent) {
            if (dragStateRef.current?.isRightDrag) {
                e.preventDefault();
            }
        }

        window.addEventListener("pointermove", onPointerMove);
        window.addEventListener("pointerup", onPointerUp);
        window.addEventListener("contextmenu", onContextMenu, true);
        return () => {
            window.removeEventListener("pointermove", onPointerMove);
            window.removeEventListener("pointerup", onPointerUp);
            window.removeEventListener("contextmenu", onContextMenu, true);
        };
    }, [dragState !== null]); // eslint-disable-line react-hooks/exhaustive-deps

    return (
        <Flex
            direction="column"
            className="h-full bg-qt-window text-qt-text select-none"
        >
            {/* 标题栏 */}
            <Flex
                align="center"
                justify="between"
                className="h-8 px-2 border-b border-qt-border shrink-0"
            >
                <Text size="2" weight="medium" className="truncate">
                    {(t as (key: string) => string)("fb_title")}
                </Text>
                <Flex align="center" gap="1">
                    <IconButton
                        size="1"
                        variant="ghost"
                        color="gray"
                        title={(t as (key: string) => string)("fb_open_folder")}
                        onClick={handleOpenFolder}
                    >
                        <FolderIcon />
                    </IconButton>
                    <IconButton
                        size="1"
                        variant="ghost"
                        color="gray"
                        title={(t as (key: string) => string)("fb_refresh")}
                        onClick={handleRefresh}
                    >
                        <ReloadIcon />
                    </IconButton>
                    <IconButton
                        size="1"
                        variant="ghost"
                        color="gray"
                        title={t("fb_close")}
                        onClick={() => dispatch(setVisible(false))}
                    >
                        <Cross2Icon />
                    </IconButton>
                </Flex>
            </Flex>

            {/* 搜索栏 */}
            <div className="px-2 py-1 border-b border-qt-border shrink-0">
                <TextField.Root
                    ref={searchInputRef}
                    size="1"
                    placeholder={(t as (key: string) => string)(
                        "fb_search_placeholder",
                    )}
                    value={fb.searchQuery}
                    onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
                        const q = e.target.value;
                        dispatch(setSearchQuery(q));
                        if (debounceRef.current)
                            clearTimeout(debounceRef.current);
                        if (q.trim() && fb.currentPath) {
                            const backendQuery = fb.regexEnabled
                                ? ""
                                : q.trim();
                            debounceRef.current = setTimeout(() => {
                                void dispatch(
                                    searchFilesRecursive({
                                        dirPath: fb.currentPath,
                                        query: backendQuery,
                                    }),
                                );
                            }, 300);
                        }
                    }}
                    style={{ backgroundColor: "var(--qt-base)" }}
                >
                    <TextField.Slot>
                        <MagnifyingGlassIcon height="12" width="12" />
                    </TextField.Slot>
                    {fb.searchQuery && (
                        <TextField.Slot>
                            <IconButton
                                size="1"
                                variant="ghost"
                                color="gray"
                                onClick={() => dispatch(setSearchQuery(""))}
                                style={{ width: 16, height: 16 }}
                            >
                                <Cross2Icon width="10" height="10" />
                            </IconButton>
                        </TextField.Slot>
                    )}
                </TextField.Root>

                {/* 正则切换 + 排序 */}
                <Flex align="center" gap="1" mt="1">
                    <IconButton
                        size="1"
                        variant={fb.regexEnabled ? "solid" : "ghost"}
                        color="gray"
                        title={tAny("fb_regex")}
                        onClick={() => {
                            const nextRegexEnabled = !fb.regexEnabled;
                            dispatch(toggleRegex());

                            if (debounceRef.current) {
                                clearTimeout(debounceRef.current);
                            }

                            if (trimmedSearchQuery && fb.currentPath) {
                                void dispatch(
                                    searchFilesRecursive({
                                        dirPath: fb.currentPath,
                                        query: nextRegexEnabled
                                            ? ""
                                            : trimmedSearchQuery,
                                    }),
                                );
                            }
                        }}
                        style={{
                            fontFamily: "monospace",
                            fontSize: 10,
                            width: 22,
                            height: 22,
                        }}
                    >
                        .*
                    </IconButton>
                    <IconButton
                        size="1"
                        variant={fb.audioOnly ? "solid" : "ghost"}
                        color="gray"
                        title={tAny("fb_audio_only")}
                        onClick={() => dispatch(toggleAudioOnly())}
                        style={{
                            width: 22,
                            height: 22,
                        }}
                    >
                        <AudioIcon />
                    </IconButton>
                    <Select.Root
                        value={fb.sortMode}
                        size="1"
                        onValueChange={(v) =>
                            dispatch(setSortMode(v as SortMode))
                        }
                    >
                        <Select.Trigger
                            style={{ fontSize: 11, height: 22, flex: 1 }}
                        />
                        <Select.Content>
                            <Select.Item value="name">
                                {tAny("fb_sort_name")}
                            </Select.Item>
                            <Select.Item value="date">
                                {tAny("fb_sort_date")}
                            </Select.Item>
                            <Select.Item value="size">
                                {tAny("fb_sort_size")}
                            </Select.Item>
                        </Select.Content>
                    </Select.Root>
                </Flex>

                {hasRegexError && (
                    <Text size="1" color="red" mt="1">
                        {tAny("fb_regex_error")}
                    </Text>
                )}
            </div>

            {/* 路径栏 */}
            {fb.currentPath && (
                <Flex
                    align="center"
                    gap="1"
                    className="px-2 py-1 border-b border-qt-border shrink-0 min-h-[28px]"
                >
                    <IconButton
                        size="1"
                        variant="ghost"
                        color="gray"
                        title={(t as (key: string) => string)("fb_parent_dir")}
                        onClick={handleParentDir}
                    >
                        <ChevronUpIcon />
                    </IconButton>
                    <Text
                        size="1"
                        color="gray"
                        className="truncate flex-1"
                        title={fb.currentPath}
                    >
                        {fb.currentPath}
                    </Text>
                </Flex>
            )}

            {/* 文件列表 */}
            <ScrollArea className="flex-1 min-h-0" scrollbars="vertical">
                <div className="py-1">
                    {fb.loading ? (
                        <Text
                            size="1"
                            color="gray"
                            className="px-3 py-4 block text-center"
                        >
                            {(t as (key: string) => string)("fb_loading")}
                        </Text>
                    ) : fb.error ? (
                        <Text
                            size="1"
                            color="red"
                            className="px-3 py-4 block text-center"
                        >
                            {(t as (key: string) => string)("fb_error")}:{" "}
                            {fb.error}
                        </Text>
                    ) : !fb.currentPath ? (
                        <Text
                            size="1"
                            color="gray"
                            className="px-3 py-4 block text-center"
                        >
                            {(t as (key: string) => string)("fb_no_folder")}
                        </Text>
                    ) : isSearchMode && fb.searchLoading ? (
                        <Text
                            size="1"
                            color="gray"
                            className="px-3 py-4 block text-center"
                        >
                            {(t as (key: string) => string)("fb_searching")}
                        </Text>
                    ) : displayEntries.length === 0 ? (
                        <Text
                            size="1"
                            color="gray"
                            className="px-3 py-4 block text-center"
                        >
                            {isSearchMode
                                ? (t as (key: string) => string)(
                                      "fb_no_results",
                                  )
                                : (t as (key: string) => string)(
                                      "fb_empty_folder",
                                  )}
                        </Text>
                    ) : (
                        displayEntries.map((entry) => (
                            <FileEntryRow
                                key={entry.path}
                                entry={entry}
                                isPlaying={fb.previewingFile === entry.path}
                                isSelected={selectedPaths.has(entry.path)}
                                onDoubleClickDir={handleEnterDir}
                                onClickAudio={handleClickAudio}
                                onPointerDownForDrag={handlePointerDownForDrag}
                                isDragging={
                                    dragState?.active === true &&
                                    dragState.allFilePaths.includes(entry.path)
                                }
                                pathHint={
                                    isSearchMode
                                        ? getRelativeDirHint(entry.path)
                                        : undefined
                                }
                            />
                        ))
                    )}
                </div>
            </ScrollArea>

            {/* 底部音量滑块 */}
            <Flex
                align="center"
                gap="2"
                className="px-2 py-1.5 border-t border-qt-border shrink-0"
            >
                <SpeakerLoudIcon
                    width="14"
                    height="14"
                    className="text-qt-text-muted shrink-0"
                />
                <Slider
                    size="1"
                    min={0}
                    max={100}
                    step={1}
                    value={[Math.round(fb.previewVolume * 100)]}
                    onValueChange={(values: number[]) => {
                        dispatch(setPreviewVolume(values[0] / 100));
                    }}
                    className="flex-1"
                />
                <Text
                    size="1"
                    color="gray"
                    className="w-[32px] text-right shrink-0"
                >
                    {Math.round(fb.previewVolume * 100)}%
                </Text>
            </Flex>

            {/* 拖拽 ghost 元素 */}
            {dragState?.active && (
                <div
                    ref={ghostRef}
                    style={{
                        position: "fixed",
                        left: 0,
                        top: 0,
                        pointerEvents: "none",
                        zIndex: 99999,
                        background: "var(--qt-highlight)",
                        color: "#fff",
                        padding: "2px 8px",
                        borderRadius: 4,
                        fontSize: 11,
                        whiteSpace: "nowrap",
                        opacity: 0.9,
                        boxShadow: "0 2px 8px rgba(0,0,0,0.3)",
                    }}
                >
                    🎵 {dragState.allFilePaths.length > 1
                        ? `${dragState.fileName} (+${dragState.allFilePaths.length - 1})`
                        : dragState.fileName}
                </div>
            )}
        </Flex>
    );
};

// ============================================================
// 文件条目行组件
// ============================================================

interface FileEntryRowProps {
    entry: FileEntry;
    isPlaying: boolean;
    isSelected?: boolean;
    onDoubleClickDir: (dirPath: string) => void;
    onClickAudio: (entry: FileEntry, ev?: React.MouseEvent) => void;
    onPointerDownForDrag: (
        e: React.PointerEvent<HTMLDivElement>,
        entry: FileEntry,
    ) => void;
    isDragging: boolean;
    pathHint?: string;
}

const FileEntryRow: React.FC<FileEntryRowProps> = React.memo(
    ({
        entry,
        isPlaying,
        isSelected,
        onDoubleClickDir,
        onClickAudio,
        onPointerDownForDrag,
        isDragging,
        pathHint,
    }) => {
        const isAudio = isAudioFile(entry);

        return (
            <div
                className={[
                    "flex items-center gap-1.5 px-2 py-[3px] cursor-default",
                    "hover:bg-[color-mix(in_oklab,var(--qt-highlight)_12%,transparent)]",
                    isSelected
                        ? "bg-[color-mix(in_oklab,var(--qt-highlight)_25%,transparent)]"
                        : isPlaying
                        ? "bg-[color-mix(in_oklab,var(--qt-highlight)_20%,transparent)]"
                        : "",
                    isDragging ? "opacity-50" : "",
                    !entry.isDir && !isAudio ? "opacity-50" : "",
                ]
                    .filter(Boolean)
                    .join(" ")}
                onPointerDown={
                    isAudio ? (e) => onPointerDownForDrag(e, entry) : undefined
                }
                onDoubleClick={
                    entry.isDir ? () => onDoubleClickDir(entry.path) : undefined
                }
                onClick={isAudio ? (ev: React.MouseEvent) => onClickAudio(entry, ev) : undefined}
            >
                {/* 图标 */}
                <span className="shrink-0 w-[14px] flex items-center justify-center">
                    {entry.isDir ? (
                        <FolderIcon className="text-yellow-500" />
                    ) : isAudio ? (
                        isPlaying ? (
                            <StopIcon
                                width="12"
                                height="12"
                                className="text-qt-highlight"
                            />
                        ) : (
                            <AudioIcon className="text-blue-400" />
                        )
                    ) : (
                        <FileIcon
                            width="12"
                            height="12"
                            className="text-qt-text-muted"
                        />
                    )}
                </span>

                {/* 文件名 + 路径提示 */}
                <div className="flex flex-col min-w-0 flex-1">
                    <Text size="1" className="truncate" title={entry.name}>
                        {entry.name}
                        {entry.isDir ? "/" : ""}
                    </Text>
                    {pathHint && (
                        <Text
                            size="1"
                            color="gray"
                            className="truncate leading-none"
                            style={{ fontSize: 10 }}
                        >
                            {pathHint}
                        </Text>
                    )}
                </div>

                {/* 右侧信息 */}
                {!entry.isDir && entry.size != null && (
                    <Text
                        size="1"
                        color="gray"
                        className="shrink-0 text-[10px]"
                    >
                        {formatSize(entry.size)}
                    </Text>
                )}

                {/* 音频播放指示 */}
                {isPlaying && (
                    <PlayIcon
                        width="10"
                        height="10"
                        className="shrink-0 text-qt-highlight animate-pulse"
                    />
                )}
            </div>
        );
    },
);

FileEntryRow.displayName = "FileEntryRow";
