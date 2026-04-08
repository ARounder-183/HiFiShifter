import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Text, Select, IconButton } from "@radix-ui/themes";
import { MagnifyingGlassIcon } from "@radix-ui/react-icons";
import { useAppDispatch, useAppSelector } from "../../app/hooks";
import type { RootState } from "../../app/store";
import { useI18n } from "../../i18n/I18nProvider";
import {
    selectMergedKeybindings,
    matchesKeybinding,
    formatKeybinding,
} from "../../features/keybindings";
import type { Keybinding } from "../../features/keybindings";
import { searchFilesRecursive } from "../../features/fileBrowser/fileBrowserSlice";
import { audioPreview } from "../../features/fileBrowser/audioPreview";
import { importAudioAtPosition } from "../../features/session/thunks/importThunks";
import type { FileEntry } from "../../services/api/fileBrowser";
import {
    getQuickSearchInitialPosition,
    QUICK_SEARCH_POPUP_HEIGHT,
    QUICK_SEARCH_POPUP_WIDTH,
} from "./quickSearchPosition";
import { applySelectWheelChange } from "../../utils/selectWheel";

/** 支持的音频扩展名 */
const AUDIO_EXTENSIONS = new Set(["wav", "mp3", "flac", "ogg", "aac", "aif", "aiff", "m4a"]);
const SORT_MODE_OPTIONS = ["name", "date", "size"] as const;

function isAudioFile(entry: FileEntry): boolean {
    return !entry.isDir && !!entry.extension && AUDIO_EXTENSIONS.has(entry.extension);
}

interface QuickSearchPopupProps {
    open: boolean;
    onClose: () => void;
}

/**
 * 快速搜索弹窗组件
 * - 在鼠标位置弹出浮动搜索框
 * - 搜索当前文件管理选中文件夹下的音频文件
 * - ↑/↓ 切换候选项，空格预览，回车放置到当前轨道+playhead位置
 */
export const QuickSearchPopup: React.FC<QuickSearchPopupProps> = ({ open, onClose }) => {
    const dispatch = useAppDispatch();
    const { t } = useI18n();
    const tAny = t as (key: string) => string;

    const keybindings = useAppSelector(selectMergedKeybindings);

    const currentPath = useAppSelector((state: RootState) => state.fileBrowser.currentPath);
    const selectedTrackId = useAppSelector((state: RootState) => state.session.selectedTrackId);
    const playheadSec = useAppSelector((state: RootState) => state.session.playheadSec);

    const [query, setQuery] = useState("");
    const [results, setResults] = useState<FileEntry[]>([]);
    const [selectedIndex, setSelectedIndex] = useState(0);
    const [loading, setLoading] = useState(false);
    const [regexEnabled, setRegexEnabled] = useState(false);
    const [sortMode, setSortMode] = useState<"name" | "date" | "size">("name");
    const [position, setPosition] = useState<{ x: number; y: number }>(() =>
        getQuickSearchInitialPosition({
            viewportWidth:
                typeof window === "undefined" ? QUICK_SEARCH_POPUP_WIDTH : window.innerWidth,
            viewportHeight:
                typeof window === "undefined" ? QUICK_SEARCH_POPUP_HEIGHT : window.innerHeight,
            pointer: null,
        }),
    );
    const [previewingPath, setPreviewingPath] = useState<string | null>(null);

    const inputRef = useRef<HTMLInputElement>(null);
    const listRef = useRef<HTMLDivElement>(null);
    const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
    const popupRef = useRef<HTMLDivElement>(null);
    const lastPointerRef = useRef<{ x: number; y: number } | null>(null);

    useEffect(() => {
        const handlePointerMove = (event: PointerEvent) => {
            lastPointerRef.current = { x: event.clientX, y: event.clientY };
        };

        window.addEventListener("pointermove", handlePointerMove);
        return () => {
            window.removeEventListener("pointermove", handlePointerMove);
        };
    }, []);

    // 打开时使用最近一次鼠标位置，若没有则回退到窗口中心
    useEffect(() => {
        if (!open) return;

        setPosition(
            getQuickSearchInitialPosition({
                viewportWidth: window.innerWidth,
                viewportHeight: window.innerHeight,
                pointer: lastPointerRef.current,
            }),
        );

        // 重置状态
        setQuery("");
        setResults([]);
        setSelectedIndex(0);
        setLoading(false);
        setPreviewingPath(null);

        // 聚焦输入框
        requestAnimationFrame(() => {
            inputRef.current?.focus();
        });

        return () => {};
    }, [open]);

    // 关闭时停止预览
    useEffect(() => {
        if (!open && previewingPath) {
            audioPreview.stop();
            setPreviewingPath(null);
        }
    }, [open]);

    useEffect(() => {
        if (open) {
            document.body.setAttribute("data-quick-search-open", "1");
        } else {
            document.body.removeAttribute("data-quick-search-open");
        }
        return () => {
            document.body.removeAttribute("data-quick-search-open");
        };
    }, [open]);

    // 点击外部关闭由全屏遮罩层处理，见 render 部分

    // 搜索逻辑（带防抖）
    const doSearch = useCallback(
        (q: string) => {
            if (debounceRef.current) clearTimeout(debounceRef.current);
            if (!q.trim() || !currentPath) {
                setResults([]);
                setSelectedIndex(0);
                setLoading(false);
                return;
            }
            setLoading(true);
            debounceRef.current = setTimeout(async () => {
                try {
                    const action = await dispatch(
                        searchFilesRecursive({
                            dirPath: currentPath,
                            query: regexEnabled ? "" : q.trim(),
                        }),
                    );
                    if (searchFilesRecursive.fulfilled.match(action)) {
                        let audioResults = (action.payload as FileEntry[]).filter(isAudioFile);
                        // 正则模式下进行客户端过滤
                        if (regexEnabled) {
                            try {
                                const re = new RegExp(q.trim(), "i");
                                audioResults = audioResults.filter((e) => {
                                    const name = e.name || "";
                                    const dot = name.lastIndexOf(".");
                                    const stem = dot > 0 ? name.substring(0, dot) : name;
                                    return re.test(stem);
                                });
                            } catch {
                                // 正则无效，返回空结果
                                audioResults = [];
                            }
                        }
                        setResults(audioResults);
                        setSelectedIndex(0);
                    }
                } catch {
                    // 忽略搜索错误
                } finally {
                    setLoading(false);
                }
            }, 200);
        },
        [dispatch, currentPath, regexEnabled],
    );

    // 输入变化
    const handleInputChange = useCallback(
        (e: React.ChangeEvent<HTMLInputElement>) => {
            const value = e.target.value;
            setQuery(value);
            doSearch(value);
        },
        [doSearch],
    );

    // 当 regexEnabled 变化时根据当前查询重新搜索
    useEffect(() => {
        if (query.trim()) {
            doSearch(query);
        }
    }, [regexEnabled, query, doSearch]);

    // 排序后的结果
    const sortedResults = useMemo(() => {
        const sorted = [...results];
        switch (sortMode) {
            case "name":
                sorted.sort((a, b) => a.name.localeCompare(b.name));
                break;
            case "date":
                sorted.sort((a, b) => (b.modifiedTime ?? 0) - (a.modifiedTime ?? 0));
                break;
            case "size":
                sorted.sort((a, b) => (b.size ?? 0) - (a.size ?? 0));
                break;
        }
        return sorted;
    }, [results, sortMode]);

    // 预览播放（始终从头重新播放）
    const handlePreview = useCallback((filePath: string) => {
        audioPreview.stop();
        setPreviewingPath(filePath);
        void audioPreview.play(filePath, () => {
            setPreviewingPath(null);
        });
    }, []);

    // 确认放置音频
    const handleConfirm = useCallback(
        (entry: FileEntry) => {
            if (!selectedTrackId) return;
            audioPreview.stop();
            setPreviewingPath(null);
            void dispatch(
                importAudioAtPosition({
                    audioPath: entry.path,
                    trackId: selectedTrackId,
                    startSec: playheadSec ?? 0,
                }),
            );
            onClose();
        },
        [dispatch, selectedTrackId, playheadSec, onClose],
    );

    const focusSearchInput = useCallback(() => {
        if (inputRef.current?.disabled) return;

        requestAnimationFrame(() => {
            inputRef.current?.focus();
        });
    }, []);

    // 将原生 React.KeyboardEvent 适配为 DOM KeyboardEvent 进行匹配
    const matchKey = useCallback(
        (e: React.KeyboardEvent<HTMLInputElement>, kb: Keybinding): boolean => {
            return matchesKeybinding(e.nativeEvent, kb);
        },
        [],
    );

    // 键盘事件处理
    const handleKeyDown = useCallback(
        (e: React.KeyboardEvent<HTMLInputElement>) => {
            if (matchKey(e, keybindings["quickSearch.navigate.down"])) {
                e.preventDefault();
                setSelectedIndex((prev) => {
                    const next = Math.min(prev + 1, sortedResults.length - 1);
                    const entry = sortedResults[next];
                    if (entry && isAudioFile(entry)) {
                        audioPreview.stop();
                        setPreviewingPath(entry.path);
                        void audioPreview.play(entry.path, () => setPreviewingPath(null));
                    }
                    return next;
                });
            } else if (matchKey(e, keybindings["quickSearch.navigate.up"])) {
                e.preventDefault();
                setSelectedIndex((prev) => {
                    const next = Math.max(prev - 1, 0);
                    const entry = sortedResults[next];
                    if (entry && isAudioFile(entry)) {
                        audioPreview.stop();
                        setPreviewingPath(entry.path);
                        void audioPreview.play(entry.path, () => setPreviewingPath(null));
                    }
                    return next;
                });
            } else if (matchKey(e, keybindings["quickSearch.preview"])) {
                // 预览试听（仅当有结果时）
                if (sortedResults.length > 0) {
                    e.preventDefault();
                    const entry = sortedResults[selectedIndex];
                    if (entry) handlePreview(entry.path);
                }
            } else if (matchKey(e, keybindings["quickSearch.confirm"])) {
                e.preventDefault();
                if (sortedResults.length > 0 && sortedResults[selectedIndex]) {
                    handleConfirm(sortedResults[selectedIndex]);
                }
            } else if (matchKey(e, keybindings["quickSearch.close"])) {
                e.preventDefault();
                audioPreview.stop();
                setPreviewingPath(null);
                onClose();
            }
        },
        [
            sortedResults,
            selectedIndex,
            handlePreview,
            handleConfirm,
            onClose,
            keybindings,
            matchKey,
        ],
    );

    // 滚动选中项到可见区域
    useEffect(() => {
        if (!listRef.current) return;
        const items = listRef.current.querySelectorAll("[data-qs-item]");
        const activeItem = items[selectedIndex] as HTMLElement | undefined;
        activeItem?.scrollIntoView({ block: "nearest" });
    }, [selectedIndex]);

    // 清理 debounce
    useEffect(
        () => () => {
            if (debounceRef.current) clearTimeout(debounceRef.current);
        },
        [],
    );

    if (!open) return null;

    const noFolder = !currentPath;

    return (
        <>
            {/* 全屏透明遮罩层 —— 点击即关闭弹窗 */}
            <div
                className="fixed inset-0 z-[99998]"
                style={{ background: "transparent" }}
                onMouseDown={(e) => {
                    e.stopPropagation();
                    audioPreview.stop();
                    setPreviewingPath(null);
                    onClose();
                }}
            />
            <div
                ref={popupRef}
                className="fixed z-[99999] flex flex-col"
                style={{
                    left: position.x,
                    top: position.y,
                    width: QUICK_SEARCH_POPUP_WIDTH,
                    maxHeight: QUICK_SEARCH_POPUP_HEIGHT,
                    background: "var(--qt-panel)",
                    border: "1px solid var(--qt-border)",
                    borderRadius: 10,
                    boxShadow: "0 20px 44px rgba(0,0,0,0.28)",
                    overflow: "hidden",
                }}
            >
                {/* 搜索输入框 */}
                <div className="flex items-center gap-1.5 px-2 py-1.5 border-b border-qt-border">
                    <MagnifyingGlassIcon
                        width="14"
                        height="14"
                        className="text-qt-text-muted shrink-0"
                    />
                    <input
                        ref={inputRef}
                        type="text"
                        value={query}
                        onChange={handleInputChange}
                        onKeyDown={handleKeyDown}
                        placeholder={
                            noFolder
                                ? (t as (key: string) => string)("qs_no_folder") || "请先选择文件夹"
                                : (t as (key: string) => string)("qs_placeholder") ||
                                  "搜索音频文件..."
                        }
                        disabled={noFolder}
                        className="flex-1 bg-transparent border-none outline-none text-qt-text text-xs placeholder:text-qt-text-muted"
                        autoComplete="off"
                        spellCheck={false}
                    />
                    {/* 正则切换 */}
                    <IconButton
                        size="1"
                        variant={regexEnabled ? "solid" : "ghost"}
                        color="gray"
                        title={tAny("fb_regex") || "Regex"}
                        onClick={() => {
                            setRegexEnabled((v) => !v);
                            focusSearchInput();
                        }}
                        style={{
                            fontFamily: "monospace",
                            fontSize: 10,
                            width: 20,
                            height: 20,
                            flexShrink: 0,
                        }}
                    >
                        .*
                    </IconButton>
                    {/* 排序 */}
                    <Select.Root
                        value={sortMode}
                        size="1"
                        onValueChange={(v) => {
                            setSortMode(v as "name" | "date" | "size");
                        }}
                    >
                        <Select.Trigger
                            style={{
                                fontSize: 10,
                                height: 20,
                                minWidth: 52,
                                flexShrink: 0,
                            }}
                            onWheel={(event) => {
                                applySelectWheelChange({
                                    event,
                                    currentValue: sortMode,
                                    options: SORT_MODE_OPTIONS,
                                    onChange: (next) => {
                                        setSortMode(next as "name" | "date" | "size");
                                        focusSearchInput();
                                    },
                                });
                            }}
                        />
                        <Select.Content
                            onCloseAutoFocus={(event) => {
                                event.preventDefault();
                                focusSearchInput();
                            }}
                        >
                            <Select.Item value="name">{tAny("fb_sort_name") || "Name"}</Select.Item>
                            <Select.Item value="date">{tAny("fb_sort_date") || "Date"}</Select.Item>
                            <Select.Item value="size">{tAny("fb_sort_size") || "Size"}</Select.Item>
                        </Select.Content>
                    </Select.Root>
                    {loading && (
                        <span className="text-[10px] text-qt-text-muted shrink-0">...</span>
                    )}
                </div>

                {/* 候选列表 */}
                <div
                    ref={listRef}
                    className="flex-1 overflow-y-auto min-h-0"
                    style={{ maxHeight: 340 }}
                >
                    {noFolder ? (
                        <Text size="1" color="gray" className="px-3 py-4 block text-center">
                            {(t as (key: string) => string)("qs_no_folder_hint") ||
                                "请先在文件管理器中选择目录"}
                        </Text>
                    ) : !query.trim() ? (
                        <Text size="1" color="gray" className="px-3 py-4 block text-center">
                            {(t as (key: string) => string)("qs_type_to_search") ||
                                "输入关键词搜索音频文件"}
                        </Text>
                    ) : loading ? (
                        <Text size="1" color="gray" className="px-3 py-4 block text-center">
                            {(t as (key: string) => string)("fb_searching") || "搜索中..."}
                        </Text>
                    ) : sortedResults.length === 0 ? (
                        <Text size="1" color="gray" className="px-3 py-4 block text-center">
                            {(t as (key: string) => string)("fb_no_results") || "无匹配文件"}
                        </Text>
                    ) : (
                        sortedResults.map((entry, index) => (
                            <div
                                key={entry.path}
                                data-qs-item
                                className={[
                                    "flex items-center gap-1.5 px-2 py-[4px] cursor-pointer text-xs",
                                    index === selectedIndex
                                        ? "bg-[color-mix(in_oklab,var(--qt-highlight)_25%,transparent)]"
                                        : "hover:bg-[color-mix(in_oklab,var(--qt-highlight)_10%,transparent)]",
                                    previewingPath === entry.path
                                        ? "text-qt-highlight"
                                        : "text-qt-text",
                                ]
                                    .filter(Boolean)
                                    .join(" ")}
                                onClick={() => handleConfirm(entry)}
                                onMouseEnter={() => setSelectedIndex(index)}
                            >
                                {/* 音频图标 */}
                                <svg
                                    width="12"
                                    height="12"
                                    viewBox="0 0 15 15"
                                    fill="none"
                                    className="shrink-0"
                                >
                                    <path
                                        d="M7.5 0.75L7.5 14.25M10.5 3L10.5 12M4.5 3L4.5 12M13.5 5.5L13.5 9.5M1.5 5.5L1.5 9.5"
                                        stroke="currentColor"
                                        strokeWidth="1.2"
                                        strokeLinecap="round"
                                    />
                                </svg>
                                {/* 文件名 */}
                                <span className="truncate flex-1" title={entry.name}>
                                    {entry.name}
                                </span>
                                {/* 预览指示 */}
                                {previewingPath === entry.path && (
                                    <span className="shrink-0 text-[10px] text-qt-highlight animate-pulse">
                                        ♫
                                    </span>
                                )}
                            </div>
                        ))
                    )}
                </div>

                {/* 底部提示栏 */}
                {sortedResults.length > 0 && (
                    <div className="px-2 py-1 border-t border-qt-border flex items-center gap-2">
                        <Text size="1" color="gray" className="text-[10px]">
                            {formatKeybinding(keybindings["quickSearch.navigate.up"])}/
                            {formatKeybinding(keybindings["quickSearch.navigate.down"])}{" "}
                            {(t as (key: string) => string)("qs_hint_nav") || "导航"}
                            {"  "}
                            {formatKeybinding(keybindings["quickSearch.preview"])}{" "}
                            {(t as (key: string) => string)("qs_hint_preview") || "预览"}
                            {"  "}
                            {formatKeybinding(keybindings["quickSearch.confirm"])}{" "}
                            {(t as (key: string) => string)("qs_hint_place") || "放置"}
                            {"  "}
                            {formatKeybinding(keybindings["quickSearch.close"])}{" "}
                            {(t as (key: string) => string)("qs_hint_close") || "关闭"}
                        </Text>
                    </div>
                )}
            </div>
        </>
    );
};
