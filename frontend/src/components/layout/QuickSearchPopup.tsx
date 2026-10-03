import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { ChevronDownIcon, ChevronUpIcon, MagnifyingGlassIcon } from "@radix-ui/react-icons";
import { useAppDispatch, useAppSelector } from "../../app/hooks";
import type { RootState } from "../../app/store";
import { useI18n } from "../../i18n/I18nProvider";
import { selectMergedKeybindings, matchesKeybinding } from "../../features/keybindings";
import type { Keybinding } from "../../features/keybindings";
import {
    searchFilesRecursive,
    FILE_BROWSER_COMPUTER_PATH,
} from "../../features/fileBrowser/fileBrowserSlice";
import { searchOptionsPayload } from "../../features/search/searchSettings";
import { SearchTranslitToggle } from "./search/SearchTranslitToggle";
import { matchReasonOf } from "./search/matchReason";
import { usePreviewToggle } from "../../features/fileBrowser/usePreviewToggle";
import { isAudioFile, isMediaFile, isMidiFile } from "../../features/fileBrowser/fileKinds";
import { compareFileNames } from "../../features/fileBrowser/fileNameCompare";
import { DEFAULT_SORT_DESCENDING } from "../../features/fileBrowser/fileBrowserViewOptions";
import { emitImportMidiRequest } from "../../features/session/projectOpenEvents";
import { FileKindIcon } from "./fileBrowser/fileIcons";
import { importAudioAtPosition } from "../../features/session/thunks/importThunks";
import {
    persistUiSettings,
    setSearchSettings,
    setSearchSettingsDialogOpen,
    toggleQuickSearchAutoNormalize,
} from "../../features/session/sessionSlice";
import type { FileEntry } from "../../services/api/fileBrowser";
import {
    getQuickSearchInitialPosition,
    QUICK_SEARCH_POPUP_HEIGHT,
    QUICK_SEARCH_POPUP_WIDTH,
} from "./quickSearchPosition";
import { AppBusy, AppEmptyState, AppIconButton, AppSelect } from "../../ui";
import { AppForm, AppSwitchRow } from "../../ui/Field";
import { useShortcutSuppression } from "../../ui/shortcutScope";

interface QuickSearchPopupProps {
    open: boolean;
    onClose: () => void;
}

/**
 * 快速搜索弹窗组件
 * - 在鼠标位置弹出浮动搜索框
 * - 搜索当前文件管理选中文件夹下的音频/视频媒体文件
 * - ↑/↓ 切换候选项，空格预览，回车放置到当前轨道+playhead位置
 */
export const QuickSearchPopup: React.FC<QuickSearchPopupProps> = ({ open, onClose }) => {
    const dispatch = useAppDispatch();
    const { t, tVars } = useI18n();

    const keybindings = useAppSelector(selectMergedKeybindings);

    const currentPath = useAppSelector((state: RootState) => state.fileBrowser.currentPath);
    const selectedTrackId = useAppSelector((state: RootState) => state.session.selectedTrackId);
    const playheadSec = useAppSelector((state: RootState) => state.session.playheadSec);
    const quickSearchAutoNormalizeEnabled = useAppSelector(
        (state: RootState) => state.session.quickSearchAutoNormalizeEnabled,
    );
    const searchSettings = useAppSelector((state: RootState) => state.session.searchSettings);

    const [query, setQuery] = useState("");
    const [results, setResults] = useState<FileEntry[]>([]);
    const [selectedIndex, setSelectedIndex] = useState(0);
    const [loading, setLoading] = useState(false);
    const [regexEnabled, setRegexEnabled] = useState(false);
    const [sortMode, setSortMode] = useState<"name" | "date" | "size">("name");
    // 初值与文件浏览器一致：名称升序（切换依据时按该依据的自然方向重置）。
    const [sortDescending, setSortDescending] = useState(DEFAULT_SORT_DESCENDING.name);
    const [position, setPosition] = useState<{ x: number; y: number }>(() =>
        getQuickSearchInitialPosition({
            viewportWidth:
                typeof window === "undefined" ? QUICK_SEARCH_POPUP_WIDTH : window.innerWidth,
            viewportHeight:
                typeof window === "undefined" ? QUICK_SEARCH_POPUP_HEIGHT : window.innerHeight,
            pointer: null,
        }),
    );
    // 试听状态与引擎调用统一走共享 hook（与文件浏览器同一份实现与同一份真值）。
    const {
        previewingFile: previewingPath,
        play: playPreview,
        stop: stopPreview,
    } = usePreviewToggle();

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
        stopPreview();

        // 聚焦输入框
        requestAnimationFrame(() => {
            inputRef.current?.focus();
        });

        return () => {};
    }, [open, stopPreview]);

    // 关闭时停止预览
    useEffect(() => {
        if (!open) stopPreview();
    }, [open, stopPreview]);

    // 抑制全局快捷键，交给弹窗自身输入框处理（避免 ↑/↓ 与时间轴缩放冲突）。
    // 走统一作用域，取代此前的 `data-quick-search-open` body 属性。
    useShortcutSuppression(open);

    // 点击外部关闭由全屏遮罩层处理，见 render 部分

    /*
     * 下发给后端的匹配参数。正则模式下强制 `off`：正则作用于**原文**，与转写互斥
     * （把 `zhuge` 当正则去匹配「主歌」没有意义）。
     */
    const searchOptions = useMemo(() => {
        const payload = searchOptionsPayload(searchSettings);
        return regexEnabled ? { ...payload, mode: "off" as const } : payload;
    }, [searchSettings, regexEnabled]);

    // 搜索逻辑（带防抖）
    const doSearch = useCallback(
        (q: string) => {
            if (debounceRef.current) clearTimeout(debounceRef.current);
            if (!q.trim() || !currentPath || currentPath === FILE_BROWSER_COMPUTER_PATH) {
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
                            options: searchOptions,
                        }),
                    );
                    if (searchFilesRecursive.fulfilled.match(action)) {
                        // 与文件浏览器开启「仅显示媒体文件」时**同一判据**
                        // （`isMediaFile` = 音频/视频 + MIDI）：两个界面对同一份目录
                        // 必须给出同一批候选。
                        let mediaResults = (action.payload as FileEntry[]).filter(isMediaFile);
                        // 正则模式下进行客户端过滤
                        if (regexEnabled) {
                            try {
                                const re = new RegExp(q.trim(), "i");
                                mediaResults = mediaResults.filter((e) => {
                                    const name = e.name || "";
                                    const dot = name.lastIndexOf(".");
                                    const stem = dot > 0 ? name.substring(0, dot) : name;
                                    return re.test(stem);
                                });
                            } catch {
                                // 正则无效，返回空结果
                                mediaResults = [];
                            }
                        }
                        setResults(mediaResults);
                        setSelectedIndex(0);
                    }
                } catch {
                    // 忽略搜索错误
                } finally {
                    setLoading(false);
                }
            }, 200);
        },
        [dispatch, currentPath, regexEnabled, searchOptions],
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

    // 匹配方式变化时同样重跑：结果是后端算好的，不重跑就看不到任何变化。
    const searchOptionsKey = `${searchOptions.mode}|${searchOptions.heteronym}|${searchOptions.japaneseLongVowel}|${searchOptions.koreanChoseong}`;
    const lastSearchOptionsKeyRef = useRef(searchOptionsKey);
    useEffect(() => {
        if (lastSearchOptionsKeyRef.current === searchOptionsKey) return;
        lastSearchOptionsKeyRef.current = searchOptionsKey;
        if (query.trim()) doSearch(query);
    }, [searchOptionsKey, query, doSearch]);

    // 排序后的结果
    const sortedResults = useMemo(() => {
        const sorted = [...results];
        const direction = sortDescending ? -1 : 1;
        switch (sortMode) {
            case "name":
                // 与文件浏览器同一个比较器：裸 `localeCompare` 既不认数字（take10
                // 会排在 take2 前面），也把汉字排在拉丁之前 —— 同一个目录在两个
                // 界面里给出两种顺序，是没理由的。
                sorted.sort((a, b) => direction * compareFileNames(a.name, b.name));
                break;
            case "date":
                sorted.sort((a, b) => direction * ((a.modifiedTime ?? 0) - (b.modifiedTime ?? 0)));
                break;
            case "size":
                sorted.sort((a, b) => direction * ((a.size ?? 0) - (b.size ?? 0)));
                break;
        }
        return sorted;
    }, [results, sortMode, sortDescending]);

    /**
     * 预览播放（始终从头重新播放）。
     *
     * 【为什么只对音频生效】试听引擎解码的是音频流；候选列表现在也包含 MIDI，
     * 而 MIDI 没有可解码的音频 —— 送进去只会失败并弹一条前端错误日志。
     * 不试听、不报错，静默跳过是对"这条不能试听"最不打扰的表达。
     */
    const handlePreview = useCallback(
        (entry: FileEntry) => {
            if (!isAudioFile(entry)) return;
            playPreview(entry.path);
        },
        [playPreview],
    );

    // 确认放置：音频直接放到时间轴；MIDI 交给导入对话框（它需要选轨与 BPM 处理，
    // 不能按音频那样直接放）。候选列表现在与文件浏览器一致地包含 MIDI，
    // 因此这条分支是必需的 —— 否则选中一条 MIDI 会走音频导入并失败。
    const handleConfirm = useCallback(
        (entry: FileEntry) => {
            stopPreview();
            if (isMidiFile(entry)) {
                emitImportMidiRequest({
                    path: entry.path,
                    startSec: playheadSec ?? 0,
                    trackId: selectedTrackId,
                });
                onClose();
                return;
            }
            if (!selectedTrackId) return;
            void dispatch(
                importAudioAtPosition({
                    audioPath: entry.path,
                    trackId: selectedTrackId,
                    startSec: playheadSec ?? 0,
                    normalizeAfterImport: quickSearchAutoNormalizeEnabled,
                }),
            );
            onClose();
        },
        [
            dispatch,
            onClose,
            playheadSec,
            quickSearchAutoNormalizeEnabled,
            selectedTrackId,
            stopPreview,
        ],
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
                        playPreview(entry.path);
                    }
                    return next;
                });
            } else if (matchKey(e, keybindings["quickSearch.navigate.up"])) {
                e.preventDefault();
                setSelectedIndex((prev) => {
                    const next = Math.max(prev - 1, 0);
                    const entry = sortedResults[next];
                    if (entry && isAudioFile(entry)) {
                        playPreview(entry.path);
                    }
                    return next;
                });
            } else if (matchKey(e, keybindings["quickSearch.preview"])) {
                // 预览试听（仅当有结果时）
                if (sortedResults.length > 0) {
                    e.preventDefault();
                    const entry = sortedResults[selectedIndex];
                    if (entry) handlePreview(entry);
                }
            } else if (matchKey(e, keybindings["quickSearch.confirm"])) {
                e.preventDefault();
                if (sortedResults.length > 0 && sortedResults[selectedIndex]) {
                    handleConfirm(sortedResults[selectedIndex]);
                }
            } else if (matchKey(e, keybindings["quickSearch.close"])) {
                e.preventDefault();
                stopPreview();
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
            playPreview,
            stopPreview,
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

    // 「计算机」虚拟层（盘符列表）没有可递归搜索的目录，与未选文件夹同样对待。
    const noFolder = !currentPath || currentPath === FILE_BROWSER_COMPUTER_PATH;

    return (
        <>
            {/* 全屏透明遮罩层 —— 点击即关闭弹窗 */}
            <div
                className="fixed inset-0 z-[99998]"
                style={{ background: "transparent" }}
                onMouseDown={(e) => {
                    e.stopPropagation();
                    stopPreview();
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
                    borderRadius: "var(--qt-radius-lg)",
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
                        placeholder={noFolder ? t("qs_no_folder") : t("qs_placeholder")}
                        disabled={noFolder}
                        className="flex-1 bg-transparent border-none outline-none text-qt-text text-qt-xs placeholder:text-qt-text-muted"
                        autoComplete="off"
                        spellCheck={false}
                    />
                    {/* 正则切换 */}
                    <AppIconButton
                        active={regexEnabled}
                        tooltip={t("fb_regex")}
                        onClick={() => {
                            setRegexEnabled((v) => !v);
                            focusSearchInput();
                        }}
                        style={{
                            fontFamily: "monospace",
                            fontSize: "var(--qt-fs-micro)",
                            width: 20,
                            height: 20,
                            flexShrink: 0,
                        }}
                        icon=".*"
                    />
                    {/* 拼音匹配开关（与两侧的正则 / 仅媒体同为「点击 = 开/关」） */}
                    <SearchTranslitToggle
                        size={20}
                        settings={searchSettings}
                        onChange={(patch) => {
                            dispatch(setSearchSettings(patch));
                            void dispatch(persistUiSettings());
                            focusSearchInput();
                        }}
                        regexActive={regexEnabled}
                        onOpenSettings={() => {
                            onClose();
                            dispatch(setSearchSettingsDialogOpen(true));
                        }}
                    />
                    {/* 排序 */}
                    <AppSelect
                        fullWidth={false}
                        // 紧凑搜索行里的控件，不是工具条子项 —— 显式声明密度
                        density="compact"
                        value={sortMode}
                        onValueChange={(v) => {
                            const next = v as "name" | "date" | "size";
                            setSortMode(next);
                            // 与文件浏览器同一规则：切换排序依据时采用该依据的自然方向
                            // （名称升序、日期/大小降序），而不是沿用上一次的方向。
                            setSortDescending(DEFAULT_SORT_DESCENDING[next]);
                            focusSearchInput();
                        }}
                        options={[
                            { value: "name", label: t("fb_sort_name") },
                            { value: "date", label: t("fb_sort_date") },
                            { value: "size", label: t("fb_sort_size") },
                        ]}
                    />
                    {/* 升降序：箭头与提示都表达**当前**方向 */}
                    <AppIconButton
                        active={sortDescending}
                        tooltip={sortDescending ? t("fb_sort_descending") : t("fb_sort_ascending")}
                        onClick={() => {
                            setSortDescending((v) => !v);
                            focusSearchInput();
                        }}
                        style={{ width: 20, height: 20, flexShrink: 0 }}
                        icon={sortDescending ? <ChevronDownIcon /> : <ChevronUpIcon />}
                    />
                    {loading && <AppBusy className="shrink-0" />}
                </div>

                {/* 候选列表 */}
                <div
                    ref={listRef}
                    className="hs-scroll-gutter flex-1 overflow-y-auto min-h-0"
                    style={{ maxHeight: 340 }}
                >
                    {noFolder ? (
                        <AppEmptyState>{t("qs_no_folder_hint")}</AppEmptyState>
                    ) : !query.trim() ? (
                        <AppEmptyState>{t("qs_type_to_search")}</AppEmptyState>
                    ) : loading ? (
                        <AppEmptyState>{t("fb_searching")}</AppEmptyState>
                    ) : sortedResults.length === 0 ? (
                        <AppEmptyState>{t("fb_no_results")}</AppEmptyState>
                    ) : (
                        sortedResults.map((entry, index) => {
                            const reason = matchReasonOf(
                                entry.matchInfo,
                                searchSettings.showMatchReason,
                            );
                            return (
                                <div
                                    key={entry.path}
                                    data-qs-item
                                    className={[
                                        "flex items-center gap-1.5 px-2 py-[4px] cursor-pointer text-qt-xs",
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
                                    {/* 图标按类型区分（音频 / 视频 / MIDI）——
                                        与文件浏览器同一份图标来源，不在这里再内联一份 */}
                                    <span className="shrink-0 flex items-center justify-center">
                                        <FileKindIcon entry={entry} />
                                    </span>
                                    {/* 文件名 + 命中原因 */}
                                    <span
                                        className="truncate flex-1 flex items-baseline gap-1"
                                        data-tooltip={entry.name}
                                    >
                                        <span className="truncate">{entry.name}</span>
                                        {reason && (
                                            <span
                                                className="shrink-0 text-qt-text-muted"
                                                style={{ fontSize: "var(--qt-fs-micro)" }}
                                            >
                                                {tVars(reason.key, reason.vars)}
                                            </span>
                                        )}
                                    </span>
                                    {/* 预览指示 */}
                                    {previewingPath === entry.path && (
                                        <span className="shrink-0 text-qt-micro text-qt-highlight animate-pulse">
                                            ♫
                                        </span>
                                    )}
                                </div>
                            );
                        })
                    )}
                </div>

                {/* 底部：放置时自动规格化。
                    快捷键提示此前放在这里，已移除 —— 它是常驻的噪音：用户已经打开
                    了这个弹窗，方向键/回车的行为在第一次使用时就会建立，不需要每次
                    都读一遍；而且它把底栏撑得很宽，把开关挤到左边。 */}
                <div className="px-2 py-1 border-t border-qt-border flex items-center gap-2">
                    <AppForm booleanRow="leading">
                        <AppSwitchRow
                            control="checkbox"
                            label={t("qs_auto_normalize")}
                            checked={quickSearchAutoNormalizeEnabled}
                            onCheckedChange={() => {
                                dispatch(toggleQuickSearchAutoNormalize());
                                void dispatch(persistUiSettings());
                                focusSearchInput();
                            }}
                        />
                    </AppForm>
                </div>
            </div>
        </>
    );
};
