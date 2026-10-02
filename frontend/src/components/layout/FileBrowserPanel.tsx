import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Flex, IconButton, TextField } from "@radix-ui/themes";
import {
    ChevronDownIcon,
    ChevronLeftIcon,
    ChevronRightIcon,
    ChevronUpIcon,
    Cross2Icon,
    GearIcon,
    MagnifyingGlassIcon,
    ReloadIcon,
    SpeakerLoudIcon,
    StarIcon,
} from "@radix-ui/react-icons";
import { useAppDispatch, useAppSelector } from "../../app/hooks";
import type { RootState } from "../../app/store";
import { useI18n } from "../../i18n/I18nProvider";
import {
    loadDirectory,
    setPreviewVolume,
    setSearchQuery,
    searchFilesRecursive,
    toggleRegex,
    FILE_BROWSER_COMPUTER_PATH,
} from "../../features/fileBrowser/fileBrowserSlice";
import { audioPreview } from "../../features/fileBrowser/audioPreview";
import { usePreviewToggle } from "../../features/fileBrowser/usePreviewToggle";
import { searchOptionsPayload } from "../../features/search/searchSettings";
import {
    isAudioFile,
    isMidiFile,
    isProjectFile,
    isReaperFile,
    isVocalShifterFile,
} from "../../features/fileBrowser/fileKinds";
import {
    buildFileBrowserContextMenu,
    type FileBrowserMenuActions,
} from "../../features/fileBrowser/fileBrowserMenu";
import type { FileBrowserViewOptions } from "../../features/fileBrowser/fileBrowserViewOptions";
import { rowDensityOf } from "../../features/fileBrowser/fileBrowserViewOptions";
import { locationLabel, parentDirOf } from "../../features/fileBrowser/fileBrowserPaths";
import { computeListWindow } from "../../features/fileBrowser/listWindow";
import {
    emitExternalFileAction,
    emitImportMidiRequest,
    emitImportProjectPick,
    emitOpenProjectPath,
} from "../../features/session/projectOpenEvents";
import {
    importAudioAtPosition,
    importMultipleAudioAtPosition,
} from "../../features/session/thunks/importThunks";
import { SearchTranslitToggle } from "./search/SearchTranslitToggle";
import { matchReasonOf } from "./search/matchReason";
import {
    persistUiSettings,
    setFileBrowserFavorites,
    setFileBrowserView,
    setSearchSettings,
    setSearchSettingsDialogOpen,
} from "../../features/session/sessionSlice";
import { PanelToolbar, PanelToolbarButton } from "./shared/PanelToolbar";
import { fileBrowserApi, type FileEntry } from "../../services/api/fileBrowser";
import {
    AppContextMenu,
    AppDialog,
    AppEmptyState,
    AppIconButton,
    AppSelect,
    AppSlider,
    AppSliderReadout,
    type AppMenuItemSpec,
} from "../../ui";
import { isPrimaryModifierDown } from "../../utils/platform";
import { copyTextToClipboard } from "../../utils/copyText";
import { DockInlineRename } from "../dock/DockInlineRename";
import { FileEntryRow } from "./fileBrowser/FileEntryRow";
import { formatModified, formatSize } from "./fileBrowser/formatFile";
import { FileKindIcon, FolderIcon } from "./fileBrowser/fileIcons";
import { FilePropertiesDialog } from "./fileBrowser/FilePropertiesDialog";
import { FileBrowserViewOptionsDialog } from "./fileBrowser/FileBrowserViewOptionsDialog";
import {
    TYPE_AHEAD_RESET_MS,
    isFileListActivationKey,
    isFileListNavKey,
    nextActiveIndex,
    nextTypeAhead,
} from "./fileBrowserKeyboardNav";

/**
 * 自然序排序器：`take2` 排在 `take10` 之前。
 *
 * 【为什么不用裸 `localeCompare`】默认比较是纯字典序，`take10` 会排在 `take2`
 * 前面 —— 音频素材几乎总是带序号（take01…take12），这是每天都会撞上的错序。
 * `numeric: true` 打开数字分段比较，`sensitivity: "base"` 让大小写与变音符号
 * 不参与排序（"Apple" 与 "apple" 相邻）。构造一次即可，Collator 本身可复用。
 */
const NAME_COLLATOR = new Intl.Collator(undefined, { numeric: true, sensitivity: "base" });

/** 键盘移动后自动试听的防抖（毫秒）。 */
const PREVIEW_NAV_DEBOUNCE_MS = 160;

/** 面板内的行内编辑状态。 */
type EditingState =
    | { kind: "rename"; path: string; initial: string }
    | { kind: "newFolder" }
    | null;

export const FileBrowserPanel: React.FC = () => {
    const dispatch = useAppDispatch();
    const { t, tf, tVars, plural, number } = useI18n();
    const fb = useAppSelector((state: RootState) => state.fileBrowser);
    const searchSettings = useAppSelector((state: RootState) => state.session.searchSettings);
    const view = useAppSelector((state: RootState) => state.session.fileBrowserView);
    const favorites = useAppSelector((state: RootState) => state.session.fileBrowserFavorites);
    const selectedTrackId = useAppSelector((state: RootState) => state.session.selectedTrackId);
    const playheadSec = useAppSelector((state: RootState) => state.session.playheadSec);

    const searchInputRef = useRef<HTMLInputElement>(null);
    const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
    const previewNavTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

    /*
     * 下发给后端的匹配参数。
     *
     * 【为什么正则模式下强制 off】正则作用于**原文**，与转写互斥：把 `zhuge` 当正则
     * 去匹配「主歌」没有任何意义。前端仍然把 query 传空串（沿用旧行为：后端不过滤，
     * 由前端做正则过滤），匹配模式一并降为 off，让后端走最便宜的路径。
     */
    const searchOptions = useMemo(() => {
        const payload = searchOptionsPayload(searchSettings);
        return fb.regexEnabled ? { ...payload, mode: "off" as const } : payload;
    }, [searchSettings, fb.regexEnabled]);

    /** 把后端的命中信息格式化成「匹配拼音 zhuge」；不需要解释时返回 undefined。 */
    const formatMatchReason = useCallback(
        (entry: FileEntry): string | undefined => {
            const reason = matchReasonOf(entry.matchInfo, searchSettings.showMatchReason);
            return reason ? tVars(reason.key, reason.vars) : undefined;
        },
        [searchSettings.showMatchReason, tVars],
    );

    // 清除 debounce / 自动试听定时器
    useEffect(
        () => () => {
            if (debounceRef.current) clearTimeout(debounceRef.current);
            if (previewNavTimerRef.current) clearTimeout(previewNavTimerRef.current);
        },
        [],
    );

    // 试听切换（播放 / 停止的唯一实现，见 usePreviewToggle）
    const previewToggle = usePreviewToggle();

    // 预览音量同步
    useEffect(() => {
        audioPreview.setVolume(fb.previewVolume);
    }, [fb.previewVolume]);

    // ── 导航历史（后退 / 前进） ─────────────────────────────────────────────
    // 历史是**面板局部**的会话态：不持久化、不进 Redux（换面板布局时丢掉可以接受）。
    const [history, setHistory] = useState<{ entries: string[]; index: number }>({
        entries: [],
        index: -1,
    });
    const canGoBack = history.index > 0;
    const canGoForward = history.index >= 0 && history.index < history.entries.length - 1;

    /**
     * 导航到某个目录。
     *
     * 【为什么所有导航都收口到这里】历史、搜索词清空、防抖取消这三件事必须与
     * "加载目录"同时发生；散在六处各写一遍，迟早有一处漏掉（例如从搜索模式进入
     * 目录后搜索框还留着旧词）。`record: false` 供历史前进/后退使用 —— 它们不该
     * 再往历史里追加。
     */
    const navigateTo = useCallback(
        (path: string, record = true) => {
            if (!path) return;
            if (record) {
                setHistory((prev) => {
                    const base = prev.entries.slice(0, prev.index + 1);
                    if (base[base.length - 1] === path) return prev;
                    const entries = [...base, path];
                    return { entries, index: entries.length - 1 };
                });
            }
            if (debounceRef.current) clearTimeout(debounceRef.current);
            dispatch(setSearchQuery(""));
            void dispatch(loadDirectory(path));
        },
        [dispatch],
    );

    // 组件挂载时，如果有上次的路径，自动加载并把它作为历史起点。
    useEffect(() => {
        if (fb.currentPath) {
            setHistory({ entries: [fb.currentPath], index: 0 });
            void dispatch(loadDirectory(fb.currentPath));
        }
    }, []); // eslint-disable-line react-hooks/exhaustive-deps

    const goBack = useCallback(() => {
        if (history.index <= 0) return;
        const next = history.index - 1;
        setHistory({ ...history, index: next });
        void dispatch(loadDirectory(history.entries[next]));
    }, [history, dispatch]);

    const goForward = useCallback(() => {
        if (history.index >= history.entries.length - 1) return;
        const next = history.index + 1;
        setHistory({ ...history, index: next });
        void dispatch(loadDirectory(history.entries[next]));
    }, [history, dispatch]);

    // 根据搜索模式决定展示配表
    const isSearchMode = fb.searchQuery.trim().length > 0;
    const trimmedSearchQuery = fb.searchQuery.trim();
    // 「计算机」虚拟层（盘符列表）：目录不存在，递归搜索在这里没有意义。
    const isComputerLevel = fb.currentPath === FILE_BROWSER_COMPUTER_PATH;

    /*
     * 匹配方式变化 → 用新参数重跑一次搜索。
     *
     * 【为什么必须重跑】结果是后端算好的：用户把「智能」改成「模糊」，不重跑就什么
     * 都不会变，看起来像开关坏了。用字符串键做守卫，避免把「输入框内容变化」也算成
     * 一次设置变化。
     */
    const searchOptionsKey = `${searchOptions.mode}|${searchOptions.heteronym}|${searchOptions.japaneseLongVowel}|${searchOptions.koreanChoseong}`;
    const lastSearchOptionsKeyRef = useRef(searchOptionsKey);
    useEffect(() => {
        if (lastSearchOptionsKeyRef.current === searchOptionsKey) return;
        lastSearchOptionsKeyRef.current = searchOptionsKey;
        // 正则模式下后端不过滤（query 传空串），重跑没有意义。
        if (fb.regexEnabled) return;
        if (!trimmedSearchQuery || !fb.currentPath || isComputerLevel) return;
        if (debounceRef.current) clearTimeout(debounceRef.current);
        void dispatch(
            searchFilesRecursive({
                dirPath: fb.currentPath,
                query: trimmedSearchQuery,
                options: searchOptions,
            }),
        );
    }, [
        searchOptionsKey,
        searchOptions,
        trimmedSearchQuery,
        fb.currentPath,
        fb.regexEnabled,
        isComputerLevel,
        dispatch,
    ]);

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
        const rawEntries = isSearchMode ? (fb.searchResults ?? []) : fb.entries;
        if (!isSearchMode || !fb.regexEnabled || !trimmedSearchQuery) {
            return rawEntries;
        }
        try {
            const re = new RegExp(trimmedSearchQuery, "i");
            return rawEntries.filter((e) => re.test(e.name));
        } catch {
            return [];
        }
    }, [isSearchMode, fb.entries, fb.searchResults, fb.regexEnabled, trimmedSearchQuery]);

    // 媒体过滤
    const mediaFilteredEntries = useMemo(() => {
        if (!view.mediaOnly) return regexFilteredEntries;
        // “仅显示媒体文件”：音频/视频 + MIDI（MIDI 可导入时间轴/参数编辑器）。
        return regexFilteredEntries.filter((e) => e.isDir || isAudioFile(e) || isMidiFile(e));
    }, [regexFilteredEntries, view.mediaOnly]);

    // 排序
    const displayEntries = useMemo(() => {
        const sorted = [...mediaFilteredEntries];
        const direction = view.sortDescending ? -1 : 1;
        switch (view.sortMode) {
            case "name":
                sorted.sort((a, b) => direction * NAME_COLLATOR.compare(a.name, b.name));
                break;
            case "date":
                sorted.sort((a, b) => direction * ((a.modifiedTime ?? 0) - (b.modifiedTime ?? 0)));
                break;
            case "size":
                sorted.sort((a, b) => direction * ((a.size ?? 0) - (b.size ?? 0)));
                break;
        }
        if (view.foldersFirst) {
            // 稳定排序：同组内保留上面的排序结果。
            sorted.sort((a, b) => (a.isDir === b.isDir ? 0 : a.isDir ? -1 : 1));
        }
        return sorted;
    }, [mediaFilteredEntries, view.sortMode, view.sortDescending, view.foldersFirst]);

    // 计算展示相对路径（搜索模式下显示文件所在目录）
    function getRelativeDirHint(fullPath: string): string {
        const normalFull = fullPath.replace(/\\/g, "/");
        const normalBase = fb.currentPath.replace(/\\/g, "/").replace(/\/$/, "");
        if (normalFull.toLowerCase().startsWith(normalBase.toLowerCase() + "/")) {
            const rel = normalFull.slice(normalBase.length + 1);
            const lastSlash = rel.lastIndexOf("/");
            return lastSlash >= 0 ? rel.slice(0, lastSlash) : "";
        }
        return "";
    }

    // 选择文件夹（通过后端 rfd dialog）
    const handleOpenFolder = useCallback(async () => {
        try {
            const result = await fileBrowserApi.pickDirectory();
            if (result.ok && !result.canceled && result.path) {
                navigateTo(result.path);
            }
        } catch {
            // 忽略错误
        }
    }, [navigateTo]);

    // 刷新当前目录
    const handleRefresh = useCallback(() => {
        if (fb.currentPath) {
            void dispatch(loadDirectory(fb.currentPath));
        }
    }, [dispatch, fb.currentPath]);

    // 返回上级目录
    const handleParentDir = useCallback(() => {
        if (!fb.currentPath) return;
        if (fb.currentPath === FILE_BROWSER_COMPUTER_PATH) return; // 已是顶层
        // 处理 Windows 和 Unix 路径
        const normalized = fb.currentPath.replace(/\\/g, "/");
        const parts = normalized.split("/").filter(Boolean);
        // Windows 盘符根（C:\）的上一级是「计算机」（列出全部盘符）；Unix 的 /
        // 已是文件系统顶端（parts 为空），再往上没有这一层，原地不动。
        if (parts.length === 1 && /^[A-Za-z]:$/.test(parts[0])) {
            navigateTo(FILE_BROWSER_COMPUTER_PATH);
            return;
        }
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
        navigateTo(parentPath);
    }, [fb.currentPath, navigateTo]);

    // 进入子目录
    const handleEnterDir = useCallback(
        (dirPath: string) => {
            navigateTo(dirPath);
        },
        [navigateTo],
    );

    // ── 多选状态 ───────────────────────────────────────────────────────────
    const [selectedPaths, setSelectedPaths] = useState<Set<string>>(new Set());
    const lastClickedIndexRef = useRef<number>(-1);

    // ── 键盘导航（roving tabindex） ─────────────────────────────────────────
    // activeIndex 指向当前活动行（-1 = 尚无）。只有活动行可 Tab 进入（tabIndex=0），
    // 方向键在同一列表内移动它；行获得焦点时同步回来，鼠标与键盘共用一套"当前行"。
    const [activeIndex, setActiveIndex] = useState(-1);
    const rowRefs = useRef<(HTMLDivElement | null)[]>([]);
    const registerRowRef = useCallback((index: number, el: HTMLDivElement | null) => {
        rowRefs.current[index] = el;
    }, []);

    // ── 列表窗口化 ─────────────────────────────────────────────────────────
    // 一个两万文件的目录若全量渲染，DOM 会到十几万个节点，首次挂载要数秒、整机
    // 跟着卡。窗口化后 DOM 规模只与视口有关，与目录大小脱钩（见 listWindow.ts）。
    const listScrollRef = useRef<HTMLDivElement | null>(null);
    const [viewport, setViewport] = useState({ scrollTop: 0, height: 0 });
    /** 实测行高（0 = 还没测到，用估值）。 */
    const [measuredRowHeight, setMeasuredRowHeight] = useState(0);
    /** 键盘跳到窗口之外的行时，等它渲染出来再聚焦。 */
    const pendingFocusRef = useRef<number | null>(null);

    /**
     * 这一模式下是否可能出现第二行（所在目录 / 命中原因）。
     *
     * 一处定义、两处使用：既决定行高的**估值**，也决定每行是否**预留**第二行 ——
     * 两者必须一致，否则窗口换算与真实高度对不上（见 FileEntryRow 的
     * `reserveSecondLine`）。
     */
    const reserveSecondLine = view.showPathHint || isSearchMode || searchSettings.showMatchReason;

    /**
     * 行高估值：按密度取基准，第二行存在时再加一行。
     *
     * 【为什么需要估值】首帧还没有 DOM 可测；没有行高就无法换算下标。估值只用来
     * 决定首帧渲染多少行，随后被实测值取代（下面的 ResizeObserver）。
     */
    const estimatedRowHeight = useMemo(() => {
        const base = rowDensityOf(view.density) === "default" ? 24 : 22;
        return base + (reserveSecondLine ? 14 : 0);
    }, [view.density, reserveSecondLine]);
    const rowHeight = measuredRowHeight || estimatedRowHeight;

    const listWindow = useMemo(
        () =>
            computeListWindow({
                total: displayEntries.length,
                rowHeight,
                scrollTop: viewport.scrollTop,
                viewportHeight: viewport.height,
            }),
        [displayEntries.length, rowHeight, viewport.scrollTop, viewport.height],
    );

    const handleListScroll = useCallback((event: React.UIEvent<HTMLDivElement>) => {
        const el = event.currentTarget;
        setViewport({ scrollTop: el.scrollTop, height: el.clientHeight });
    }, []);

    // 视口尺寸：滚动事件不会为"首次布局 / 面板被拖宽"触发，另用 ResizeObserver 兜。
    useEffect(() => {
        const el = listScrollRef.current;
        if (!el) return;
        const sync = () => setViewport({ scrollTop: el.scrollTop, height: el.clientHeight });
        // 首帧同步放到下一帧：effect 体内同步 setState 会触发级联渲染（React 明确
        // 不建议），而一帧的延迟在这里不可感知。
        const raf = requestAnimationFrame(sync);
        const observer = typeof ResizeObserver !== "undefined" ? new ResizeObserver(sync) : null;
        observer?.observe(el);
        return () => {
            cancelAnimationFrame(raf);
            observer?.disconnect();
        };
    }, []);

    /*
     * 实测行高。
     *
     * 【为什么用 ResizeObserver 而不是"每次渲染后读一次"】后者要在 effect 体里
     * 同步 setState（React 明确不建议，lint 也会拦）。ResizeObserver 在**回调**里
     * 给出尺寸，没有级联渲染问题；它还会在开始观察时立刻回调一次，正是需要的"首测"。
     * 密度变化导致行高变化时也会再次回调。
     *
     * 【为什么订阅随 first 变化重建】窗口滑动时首行换了 key，React 换掉的是另一个
     * DOM 节点，旧节点上的观察不再有意义。重建一次观察是微秒级开销，远小于
     * "行高失准导致整列错位"的代价。
     *
     * 无 ResizeObserver 的环境（jsdom）退化为使用估值 —— 那里本来也不测布局。
     */
    useEffect(() => {
        const el = rowRefs.current[listWindow.first];
        if (!el || typeof ResizeObserver === "undefined") return;
        const observer = new ResizeObserver(() => {
            const height = el.offsetHeight;
            if (height > 0) setMeasuredRowHeight((prev) => (prev === height ? prev : height));
        });
        observer.observe(el);
        return () => observer.disconnect();
    }, [listWindow.first, estimatedRowHeight, displayEntries.length]);

    // 窗口重算后补上"跳到窗口外的行"的聚焦。
    useEffect(() => {
        const index = pendingFocusRef.current;
        if (index == null) return;
        const el = rowRefs.current[index];
        if (!el) return;
        pendingFocusRef.current = null;
        try {
            el.scrollIntoView({ block: "nearest" });
        } catch {
            /* jsdom 无布局 */
        }
        el.focus({ preventScroll: true });
    });

    /*
     * 焦点兜底：窗口滑动把**正在聚焦的行**移出 DOM 时，浏览器把焦点丢回 `<body>`，
     * 此后方向键再也到不了列表（面板级处理器要求目标在面板内），用户看到的是
     * "列表突然按不动了"。此时把焦点收到滚动容器上 —— 它在面板内，键盘模型照常。
     *
     * 只在"焦点确实已经落到 body"时接管：用户主动点到别处（工具栏、搜索框）时
     * `activeElement` 不是 body，这里不会去抢。
     */
    useEffect(() => {
        if (activeIndex < 0) return;
        const container = listScrollRef.current;
        if (!container) return;
        if (document.activeElement === document.body && !rowRefs.current[activeIndex]) {
            container.focus({ preventScroll: true });
        }
    }, [activeIndex, listWindow.first, listWindow.last]);

    const handleRowFocus = useCallback((index: number) => {
        setActiveIndex(index);
    }, []);

    /**
     * 把键盘光标移到第 `index` 行并聚焦它。
     *
     * 【为什么要显式滚动】裸 `focus()` 会让浏览器用自己的算法把行滚进视口，
     * 在滚动容器里表现为整块跳变。`block: "nearest"` 是最小滚动 —— 行已在视口内
     * 就完全不动。这是全仓既有做法（QuickSearchPopup / UndoHistoryPanel /
     * KeybindingsDialog 三处），文件浏览器此前是唯一没接的。
     *
     * 【为什么滚动要包 try/catch】jsdom 没有布局实现，`scrollIntoView` 在单测里
     * 会抛（KeybindingsDialog 同样处理）。焦点移动才是语义要求，滚动只是观感，
     * 因此让滚动失败不阻断聚焦。
     *
     * 【为什么 focus 带 preventScroll】滚动已由上一行显式完成，再让浏览器在聚焦时
     * 滚一次会与它抢，产生二次跳动。
     *
     * 【窗口化带来的第三态】目标行可能**根本没被渲染**（在窗口之外）。此时不能
     * 直接聚焦：先把容器滚到它附近，记下待办，等窗口重算并渲染出该行后再聚焦
     * （见下面的 `pendingFocusRef` effect）。键盘"End 跳到末行"走的正是这条路。
     */
    const focusRow = useCallback(
        (index: number) => {
            if (index < 0) return;
            const el = rowRefs.current[index];
            if (el) {
                try {
                    el.scrollIntoView({ block: "nearest" });
                } catch {
                    /* jsdom 无布局：滚动不是语义要求，忽略 */
                }
                el.focus({ preventScroll: true });
                return;
            }
            const container = listScrollRef.current;
            if (container && rowHeight > 0) {
                const rowTop = index * rowHeight;
                const margin = rowHeight * 2;
                if (rowTop < container.scrollTop + margin) {
                    container.scrollTop = Math.max(0, rowTop - margin);
                } else if (
                    rowTop + rowHeight >
                    container.scrollTop + container.clientHeight - margin
                ) {
                    container.scrollTop = rowTop + rowHeight - container.clientHeight + margin;
                }
                setViewport({
                    scrollTop: container.scrollTop,
                    height: container.clientHeight,
                });
            }
            pendingFocusRef.current = index;
        },
        [rowHeight],
    );

    /**
     * 键盘移动后按需自动试听。
     *
     * 【为什么要防抖】按住方向键浏览一屏文件会连续触发十几次 —— 每次都要取数、
     * 解码、起播，既卡又吵。160ms 的窗口让"停在哪一条"才出声。
     */
    const maybePreviewOnNavigate = useCallback(
        (entry: FileEntry | undefined) => {
            if (previewNavTimerRef.current) clearTimeout(previewNavTimerRef.current);
            if (!view.previewOnNavigate || !entry || !isAudioFile(entry)) return;
            previewNavTimerRef.current = setTimeout(() => {
                previewToggle.play(entry.path);
            }, PREVIEW_NAV_DEBOUNCE_MS);
        },
        [view.previewOnNavigate, previewToggle],
    );

    /**
     * 单击一行。
     *
     * 【为什么选中与试听分开】行的左键语义是"选中它"（让 Ctrl+C / Delete / F2 /
     * 拖拽有作用对象）；音频文件在此之上**额外**试听。此前只有音频行接了点击，
     * 点一个 `.txt` 什么都不发生 —— 而右键菜单却能对它操作，两条路径的可用性
     * 不一致。
     *
     * 【为什么下标空间必须统一】此前 `lastClickedIndexRef` 记的是 **audioEntries**
     * 的下标，而键盘光标用的是 displayEntries 的下标；Shift 范围选择因此只在音频
     * 之间连线，与列表里看到的顺序不是一回事。现在统一用 displayEntries。
     */
    const handleRowClick = useCallback(
        (entry: FileEntry, ev: React.MouseEvent) => {
            const index = displayEntries.findIndex((candidate) => candidate.path === entry.path);

            if (isPrimaryModifierDown(ev)) {
                // macOS: Command+click / Windows: Ctrl+click — 加选 / 减选
                setSelectedPaths((prev) => {
                    const next = new Set(prev);
                    if (next.has(entry.path)) next.delete(entry.path);
                    else next.add(entry.path);
                    return next;
                });
                lastClickedIndexRef.current = index;
                return;
            }

            if (ev.shiftKey && lastClickedIndexRef.current >= 0) {
                // Shift+click：从锚点到这一行的范围选择（含目录，与列表所见一致）
                const start = Math.min(lastClickedIndexRef.current, index);
                const end = Math.max(lastClickedIndexRef.current, index);
                setSelectedPaths((prev) => {
                    const next = new Set(prev);
                    for (let i = start; i <= end; i++) {
                        next.add(displayEntries[i].path);
                    }
                    return next;
                });
                return;
            }

            // 普通点击：单选这一行；音频文件再切换试听
            // （"再点一次停止"由 `usePreviewToggle` 统一实现：此前这里只有播放
            // 分支，重复点击会从头重放并与在播的旧音源叠加。）
            setSelectedPaths(new Set([entry.path]));
            lastClickedIndexRef.current = index;
            if (isAudioFile(entry)) previewToggle.toggle(entry.path);
        },
        [displayEntries, previewToggle],
    );

    /**
     * 键盘激活一行：与鼠标走**同一条**路径 —— 目录进入子目录，音频文件切换试听。
     * 不复制这两段逻辑，只做选择。
     */
    const activateEntry = useCallback(
        (entry: FileEntry) => {
            if (entry.isDir) {
                handleEnterDir(entry.path);
            } else if (isAudioFile(entry)) {
                previewToggle.toggle(entry.path);
            }
        },
        [handleEnterDir, previewToggle],
    );

    /**
     * 列表容器的键盘模型：方向键 / Home / End 移动活动行（夹紧，不环绕），
     * Enter 与空格激活。方向键必须 `preventDefault`，否则 ScrollArea 会跟着滚动。
     */
    const handleListKeyDown = useCallback(
        (event: React.KeyboardEvent<HTMLDivElement>) => {
            if (isFileListNavKey(event.key)) {
                event.preventDefault();
                const next = nextActiveIndex(activeIndex, event.key, displayEntries.length);
                if (next < 0) return;
                setActiveIndex(next);
                focusRow(next);
                maybePreviewOnNavigate(displayEntries[next]);
                return;
            }
            if (isFileListActivationKey(event.key)) {
                const entry = displayEntries[activeIndex];
                if (!entry) return;
                event.preventDefault();
                activateEntry(entry);
            }
        },
        [activeIndex, displayEntries, activateEntry, focusRow, maybePreviewOnNavigate],
    );

    // Clear selection when directory changes
    useEffect(() => {
        setSelectedPaths(new Set());
        lastClickedIndexRef.current = -1;
        setActiveIndex(-1);
        setEditing(null);
        // 上一次目录里的失败提示（重名 / 非法名）不该跟着走进新目录。
        setError(null);
    }, [fb.currentPath]);

    // ── 输入字母快速跳转（type-ahead，与资源管理器一致） ─────────────────────
    // 名单与 displayEntries 同步，供增量搜索逐键匹配。
    const entryNames = useMemo(() => displayEntries.map((e) => e.name), [displayEntries]);
    const typeAheadBufferRef = useRef("");
    const typeAheadLastKeyAtRef = useRef(0);
    const panelRootRef = useRef<HTMLDivElement>(null);

    /**
     * 面板级键盘捕获。
     *
     * 【顺序很重要】先判"焦点在不在文本控件里"，再判按键类别。此前是反过来的
     * （先看 `key.length === 1`），于是 Ctrl+A 这类组合键在搜索框里也会被
     * 面板抢走 —— 输入框里"全选文本"变成了"全选文件"。
     */
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
        // eslint-disable-next-line react-hooks/exhaustive-deps -- selectAll / 等回调在下方定义，见其自身 useCallback
        [
            displayEntries,
            activeIndex,
            entryNames,
            selectedPaths,
            isComputerLevel,
            focusRow,
            goBack,
            goForward,
            handleParentDir,
            handleRefresh,
            maybePreviewOnNavigate,
        ],
    );

    // 列表内容变化（搜索、排序、过滤）后，活动行可能越界：收回为"无活动行"。
    useEffect(() => {
        setActiveIndex((current) => (current >= displayEntries.length ? -1 : current));
    }, [displayEntries.length]);

    // ── 选择 ───────────────────────────────────────────────────────────────
    const selectAll = useCallback(() => {
        setSelectedPaths(new Set(displayEntries.map((entry) => entry.path)));
    }, [displayEntries]);

    // ── 右键菜单 ───────────────────────────────────────────────────────────
    const [menu, setMenu] = useState<{ x: number; y: number; entry: FileEntry | null } | null>(
        null,
    );

    const handleRowContextMenu = useCallback(
        (event: React.MouseEvent, entry: FileEntry) => {
            event.preventDefault();
            event.stopPropagation();
            // Explorer 语义：右键未选中项 → 先把它选中，菜单作用于它。
            setSelectedPaths((prev) => (prev.has(entry.path) ? prev : new Set([entry.path])));
            setActiveIndex(displayEntries.findIndex((candidate) => candidate.path === entry.path));
            setMenu({ x: event.clientX, y: event.clientY, entry });
        },
        [displayEntries],
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

    // ── 对话框与行内编辑 ───────────────────────────────────────────────────
    const [propertiesEntry, setPropertiesEntry] = useState<FileEntry | null>(null);
    const [viewOptionsOpen, setViewOptionsOpen] = useState(false);
    const [deleteRequest, setDeleteRequest] = useState<string[] | null>(null);
    const [editing, setEditing] = useState<EditingState>(null);
    const [pathDraft, setPathDraft] = useState<string | null>(null);
    /** 写操作失败的一次性提示（重名 / 非法名 / 受保护路径）。 */
    const [transientError, setError] = useState<string | null>(null);

    const selectedEntries = useMemo(
        () => displayEntries.filter((entry) => selectedPaths.has(entry.path)),
        [displayEntries, selectedPaths],
    );

    const handleRenameCommit = useCallback(
        async (entry: FileEntry, newName: string) => {
            setEditing(null);
            const trimmed = newName.trim();
            if (!trimmed || trimmed === entry.name) return;
            try {
                const newPath = await fileBrowserApi.renamePath(entry.path, trimmed);
                setSelectedPaths(new Set([newPath]));
                await dispatch(loadDirectory(fb.currentPath));
            } catch {
                // 后端已把非法名 / 重名 / 受保护路径拒绝掉了，这里只需让用户看到结果。
                setError(tf("fb_rename_failed"));
            }
        },
        [dispatch, fb.currentPath, tf],
    );

    const handleNewFolderCommit = useCallback(
        async (name: string) => {
            setEditing(null);
            const trimmed = name.trim();
            if (!trimmed) return;
            try {
                const created = await fileBrowserApi.createDirectory(fb.currentPath, trimmed);
                setSelectedPaths(new Set([created]));
                await dispatch(loadDirectory(fb.currentPath));
            } catch {
                setError(tf("fb_create_folder_failed"));
            }
        },
        [dispatch, fb.currentPath, tf],
    );

    const handleDelete = useCallback(
        async (permanent: boolean) => {
            const paths = deleteRequest;
            setDeleteRequest(null);
            if (!paths || paths.length === 0) return;
            try {
                const result = await fileBrowserApi.deletePaths(paths, permanent);
                if (!result.ok) setError(tf("fb_delete_failed"));
            } catch {
                setError(tf("fb_delete_failed"));
            }
            setSelectedPaths(new Set());
            await dispatch(loadDirectory(fb.currentPath));
        },
        [deleteRequest, dispatch, fb.currentPath, tf],
    );

    /** 菜单动作集合：菜单只决定"显示什么"，这里决定"做什么"。 */
    const menuActions: FileBrowserMenuActions = useMemo(
        () => ({
            openEntry: (entry) => {
                if (entry.isDir) {
                    handleEnterDir(entry.path);
                } else if (isAudioFile(entry)) {
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
        ],
    );

    const menuItems = useMemo(
        () =>
            menu
                ? buildFileBrowserContextMenu(menu.entry, {
                      t,
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
            const paths =
                selectedPaths.size > 0 && selectedPaths.has(entry.path)
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
                fileBrowserApi
                    .getAudioFileInfo(ds.filePath)
                    .then((info) => {
                        if (info && dragStateRef.current?.filePath === ds.filePath) {
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
            }

            // 更新 ghost 位置（clamp 到窗口可视范围内，鼠标超出界面时 ghost 停在边缘。
            // 余量按 ghost 实测尺寸计算：文件名长短不一，固定余量会让 ghost
            // 在窗口右缘脱离光标，长文件名仍会溢出右边界）。
            if (ghostRef.current) {
                const ghost = ghostRef.current;
                const clampedX = Math.max(
                    0,
                    Math.min(e.clientX + 12, window.innerWidth - ghost.offsetWidth - 4),
                );
                const clampedY = Math.max(
                    0,
                    Math.min(e.clientY + 12, window.innerHeight - ghost.offsetHeight - 4),
                );
                ghost.style.left = `${clampedX}px`;
                ghost.style.top = `${clampedY}px`;
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

        // 指针在窗口外（任务栏/另一显示器）松开时 pointerup 不会派发，
        // pointercancel / lostpointercapture 是唯一可靠的收尾信号 —— 否则
        // 拖拽态永久卡死（ghost 滞留、列表行保持半透明、move 事件持续派发）。
        function onPointerCancel() {
            const ds = dragStateRef.current;
            if (ds?.active) {
                window.dispatchEvent(
                    new CustomEvent("hifi-file-drag", {
                        detail: {
                            type: "drop",
                            filePath: ds.filePath,
                            fileName: ds.fileName,
                            filePaths: ds.allFilePaths,
                            clientX: ds.startX,
                            clientY: ds.startY,
                            isRightDrag: ds.isRightDrag,
                            canceled: true,
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
        window.addEventListener("pointercancel", onPointerCancel);
        window.addEventListener("blur", onPointerCancel);
        window.addEventListener("contextmenu", onContextMenu, true);
        return () => {
            window.removeEventListener("pointermove", onPointerMove);
            window.removeEventListener("pointerup", onPointerUp);
            window.removeEventListener("pointercancel", onPointerCancel);
            window.removeEventListener("blur", onPointerCancel);
            window.removeEventListener("contextmenu", onContextMenu, true);
        };
    }, [dragState !== null]); // eslint-disable-line react-hooks/exhaustive-deps

    // 列表真正渲染出条目时，容器才承担 listbox 语义（加载/错误/空态不是列表）。
    const showEntries =
        !fb.loading &&
        !fb.error &&
        !!fb.currentPath &&
        !(isSearchMode && fb.searchLoading) &&
        displayEntries.length > 0;

    // roving tabindex 的起点：尚无活动行时首行可 Tab 进入。
    const tabbableIndex = activeIndex >= 0 ? activeIndex : 0;

    const detailTextOf = useCallback(
        (entry: FileEntry): string | undefined => {
            if (view.detailsColumn === "none" || entry.isDir) return undefined;
            if (view.detailsColumn === "date") return formatModified(entry.modifiedTime);
            return formatSize(entry.size);
        },
        [view.detailsColumn],
    );

    /*
     * 状态行的数量按**当前语系**格式化（`20,000` 而不是 `20000`）。
     *
     * 项数走 `plural` —— 它内部已用 `Intl.NumberFormat` 回填 `{count}`；
     * 选中数走 `tVars`，它只做字符串替换、不认识数字，所以要在这里先格式化。
     */
    const statusText = useMemo(() => {
        const parts = [plural("fb_status_items", displayEntries.length)];
        if (selectedPaths.size > 0) {
            parts.push(tVars("fb_status_selected", { count: number(selectedPaths.size) }));
        }
        return parts.join(" · ");
    }, [displayEntries.length, selectedPaths.size, plural, tVars, number]);

    return (
        <Flex
            ref={panelRootRef}
            direction="column"
            className="h-full bg-qt-window text-qt-text select-none"
            onKeyDown={handlePanelKeyDown}
        >
            {/* 工具条：只放本面板**独有**的功能按钮。
                标题与关闭属于窗框（停靠时是标签行、浮动时是浮动标题栏、独立窗口时
                是系统标题栏），在这里再画一遍就是重复展示 —— 用户看到两个标题、两个
                关闭键会困惑。 */}
            <PanelToolbar
                trailing={
                    <>
                        <PanelToolbarButton
                            icon={<StarIcon />}
                            tooltip={t("fb_locations")}
                            buttonRef={locationsButtonRef}
                            onClick={() => {
                                const rect = locationsButtonRef.current?.getBoundingClientRect();
                                setLocationsAt(
                                    rect ? { x: rect.left, y: rect.bottom + 2 } : { x: 40, y: 40 },
                                );
                            }}
                        />
                        <PanelToolbarButton
                            icon={<FolderIcon />}
                            tooltip={tf("fb_open_folder")}
                            onClick={() => void handleOpenFolder()}
                        />
                        <PanelToolbarButton
                            icon={<ReloadIcon />}
                            tooltip={tf("fb_refresh")}
                            onClick={handleRefresh}
                        />
                        <PanelToolbarButton
                            icon={<GearIcon />}
                            tooltip={t("fb_view_options")}
                            onClick={() => setViewOptionsOpen(true)}
                        />
                    </>
                }
            />

            {/* 搜索栏 */}
            <div className="px-2 py-1 border-b border-qt-border shrink-0">
                <TextField.Root
                    ref={searchInputRef}
                    size="1"
                    placeholder={tf("fb_search_placeholder")}
                    value={fb.searchQuery}
                    onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
                        const q = e.target.value;
                        dispatch(setSearchQuery(q));
                        if (debounceRef.current) clearTimeout(debounceRef.current);
                        if (q.trim() && fb.currentPath && !isComputerLevel) {
                            const backendQuery = fb.regexEnabled ? "" : q.trim();
                            debounceRef.current = setTimeout(() => {
                                void dispatch(
                                    searchFilesRecursive({
                                        dirPath: fb.currentPath,
                                        query: backendQuery,
                                        options: searchOptions,
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

                {/* 正则切换 + 转写 + 媒体过滤 + 排序 */}
                <Flex align="center" gap="1" mt="1">
                    <AppIconButton
                        active={fb.regexEnabled}
                        tooltip={tf("fb_regex")}
                        onClick={() => {
                            const nextRegexEnabled = !fb.regexEnabled;
                            dispatch(toggleRegex());

                            if (debounceRef.current) {
                                clearTimeout(debounceRef.current);
                            }

                            if (trimmedSearchQuery && fb.currentPath && !isComputerLevel) {
                                void dispatch(
                                    searchFilesRecursive({
                                        dirPath: fb.currentPath,
                                        query: nextRegexEnabled ? "" : trimmedSearchQuery,
                                        options: nextRegexEnabled
                                            ? { ...searchOptions, mode: "off" }
                                            : searchOptions,
                                    }),
                                );
                            }
                        }}
                        style={{
                            fontFamily: "monospace",
                            fontSize: "var(--qt-fs-micro)",
                            width: 22,
                            height: 22,
                        }}
                        icon=".*"
                    />
                    <SearchTranslitToggle
                        settings={searchSettings}
                        onChange={(patch) => {
                            dispatch(setSearchSettings(patch));
                            void dispatch(persistUiSettings());
                        }}
                        regexActive={fb.regexEnabled}
                        onOpenSettings={() => dispatch(setSearchSettingsDialogOpen(true))}
                    />
                    <AppIconButton
                        active={view.mediaOnly}
                        tooltip={tf("fb_audio_only")}
                        onClick={() => {
                            dispatch(setFileBrowserView({ mediaOnly: !view.mediaOnly }));
                            void dispatch(persistUiSettings());
                        }}
                        style={{
                            width: 22,
                            height: 22,
                        }}
                        icon={
                            <svg width="14" height="14" viewBox="0 0 15 15" fill="none">
                                <path
                                    d="M7.5 0.75L7.5 14.25M10.5 3L10.5 12M4.5 3L4.5 12M13.5 5.5L13.5 9.5M1.5 5.5L1.5 9.5"
                                    stroke="currentColor"
                                    strokeWidth="1.2"
                                    strokeLinecap="round"
                                />
                            </svg>
                        }
                    />
                    <AppSelect
                        fullWidth={false}
                        className="flex-1"
                        value={view.sortMode}
                        onValueChange={(value) => {
                            dispatch(
                                setFileBrowserView({
                                    sortMode: value as FileBrowserViewOptions["sortMode"],
                                }),
                            );
                            void dispatch(persistUiSettings());
                        }}
                        options={[
                            { value: "name", label: tf("fb_sort_name") },
                            { value: "date", label: tf("fb_sort_date") },
                            { value: "size", label: tf("fb_sort_size") },
                        ]}
                    />
                    <AppIconButton
                        active={view.sortDescending}
                        tooltip={t("fb_sort_descending")}
                        onClick={() => {
                            dispatch(setFileBrowserView({ sortDescending: !view.sortDescending }));
                            void dispatch(persistUiSettings());
                        }}
                        style={{ width: 22, height: 22 }}
                        icon={<ChevronDownIcon />}
                    />
                </Flex>

                {hasRegexError && (
                    <span className="hs-type-label" style={{ color: "var(--qt-danger-text)" }}>
                        {tf("fb_regex_error")}
                    </span>
                )}
                {transientError && (
                    <span className="hs-type-label" style={{ color: "var(--qt-danger-text)" }}>
                        {transientError}
                    </span>
                )}
            </div>

            {/* 路径栏：后退 / 前进 / 上级 + 可编辑路径 */}
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
                        data-tooltip={t("fb_nav_back")}
                        disabled={!canGoBack}
                        onClick={goBack}
                    >
                        <ChevronLeftIcon />
                    </IconButton>
                    <IconButton
                        size="1"
                        variant="ghost"
                        color="gray"
                        data-tooltip={t("fb_nav_forward")}
                        disabled={!canGoForward}
                        onClick={goForward}
                    >
                        <ChevronRightIcon />
                    </IconButton>
                    <IconButton
                        size="1"
                        variant="ghost"
                        color="gray"
                        data-tooltip={tf("fb_parent_dir")}
                        onClick={handleParentDir}
                    >
                        <ChevronUpIcon />
                    </IconButton>
                    {pathDraft === null ? (
                        <span
                            className="hs-type-label truncate flex-1 cursor-text"
                            data-tooltip={isComputerLevel ? tf("fb_computer") : fb.currentPath}
                            onClick={() => setPathDraft(isComputerLevel ? "" : fb.currentPath)}
                        >
                            {isComputerLevel ? tf("fb_computer") : fb.currentPath}
                        </span>
                    ) : (
                        <input
                            autoFocus
                            className="hs-type-label flex-1 min-w-0 bg-qt-base rounded px-1 outline-none"
                            style={{ border: "1px solid var(--qt-border)" }}
                            value={pathDraft}
                            aria-label={t("fb_path_edit_tooltip")}
                            onChange={(e) => setPathDraft(e.target.value)}
                            onKeyDown={(e) => {
                                if (e.key === "Enter") {
                                    e.preventDefault();
                                    const next = pathDraft.trim();
                                    setPathDraft(null);
                                    if (next) navigateTo(next);
                                } else if (e.key === "Escape") {
                                    e.preventDefault();
                                    setPathDraft(null);
                                }
                            }}
                            onBlur={() => setPathDraft(null)}
                        />
                    )}
                </Flex>
            )}

            {/* 文件列表。
                用原生滚动容器而不是 Radix ScrollArea：窗口化需要**自己**读写
                `scrollTop` / `clientHeight`，而 ScrollArea 的滚动元素是它内部
                自绘的（要靠 `[data-radix-scroll-area-viewport]` 这种内部属性去找，
                一旦上游改名，窗口就永远不动、列表看起来卡死）。原生容器还给回
                全仓统一的主题滚动条（见 index.css 的滚动条说明）。 */}
            <div
                ref={listScrollRef}
                className="hs-scroll-gutter flex-1 min-h-0 overflow-y-auto"
                onScroll={handleListScroll}
                /*
                 * 键盘模型挂在这一层而不是内层 listbox 上。
                 *
                 * 【为什么】窗口滑动会把正在聚焦的行移出 DOM，浏览器随之把焦点丢回
                 * `<body>`。下面的恢复逻辑把焦点收到**本容器**（`tabIndex={-1}`），
                 * 于是方向键仍然需要在这里被接住 —— 若处理器留在内层 div 上，事件
                 * 从容器发出时根本不会经过它，列表会显得"突然按不动了"。
                 */
                tabIndex={showEntries ? -1 : undefined}
                onKeyDown={showEntries ? handleListKeyDown : undefined}
            >
                <div
                    className="py-1"
                    role={showEntries ? "listbox" : undefined}
                    aria-label={showEntries ? tf("fb_file_list") : undefined}
                    // 列表本就支持 Ctrl/Shift 多选，声明多选语义以免读屏按单选播报。
                    aria-multiselectable={showEntries ? true : undefined}
                    onContextMenu={handleBackgroundContextMenu}
                >
                    {fb.loading ? (
                        <AppEmptyState>{tf("fb_loading")}</AppEmptyState>
                    ) : fb.error ? (
                        <AppEmptyState tone="danger">
                            {tf("fb_error")}: {fb.error}
                        </AppEmptyState>
                    ) : !fb.currentPath ? (
                        <AppEmptyState>{tf("fb_no_folder")}</AppEmptyState>
                    ) : isSearchMode && fb.searchLoading ? (
                        <AppEmptyState>{tf("fb_searching")}</AppEmptyState>
                    ) : displayEntries.length === 0 && editing?.kind !== "newFolder" ? (
                        <AppEmptyState>
                            {isSearchMode ? tf("fb_no_results") : tf("fb_empty_folder")}
                        </AppEmptyState>
                    ) : (
                        <>
                            {editing?.kind === "newFolder" && (
                                <Flex
                                    align="center"
                                    gap="1.5"
                                    className="px-2 py-qt-1 min-h-[22px]"
                                >
                                    <FolderIcon className="text-yellow-500 shrink-0" />
                                    <DockInlineRename
                                        initial=""
                                        placeholder={t("fb_new_folder_default")}
                                        ariaLabel={t("fb_ctx_new_folder")}
                                        onCommit={(name) => void handleNewFolderCommit(name)}
                                        onCancel={() => setEditing(null)}
                                    />
                                </Flex>
                            )}
                            {/* 全量高度撑起滚动条，窗口内容整体偏移到对应位置。 */}
                            <div
                                style={{
                                    position: "relative",
                                    height: listWindow.totalHeight,
                                }}
                            >
                                <div
                                    style={{
                                        transform: `translateY(${listWindow.offsetTop}px)`,
                                    }}
                                >
                                    {displayEntries
                                        .slice(listWindow.first, listWindow.last)
                                        .map((entry, offset) => {
                                            const index = listWindow.first + offset;
                                            return editing?.kind === "rename" &&
                                                editing.path === entry.path ? (
                                                <Flex
                                                    key={entry.path}
                                                    align="center"
                                                    gap="1.5"
                                                    className="px-2 py-qt-1 min-h-[22px]"
                                                >
                                                    <FileKindIcon entry={entry} />
                                                    <DockInlineRename
                                                        initial={editing.initial}
                                                        ariaLabel={t("fb_ctx_rename")}
                                                        onCommit={(name) =>
                                                            void handleRenameCommit(entry, name)
                                                        }
                                                        onCancel={() => setEditing(null)}
                                                    />
                                                </Flex>
                                            ) : (
                                                <FileEntryRow
                                                    key={entry.path}
                                                    entry={entry}
                                                    index={index}
                                                    tabIndex={index === tabbableIndex ? 0 : -1}
                                                    ariaPosInSet={index + 1}
                                                    ariaSetSize={displayEntries.length}
                                                    active={index === activeIndex}
                                                    density={rowDensityOf(view.density)}
                                                    onFocus={handleRowFocus}
                                                    registerRowRef={registerRowRef}
                                                    isPlaying={fb.previewingFile === entry.path}
                                                    isSelected={selectedPaths.has(entry.path)}
                                                    onDoubleClickDir={handleEnterDir}
                                                    onRowClick={handleRowClick}
                                                    onPointerDownForDrag={handlePointerDownForDrag}
                                                    onContextMenu={handleRowContextMenu}
                                                    isDragging={
                                                        dragState?.active === true &&
                                                        dragState.allFilePaths.includes(entry.path)
                                                    }
                                                    pathHint={
                                                        view.showPathHint || isSearchMode
                                                            ? getRelativeDirHint(entry.path)
                                                            : undefined
                                                    }
                                                    matchReason={formatMatchReason(entry)}
                                                    detailText={detailTextOf(entry)}
                                                    reserveSecondLine={reserveSecondLine}
                                                />
                                            );
                                        })}
                                </div>
                            </div>
                        </>
                    )}
                </div>
            </div>

            {/* 状态行：项数 / 选中数 */}
            {view.statusBarVisible && (
                <div className="px-2 py-0.5 border-t border-qt-border shrink-0 hs-type-caption">
                    {statusText}
                </div>
            )}

            {/* 底部音量滑块 */}
            <Flex align="center" gap="2" className="px-2 py-1.5 border-t border-qt-border shrink-0">
                <SpeakerLoudIcon width="14" height="14" className="text-qt-text-muted shrink-0" />
                <AppSlider
                    value={Math.round(fb.previewVolume * 100)}
                    unit="percent"
                    min={0}
                    max={100}
                    ariaLabel={tf("fb_preview_volume")}
                    onChange={(next) => {
                        dispatch(setPreviewVolume(next / 100));
                    }}
                />
                <AppSliderReadout>{Math.round(fb.previewVolume * 100)}%</AppSliderReadout>
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
                        color: "var(--qt-text)",
                        padding: "2px 8px",
                        borderRadius: "var(--qt-radius-sm)",
                        fontSize: "var(--qt-fs-xs)",
                        whiteSpace: "nowrap",
                        opacity: 0.9,
                        boxShadow: "0 2px 8px rgba(0,0,0,0.3)",
                    }}
                >
                    🎵{" "}
                    {dragState.allFilePaths.length > 1
                        ? `${dragState.fileName} (+${dragState.allFilePaths.length - 1})`
                        : dragState.fileName}
                </div>
            )}

            {/* 右键菜单 */}
            {menu && (
                <AppContextMenu
                    x={menu.x}
                    y={menu.y}
                    ariaLabel={tf("fb_file_list")}
                    items={menuItems}
                    onClose={() => setMenu(null)}
                />
            )}

            {/* 常用位置 */}
            {locationsAt && (
                <AppContextMenu
                    x={locationsAt.x}
                    y={locationsAt.y}
                    minWidth={220}
                    ariaLabel={t("fb_locations")}
                    items={locationItems}
                    onClose={() => setLocationsAt(null)}
                />
            )}

            {/* 属性。key 让换条目时重新挂载 —— 探测结果（音频信息 / 目录条目数）
                随之重置，不必在对话框内部用 effect 清 state。 */}
            <FilePropertiesDialog
                key={propertiesEntry?.path ?? "none"}
                open={propertiesEntry !== null}
                onOpenChange={(open) => {
                    if (!open) setPropertiesEntry(null);
                }}
                entry={propertiesEntry}
            />

            {/* 视图选项 */}
            <FileBrowserViewOptionsDialog
                open={viewOptionsOpen}
                onOpenChange={setViewOptionsOpen}
            />

            {/* 删除确认：默认进回收站，永久删除是次要（靠左、危险色）动作 */}
            <AppDialog
                open={deleteRequest !== null}
                onOpenChange={(open) => {
                    if (!open) setDeleteRequest(null);
                }}
                title={t("fb_delete_confirm_title")}
                message={plural("fb_delete_confirm_message", deleteRequest?.length ?? 0)}
                tone="danger"
                size="sm"
                actions={[
                    {
                        id: "permanent",
                        label: t("fb_delete_permanent"),
                        intent: "danger",
                        align: "start",
                        onClick: () => void handleDelete(true),
                    },
                    { id: "cancel", label: t("cancel"), onClick: () => setDeleteRequest(null) },
                    {
                        id: "trash",
                        label: t("fb_ctx_delete"),
                        intent: "primary",
                        onClick: () => void handleDelete(false),
                    },
                ]}
            />
        </Flex>
    );
};
