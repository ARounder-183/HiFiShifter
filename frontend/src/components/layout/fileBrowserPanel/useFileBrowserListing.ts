// FileBrowser 列表、搜索与目录导航状态。拆自 FileBrowserPanel，保持原有行为不变。
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useAppDispatch, useAppSelector } from "../../../app/hooks";
import type { RootState } from "../../../app/store";
import { useI18n } from "../../../i18n/I18nProvider";
import {
    loadDirectory,
    setFileBrowserError,
    setSearchQuery,
    searchFilesRecursive,
    FILE_BROWSER_COMPUTER_PATH,
} from "../../../features/fileBrowser/fileBrowserSlice";
import { audioPreview } from "../../../features/fileBrowser/audioPreview";
import { usePreviewToggle } from "../../../features/fileBrowser/usePreviewToggle";
import {
    fileBrowserSearchOptions,
    fileBrowserSearchRequest,
    visibleFileBrowserEntries,
} from "../../../features/fileBrowser/fileBrowserSearchOptions";
import { compareFileNames } from "../../../features/fileBrowser/fileNameCompare";
import type { FileEntry } from "../../../services/api/fileBrowser";
import { fileBrowserApi } from "../../../services/api/fileBrowser";
import { matchReasonOf } from "../search/matchReason";

export function useFileBrowserListing() {
    const dispatch = useAppDispatch();
    const { t, tf, tVars, plural, number } = useI18n();
    const fb = useAppSelector((state: RootState) => state.fileBrowser);
    const searchSettings = useAppSelector((state: RootState) => state.session.searchSettings);
    const view = useAppSelector((state: RootState) => state.session.fileBrowserView);
    const favorites = useAppSelector((state: RootState) => state.session.fileBrowserFavorites);
    const selectedTrackId = useAppSelector((state: RootState) => state.session.selectedTrackId);
    const playheadSec = useAppSelector((state: RootState) => state.session.playheadSec);
    /** 轨道列表（DFS 显示顺序）：算目录导入的落点用。 */
    const tracks = useAppSelector((state: RootState) => state.session.tracks);

    const searchInputRef = useRef<HTMLInputElement>(null);
    const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
    const previewNavTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

    /*
     * 下发给后端的匹配参数。
     *
     * 【为什么正则模式下强制 off】正则作用于**原文**，与转写互斥：把 `zhuge` 当正则
     * 去匹配「主歌」没有任何意义。前端仍然把 query 传空串（沿用旧行为：后端不过滤，
     * 由前端做正则过滤），匹配模式一并降为 off，让后端走最便宜的路径。
     *
     * 【为什么 includeDirs 绑 mediaOnly】见 `fileBrowserSearchOptions` 的文件头：
     * 关掉「仅显示媒体文件」= 我要在这个目录下找东西（子目录是合法命中），
     * 打开 = 我只要可导入的媒体（目录不是媒体文件）。
     */
    const searchOptions = useMemo(
        () =>
            fileBrowserSearchOptions({
                settings: searchSettings,
                regexEnabled: fb.regexEnabled,
                mediaOnly: view.mediaOnly,
                showHiddenFiles: view.showHiddenFiles,
            }),
        [searchSettings, fb.regexEnabled, view.mediaOnly, view.showHiddenFiles],
    );

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
     *
     * 【为什么键里含 includeDirs / includeHidden】这两个也是后端算结果时的输入：
     * 搜索途中切「仅显示媒体文件」，目录该出现或该消失；切「显示隐藏文件」同理。
     * 漏掉它们，开关就会变成"点了没反应"。
     */
    const searchOptionsKey = `${searchOptions.mode}|${searchOptions.heteronym}|${searchOptions.japaneseLongVowel}|${searchOptions.koreanChoseong}|${searchOptions.includeDirs}|${searchOptions.includeHidden}`;
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

    // 媒体过滤（搜索模式下由后端的 includeDirs 承担，见 fileBrowserSearchOptions）
    const mediaFilteredEntries = useMemo(
        () =>
            visibleFileBrowserEntries(regexFilteredEntries, {
                isSearchMode,
                mediaOnly: view.mediaOnly,
            }),
        [regexFilteredEntries, isSearchMode, view.mediaOnly],
    );

    // 排序
    const displayEntries = useMemo(() => {
        const sorted = [...mediaFilteredEntries];
        const direction = view.sortDescending ? -1 : 1;
        switch (view.sortMode) {
            case "name":
                sorted.sort((a, b) => direction * compareFileNames(a.name, b.name));
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
        } catch (error) {
            dispatch(setFileBrowserError(String(error)));
        }
    }, [navigateTo, dispatch]);

    /**
     * 重新拉取**列表当前显示的内容**（刷新按钮 / 写操作之后）。
     *
     * 【为什么不能只 reload 当前目录】搜索模式下列表来自 `fb.searchResults`，
     * 而 `loadDirectory` 只更新 `fb.entries` —— 被删掉的项会**继续留在列表里**，
     * 用户看到的就是"确认了删除，但文件夹还在"（再点一次才报"找不到"），
     * 甚至按了刷新也还在。因此搜索模式下必须用同一个查询重跑一次搜索。
     */
    const refreshListing = useCallback(async () => {
        // 没有当前目录（首次启动、还没选过文件夹）时什么都不做 —— 直接
        // `loadDirectory("")` 会被后端拒绝，把面板推进错误态。
        if (!fb.currentPath) return;
        await dispatch(loadDirectory(fb.currentPath));
        if (!isSearchMode || isComputerLevel) return;
        await dispatch(
            searchFilesRecursive(
                fileBrowserSearchRequest({
                    dirPath: fb.currentPath,
                    query: trimmedSearchQuery,
                    regexEnabled: fb.regexEnabled,
                    options: searchOptions,
                }),
            ),
        );
    }, [
        dispatch,
        fb.currentPath,
        fb.regexEnabled,
        isComputerLevel,
        isSearchMode,
        searchOptions,
        trimmedSearchQuery,
    ]);

    // 刷新当前目录
    const handleRefresh = useCallback(() => {
        void refreshListing();
    }, [refreshListing]);

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

    return {
        dispatch,
        t,
        tf,
        tVars,
        plural,
        number,
        fb,
        searchSettings,
        view,
        favorites,
        selectedTrackId,
        playheadSec,
        tracks,
        searchInputRef,
        debounceRef,
        previewNavTimerRef,
        searchOptions,
        formatMatchReason,
        previewToggle,
        history,
        canGoBack,
        canGoForward,
        navigateTo,
        goBack,
        goForward,
        isSearchMode,
        trimmedSearchQuery,
        isComputerLevel,
        hasRegexError,
        displayEntries,
        getRelativeDirHint,
        handleOpenFolder,
        refreshListing,
        handleRefresh,
        handleParentDir,
        handleEnterDir,
    };
}

export type FileBrowserListing = ReturnType<typeof useFileBrowserListing>;
