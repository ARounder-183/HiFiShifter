// FileBrowser 的窗口化列表、多选与键盘导航。拆自 FileBrowserPanel。
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { computeListWindow } from "../../../features/fileBrowser/listWindow";
import { rowDensityOf } from "../../../features/fileBrowser/fileBrowserViewOptions";
import type { FileEntry } from "../../../services/api/fileBrowser";
import { isAudioFile } from "../../../features/fileBrowser/fileKinds";
import { isPrimaryModifierDown } from "../../../utils/platform";
import {
    isFileListActivationKey,
    isFileListNavKey,
    nextActiveIndex,
    rangeIndexes,
    selectionIntentOf,
    type FileListSelectionIntent,
} from "../fileBrowserKeyboardNav";
import type { FileBrowserListing } from "./useFileBrowserListing";

/** 键盘移动后自动试听的防抖（毫秒）。 */
const PREVIEW_NAV_DEBOUNCE_MS = 160;

export function useFileBrowserSelection(listing: FileBrowserListing) {
    const {
        view,
        searchSettings,
        isSearchMode,
        displayEntries,
        handleEnterDir,
        previewToggle,
        previewNavTimerRef,
    } = listing;
    const [selectedPaths, setSelectedPaths] = useState<Set<string>>(new Set());
    /**
     * 选区锚点：Shift 扩展与 Shift + 点击都从这里算起。
     *
     * 普通点击 / 普通方向键会把锚点移到光标处；带 Shift 的操作只读不动它 ——
     * 这正是资源管理器里「连续 Shift 扩展」的语义。
     */
    const selectionAnchorRef = useRef<number>(-1);

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
        [view.previewOnNavigate, previewToggle, previewNavTimerRef],
    );

    /**
     * 本次"激活一个音频文件"要不要出声。
     *
     * 【为什么收口到一处】点击试听此前散在 handleRowClick 与 activateEntry 两处，
     * 各自判 `isAudioFile`。开关若只加在其中一处，另一处仍会出声 —— 用户看到的
     * 是"我关掉了，但它有时响"。右键菜单的「试听」是**显式命令**（用户点名要听），
     * 不受这个开关管辖，因此 openEntry 不经过这里。
     */
    const shouldPreviewOnActivate = useCallback(
        (entry: FileEntry) => view.previewOnClick && isAudioFile(entry),
        [view.previewOnClick],
    );

    /** 选区落在闭区间 `[a, b]` 内的全部路径。 */
    const selectedPathsInRange = useCallback(
        (a: number, b: number) =>
            new Set(
                rangeIndexes(a, b)
                    .map((i) => displayEntries[i]?.path)
                    .filter((path): path is string => Boolean(path)),
            ),
        [displayEntries],
    );

    /**
     * 单击一行。
     *
     * 【为什么选中与试听分开】行的左键语义是"选中它"（让 Ctrl+C / Delete / F2 /
     * 拖拽有作用对象）；音频文件在此之上**额外**试听。此前只有音频行接了点击，
     * 点一个 `.txt` 什么都不发生 —— 而右键菜单却能对它操作，两条路径的可用性
     * 不一致。
     *
     * 【为什么下标空间必须统一】锚点此前记的是 **audioEntries** 的下标，而键盘
     * 光标用的是 displayEntries 的下标；Shift 范围选择因此只在音频之间连线，与
     * 列表里看到的顺序不是一回事。现在统一用 displayEntries。
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
                selectionAnchorRef.current = index;
                return;
            }

            if (ev.shiftKey && selectionAnchorRef.current >= 0) {
                // Shift+click：从锚点到这一行的范围选择（含目录，与列表所见一致）
                setSelectedPaths(selectedPathsInRange(selectionAnchorRef.current, index));
                return;
            }

            // 普通点击：单选这一行；音频文件再切换试听
            // （"再点一次停止"由 `usePreviewToggle` 统一实现：此前这里只有播放
            // 分支，重复点击会从头重放并与在播的旧音源叠加。）
            setSelectedPaths(new Set([entry.path]));
            selectionAnchorRef.current = index;
            if (shouldPreviewOnActivate(entry)) previewToggle.toggle(entry.path);
        },
        [displayEntries, previewToggle, selectedPathsInRange, shouldPreviewOnActivate],
    );

    /**
     * 键盘激活一行：与鼠标走**同一条**路径 —— 目录进入子目录，音频文件切换试听。
     * 不复制这两段逻辑，只做选择。
     */
    const activateEntry = useCallback(
        (entry: FileEntry) => {
            if (entry.isDir) {
                handleEnterDir(entry.path);
            } else if (shouldPreviewOnActivate(entry)) {
                previewToggle.toggle(entry.path);
            }
        },
        [handleEnterDir, previewToggle, shouldPreviewOnActivate],
    );

    /**
     * 把一次按键的**选区意图**应用到当前光标行上。
     *
     * 规则见 `selectionIntentOf`：普通方向键替换选区（资源管理器模型）、`Shift`
     * 从锚点扩展、`Ctrl` 只移动光标、`Ctrl`+空格加/减选。
     */
    const applySelectionIntent = useCallback(
        (intent: FileListSelectionIntent | null, index: number) => {
            const entry = displayEntries[index];
            if (!entry || intent === null || intent === "moveOnly") return;
            if (intent === "extend") {
                const anchor = selectionAnchorRef.current >= 0 ? selectionAnchorRef.current : index;
                setSelectedPaths(selectedPathsInRange(anchor, index));
                return;
            }
            if (intent === "toggle") {
                setSelectedPaths((prev) => {
                    const next = new Set(prev);
                    if (next.has(entry.path)) next.delete(entry.path);
                    else next.add(entry.path);
                    return next;
                });
                selectionAnchorRef.current = index;
                return;
            }
            // replace
            selectionAnchorRef.current = index;
            setSelectedPaths(new Set([entry.path]));
        },
        [displayEntries, selectedPathsInRange],
    );

    // ── 对话框、行内编辑与选择状态 ─────────────────────────────────────────
    // 这些 setter 会被下方的键盘模型与目录切换 effect 引用；声明必须在它们之前，
    // 否则 React Compiler 会把「闭包引用后声明的 const」判成 TDZ 风险并跳过优化。
    const selectAll = useCallback(() => {
        setSelectedPaths(new Set(displayEntries.map((entry) => entry.path)));
    }, [displayEntries]);

    /**
     * 列表容器的键盘模型。
     *
     * 移动：方向键 / Home / End（夹紧，不环绕）。方向键必须 `preventDefault`，
     * 否则滚动容器会跟着滚。
     * 选区：普通方向键 = 移动并选中该行，`Shift` + 方向键 = 从锚点扩展，
     * `Ctrl` + 方向键 = 只移动光标，`Ctrl` + 空格 = 加/减选（均见 `selectionIntentOf`）。
     * 激活：Enter / 空格（不带修饰键时）。
     */
    const handleListKeyDown = useCallback(
        (event: React.KeyboardEvent<HTMLDivElement>) => {
            const intent = selectionIntentOf(event.key, {
                shift: event.shiftKey,
                ctrl: event.ctrlKey,
                meta: event.metaKey,
            });

            if (isFileListNavKey(event.key)) {
                event.preventDefault();
                const next = nextActiveIndex(activeIndex, event.key, displayEntries.length);
                if (next < 0) return;
                setActiveIndex(next);
                focusRow(next);
                maybePreviewOnNavigate(displayEntries[next]);
                applySelectionIntent(intent, next);
                return;
            }
            // Ctrl+空格先于激活分支：它带修饰键，语义是"加/减选"而不是"试听"。
            if (intent === "toggle" && activeIndex >= 0) {
                event.preventDefault();
                applySelectionIntent(intent, activeIndex);
                return;
            }
            if (isFileListActivationKey(event.key)) {
                const entry = displayEntries[activeIndex];
                if (!entry) return;
                event.preventDefault();
                activateEntry(entry);
            }
        },
        [
            activeIndex,
            displayEntries,
            activateEntry,
            applySelectionIntent,
            focusRow,
            maybePreviewOnNavigate,
        ],
    );

    const selectedEntries = useMemo(
        () => displayEntries.filter((entry) => selectedPaths.has(entry.path)),
        [displayEntries, selectedPaths],
    );

    return {
        selectedPaths,
        setSelectedPaths,
        selectionAnchorRef,
        activeIndex,
        setActiveIndex,
        rowRefs,
        registerRowRef,
        listScrollRef,
        viewport,
        listWindow,
        rowHeight,
        reserveSecondLine,
        handleListScroll,
        handleRowFocus,
        focusRow,
        maybePreviewOnNavigate,
        shouldPreviewOnActivate,
        selectedPathsInRange,
        handleRowClick,
        activateEntry,
        applySelectionIntent,
        selectAll,
        handleListKeyDown,
        selectedEntries,
    };
}

export type FileBrowserSelection = ReturnType<typeof useFileBrowserSelection>;
