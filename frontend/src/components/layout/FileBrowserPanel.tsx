// 文件浏览器复用 App 界面；插件原生目录选择失败也要显示原因。
// 列表/搜索、选择、菜单、拖拽分别拆到 fileBrowserPanel/ 下的 hook。
import React, { useCallback, useMemo } from "react";
import { Flex, IconButton, TextField } from "@radix-ui/themes";
import {
    ChevronDownIcon,
    ChevronLeftIcon,
    ChevronRightIcon,
    ChevronUpIcon,
    Cross2Icon,
    GearIcon,
    MagnifyingGlassIcon,
    PlayIcon,
    ReloadIcon,
    SpeakerLoudIcon,
    StarIcon,
} from "@radix-ui/react-icons";
import {
    setPreviewVolume,
    setSearchQuery,
    searchFilesRecursive,
    toggleRegex,
} from "../../features/fileBrowser/fileBrowserSlice";
import { fileBrowserSearchRequest } from "../../features/fileBrowser/fileBrowserSearchOptions";
import {
    rowDensityOf,
    type FileBrowserViewOptions,
} from "../../features/fileBrowser/fileBrowserViewOptions";
import {
    persistUiSettings,
    setFileBrowserView,
    setSearchSettings,
    setSearchSettingsDialogOpen,
} from "../../features/session/sessionSlice";
import type { FileEntry } from "../../services/api/fileBrowser";
import { SearchTranslitToggle } from "./search/SearchTranslitToggle";
import { PanelToolbar, PanelToolbarButton } from "./shared/PanelToolbar";
import {
    AppContextMenu,
    AppDialog,
    AppEmptyState,
    AppIconButton,
    AppSelect,
    AppSlider,
    AppSliderReadout,
} from "../../ui";
import { DockInlineRename } from "../dock/DockInlineRename";
import { FileEntryRow } from "./fileBrowser/FileEntryRow";
import { formatModified, formatSize } from "./fileBrowser/formatFile";
import { FileKindIcon, FolderIcon, MediaIcon } from "./fileBrowser/fileIcons";
import { FilePropertiesDialog } from "./fileBrowser/FilePropertiesDialog";
import { FileBrowserViewOptionsDialog } from "./fileBrowser/FileBrowserViewOptionsDialog";
import { useFileBrowserListing } from "./fileBrowserPanel/useFileBrowserListing";
import { useFileBrowserSelection } from "./fileBrowserPanel/useFileBrowserSelection";
import { useFileBrowserActions } from "./fileBrowserPanel/useFileBrowserActions";
import { useFileBrowserDrag } from "./fileBrowserPanel/useFileBrowserDrag";

export const FileBrowserPanel: React.FC = () => {
    const listing = useFileBrowserListing();
    const selection = useFileBrowserSelection(listing);
    const actions = useFileBrowserActions(listing, selection);
    const drag = useFileBrowserDrag(listing, selection);

    const {
        dispatch,
        t,
        tf,
        tVars,
        plural,
        number,
        fb,
        searchSettings,
        view,
        searchInputRef,
        debounceRef,
        searchOptions,
        formatMatchReason,
        canGoBack,
        canGoForward,
        goBack,
        goForward,
        handleParentDir,
        handleOpenFolder,
        handleRefresh,
        navigateTo,
        handleEnterDir,
        isSearchMode,
        trimmedSearchQuery,
        hasRegexError,
        isComputerLevel,
        displayEntries,
        getRelativeDirHint,
    } = listing;
    const {
        selectedPaths,
        activeIndex,
        registerRowRef,
        listScrollRef,
        listWindow,
        handleListScroll,
        handleRowFocus,
        handleRowClick,
        reserveSecondLine,
        handleListKeyDown,
    } = selection;
    const {
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
        menuItems,
    } = actions;
    const { dragState, ghostRef, handlePointerDownForDrag } = drag;

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
            onPointerDown={handlePanelPointerDown}
            onMouseDown={handlePanelMouseDown}
            /*
             * 这两个属性挂在**面板根**而不是列表容器上：面板的键盘模型（
             * `handlePanelKeyDown`）本来就覆盖整个面板 —— 焦点在工具条按钮、路径栏
             * 留白上时打字同样应该跳转。声明在列表上会让"点过工具条背景之后打不了字"。
             */
            data-hs-typeahead="1"
            data-hs-surface="fileBrowser"
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

            {/* 搜索栏。
                两排，按"回答哪个问题"分组，并按**主次**分层：
                  第一排 = 搜索框 + 正则 + 匹配方式（"怎么搜"）——搜索框是本面板的
                    主要动作，因此它是这一屏里**最大**的控件（Radix size 2 = 32px），
                    两侧的正则 / 匹配方式按钮是它的修饰，统一 24px 退居次位；
                  第二排 = 仅显示媒体文件 + 排序方式 + 升降序（"列什么、怎么排"），
                    整排 24px。过滤在最左：先筛出哪些行、再决定怎么排，与用户的心智
                    顺序一致。
                此前排序下拉比搜索框还大：`AppSelect` 的密度是从**容器**继承的，而
                这两排不在 `PanelToolbar` 里（只有它下发 compact），于是下拉落到默认
                的 form 密度、渲染成 32px，搜索框却按 `size="1"` 是 24px —— 次要控件
                反而更抢眼。现在下拉显式声明 `density="compact"`。 */}
            <div className="px-2 py-1 border-b border-qt-border shrink-0">
                <Flex align="center" gap="1">
                    <TextField.Root
                        ref={searchInputRef}
                        size="2"
                        className="flex-1"
                        placeholder={tf("fb_search_placeholder")}
                        value={fb.searchQuery}
                        onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
                            const q = e.target.value;
                            dispatch(setSearchQuery(q));
                            if (debounceRef.current) clearTimeout(debounceRef.current);
                            if (q.trim() && fb.currentPath && !isComputerLevel) {
                                const request = fileBrowserSearchRequest({
                                    dirPath: fb.currentPath,
                                    query: q.trim(),
                                    regexEnabled: fb.regexEnabled,
                                    options: searchOptions,
                                });
                                debounceRef.current = setTimeout(() => {
                                    void dispatch(searchFilesRecursive(request));
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
                                    searchFilesRecursive(
                                        fileBrowserSearchRequest({
                                            dirPath: fb.currentPath,
                                            query: trimmedSearchQuery,
                                            regexEnabled: nextRegexEnabled,
                                            // 正则模式下匹配模式降为 off（与 searchOptions 同源，
                                            // 但这里用的是**切换后**的取值，所以显式给一份）。
                                            options: nextRegexEnabled
                                                ? { ...searchOptions, mode: "off" }
                                                : searchOptions,
                                        }),
                                    ),
                                );
                            }
                        }}
                        style={{
                            fontFamily: "monospace",
                            fontSize: "var(--qt-fs-micro)",
                            width: "var(--qt-ctl-md)",
                            height: "var(--qt-ctl-md)",
                            flexShrink: 0,
                        }}
                        icon=".*"
                    />
                    <SearchTranslitToggle
                        size="var(--qt-ctl-md)"
                        settings={searchSettings}
                        onChange={(patch) => {
                            dispatch(setSearchSettings(patch));
                            void dispatch(persistUiSettings());
                        }}
                        regexActive={fb.regexEnabled}
                        onOpenSettings={() => dispatch(setSearchSettingsDialogOpen(true))}
                    />
                </Flex>

                {/* 第二排：仅显示媒体文件 + 排序方式 + 升降序 —— 都在回答"列什么、怎么排"。
                    过滤开关放在最左：它决定"有哪些行"，排序决定"这些行怎么排" ——
                    先筛后排，顺序与用户的心智顺序一致。 */}
                <Flex align="center" gap="1" mt="1">
                    <AppIconButton
                        active={view.mediaOnly}
                        tooltip={tf("fb_audio_only")}
                        onClick={() => {
                            dispatch(setFileBrowserView({ mediaOnly: !view.mediaOnly }));
                            void dispatch(persistUiSettings());
                        }}
                        style={{
                            width: "var(--qt-ctl-md)",
                            height: "var(--qt-ctl-md)",
                            flexShrink: 0,
                        }}
                        icon={<MediaIcon />}
                    />
                    <AppSelect
                        fullWidth={false}
                        className="flex-1"
                        // 本面板的这两排不在 PanelToolbar 内，继承不到 compact 密度；
                        // 不声明就会落到 form（32px），比搜索框还大。
                        density="compact"
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
                    {/* 方向按钮：箭头与提示都表达**当前**方向，不必点开才知道。 */}
                    <AppIconButton
                        active={view.sortDescending}
                        tooltip={
                            view.sortDescending ? t("fb_sort_descending") : t("fb_sort_ascending")
                        }
                        onClick={() => {
                            dispatch(setFileBrowserView({ sortDescending: !view.sortDescending }));
                            void dispatch(persistUiSettings());
                        }}
                        style={{
                            width: "var(--qt-ctl-md)",
                            height: "var(--qt-ctl-md)",
                            flexShrink: 0,
                        }}
                        icon={view.sortDescending ? <ChevronDownIcon /> : <ChevronUpIcon />}
                    />
                </Flex>

                {hasRegexError && (
                    <span className="hs-type-label" style={{ color: "var(--qt-danger-text)" }}>
                        {tf("fb_regex_error")}
                    </span>
                )}
                {transientError && (
                    /* 失败原因可能很长（后端逐条汇总路径 + 系统错误原文）：
                       单行截断显示，完整内容走 tooltip，避免把这一屏撑破。 */
                    <span
                        className="hs-type-label truncate block"
                        style={{ color: "var(--qt-danger-text)" }}
                        data-tooltip={transientError}
                    >
                        {transientError}
                    </span>
                )}
            </div>

            {/* 导航栏：后退 / 前进 / 上级 + 可编辑路径 + 「打开文件夹」。
                「打开文件夹」放在这里（路径输入框最右侧）：它回答的正是"去哪"，
                与这一排的导航动作同属一件事 —— 放在顶部工具条里反而离它的语义最远。
                整排**始终渲染**：路径为空时它是"打开一个文件夹"的唯一常驻入口。 */}
            <Flex
                align="center"
                gap="1"
                className="px-2 py-1 border-b border-qt-border shrink-0 min-h-[28px]"
            >
                {fb.currentPath && (
                    <>
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
                            /*
                             * 路径过长时显示**末尾**（`…\assets\audio\takes`），而不是
                             * 开头：辨认一条路径靠的是最后几段，`C:\Users\…` 几乎没有
                             * 信息量。
                             *
                             * 做法：`row-reverse` 的 flex 容器 + 子元素 `margin-right: auto`。
                             * - 文本装得下时，auto 边距吃掉剩余空间 → 子元素被推到左边
                             *   （与普通左对齐一样）；
                             * - 文本装不下时，负剩余空间让 auto 边距归零 → 子元素贴住容器
                             *   右侧、向左溢出，容器裁掉左侧 → 露出的正是路径末尾。
                             *
                             * 【为什么不用 `direction: rtl`】那是同类需求最常见的写法，
                             * 但它会让路径里的中性字符（`:` `\`）按双向文本规则重排 ——
                             * 例如 `D:\` 会被渲染成 `\D:`。
                             */
                            <div
                                className="flex min-w-0 flex-1 overflow-hidden cursor-text"
                                style={{
                                    flexDirection: "row-reverse",
                                    justifyContent: "flex-start",
                                }}
                                data-tooltip={isComputerLevel ? tf("fb_computer") : fb.currentPath}
                                onClick={() => setPathDraft(isComputerLevel ? "" : fb.currentPath)}
                            >
                                {/* `flex: none`：不让它被压缩，否则文本会退回"从开头显示"。 */}
                                <span
                                    className="hs-type-label whitespace-nowrap"
                                    style={{ flex: "none", marginRight: "auto" }}
                                >
                                    {isComputerLevel ? tf("fb_computer") : fb.currentPath}
                                </span>
                            </div>
                        ) : (
                            <input
                                autoFocus
                                className="hs-type-label flex-1 min-w-0 bg-qt-base rounded px-1 outline-hidden"
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
                    </>
                )}
                {/* `marginLeft: auto`：路径为空时这一排没有 `flex-1` 元素，
                    按钮仍应贴在右侧（与路径非空时的位置一致）。 */}
                <AppIconButton
                    tooltip={tf("fb_open_folder")}
                    onClick={() => void handleOpenFolder()}
                    style={{ flexShrink: 0, marginLeft: "auto" }}
                    icon={<FolderIcon />}
                />
            </Flex>

            {/* 文件列表。
                用原生滚动容器而不是 Radix ScrollArea：窗口化需要**自己**读写
                `scrollTop` / `clientHeight`，而 ScrollArea 的滚动元素是它内部
                自绘的（要靠 `[data-radix-scroll-area-viewport]` 这种内部属性去找，
                一旦上游改名，窗口就永远不动、列表看起来卡死）。原生容器还给回
                全仓统一的主题滚动条（见 index.css 的滚动条说明）。 */}
            <div
                ref={listScrollRef}
                // `focus:outline-hidden`：容器只是"焦点落点"，视觉指示由**活动行**的
                // 描边负责（`index.css` 的 `[data-active]:focus`）。不关掉的话，浏览器
                // 会给这个可聚焦的 div 画一圈默认的黑色描边 —— 那不是本应用的焦点样式。
                className="hs-scroll-gutter flex-1 min-h-0 overflow-y-auto focus:outline-hidden"
                onScroll={handleListScroll}
                /*
                 * listbox 角色挂在**滚动容器**上（内层只是 `role="presentation"` 的
                 * 排版壳）。
                 *
                 * 【为什么】`listbox` 同时是"这个表面拥有方向键"的判据
                 * （`useKeybindings` 的 ARROW_OWNING_SELECTOR）与"焦点落点"。
                 * 角色挂在里层时，焦点在容器上（点面板留白后就是这样）就找不到
                 * 归属，方向键会漏给全局绑定。挂在容器上则两种焦点位置都成立；
                 * `presentation` 让里层壳在无障碍树里透明，选项仍是 listbox 的直接子项。
                 */
                role={showEntries ? "listbox" : undefined}
                aria-label={showEntries ? tf("fb_file_list") : undefined}
                // 列表本就支持 Ctrl/Shift 多选，声明多选语义以免读屏按单选播报。
                aria-multiselectable={showEntries ? true : undefined}
                // 始终可聚焦：面板里点在没有焦点的空白处时，焦点要落回这里
                // （见 handlePanelPointerDown），键盘模型才活得起来。
                tabIndex={-1}
                onKeyDown={showEntries ? handleListKeyDown : undefined}
            >
                <div
                    className="py-1"
                    role={showEntries ? "presentation" : undefined}
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
                                                    allowDrag={!isComputerLevel}
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

            {/* 底部：点击试听开关 + 音量滑块 —— 两者回答同一个问题（"点一下会发生什么"），
                所以同一行。开关放行首（音量控件的最左侧）。 */}
            <Flex align="center" gap="2" className="px-2 py-1.5 border-t border-qt-border shrink-0">
                <AppIconButton
                    active={view.previewOnClick}
                    // 表达"这个功能开着"，必须显式声明 accent（默认 neutral 是灰的）。
                    emphasis="accent"
                    tooltip={tf("fb_preview_on_click")}
                    size="sm"
                    onClick={() => {
                        dispatch(setFileBrowserView({ previewOnClick: !view.previewOnClick }));
                        void dispatch(persistUiSettings());
                    }}
                    style={{ width: "var(--qt-ctl-sm)", height: "var(--qt-ctl-sm)", flexShrink: 0 }}
                    icon={<PlayIcon />}
                />
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
