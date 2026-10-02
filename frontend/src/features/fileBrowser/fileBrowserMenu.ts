/**
 * 文件浏览器右键菜单的**结构**（纯函数，不渲染）。
 *
 * 【为什么把菜单拆出来】菜单内容取决于"点的是什么"（目录 / 音频 / MIDI / 工程 /
 * 多选 / 空白区）与"当前在哪"（计算机虚拟层 / 搜索模式 / 是否正在试听）。
 * 把这些判断留在面板组件里，就只能靠渲染整个面板来测 —— 而面板依赖 Redux、
 * Tauri 与 AudioContext。抽成"给定目标与上下文，返回菜单项数组"的纯函数后，
 * 每一条分派规则都能直接单测（见 `fileBrowserMenu.test.ts`）。
 *
 * 【为什么用平铺 + `heading` 而不是子菜单】`AppContextMenu` 的 items API 不支持
 * 嵌套子菜单（`AppSubMenu` 是给手写 JSX 菜单用的另一个组件）。`AppMenuItemSpec`
 * 自带 `heading`（不可选中的分组小标题），正是为这种"平面菜单里的分段"准备的。
 * 用它与用子菜单表达同一件事，且不必为此改动菜单原语。
 */

import type { FileEntry } from "../../services/api/fileBrowser";
import type { MessageKey } from "../../i18n/messages";
import type { AppMenuItemSpec } from "../../ui";
import { FILE_BROWSER_SORT_LABEL_KEY, type FileBrowserViewOptions } from "./fileBrowserViewOptions";
import { isAudioFile, isMidiFile, isProjectFile } from "./fileKinds";

/** 多文件插入时间轴的三种排布方式（与 `importMultipleAudioAtPosition` 对齐）。 */
export type MultiInsertMode = "across-time" | "across-tracks" | "as-takes";

/** 菜单项要触发的动作。面板负责实现，菜单只负责决定"什么时候显示哪一条"。 */
export interface FileBrowserMenuActions {
    /** 目录 → 进入；音频 → 试听；工程 → 打开。 */
    openEntry(entry: FileEntry): void;
    insertAtPlayhead(entries: FileEntry[]): void;
    insertOnNewTrack(entries: FileEntry[]): void;
    insertMultiple(entries: FileEntry[], mode: MultiInsertMode): void;
    /**
     * 把一批**目录**展开后导入（打开选项对话框）。
     *
     * 【为什么目录只有一条菜单项，而文件有三条】文件那三条（依次排列 / 分到多轨 /
     * 叠成 Take）就是全部的排布选择，列在菜单里省一次点击。目录还多出两个**正交**的
     * 问题（要不要下钻子目录、要不要为文件夹建轨道组），塞不进菜单项；列三条模式
     * 只会变成"点了之后还要再选一次"或者"静默用记住的选项"。所以目录走对话框，
     * 三个模式在对话框里选 —— 与 REAPER 在这件事上的做法一致。
     */
    importFolder(entries: FileEntry[]): void;
    togglePreview(entry: FileEntry): void;
    reveal(paths: string[]): void;
    /** 用系统默认程序打开一个路径。 */
    openWithDefaultApp(path: string): void;
    copyPaths(paths: string[]): void;
    copyName(name: string): void;
    openProject(path: string): void;
    importProject(path: string): void;
    importMidi(path: string): void;
    /** 搜索模式下跳到该条目所在的目录。 */
    openContainingFolder(entry: FileEntry): void;
    rename(entry: FileEntry): void;
    remove(entries: FileEntry[]): void;
    showProperties(entry: FileEntry): void;
    newFolder(): void;
    refresh(): void;
    openFolderDialog(): void;
    selectAll(): void;
    clearSelection(): void;
    setSortMode(mode: FileBrowserViewOptions["sortMode"]): void;
    setSortDescending(descending: boolean): void;
    patchView(patch: Partial<FileBrowserViewOptions>): void;
    openViewOptions(): void;
}

export interface FileBrowserMenuContext {
    /** 已本地化的翻译函数。 */
    t: (key: MessageKey) => string;
    /**
     * 带数量文案的复数形式（`{count}` 由它回填，并按语系格式化数字）。
     *
     * 【为什么菜单也需要它】"删除 N 项"是**带数量**的条目：只传 `t` 会把模板原样
     * 渲染出来 —— 界面上出现的就是 `Delete {n} Items` 这种带花括号的文案。
     */
    plural: (key: MessageKey, count: number) => string;
    view: FileBrowserViewOptions;
    /** 是否在「计算机」虚拟层（盘符列表）—— 写操作与"新建文件夹"在此无意义。 */
    isComputerLevel: boolean;
    /** 是否处于搜索模式（结果可能来自子目录）。 */
    isSearchMode: boolean;
    /** 当前正在试听的路径。 */
    previewingPath: string | null;
    /** 当前多选集合，按列表顺序。 */
    selected: FileEntry[];
    /** 当前目录（背景菜单的"在文件管理器中显示当前目录"用）。 */
    currentPath: string;
    actions: FileBrowserMenuActions;
}

/**
 * 右键落在一行上时，这一行操作的是谁。
 *
 * 【Explorer 语义】右键未选中的行 → 先把它选中，菜单只作用于它；右键已选中的行 →
 * 菜单作用于整个选中集合。这条规则决定了"删除 5 个文件"和"删除这一个"的区别，
 * 因此必须由菜单自己算，而不是让每个调用点各判一次。
 */
export function resolveMenuTargets(clicked: FileEntry, selected: FileEntry[]): FileEntry[] {
    const inSelection = selected.some((entry) => entry.path === clicked.path);
    return inSelection && selected.length > 0 ? selected : [clicked];
}

/** 目录、音频、MIDI、工程各自的"主操作"标签键。 */
function primaryActionKey(entry: FileEntry): MessageKey {
    if (entry.isDir) return "fb_ctx_open";
    if (isProjectFile(entry)) return "fb_ctx_open_project";
    if (isMidiFile(entry)) return "fb_ctx_import_midi";
    return "fb_ctx_preview";
}

/** 一行条目的菜单。 */
function entryMenu(targets: FileEntry[], ctx: FileBrowserMenuContext): AppMenuItemSpec[] {
    const { t, actions } = ctx;
    const single = targets.length === 1 ? targets[0] : null;
    const items: AppMenuItemSpec[] = [];

    // ── 主操作 ──────────────────────────────────────────────────────────
    if (single) {
        items.push({
            key: "primary",
            label: t(primaryActionKey(single)),
            onSelect: () => actions.openEntry(single),
        });
    }

    // ── 插入到时间轴（只有音频/视频能直接插入；MIDI 与工程各有自己的入口）──
    const insertable = targets.filter(isAudioFile);
    // 目录：整批（可含散文件）交给目录导入的选项对话框。
    const folders = targets.filter((entry) => entry.isDir);
    if (folders.length > 0) {
        items.push({
            key: "import-folder",
            label: t("fb_ctx_import_folder"),
            onSelect: () => actions.importFolder(folders),
            separatorBefore: items.length > 0,
        });
    }
    if (insertable.length > 0) {
        if (single) {
            // 单个文件：两种落点直接列出来，不必先选"排布方式"。
            items.push(
                {
                    key: "insert-playhead",
                    label: t("fb_ctx_insert_playhead"),
                    onSelect: () => actions.insertAtPlayhead(insertable),
                },
                {
                    key: "insert-new-track",
                    label: t("fb_ctx_insert_new_track"),
                    onSelect: () => actions.insertOnNewTrack(insertable),
                },
            );
        } else {
            // 多选：三种排布方式各一条，标题说明它们属于同一组。
            items.push(
                { key: "insert-heading", label: t("fb_ctx_insert_heading"), heading: true },
                {
                    key: "insert-across-time",
                    label: t("fb_ctx_insert_across_time"),
                    onSelect: () => actions.insertMultiple(insertable, "across-time"),
                },
                {
                    key: "insert-across-tracks",
                    label: t("fb_ctx_insert_across_tracks"),
                    onSelect: () => actions.insertMultiple(insertable, "across-tracks"),
                },
                {
                    key: "insert-as-takes",
                    label: t("fb_ctx_insert_as_takes"),
                    onSelect: () => actions.insertMultiple(insertable, "as-takes"),
                },
            );
        }
    }

    // ── 试听（单个音频文件才有意义）────────────────────────────────────
    if (single && isAudioFile(single)) {
        const playing = ctx.previewingPath === single.path;
        items.push({
            key: "preview",
            label: playing ? t("fb_ctx_stop_preview") : t("fb_ctx_preview"),
            onSelect: () => actions.togglePreview(single),
        });
    }

    // ── 工程文件的第二个入口：导入 ──────────────────────────────────────
    if (single && isProjectFile(single)) {
        items.push({
            key: "import-project",
            label: t("fb_ctx_import_project"),
            onSelect: () => actions.importProject(single.path),
        });
    }

    // ── 搜索模式下跳回文件所在目录 ──────────────────────────────────────
    if (single && ctx.isSearchMode) {
        items.push({
            key: "open-containing",
            label: t("fb_ctx_open_containing_folder"),
            onSelect: () => actions.openContainingFolder(single),
        });
    }

    // ── 系统集成 ────────────────────────────────────────────────────────
    const paths = targets.map((entry) => entry.path);
    items.push(
        {
            key: "reveal",
            label: t("fb_ctx_reveal"),
            separatorBefore: true,
            onSelect: () => actions.reveal(paths),
        },
        {
            key: "copy-paths",
            label: single ? t("fb_ctx_copy_path") : t("fb_ctx_copy_paths"),
            onSelect: () => actions.copyPaths(paths),
        },
    );
    if (single) {
        items.push({
            key: "copy-name",
            label: t("fb_ctx_copy_name"),
            onSelect: () => actions.copyName(single.name),
        });
        // 交给系统默认程序：`.txt` / `.rpp` 这类本应用不能导入的文件，
        // 用记事本打开往往正是用户想做的事。
        if (!single.isDir) {
            items.push({
                key: "open-default-app",
                label: t("fb_ctx_open_with_default_app"),
                onSelect: () => actions.openWithDefaultApp(single.path),
            });
        }
    }

    // ── 写操作（「计算机」层没有真实文件可改）──────────────────────────
    if (!ctx.isComputerLevel) {
        if (single) {
            items.push({
                key: "rename",
                label: t("fb_ctx_rename"),
                shortcut: "F2",
                separatorBefore: true,
                onSelect: () => actions.rename(single),
            });
        }
        items.push({
            key: "delete",
            label: single ? t("fb_ctx_delete") : ctx.plural("fb_ctx_delete_items", targets.length),
            danger: true,
            separatorBefore: single ? false : true,
            onSelect: () => actions.remove(targets),
        });
    }

    // ── 属性 ────────────────────────────────────────────────────────────
    if (single) {
        items.push({
            key: "properties",
            label: t("fb_ctx_properties"),
            separatorBefore: true,
            onSelect: () => actions.showProperties(single),
        });
    }

    return items;
}

/** 空白区（列表背景）的菜单。 */
function backgroundMenu(ctx: FileBrowserMenuContext): AppMenuItemSpec[] {
    const { t, view, actions } = ctx;
    const items: AppMenuItemSpec[] = [];

    if (!ctx.isComputerLevel) {
        items.push({
            key: "new-folder",
            label: t("fb_ctx_new_folder"),
            shortcut: "Ctrl+Shift+N",
            onSelect: actions.newFolder,
        });
    }
    items.push({
        key: "refresh",
        label: t("fb_ctx_refresh"),
        shortcut: "F5",
        onSelect: actions.refresh,
    });

    // ── 排序 ────────────────────────────────────────────────────────────
    items.push(
        { key: "sort-heading", label: t("fb_sort_label"), heading: true, separatorBefore: true },
        ...(["name", "date", "size"] as const).map((mode) => ({
            key: `sort-${mode}`,
            label: t(FILE_BROWSER_SORT_LABEL_KEY[mode]),
            checked: view.sortMode === mode,
            onSelect: () => actions.setSortMode(mode),
        })),
        {
            key: "sort-descending",
            label: t("fb_sort_descending"),
            checked: view.sortDescending,
            onSelect: () => actions.setSortDescending(!view.sortDescending),
        },
    );

    // ── 常用显示开关 ────────────────────────────────────────────────────
    items.push(
        {
            key: "folders-first",
            label: t("fb_folders_first"),
            checked: view.foldersFirst,
            separatorBefore: true,
            onSelect: () => actions.patchView({ foldersFirst: !view.foldersFirst }),
        },
        {
            key: "show-hidden",
            label: t("fb_show_hidden"),
            checked: view.showHiddenFiles,
            onSelect: () => actions.patchView({ showHiddenFiles: !view.showHiddenFiles }),
        },
        {
            key: "media-only",
            label: t("fb_audio_only"),
            checked: view.mediaOnly,
            onSelect: () => actions.patchView({ mediaOnly: !view.mediaOnly }),
        },
        {
            key: "view-options",
            label: t("fb_view_options"),
            separatorBefore: true,
            onSelect: actions.openViewOptions,
        },
    );

    // ── 选择与目录 ──────────────────────────────────────────────────────
    items.push(
        {
            key: "select-all",
            label: t("fb_ctx_select_all"),
            shortcut: "Ctrl+A",
            separatorBefore: true,
            disabled: ctx.selected.length === 0 && ctx.currentPath === "",
            onSelect: actions.selectAll,
        },
        {
            key: "clear-selection",
            label: t("fb_ctx_clear_selection"),
            disabled: ctx.selected.length === 0,
            onSelect: actions.clearSelection,
        },
    );

    if (!ctx.isComputerLevel && ctx.currentPath) {
        items.push(
            {
                key: "reveal-current",
                label: t("fb_ctx_reveal_current_folder"),
                separatorBefore: true,
                onSelect: () => actions.reveal([ctx.currentPath]),
            },
            {
                key: "open-folder-dialog",
                label: t("fb_open_folder"),
                onSelect: actions.openFolderDialog,
            },
        );
    }

    return items;
}

/**
 * 构建右键菜单。
 *
 * @param clicked 右键落点所在的条目；`null` 表示点在列表空白处。
 */
export function buildFileBrowserContextMenu(
    clicked: FileEntry | null,
    ctx: FileBrowserMenuContext,
): AppMenuItemSpec[] {
    return clicked === null
        ? backgroundMenu(ctx)
        : entryMenu(resolveMenuTargets(clicked, ctx.selected), ctx);
}
