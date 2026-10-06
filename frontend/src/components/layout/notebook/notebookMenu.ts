/*
 * 记事本右键菜单的**内容**（纯函数，不渲染、不碰编辑器）。
 *
 * 【文件名的由来：不能叫 `notebookContextMenu.ts`】本目录里已经有一个
 * `NotebookContextMenu.tsx`（渲染壳）。**Windows 与 macOS 的默认文件系统不区分
 * 大小写**，而模块解析在补扩展名之前就做大小写不敏感匹配 —— 于是
 * `import { NotebookContextMenu } from "./NotebookContextMenu"` 会解析到这个
 * 纯逻辑文件，拿到 `undefined`，运行时报"Element type is invalid"（React 只会
 * 说某个组件是 undefined，不会告诉你是名字撞了）。实测踩过一次：全量测试里
 * 只有"打开 `⋯` 菜单"的那一条挂掉，其余三条照常通过。
 * 因此纯逻辑这层取名 `notebookMenu`，与 `features/fileBrowser/fileBrowserMenu.ts`
 * 的命名也正好对齐。
 *
 * 【为什么拆成纯函数】菜单内容取决于"右键点在什么上"（链接 / 表格 / 列表 /
 * 图片 / 暂存块 / 文本 / 空白）、"编辑器现在能不能编辑"、"有没有选中东西"，
 * 以及"菜单要多全"。把这些判断留在面板组件里，就只能靠渲染整个面板来测 ——
 * 而面板依赖 Redux、Tauri 与 AudioContext。抽成"给定目标与开关，返回菜单项
 * 数组"之后，每一条分派规则都能直接单测（见 `notebookMenu.test.ts`）。
 *
 * 【为什么用平铺 + `heading` 而不是子菜单】`AppContextMenu` 的 items API 不支持
 * 嵌套子菜单（`AppSubMenu` 是给手写 JSX 菜单用的另一个组件）。`AppMenuItemSpec`
 * 自带 `heading`（不可选中的分组小标题），正是为"平面菜单里的分段"准备的。
 *
 * 【actions 全部可选 —— 这是本设计的枢纽】builder 只发出"调用方提供了动作"的
 * 项。于是同一份规则可以同时服务三种调用方：
 *   - 正文面板：提供全部动作（它有 Redux、有 insertContext）；
 *   - 图片 / 暂存块 NodeView：只提供自己知道的动作（复制图片、另存为、重命名…），
 *     自然就不会出现"插入播放头时间"这种它做不到的项。
 * 若 actions 是必填的，NodeView 就得为了复用去接一整套用不到的依赖 —— 那才是
 * 真正的耦合。
 *
 * 【分隔线为什么用标记而不是 `separatorBefore`】`AppMenuItemSpec` 的分隔线挂在
 * "某一项"上，而这里的项是**条件产出**的：某组首项因为调用方没给动作而消失时，
 * 挂在它身上的分隔线会一起消失（菜单里出现两组黏在一起），或者更糟 —— 留在
 * 一个空组前面。因此这里用显式的 `SEPARATOR` 标记，由 `assemble()` 统一解析：
 * 空组前后的分隔线会被吞掉，相邻分隔线合并，首尾不留线。调用方不必再操心
 * "这一项到底会不会出现"。
 *
 * 【快捷键提示为什么不走 `useMenuShortcut`】那个钩子读的是**快捷键注册表**，
 * 而本菜单里的编辑键（Ctrl+B/I/E、Ctrl+Z、Ctrl+C/X/V/A）全部是 ProseMirror /
 * StarterKit 自己的 keymap，不在注册表里、也不可重绑。若去读注册表里同名的
 * `edit.undo` / `clip.copy`，用户一旦改绑那两项，菜单就会显示一个**在记事本里
 * 按了没反应**的键。因此这里只声明编辑器真实的键位，并用全应用唯一的格式化器
 * `formatKeybinding` 渲染（它负责 ⌘ / Ctrl 的平台差异），不另写一套拼接。
 */

import { formatKeybinding } from "../../../features/keybindings/keybindingsSlice";
import type { Keybinding } from "../../../features/keybindings/types";
import type { MessageKey } from "../../../i18n/messages";
import type { AppMenuItemSpec } from "../../../ui";
import type { NotebookContext } from "./notebookContextTarget";

/** 菜单详略。与设置项 `contextMenu` 的两档一一对应（`off` 由调用方拦截）。 */
export type NotebookMenuScope = "full" | "compact";

/** 菜单属于哪个表面。源码视图是 textarea，没有富文本语义。 */
export type NotebookMenuSurface = "rich" | "source" | "preview";

/**
 * 菜单能触发的动作。**全部可选** —— 缺哪个，对应的项就不出现。
 *
 * 这些是"键"，动作本身由调用方提供（面板 / NodeView）。builder 只决定
 * "什么时候显示哪一条"。
 */
export type NotebookMenuActionKey =
    // 撤销
    | "undo"
    | "redo"
    // 剪贴板
    | "cut"
    | "copy"
    | "copyMarkdown"
    | "copyPlain"
    | "paste"
    | "pastePlain"
    | "pasteMarkdown"
    | "pasteImage"
    | "stageClipboard"
    | "selectAll"
    // 格式
    | "bold"
    | "italic"
    | "strike"
    | "code"
    | "clearFormatting"
    // 块
    | "heading1"
    | "heading2"
    | "heading3"
    | "paragraph"
    | "indent"
    | "outdent"
    // 链接
    | "linkOpen"
    | "linkCopy"
    | "linkEdit"
    | "linkRemove"
    // 表格
    | "tableRowAbove"
    | "tableRowBelow"
    | "tableColLeft"
    | "tableColRight"
    | "tableDeleteRow"
    | "tableDeleteCol"
    | "tableDelete"
    | "tableToggleHeader"
    // 插入
    | "insertImage"
    | "insertTable"
    | "insertRule"
    | "insertTimecode"
    | "insertClipReference"
    | "insertProjectInfo"
    // 查找
    | "find"
    // 图片（NodeView 提供）
    | "imageCopy"
    | "imageSaveAs"
    | "imageWidthReset"
    | "imageEditAlt"
    | "imageRemove"
    // 暂存块（NodeView 提供）
    | "clipRestore"
    | "clipInsert"
    | "clipInsertNewTracks"
    | "clipRename"
    | "clipCopyMarkdown"
    | "clipSavePayload"
    | "clipRemove";

export type NotebookMenuActions = Partial<Record<NotebookMenuActionKey, () => void>>;

/** 置灰项：动作键 → 原因（本地化文本，进 `data-tooltip`）。缺省即可用。 */
export type NotebookMenuDisabled = Partial<Record<NotebookMenuActionKey, string>>;

export interface NotebookMenuContext {
    surface: NotebookMenuSurface;
    scope: NotebookMenuScope;
    /** 落点与开关，来自 `prepareNotebookContext`（或 NodeView 自己构造）。 */
    context: NotebookContext;
    translate: (key: MessageKey) => string;
    actions: NotebookMenuActions;
    disabled?: NotebookMenuDisabled;
}

// ─── 编辑器自身的键位（只作展示） ──────────────────────────────────────────────
//
// 这些是 ProseMirror / StarterKit 的 keymap 条目，不是用户可重绑的注册表项，
// 因此在这里声明成常量而不是进 `DEFAULT_KEYBINDINGS`。渲染统一走
// `formatKeybinding`，macOS 下自动变成 ⌘B / ⇧⌘S。

const CHORD_UNDO: Keybinding = { key: "z", ctrl: true };
const CHORD_REDO: Keybinding = { key: "z", ctrl: true, shift: true };
const CHORD_CUT: Keybinding = { key: "x", ctrl: true };
const CHORD_COPY: Keybinding = { key: "c", ctrl: true };
const CHORD_PASTE: Keybinding = { key: "v", ctrl: true };
const CHORD_SELECT_ALL: Keybinding = { key: "a", ctrl: true };
const CHORD_FIND: Keybinding = { key: "f", ctrl: true };
const CHORD_BOLD: Keybinding = { key: "b", ctrl: true };
const CHORD_ITALIC: Keybinding = { key: "i", ctrl: true };
const CHORD_STRIKE: Keybinding = { key: "s", ctrl: true, shift: true };
const CHORD_CODE: Keybinding = { key: "e", ctrl: true };

/** 全部展示用快捷键文本（导出以便面板与测试共用同一份口径）。 */
export const NOTEBOOK_MENU_SHORTCUTS = {
    undo: formatKeybinding(CHORD_UNDO),
    redo: formatKeybinding(CHORD_REDO),
    cut: formatKeybinding(CHORD_CUT),
    copy: formatKeybinding(CHORD_COPY),
    paste: formatKeybinding(CHORD_PASTE),
    selectAll: formatKeybinding(CHORD_SELECT_ALL),
    find: formatKeybinding(CHORD_FIND),
    bold: formatKeybinding(CHORD_BOLD),
    italic: formatKeybinding(CHORD_ITALIC),
    strike: formatKeybinding(CHORD_STRIKE),
    code: formatKeybinding(CHORD_CODE),
} as const;

// ─── 项构造 ────────────────────────────────────────────────────────────────────

/**
 * 分隔线标记（由 `assemble` 解析，不会出现在结果里）。
 *
 * 【为什么是对象而不是 `Symbol()`】`Symbol()` 在**未标注类型的数组字面量**里会被
 * TypeScript 拓宽成 `symbol`，即使显式声明 `const SEP: unique symbol = Symbol()`
 * 也一样 —— 实测 `const items = [mk(), SEP]` 推出 `(symbol | …)[]`，赋给
 * `MenuEntry[]` 直接报错（`tsc -b` 才看得见；vitest 只剥类型不做检查，所以测试
 * 全绿也可能构建失败）。具名接口的对象字面量不会这样拓宽。
 *
 * 配一个类型谓词而不是"与常量比相等"，是因为 TS 不为对象身份比较做窄化。
 */
interface MenuSeparator {
    readonly __menuSeparator: true;
}
const SEPARATOR: MenuSeparator = { __menuSeparator: true };
type MenuEntry = AppMenuItemSpec | null | MenuSeparator;

function isSeparator(entry: MenuEntry): entry is MenuSeparator {
    return entry !== null && "__menuSeparator" in entry;
}

interface ItemOptions {
    shortcut?: string;
    danger?: boolean;
    checked?: boolean;
    /** 置灰原因（本地化文本）。 */
    disabledReason?: string;
}

/**
 * 造一项；动作缺失时返回 null（由 `assemble` 丢掉）。
 *
 * 置灰原因取"调用方显式给的"与"`opts` 里的"之先者 —— 调用方更清楚具体
 * 语境（例如暂存块的载荷丢了），`opts` 只表达通用规则（空选区不能复制）。
 */
function makeItem(
    ctx: NotebookMenuContext,
    key: NotebookMenuActionKey,
    labelKey: MessageKey,
    opts: ItemOptions = {},
): MenuEntry {
    const action = ctx.actions[key];
    if (!action) return null;
    const reason = ctx.disabled?.[key] ?? opts.disabledReason;
    return {
        key,
        label: ctx.translate(labelKey),
        onSelect: action,
        shortcut: opts.shortcut,
        danger: opts.danger,
        checked: opts.checked,
        disabled: reason !== undefined,
        tooltip: reason,
    };
}

/** 分组标题行。 */
function headingItem(key: string, label: string): AppMenuItemSpec {
    return { key, label, heading: true };
}

/**
 * 解析标记：丢掉空项、合并相邻分隔线、剥掉首尾分隔线。
 *
 * 这是"条件产出"能安全使用分隔线的原因 —— 调用方只管在"逻辑上该分组的地方"
 * 放一个标记，不必预判那一组会不会整组消失。
 */
export function assemble(entries: MenuEntry[]): AppMenuItemSpec[] {
    const out: AppMenuItemSpec[] = [];
    let pendingSeparator = false;
    for (const entry of entries) {
        if (entry === null) continue;
        if (isSeparator(entry)) {
            if (out.length > 0) pendingSeparator = true;
            continue;
        }
        out.push(pendingSeparator ? { ...entry, separatorBefore: true } : entry);
        pendingSeparator = false;
    }
    return out;
}

/**
 * 组装记事本右键菜单。
 *
 * 顺序固定：**落点专属组 → 撤销 → 剪贴板 → 选择 → 格式 → 段落 → 插入 → 查找**。
 * 落点专属项排最前 —— 右键一张图片时，"复制图片"是他要的，"加粗"不是。
 */
export function buildNotebookContextMenu(ctx: NotebookMenuContext): AppMenuItemSpec[] {
    if (ctx.surface === "source") return buildSourceMenu(ctx);
    return buildEditorMenu(ctx);
}

/**
 * 源码视图（`<textarea>`）的菜单。
 *
 * 只有最基础的一组：源码视图里"格式 / 段落 / 插入"都没有意义（所见即源码），
 * 而它的原生右键菜单被应用全局禁用了 —— 不给这一组，用户连"复制"都要靠键盘。
 * "粘贴为纯文本"同样不出现：textarea 里本来就是纯文本。
 */
function buildSourceMenu(ctx: NotebookMenuContext): AppMenuItemSpec[] {
    return assemble([
        makeItem(ctx, "undo", "menu_undo", { shortcut: NOTEBOOK_MENU_SHORTCUTS.undo }),
        makeItem(ctx, "redo", "menu_redo", { shortcut: NOTEBOOK_MENU_SHORTCUTS.redo }),
        SEPARATOR,
        makeItem(ctx, "cut", "menu_cut", { shortcut: NOTEBOOK_MENU_SHORTCUTS.cut }),
        makeItem(ctx, "copy", "menu_copy", { shortcut: NOTEBOOK_MENU_SHORTCUTS.copy }),
        makeItem(ctx, "paste", "menu_paste", { shortcut: NOTEBOOK_MENU_SHORTCUTS.paste }),
        makeItem(ctx, "selectAll", "menu_select_all", {
            shortcut: NOTEBOOK_MENU_SHORTCUTS.selectAll,
        }),
        SEPARATOR,
        makeItem(ctx, "find", "notebook_ctx_find", { shortcut: NOTEBOOK_MENU_SHORTCUTS.find }),
    ]);
}

function buildEditorMenu(ctx: NotebookMenuContext): AppMenuItemSpec[] {
    const { flags, target } = ctx.context;
    const t = ctx.translate;
    const editable = flags.editable;
    const compactScope = ctx.scope === "compact";
    /** 空选区时剪切/复制不可用，并说明原因（而不是静默无反应）。 */
    const selectionReason = flags.selectionEmpty ? t("notebook_ctx_need_selection") : undefined;
    const inList = target.kind === "list";

    const entries: MenuEntry[] = [...targetGroup(ctx), SEPARATOR];

    // ── 撤销 ──────────────────────────────────────────────────────────────
    if (editable) {
        entries.push(
            makeItem(ctx, "undo", "menu_undo", {
                shortcut: NOTEBOOK_MENU_SHORTCUTS.undo,
                disabledReason: flags.canUndo ? undefined : t("notebook_ctx_nothing_to_undo"),
            }),
            makeItem(ctx, "redo", "menu_redo", {
                shortcut: NOTEBOOK_MENU_SHORTCUTS.redo,
                disabledReason: flags.canRedo ? undefined : t("notebook_ctx_nothing_to_redo"),
            }),
            SEPARATOR,
        );
    }

    // ── 剪贴板 ────────────────────────────────────────────────────────────
    if (editable) {
        entries.push(
            makeItem(ctx, "cut", "menu_cut", {
                shortcut: NOTEBOOK_MENU_SHORTCUTS.cut,
                disabledReason: selectionReason,
            }),
        );
    }
    entries.push(
        makeItem(ctx, "copy", "menu_copy", {
            shortcut: NOTEBOOK_MENU_SHORTCUTS.copy,
            disabledReason: selectionReason,
        }),
    );
    // 暂存块自带一条 `notebook_clip_copy_markdown`（同为 "Copy as Markdown"）。
    // 同一张菜单里两条一模一样的标签，用户分不清哪条作用于选区、哪条作用于
    // 整个块 —— 所以落在暂存块上时不发通用那一条。
    if (target.kind !== "clip") {
        entries.push(
            makeItem(ctx, "copyMarkdown", "notebook_ctx_copy_markdown", {
                disabledReason: selectionReason,
            }),
            makeItem(ctx, "copyPlain", "notebook_ctx_copy_plain", {
                disabledReason: selectionReason,
            }),
        );
    }

    if (editable) {
        entries.push(
            SEPARATOR,
            makeItem(ctx, "paste", "menu_paste", { shortcut: NOTEBOOK_MENU_SHORTCUTS.paste }),
        );
        // 三档"粘贴为…"是**单次覆盖**：它们顶掉设置里的 `plainPasteMode` /
        // `htmlPasteMode` 一次，而不改设置 —— 用户想这一次别解析 Markdown，
        // 不该顺手改掉他所有的粘贴行为。
        if (!compactScope) {
            entries.push(
                makeItem(ctx, "pastePlain", "notebook_ctx_paste_plain"),
                makeItem(ctx, "pasteMarkdown", "notebook_ctx_paste_markdown"),
                makeItem(ctx, "pasteImage", "notebook_ctx_paste_image"),
                makeItem(ctx, "stageClipboard", "notebook_toolbar_stage_clipboard"),
            );
        }
    }

    entries.push(
        SEPARATOR,
        makeItem(ctx, "selectAll", "menu_select_all", {
            shortcut: NOTEBOOK_MENU_SHORTCUTS.selectAll,
        }),
    );

    // ── 格式 / 段落 / 插入（仅富文本可编辑 + full 档） ──────────────────────
    if (editable && !compactScope) {
        const formatItems = [
            makeItem(ctx, "bold", "notebook_ctx_bold", {
                shortcut: NOTEBOOK_MENU_SHORTCUTS.bold,
                checked: flags.isBold,
            }),
            makeItem(ctx, "italic", "notebook_ctx_italic", {
                shortcut: NOTEBOOK_MENU_SHORTCUTS.italic,
                checked: flags.isItalic,
            }),
            makeItem(ctx, "strike", "notebook_ctx_strike", {
                shortcut: NOTEBOOK_MENU_SHORTCUTS.strike,
                checked: flags.isStrike,
            }),
            makeItem(ctx, "code", "notebook_ctx_inline_code", {
                shortcut: NOTEBOOK_MENU_SHORTCUTS.code,
                checked: flags.isCode,
            }),
            SEPARATOR,
            makeItem(ctx, "clearFormatting", "notebook_ctx_clear_formatting", {
                disabledReason: flags.hasMarks ? undefined : t("notebook_ctx_no_formatting"),
            }),
        ];
        pushGroup(entries, "__group-format", t("notebook_ctx_group_format"), formatItems);

        // 缩进 / 凸排只在**不在列表落点**时进这一组：列表落点已经在最上面
        // 给过它们了，同一张菜单里出现两条同名项既冗余、又会让 React 的
        // `key` 撞车（两条都叫 `indent`）。
        const blockItems = [
            makeItem(ctx, "heading1", "notebook_toolbar_heading1", {
                checked: flags.headingLevel === 1,
            }),
            makeItem(ctx, "heading2", "notebook_toolbar_heading2", {
                checked: flags.headingLevel === 2,
            }),
            makeItem(ctx, "heading3", "notebook_toolbar_heading3", {
                checked: flags.headingLevel === 3,
            }),
            makeItem(ctx, "paragraph", "notebook_toolbar_paragraph", {
                checked: flags.headingLevel === null,
            }),
            ...(inList
                ? []
                : [
                      SEPARATOR,
                      makeItem(ctx, "indent", "notebook_ctx_indent", {
                          disabledReason: flags.canIndent
                              ? undefined
                              : t("notebook_ctx_cannot_indent"),
                      }),
                      makeItem(ctx, "outdent", "notebook_ctx_outdent"),
                  ]),
        ];
        pushGroup(entries, "__group-block", t("notebook_ctx_group_block"), blockItems);

        pushGroup(entries, "__group-insert", t("notebook_ctx_group_insert"), [
            makeItem(ctx, "insertImage", "notebook_toolbar_image"),
            makeItem(ctx, "insertTable", "notebook_toolbar_table"),
            makeItem(ctx, "insertRule", "notebook_toolbar_rule"),
            SEPARATOR,
            makeItem(ctx, "insertTimecode", "notebook_toolbar_timecode"),
            makeItem(ctx, "insertClipReference", "notebook_toolbar_clip_ref"),
            makeItem(ctx, "insertProjectInfo", "notebook_toolbar_project_info"),
        ]);
    }

    entries.push(
        SEPARATOR,
        makeItem(ctx, "find", "notebook_ctx_find", { shortcut: NOTEBOOK_MENU_SHORTCUTS.find }),
    );

    return assemble(entries);
}

/** 追加一个分组；整组都是空项时连标题一起省掉（不留一个孤零零的小标题）。 */
function pushGroup(
    entries: MenuEntry[],
    headingKey: string,
    headingLabel: string,
    items: MenuEntry[],
): void {
    const resolved = assemble(items);
    if (resolved.length === 0) return;
    entries.push(SEPARATOR, headingItem(headingKey, headingLabel), ...items);
}

/**
 * 落点专属项。
 *
 * 图片与暂存块的动作由各自 NodeView 提供 —— 这里只决定"发出哪一组、什么顺序"。
 */
function targetGroup(ctx: NotebookMenuContext): MenuEntry[] {
    const { target, flags } = ctx.context;
    const t = ctx.translate;
    const editable = flags.editable;
    const cannotIndent = flags.canIndent ? undefined : t("notebook_ctx_cannot_indent");

    switch (target.kind) {
        case "link":
            return [
                makeItem(ctx, "linkOpen", "notebook_ctx_open_link"),
                makeItem(ctx, "linkCopy", "notebook_ctx_copy_link"),
                ...(editable
                    ? [
                          SEPARATOR,
                          makeItem(ctx, "linkEdit", "notebook_ctx_edit_link"),
                          makeItem(ctx, "linkRemove", "notebook_ctx_remove_link"),
                      ]
                    : []),
            ];

        case "table":
            // 只读预览里表格不可编辑，整组省掉（不是置灰 —— 置灰六项会让人
            // 以为是暂时的）。
            if (!editable) return [];
            return [
                makeItem(ctx, "tableRowAbove", "notebook_ctx_table_row_above"),
                makeItem(ctx, "tableRowBelow", "notebook_ctx_table_row_below"),
                SEPARATOR,
                makeItem(ctx, "tableColLeft", "notebook_ctx_table_col_left"),
                makeItem(ctx, "tableColRight", "notebook_ctx_table_col_right"),
                SEPARATOR,
                makeItem(ctx, "tableToggleHeader", "notebook_ctx_table_header_row", {
                    checked: target.inHeaderRow,
                }),
                SEPARATOR,
                makeItem(ctx, "tableDeleteRow", "notebook_ctx_table_delete_row"),
                makeItem(ctx, "tableDeleteCol", "notebook_ctx_table_delete_col"),
                makeItem(ctx, "tableDelete", "notebook_ctx_table_delete", { danger: true }),
            ];

        case "list":
            if (!editable) return [];
            return [
                makeItem(ctx, "indent", "notebook_ctx_indent", { disabledReason: cannotIndent }),
                makeItem(ctx, "outdent", "notebook_ctx_outdent"),
            ];

        case "image":
            return [
                makeItem(ctx, "imageCopy", "notebook_image_copy"),
                makeItem(ctx, "imageSaveAs", "notebook_image_save_as"),
                ...(editable
                    ? [
                          SEPARATOR,
                          makeItem(ctx, "imageWidthReset", "notebook_image_width_reset"),
                          makeItem(ctx, "imageEditAlt", "notebook_image_edit_alt"),
                          SEPARATOR,
                          makeItem(ctx, "imageRemove", "notebook_image_remove", { danger: true }),
                      ]
                    : []),
            ];

        case "clip":
            // 与既有的 `⋯` 菜单行为一致：只读预览里不给暂存块动作。
            if (!editable) return [];
            return [
                makeItem(ctx, "clipRestore", "notebook_clip_restore"),
                // 参数线载荷没有时间轴几何："插入到时间轴"对它是一句错话，
                // 它恢复后走的是参数编辑器通道。
                makeItem(
                    ctx,
                    "clipInsert",
                    target.param ? "notebook_clip_apply_to_param" : "notebook_clip_insert_timeline",
                ),
                makeItem(ctx, "clipInsertNewTracks", "notebook_clip_insert_new_tracks"),
                SEPARATOR,
                makeItem(ctx, "clipRename", "notebook_clip_rename"),
                makeItem(ctx, "clipCopyMarkdown", "notebook_clip_copy_markdown"),
                makeItem(ctx, "clipSavePayload", "notebook_clip_save_payload"),
                SEPARATOR,
                makeItem(ctx, "clipRemove", "notebook_clip_remove", { danger: true }),
            ];

        case "text":
        case "empty":
            return [];
    }
}
