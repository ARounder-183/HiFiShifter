/*
 * 右键落点的解析与菜单开关（flags）的快照。
 *
 * ## 为什么与菜单内容分开
 *
 * `notebookContextMenu.ts` 是纯函数：给它一个目标与一组开关，返回菜单项数组。
 * 但"目标是什么"必须问 ProseMirror 要（几何 + schema），"开关是什么"必须问
 * 编辑器状态要 —— 这两件事都依赖真实编辑器，塞进纯函数就没法单测了。
 *
 * 因此本文件是**唯一**接触编辑器实例的一层：它把编辑器状态翻译成两个可序列化
 * 的数据结构，交给纯函数去决定菜单长什么样。测试分成两边：落点解析用 jsdom +
 * 真编辑器测（本文件），菜单内容用纯数据测（builder 文件）。
 *
 * ## 顺序不能颠倒
 *
 * `applyContextSelection` 是**唯一会改编辑器状态**的步骤（把光标移到落点），
 * 而 flags 必须在它之后读 —— 否则读到的是"上一次的选区"，菜单里就会出现
 * "选中了三段却显示未选中"这类错位。这个顺序约束被编码进 `prepareNotebookContext`
 * 一个函数里，调用方无法弄错。
 */

import type { Editor } from "@tiptap/core";

import { parseInternalLink } from "./timecode";

/**
 * 右键落在什么上。
 *
 * `image` / `clip` 不带载荷：它们的菜单项由各自的 NodeView 提供（那里才知道
 * 附件 id、载荷是否还在），builder 只需要知道"该发哪一组"。把 ProseMirror
 * 节点对象塞进 target 会让 builder 依赖 schema，得不偿失。
 *
 * 唯一的例外是 `clip.param`：它不是"节点数据"，而是**决定文案**的一个事实 ——
 * 参数线载荷没有时间轴几何，"插入到时间轴"对它是一句错话（它恢复后走参数编辑器
 * 通道）。文案属于菜单内容，因此由 builder 据此选键，而不是让 NodeView 自己
 * 造一个 label 传进来。
 */
export type NotebookMenuTarget =
    | { kind: "text" }
    | { kind: "empty" }
    | { kind: "link"; href: string; internal: boolean }
    | { kind: "table"; inHeaderRow: boolean }
    | { kind: "list"; itemType: "listItem" | "taskItem" }
    | { kind: "image" }
    | { kind: "clip"; param: boolean };

/** 菜单项可用性快照。全部在打开菜单那一刻读一次。 */
export interface NotebookMenuFlags {
    /** 选区为空（决定剪切/复制是否可用）。 */
    selectionEmpty: boolean;
    /** 编辑器可编辑（分栏只读预览为 false）。 */
    editable: boolean;
    canUndo: boolean;
    canRedo: boolean;
    isBold: boolean;
    isItalic: boolean;
    isStrike: boolean;
    isCode: boolean;
    /** 选区/光标处是否带有任何可清除的标记（决定"清除格式"是否可用）。 */
    hasMarks: boolean;
    /** 当前标题级别；不在标题内为 null。 */
    headingLevel: number | null;
    /**
     * 当前列表项能否再缩进一层。
     *
     * 【为什么没有对称的 `canOutdent`】`liftListItem` 在**最外层也返回 true** ——
     * 它把列表项整个提出列表、变成一个普通段落，这正是 Word「减少缩进」在最外层
     * 的行为。既然它永远可用，一个恒为真的开关就只会带来一个永远显示不出来的
     * tooltip（"已在最外层"），所以不留。
     */
    canIndent: boolean;
}

/** 打开菜单所需的一切（目标 + 开关）。 */
export interface NotebookContext {
    target: NotebookMenuTarget;
    flags: NotebookMenuFlags;
}

/**
 * 解析右键落点：先把光标放到该去的地方，再读目标与开关。
 *
 * 选区语义（Explorer / Word 惯例）：
 * - 落点**在选区内**（含边界）→ 保持选区，菜单作用于整个选区；
 * - 落点在选区**外** → 光标移到落点，菜单作用于该处。
 *
 * 不做"落在词上就选中该词"：那是聪明的猜测，猜错就是静默改了用户的选区。
 */
export function prepareNotebookContext(editor: Editor, x: number, y: number): NotebookContext {
    applyContextSelection(editor, x, y);
    return { target: resolveContextTarget(editor), flags: readNotebookMenuFlags(editor) };
}

/**
 * 键盘入口（`ContextMenu` 键 / `Shift+F10`）：**直接用光标位置**，不做几何换算。
 *
 * 【为什么不复用 `prepareNotebookContext`】那条路径是"坐标 → 位置"，为鼠标而设。
 * 键盘没有坐标；拿 `coordsAtPos(selection.from)` 再喂回 `posAtCoords` 只是把
 * 已知答案换算成坐标再换算回来，而 `posAtCoords` 返回的是"离该坐标最近的位置"，
 * 在块与块的边界上（某行的上/下边缘）可能落到**相邻块**里。光标位置本来就是我
 * 们唯一需要的答案，直接读既更短也更确定。
 *
 * @param anchor 菜单锚点（视口坐标）。调用方用 `coordsAtPos` 算，它只影响菜单
 *   弹出的位置，不参与落点判定。
 */
export function prepareNotebookContextAtCaret(
    editor: Editor,
    anchor: { x: number; y: number },
): { context: NotebookContext; x: number; y: number } {
    return {
        context: { target: resolveContextTarget(editor), flags: readNotebookMenuFlags(editor) },
        x: anchor.x,
        y: anchor.y,
    };
}

/**
 * 把光标移到落点 —— **仅当落点不在当前选区内**。
 *
 * 这是本模块唯一会写编辑器状态的函数。`posAtCoords` 返回 null 时（点在
 * 内边距外的空白、或编辑器尚未布局）什么都不做：此时保持既有选区是最不
 * 意外的选择，而"把光标丢到 0"会让菜单里的复制悄悄复制错东西。
 */
export function applyContextSelection(editor: Editor, x: number, y: number): void {
    if (editor.isDestroyed) return;
    const hit = editor.view.posAtCoords({ left: x, top: y });
    if (!hit) return;
    const pos = hit.pos;
    const { from, to, empty } = editor.state.selection;
    // 边界含在内：右键落在选区首字符上是最常见的"我就想复制这一段"手势，
    // 用开区间会把选区收掉，用户随即看到"复制"变灰。
    if (!empty && pos >= from && pos <= to) return;
    editor.commands.setTextSelection(pos);
}

/**
 * 目标判定。
 *
 * 顺序即优先级：链接 > 表格 > 列表 > 空选区 > 文本。链接排最前是因为它的
 * 菜单组最具体（打开 / 复制地址 / 移除），而"光标在一个链接里"同时也会满足
 * "在表格里"（链接可以出现在单元格中）—— 此时用户想操作的是链接。
 */
export function resolveContextTarget(editor: Editor): NotebookMenuTarget {
    if (editor.isDestroyed) return { kind: "empty" };

    if (editor.isActive("link")) {
        const href = String(editor.getAttributes("link").href ?? "");
        return { kind: "link", href, internal: parseInternalLink(href) !== null };
    }
    if (editor.isActive("table")) {
        return { kind: "table", inHeaderRow: editor.isActive("tableHeader") };
    }
    if (editor.isActive("taskItem")) {
        return { kind: "list", itemType: "taskItem" };
    }
    if (editor.isActive("listItem")) {
        return { kind: "list", itemType: "listItem" };
    }
    return editor.state.selection.empty ? { kind: "empty" } : { kind: "text" };
}

/**
 * 读一次可用性快照。
 *
 * 【为什么快照而不是实时订阅】菜单是弹出表面，外部 pointerdown 即关闭 ——
 * 它的生命周期内编辑器不会再变（除非用户点了菜单项，而那时菜单已经关了）。
 * 为它挂一个 `editor.on("transaction")` 订阅只会带来无谓的渲染。
 */
export function readNotebookMenuFlags(editor: Editor): NotebookMenuFlags {
    if (editor.isDestroyed) {
        return {
            selectionEmpty: true,
            editable: false,
            canUndo: false,
            canRedo: false,
            isBold: false,
            isItalic: false,
            isStrike: false,
            isCode: false,
            hasMarks: false,
            headingLevel: null,
            canIndent: false,
        };
    }
    const itemType = editor.isActive("taskItem") ? "taskItem" : "listItem";
    const heading = editor.getAttributes("heading");
    const level = typeof heading.level === "number" ? heading.level : null;
    return {
        selectionEmpty: editor.state.selection.empty,
        editable: editor.isEditable,
        // `can()` 在命令不可用时返回 false（历史为空、或已到栈顶），比读
        // `editor.commands.undo()` 的返回值更安全：后者会真的执行一次撤销。
        canUndo: editor.can().undo(),
        canRedo: editor.can().redo(),
        isBold: editor.isActive("bold"),
        isItalic: editor.isActive("italic"),
        isStrike: editor.isActive("strike"),
        isCode: editor.isActive("code"),
        hasMarks: hasAnyMark(editor),
        headingLevel: level,
        canIndent: editor.can().sinkListItem(itemType),
    };
}

/** 选区（或光标处）是否带有任何标记 —— 决定"清除格式"是否可用。 */
function hasAnyMark(editor: Editor): boolean {
    const { state } = editor;
    const { from, to, empty } = state.selection;
    if (empty) return state.storedMarks?.length ? true : state.selection.$from.marks().length > 0;
    let found = false;
    state.doc.nodesBetween(from, to, (node) => {
        if (found) return false;
        if (node.isText && node.marks.length > 0) {
            found = true;
            return false;
        }
        return true;
    });
    return found;
}
