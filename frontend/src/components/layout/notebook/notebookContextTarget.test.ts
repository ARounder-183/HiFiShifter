// @vitest-environment jsdom
/*
 * 右键落点解析与开关快照的回归测试（真实 TipTap 编辑器）。
 *
 * 【要钉死什么】
 *   1. **选区语义**：右键落在选区内保持选区、落在选区外移动光标。这条规则决定了
 *      "删除这一段"和"删除我选中的五段"的区别，写反了就是静默操作错对象。
 *   2. **落点优先级**：链接 > 表格 > 列表。链接可以出现在单元格里，此时用户想
 *      操作的是链接而不是表格。
 *   3. **内部链接标记**：`hifi://` 链接要能被认出来，菜单才知道"打开"应该是
 *      跳播放头而不是开浏览器。
 *
 * 【为什么要 stub `posAtCoords`】jsdom 没有布局引擎，ProseMirror 的坐标→位置
 * 换算拿不到任何几何信息（返回 null）。落点策略是本文件的被测对象，几何本身不是
 * —— 因此把"命中哪个位置"直接喂进去，只验证由它衍生的行为。
 */

import { Editor } from "@tiptap/core";
import { test } from "vitest";

import { buildNotebookExtensions } from "./notebookExtensions.ts";
import {
    applyContextSelection,
    prepareNotebookContext,
    prepareNotebookContextAtCaret,
    readNotebookMenuFlags,
    resolveContextTarget,
} from "./notebookContextTarget.ts";

function assertEqual<T>(actual: T, expected: T, label: string): void {
    const a = JSON.stringify(actual);
    const b = JSON.stringify(expected);
    if (a !== b) throw new Error(`${label}: expected ${b}, received ${a}`);
}

function makeEditor(content: string): Editor {
    return new Editor({
        element: document.createElement("div"),
        extensions: buildNotebookExtensions({ markdownShortcuts: true, slashCommands: false }),
        // `content` 是 **Markdown**（见 `notebookLinkEdit.test.ts` 的同名说明）。
        content,
    });
}

/** 把光标放到文档末尾（不选中任何文本）。 */
function collapseToEnd(editor: Editor): void {
    editor.commands.setTextSelection(editor.state.doc.content.size - 1);
}

/** 把光标放到第 `index` 个匹配节点的内部（偏移 +1）。 */
function collapseInto(editor: Editor, predicate: (name: string) => boolean, index = 0): void {
    let seen = 0;
    let target: number | null = null;
    editor.state.doc.descendants((node, pos) => {
        if (target !== null) return false;
        if (!predicate(node.type.name)) return true;
        if (seen === index) {
            target = pos + 1;
            return false;
        }
        seen += 1;
        return true;
    });
    if (target === null) throw new Error("document has no matching node");
    editor.commands.setTextSelection(target);
}

/** 让下一次坐标查询命中指定位置。 */
function stubHit(editor: Editor, pos: number): void {
    (editor.view as unknown as { posAtCoords: () => { pos: number; inside: number } }).posAtCoords =
        () => ({ pos, inside: -1 });
}

test("components/layout/notebook/notebookContextTarget.test.ts scripted checks", () => {
    // ── 落点判定 ────────────────────────────────────────────────────────
    {
        const editor = makeEditor("一段普通文字");
        collapseToEnd(editor);
        assertEqual(resolveContextTarget(editor), { kind: "empty" }, "caret in text → empty");
        editor.commands.setTextSelection({ from: 1, to: 3 });
        assertEqual(resolveContextTarget(editor), { kind: "text" }, "selection → text");
        editor.destroy();
    }

    // ── 链接：外部 / 内部两种 ───────────────────────────────────────────
    {
        const editor = makeEditor("[外站](https://example.com/a)");
        collapseInto(editor, (name) => name === "text");
        assertEqual(
            resolveContextTarget(editor),
            { kind: "link", href: "https://example.com/a", internal: false },
            "external link",
        );
        editor.destroy();
    }
    {
        const editor = makeEditor("[跳](hifi://seek/12.5)");
        collapseInto(editor, (name) => name === "text");
        assertEqual(
            resolveContextTarget(editor),
            { kind: "link", href: "hifi://seek/12.5", internal: true },
            "internal link is flagged",
        );
        editor.destroy();
    }

    // ── 表格：正文单元格与标题行要能区分 ────────────────────────────────
    // 光标落在单元格的**文本**上（而不是直接落在 tableCell 上）：后者不是合法的
    // 文本选区端点，ProseMirror 会告警。
    {
        const editor = makeEditor("| a | b |\n| --- | --- |\n| c | d |");
        collapseInto(editor, (name) => name === "text", 0);
        assertEqual(
            resolveContextTarget(editor),
            { kind: "table", inHeaderRow: true },
            "header cell",
        );
        collapseInto(editor, (name) => name === "text", 2);
        assertEqual(
            resolveContextTarget(editor),
            { kind: "table", inHeaderRow: false },
            "body cell",
        );
        editor.destroy();
    }

    // ── 列表：普通列表项与任务项 ────────────────────────────────────────
    {
        const editor = makeEditor("- 一\n- 二");
        collapseInto(editor, (name) => name === "listItem");
        assertEqual(
            resolveContextTarget(editor),
            { kind: "list", itemType: "listItem" },
            "plain list item",
        );
        editor.destroy();
    }
    {
        const editor = makeEditor("- [ ] 待办");
        collapseInto(editor, (name) => name === "taskItem");
        assertEqual(
            resolveContextTarget(editor),
            { kind: "list", itemType: "taskItem" },
            "task item",
        );
        editor.destroy();
    }

    // ── 优先级：单元格里的链接认链接，不认表格 ──────────────────────────
    {
        const editor = makeEditor("| [站](https://example.com) |\n| --- |");
        collapseInto(editor, (name) => name === "text");
        const target = resolveContextTarget(editor);
        assertEqual(target.kind, "link", "link inside a table cell wins over table");
        editor.destroy();
    }

    // ── 选区语义：落在选区内保持，落在选区外移动 ────────────────────────
    {
        const editor = makeEditor("一二三四五");
        editor.commands.setTextSelection({ from: 2, to: 4 });
        stubHit(editor, 3);
        applyContextSelection(editor, 0, 0);
        assertEqual(
            { from: editor.state.selection.from, to: editor.state.selection.to },
            { from: 2, to: 4 },
            "a hit inside the selection keeps it",
        );

        stubHit(editor, 1);
        applyContextSelection(editor, 0, 0);
        assertEqual(
            {
                from: editor.state.selection.from,
                to: editor.state.selection.to,
                empty: editor.state.selection.empty,
            },
            { from: 1, to: 1, empty: true },
            "a hit outside the selection collapses the caret there",
        );
        editor.destroy();
    }

    // ── 选区语义：落在选区首字符上也保持（最常见的"复制这一段"手势） ────
    {
        const editor = makeEditor("一二三四五");
        editor.commands.setTextSelection({ from: 2, to: 4 });
        stubHit(editor, 2);
        applyContextSelection(editor, 0, 0);
        assertEqual(
            { from: editor.state.selection.from, to: editor.state.selection.to },
            { from: 2, to: 4 },
            "a hit exactly on the selection start keeps it",
        );
        editor.destroy();
    }

    // ── 取不到坐标时不动选区（而不是把光标丢到 0） ──────────────────────
    {
        const editor = makeEditor("一二三四五");
        editor.commands.setTextSelection({ from: 2, to: 4 });
        (editor.view as unknown as { posAtCoords: () => null }).posAtCoords = () => null;
        applyContextSelection(editor, 0, 0);
        assertEqual(
            { from: editor.state.selection.from, to: editor.state.selection.to },
            { from: 2, to: 4 },
            "no hit → selection untouched",
        );
        editor.destroy();
    }

    // ── 开关快照 ────────────────────────────────────────────────────────
    {
        const editor = makeEditor("# 标题");
        collapseToEnd(editor);
        assertEqual(readNotebookMenuFlags(editor).headingLevel, 1, "heading level");

        editor.commands.setTextSelection({ from: 1, to: 3 });
        editor.commands.setBold();
        const bold = readNotebookMenuFlags(editor);
        assertEqual(bold.isBold, true, "bold active");
        assertEqual(bold.hasMarks, true, "selection carries a mark");
        assertEqual(bold.selectionEmpty, false, "selection not empty");
        assertEqual(bold.editable, true, "editor is editable");

        // 只选纯文本：没有可清除的格式。
        editor.destroy();
    }
    {
        const editor = makeEditor("普通文字");
        editor.commands.setTextSelection({ from: 1, to: 3 });
        assertEqual(readNotebookMenuFlags(editor).hasMarks, false, "plain text has no marks");
        editor.destroy();
    }

    // ── 缩进能力：列表首项上方没有同级项，因此缩不进去 ──────────────────
    {
        const editor = makeEditor("- 一\n- 二");
        collapseInto(editor, (name) => name === "listItem", 0);
        assertEqual(readNotebookMenuFlags(editor).canIndent, false, "the first item cannot nest");
        collapseInto(editor, (name) => name === "listItem", 1);
        assertEqual(readNotebookMenuFlags(editor).canIndent, true, "a following item can nest");
        editor.destroy();
    }

    // ── prepareNotebookContext：一次拿到目标与开关，且先改选区后读开关 ────
    {
        const editor = makeEditor("一二三四五");
        editor.commands.setTextSelection({ from: 2, to: 4 });
        stubHit(editor, 5);
        const prepared = prepareNotebookContext(editor, 0, 0);
        assertEqual(prepared.target, { kind: "empty" }, "caret moved out of the selection");
        assertEqual(prepared.flags.selectionEmpty, true, "flags reflect the post-move state");
        editor.destroy();
    }

    // ── 键盘入口：落点取自光标本身，不查坐标 ────────────────────────────
    //
    // 【要钉死什么】键盘路径若拿 `coordsAtPos` 再喂回 `posAtCoords`，就把已知答案
    // 绕了一圈几何换算；而 `posAtCoords` 给的是"离该坐标最近的位置"，在块边界上
    // 可能落到相邻块里。因此这里断言：**无论坐标反查会给出什么，键盘路径都不看它**。
    {
        const editor = makeEditor("一段文字\n\n| a | b |\n| --- | --- |\n| c | d |");
        // 先取一个"落在表格里"的位置，供下面的桩使用。
        collapseInto(editor, (name) => name === "text", 0);
        const tablePos = editor.state.selection.from;
        // 光标放回第一段（普通段落）。
        editor.commands.setTextSelection(1);
        // 让坐标反查无论如何都命中表格 —— 它不该被采纳。
        stubHit(editor, tablePos);
        const prepared = prepareNotebookContextAtCaret(editor, { x: 10, y: 20 });
        assertEqual(
            prepared.context.target.kind,
            "empty",
            "caret-based resolution ignores the coordinate round-trip",
        );
        assertEqual(prepared.x, 10, "anchor x passes through");
        assertEqual(prepared.y, 20, "anchor y passes through");
        editor.destroy();
    }
    {
        // 反过来：光标确实在表格里时，键盘入口也要如实给出表格。
        const editor = makeEditor("| a | b |\n| --- | --- |\n| c | d |");
        collapseInto(editor, (name) => name === "text", 0);
        const prepared = prepareNotebookContextAtCaret(editor, { x: 0, y: 0 });
        assertEqual(
            prepared.context.target,
            { kind: "table", inHeaderRow: true },
            "caret inside a header cell still reports the table",
        );
        editor.destroy();
    }
});
