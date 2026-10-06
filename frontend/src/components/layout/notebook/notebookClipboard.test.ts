// @vitest-environment jsdom
/*
 * 剪贴板接管的边界测试。
 *
 * 【要钉死什么】面板容器的捕获阶段 paste 监听与编辑器 DOM 上的 copy/cut
 * flavor 写出器，都不得截走 **input / textarea** 里的原生复制粘贴 —— 查找条、
 * 图片 alt 编辑器、暂存块改名、链接浮层都活在编辑器/面板 DOM 内部，历史上一
 * 度把粘贴内容插进了正文。
 *
 * 需要 jsdom：要真实的 DOM 层级（input 挂在容器里）来验证 closest() 命中。
 */

import type { Editor } from "@tiptap/core";
import { test } from "vitest";

import {
    escapeMarkdownText,
    handleNotebookPaste,
    installClipboardFlavorWriter,
} from "./notebookClipboard.ts";
import type { PasteContext } from "./notebookClipboard.ts";

function assertEqual<T>(actual: T, expected: T, label: string): void {
    const a = JSON.stringify(actual);
    const b = JSON.stringify(expected);
    if (a !== b) throw new Error(`${label}: expected ${b}, received ${a}`);
}

/** 带可控 target/clipboardData 的合成事件（jsdom 的 ClipboardEvent 拿不到 data）。 */
function fakeClipboardEvent(
    type: string,
    target: EventTarget,
    clipboardData: unknown,
): ClipboardEvent {
    const event = new Event(type, { bubbles: true });
    Object.defineProperty(event, "target", { value: target });
    Object.defineProperty(event, "clipboardData", { value: clipboardData });
    return event as unknown as ClipboardEvent;
}

/** 只够 `computeClipboardFlavors` 走"空选区"路径的最小 editor 形状。 */
function fakeEmptySelectionEditor(): Editor {
    return {
        state: { selection: { from: 0, to: 0, empty: true } },
    } as unknown as Editor;
}

test("components/layout/notebook/notebookClipboard.test.ts scripted checks", () => {
    // ── paste 分流：输入框目标必须放行（返回 false = 不接管）────────
    const input = document.createElement("input");
    const ctx = { sourceMode: false } as unknown as PasteContext;
    assertEqual(
        handleNotebookPaste(ctx, fakeClipboardEvent("paste", input, null)),
        false,
        "paste into an input is not intercepted",
    );

    // ── copy flavor 写出器 ────────────────────────────────────────
    const container = document.createElement("div");
    const inner = document.createElement("input");
    container.appendChild(inner);
    document.body.appendChild(container);

    const written: string[] = [];
    const clipboardData = {
        setData: (flavor: string) => {
            written.push(flavor);
        },
    };
    const editor = fakeEmptySelectionEditor();

    const remove = installClipboardFlavorWriter(
        container,
        () => ({ copyFormat: "markdown", copyPlainTextAs: "markdown" }),
        () => editor,
    );

    // 输入框里的复制：不重写任何 flavor，原生气味留给浏览器。
    inner.dispatchEvent(fakeClipboardEvent("copy", inner, clipboardData));
    assertEqual(written, [], "copy inside an input keeps native flavors");

    // 编辑器表面（容器自身）的复制：仍要裁剪/补写 flavor。
    container.dispatchEvent(fakeClipboardEvent("copy", container, clipboardData));
    assertEqual(written, ["text/html", "text/plain"], "copy on the editor surface is rewritten");

    remove();
    container.remove();

    // ── 纯文本粘贴的转义：行首标记不能被解释成块级语法 ─────────────
    // 有序列表标记要转义分隔符（`.` / `)`）而不是数字：`\1. ` 会留下可见的
    // 反斜杠，只有 `1\. ` 才既压住列表语法又保留原文本。
    assertEqual(
        escapeMarkdownText("1. 甲\n2) 乙\n- 丙\n# 丁\n> 戊"),
        "1\\. 甲\n2\\) 乙\n\\- 丙\n\\# 丁\n\\> 戊",
        "line-leading markdown markers are escaped",
    );
    // 普通文本不受影响，且缩进保持不变。
    assertEqual(escapeMarkdownText("普通文字"), "普通文字", "plain text untouched");
    assertEqual(escapeMarkdownText("  1. 缩进"), "  1\\. 缩进", "indented ordered marker escaped");
});

/*
 * 剪切：flavor 必须取自**选区被删之前**。
 *
 * 【要钉死什么】ProseMirror 的 `cut` 处理器在**同一次事件里**就把选区删掉了
 * （它 dispatch 一条删除事务），而 flavor 写出器跑在冒泡阶段 —— 那时
 * `state.selection` 已经塌缩，`selectionMarkdown` 只能返回空串，于是写出去的
 * `text/plain` 是**空串**（浏览器会把它整个丢掉，剪贴板里只剩下 `text/html`）。
 *
 * 后果是一条混搭路径静默失效：`Ctrl+V` 还能从 html 还原，而**右键菜单的粘贴只能
 * `navigator.clipboard.readText()`** —— 读到空串，"用快捷键剪切、用菜单粘贴"什么
 * 都粘不出来。浏览器实测：修复前剪贴板类型只有 `["text/html"]`，修复后是
 * `["text/plain","text/html"]`。
 *
 * 【怎么在 jsdom 里复现这个时序】真正删选区的是 ProseMirror，这里用一个注册得
 * **更早**的冒泡监听代替它（同一元素上，先注册的先跑）：于是顺序与浏览器一致 ——
 * 写出器的捕获监听 → 删选区 → 写出器的冒泡监听。
 */
test("剪切时 text/plain 取自选区被删之前（否则为空串）", async () => {
    const { Editor } = await import("@tiptap/core");
    const { buildNotebookExtensions } = await import("./notebookExtensions.ts");

    const editor = new Editor({
        element: document.createElement("div"),
        extensions: buildNotebookExtensions({ markdownShortcuts: true, slashCommands: false }),
        content: "一段文字",
    });
    const container = editor.view.dom;
    document.body.append(container);
    // 模拟 ProseMirror：在冒泡阶段删掉选区（比写出器的冒泡监听更早注册）。
    container.addEventListener("cut", () => editor.commands.deleteSelection());

    const written: Record<string, string> = {};
    const remove = installClipboardFlavorWriter(
        container,
        () => ({ copyFormat: "markdown+html", copyPlainTextAs: "markdown" }),
        () => editor,
    );

    editor.commands.setTextSelection({ from: 1, to: 3 }); // 选中前两个字
    const target = container.querySelector("p") ?? container;
    // 派发在**后代**上：容器的捕获监听先于它的冒泡监听，与浏览器一致。
    target.dispatchEvent(
        fakeClipboardEvent("cut", target, {
            setData: (type: string, value: string) => {
                written[type] = value;
            },
        }),
    );

    assertEqual(written["text/plain"], "一段", "剪切写出的 text/plain 必须是 Markdown，而不是空串");

    remove();
    editor.destroy();
    container.remove();
});
