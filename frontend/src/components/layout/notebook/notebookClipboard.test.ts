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
    const event = new Event(type, { bubbles: true, cancelable: true });
    Object.defineProperty(event, "target", { value: target });
    Object.defineProperty(event, "clipboardData", { value: clipboardData });
    return event as unknown as ClipboardEvent;
}

/**
 * `DataTransfer` 的最小替身。
 *
 * 【为什么不能只给 `setData`】ProseMirror 的 `copy`/`cut` 处理器会在同一个事件里跑，
 * 它开头就调 `data.clearData()` —— 少了这个方法，处理器抛 `TypeError`，而那个异常
 * 发生在事件派发里、**不会被本用例的断言捕获**，只会变成 vitest 的 "Unhandled
 * Errors"（测试照样显示 passed，除非你去看 Errors 那一行）。
 */
function fakeDataTransfer(): {
    setData: (type: string, value: string) => void;
    getData: (type: string) => string;
    clearData: () => void;
    readonly types: string[];
    readonly files: File[];
} {
    const store = new Map<string, string>();
    return {
        setData: (type, value) => {
            store.set(type, value);
        },
        getData: (type) => store.get(type) ?? "",
        clearData: () => {
            store.clear();
        },
        get types() {
            return [...store.keys()];
        },
        files: [],
    };
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
 * 【怎么在 jsdom 里复现这个时序】不自己模拟删除 —— ProseMirror 自己的 `copy`/`cut`
 * 处理器就挂在同一个 DOM 上，派发合成事件时它会**真的跑**：先写 `text/html` 与
 * `text/plain`，**然后 dispatch 一条删除事务**。让它真跑，顺序就与浏览器一致
 * （写出器的捕获监听 → PM 写盘并删选区 → 写出器的冒泡监听），也就不存在
 * "模拟行为与真实行为漂移、测试却仍然绿"的风险。
 */
test("剪切时 text/plain 取自选区被删之前（否则为空串）", async () => {
    const { Editor } = await import("@tiptap/core");
    const { buildNotebookExtensions } = await import("./notebookExtensions.ts");

    const editor = new Editor({
        element: document.createElement("div"),
        extensions: buildNotebookExtensions({ markdownShortcuts: true, slashCommands: false }),
        // 正文要能区分 Markdown 与纯文本，否则"我们改写过它"这件事看不出来。
        content: "普通**加粗**文字",
    });
    const container = editor.view.dom;
    document.body.append(container);

    const remove = installClipboardFlavorWriter(
        container,
        () => ({ copyFormat: "markdown+html", copyPlainTextAs: "markdown" }),
        () => editor,
    );

    editor.commands.setTextSelection({ from: 1, to: editor.state.doc.content.size - 1 });

    // 替身必须实现 `clearData` —— PM 的处理器会调它（只实现 setData 会抛）。
    const data = fakeDataTransfer();
    const target = container.querySelector("p") ?? container;
    // 派发在**后代**上：容器的捕获监听先于它的冒泡监听，与浏览器一致。
    target.dispatchEvent(fakeClipboardEvent("cut", target, data));

    assertEqual(
        data.getData("text/plain"),
        "普通**加粗**文字",
        "剪切写出的 text/plain 必须是 Markdown（PM 自己写的是纯文本「普通加粗文字」，被我们改写）",
    );
    assertEqual(editor.state.doc.textContent, "", "ProseMirror 的 cut 处理器应已删掉选区");

    remove();
    editor.destroy();
    container.remove();
});
