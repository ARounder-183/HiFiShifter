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

import { handleNotebookPaste, installClipboardFlavorWriter } from "./notebookClipboard.ts";
import type { PasteContext } from "./notebookClipboard.ts";

function assertEqual<T>(actual: T, expected: T, label: string): void {
    const a = JSON.stringify(actual);
    const b = JSON.stringify(expected);
    if (a !== b) throw new Error(`${label}: expected ${b}, received ${a}`);
}

/** 带可控 target/clipboardData 的合成事件（jsdom 的 ClipboardEvent 拿不到 data）。 */
function fakeClipboardEvent(type: string, target: EventTarget, clipboardData: unknown): ClipboardEvent {
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
});
