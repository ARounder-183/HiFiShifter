// @vitest-environment jsdom
/*
 * WebView 原生选择准入判定的回归测试。
 *
 * 【为什么必须有】`selectstart` 的 target 是**文本节点**（nodeType 3），而早先
 * 的判定把 target 当 Element 用 —— 文本节点没有 `tagName` / `closest()`，
 * 于是"在 contenteditable 里选中文字"被判成不可选中，`selectstart` 被
 * preventDefault、选区被 mouseup 守卫清掉。用户看到的现象是：**在富文本
 * 编辑器里既不能拖选文字，点击文字也不落光标**。
 *
 * 这里锁定"文本节点必须先归一成父元素再判定"，以及各放行条件的边界。
 */

import { test } from "vitest";

import {
    allowsNativeTextSelection,
    elementFromEventTarget,
    isEditableTarget,
} from "./nativeSelectionGuards.ts";

/** 造一个与 index.css 同构的环境：整页 user-select:none，可编辑区放行。 */
function buildDom(): { editor: HTMLElement; paragraph: HTMLElement; plain: HTMLElement } {
    const style = document.createElement("style");
    style.textContent = `
        body * { -webkit-user-select: none; user-select: none; }
        body [contenteditable="true"] { -webkit-user-select: text; user-select: text; }
        body [contenteditable="true"] * { -webkit-user-select: text; user-select: text; }
        body [data-hs-selectable="true"], body [data-hs-selectable="true"] * {
            -webkit-user-select: text; user-select: text;
        }
    `;
    document.head.appendChild(style);

    const editor = document.createElement("div");
    editor.setAttribute("contenteditable", "true");
    editor.setAttribute("data-hs-selectable", "true");
    const paragraph = document.createElement("p");
    paragraph.textContent = "alpha bravo charlie";
    editor.appendChild(paragraph);
    document.body.appendChild(editor);

    const plain = document.createElement("div");
    plain.textContent = "timeline label";
    document.body.appendChild(plain);

    return { editor, paragraph, plain };
}

function firstTextNode(element: Element): Node {
    const node = element.firstChild;
    if (!node) throw new Error("element has no child node");
    return node;
}

test("utils/nativeSelectionGuards.test.ts scripted checks", async () => {
    const { paragraph, plain } = buildDom();

    // ── 文本节点归一成父元素 ─────────────────────────────────────
    const textNode = firstTextNode(paragraph);
    if (textNode.nodeType !== 3) throw new Error("fixture should hold a text node");
    if (elementFromEventTarget(textNode) !== paragraph) {
        throw new Error("text node must resolve to its parent element");
    }
    if (elementFromEventTarget(null) !== null) throw new Error("null target resolves to null");

    // ── 核心回归：contenteditable 内的文本节点必须被认成可编辑 ────
    if (!isEditableTarget(textNode)) {
        throw new Error("text node inside contenteditable must be an editable target");
    }
    if (!allowsNativeTextSelection(textNode)) {
        throw new Error(
            "text node inside contenteditable must allow native selection " +
                "(otherwise selectstart is prevented and drag-select breaks)",
        );
    }
    // 元素形态同样成立（mouseup / pointerdown 的 target 是元素）。
    if (!allowsNativeTextSelection(paragraph)) {
        throw new Error("paragraph inside contenteditable must allow native selection");
    }

    // ── data-hs-selectable 表面：子树一律放行 ─────────────────────
    const card = document.createElement("div");
    card.setAttribute("data-hs-selectable", "true");
    const cardText = document.createElement("span");
    cardText.textContent = "staged clip";
    card.appendChild(cardText);
    document.body.appendChild(card);
    if (!allowsNativeTextSelection(firstTextNode(cardText))) {
        throw new Error("text node inside data-hs-selectable must allow native selection");
    }

    // ── 普通区域（时间轴/面板文字）仍然不可选中 ───────────────────
    const plainText = firstTextNode(plain);
    if (allowsNativeTextSelection(plainText)) {
        throw new Error("text node in a plain user-select:none region must stay unselectable");
    }
    if (allowsNativeTextSelection(plain)) {
        throw new Error("plain element must stay unselectable");
    }
    if (isEditableTarget(plainText)) {
        throw new Error("plain text node must not be an editable target");
    }

    // ── 表单控件仍然放行 ─────────────────────────────────────────
    const textarea = document.createElement("textarea");
    document.body.appendChild(textarea);
    if (!isEditableTarget(textarea)) throw new Error("textarea must be editable");
    if (!allowsNativeTextSelection(textarea)) throw new Error("textarea must allow selection");

    // ── 空/异常目标不抛错 ────────────────────────────────────────
    if (allowsNativeTextSelection(null)) throw new Error("null target must not allow selection");
    const detached = document.createElement("p");
    detached.textContent = "detached";
    if (allowsNativeTextSelection(firstTextNode(detached))) {
        throw new Error("detached node must not allow selection");
    }
});
