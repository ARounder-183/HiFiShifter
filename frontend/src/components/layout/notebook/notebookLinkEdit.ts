/*
 * 链接的写入 / 移除（记事本富文本）。
 *
 * 【为什么单独成文件】这里是"用户填了一个地址之后，文档该怎么变"的全部规则，
 * 而它有三种情形（改了已有链接 / 给选中的文字加链接 / **什么都没选**），
 * 最后一种正是此前失效的那个。规则放进来就能用真实编辑器直接测 —— 挂在工具栏
 * 组件里时，只有"打开浮层"这一层能被测到，真正改文档的那两行测不到。
 */

import type { Editor } from "@tiptap/core";

import { normalizeLinkHref } from "./notebookLinkUrl";

/**
 * 把 `href` 应用到当前选区。
 *
 * 三种情形：
 * 1. **光标落在已有链接里**（或选区横跨链接）：改写那段链接的地址，文字不动；
 * 2. **选中了文字**：给选中的文字加链接；
 * 3. **什么都没选、也不在链接里**：把**地址本身**作为链接文字插入到光标处。
 *
 * 【为什么第 3 种必须显式插入文字】`setLink` 最终是 ProseMirror 的 `setMark`，
 * 而空选区上的 `setMark` 只写"存储标记"（stored marks）—— 它作用于**下一个**
 * 键入的字符，文档当场毫无变化。于是用户填完地址按回车，看到的是"什么都没发生"，
 * 只能自己再打一遍文字。常见编辑器（Docs / Notion / VS Code）在这里都是把地址
 * 本身插成链接文字，本函数照此处理。
 *
 * 【为什么先归一化】`www.bilibili.com` 这种缺协议的地址若原样存进标记，会被
 * 当作相对地址按应用 origin 解析（点开跳到 `tauri.localhost/www.bilibili.com`），
 * 导出成 HTML 后同样坏掉。见 `notebookLinkUrl.ts`。
 *
 * 【地址白名单】`setLink` 会拒绝 `javascript:` 之类的协议（见 Link 扩展的
 * `isAllowedUri`）。第 3 种情形会先插入文字，所以**先探一次** `can()`：
 * 否则非法地址会留下一段没有链接的裸文字，而用户以为链接建好了。
 */
export function applyNotebookLink(editor: Editor, href: string): void {
    const next = normalizeLinkHref(href);
    if (!next) {
        clearNotebookLink(editor);
        return;
    }
    if (!editor.can().setLink({ href: next })) return;

    const { empty } = editor.state.selection;
    if (empty && !editor.isActive("link")) {
        const from = editor.state.selection.from;
        editor
            .chain()
            .focus()
            // 显式给文本节点，不用 `insertContent(href)`：字符串会被当作 HTML
            // 解析，地址里的 `<` 与 `&` 会变成标签或实体。
            .insertContent({ type: "text", text: next })
            // 选中刚插入的文字再交给 `setLink` —— 链接由**同一个命令**建立，
            // 白名单与 attrs 处理不会出现第二套。
            .setTextSelection({ from, to: from + next.length })
            .setLink({ href: next })
            .run();
        return;
    }

    editor.chain().focus().extendMarkRange("link").setLink({ href: next }).run();
}

/**
 * 移除当前链接。
 *
 * `extendMarkRange` 让"光标停在链接中间"也能整段摘掉链接，而不是只摘一个字 ——
 * 与 `setLink` 那一支的取范围方式保持一致。
 */
export function clearNotebookLink(editor: Editor): void {
    editor.chain().focus().extendMarkRange("link").unsetLink().run();
}

/** 当前选区是否已是一个链接（用于浮层的初值与按钮激活态）。 */
export function currentLinkHref(editor: Editor): string {
    return String(editor.getAttributes("link").href ?? "");
}
