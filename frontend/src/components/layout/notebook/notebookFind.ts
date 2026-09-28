/*
 * 面板内查找的匹配逻辑。
 *
 * 独立成模块（而不是放在 `NotebookFindBar.tsx` 里）：组件文件只导出组件才能
 * 让 React Fast Refresh 正常工作；而且这段是纯函数，可以在 node 环境直接测。
 */

import type { Editor } from "@tiptap/core";

export interface TextMatch {
    from: number;
    to: number;
}

/** 转义正则元字符（用户输入的查询词当字面量处理）。 */
function escapeRegExp(text: string): string {
    return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/**
 * 在 `text` 里找出 query 的全部大小写不敏感匹配，偏移直接取自**原字符串**。
 *
 * 【为什么不用 toLowerCase 再 indexOf】某些字符变小写会**变长**（土耳其
 * `İ` → "i̇"，1 个码点变 2 个），toLowerCase 后的下标与原串错位，用错位下标
 * 回原串上选区就会偏。大小写不敏感的正则直接在原串上匹配，索引天然可用。
 */
function findMatches(text: string, query: string): TextMatch[] {
    if (!query) return [];
    const matches: TextMatch[] = [];
    const pattern = new RegExp(escapeRegExp(query), "gi");
    let match = pattern.exec(text);
    while (match) {
        matches.push({ from: match.index, to: match.index + match[0].length });
        match = pattern.exec(text);
    }
    return matches;
}

/**
 * 在编辑器文档里找出全部匹配，返回**文档位置**区间。
 *
 * 逐文本节点搜索而不是拿 `doc.textContent` 做 indexOf：后者得到的偏移与
 * ProseMirror 的位置体系无关（块边界会占位），选中范围会错位。
 */
export function findMatchesInDoc(editor: Editor, query: string): TextMatch[] {
    if (!query) return [];
    const matches: TextMatch[] = [];
    editor.state.doc.descendants((node, pos) => {
        if (!node.isText) return true;
        for (const m of findMatches(node.text ?? "", query)) {
            matches.push({ from: pos + m.from, to: pos + m.to });
        }
        return true;
    });
    return matches;
}

/** 在纯文本里找出全部匹配（源码视图用）。 */
export function findMatchesInText(text: string, query: string): TextMatch[] {
    return findMatches(text, query);
}
