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

/**
 * 在编辑器文档里找出全部匹配，返回**文档位置**区间。
 *
 * 逐文本节点搜索而不是拿 `doc.textContent` 做 indexOf：后者得到的偏移与
 * ProseMirror 的位置体系无关（块边界会占位），选中范围会错位。
 */
export function findMatchesInDoc(editor: Editor, query: string): TextMatch[] {
    if (!query) return [];
    const matches: TextMatch[] = [];
    const needle = query.toLowerCase();
    editor.state.doc.descendants((node, pos) => {
        if (!node.isText) return true;
        const text = node.text ?? "";
        const haystack = text.toLowerCase();
        let index = haystack.indexOf(needle);
        while (index >= 0) {
            matches.push({ from: pos + index, to: pos + index + query.length });
            index = haystack.indexOf(needle, index + Math.max(1, needle.length));
        }
        return true;
    });
    return matches;
}

/** 在纯文本里找出全部匹配（源码视图用）。 */
export function findMatchesInText(text: string, query: string): TextMatch[] {
    if (!query) return [];
    const matches: TextMatch[] = [];
    const haystack = text.toLowerCase();
    const needle = query.toLowerCase();
    let index = haystack.indexOf(needle);
    while (index >= 0) {
        matches.push({ from: index, to: index + query.length });
        index = haystack.indexOf(needle, index + Math.max(1, needle.length));
    }
    return matches;
}
