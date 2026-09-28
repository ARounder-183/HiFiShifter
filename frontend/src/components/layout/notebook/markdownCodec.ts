/*
 * `tiptap-markdown` 存储的类型化访问。
 *
 * `tiptap-markdown` 把解析器/序列化器挂在 `editor.storage.markdown` 上，但它
 * 的 `Extension` 泛型只声明了 `MarkdownStorage`（含 `getMarkdown`），
 * `serializer` 没进类型。项目里要序列化**选区**（复制）而不只是整篇文档，
 * 因此这里做一次集中的类型收口 —— 而不是在每个调用点各写一遍
 * `as unknown as ...`。
 */

import type { Editor } from "@tiptap/core";

/** 解析器的可用表面（markdown 字符串 → HTML，供内容替换等场景复用）。 */
export interface MarkdownParserLike {
    parse: (content: string) => string;
}

/** 序列化器的可用表面（`prosemirror-markdown` 的 MarkdownSerializer）。 */
export interface MarkdownSerializerLike {
    serialize: (content: unknown) => string;
}

export interface MarkdownStorageLike {
    getMarkdown: () => string;
    serializer: MarkdownSerializerLike;
    parser: MarkdownParserLike;
}

/** 取记事本的 Markdown 存储（编辑器未就绪时抛错，调用方需先判空）。 */
export function markdownStorage(editor: Editor): MarkdownStorageLike {
    return (editor.storage as unknown as { markdown: MarkdownStorageLike }).markdown;
}

/** 整篇文档的 Markdown。 */
export function documentMarkdown(editor: Editor): string {
    return markdownStorage(editor).getMarkdown();
}
