/*
 * `link` mark（记事本版）。
 *
 * tiptap-markdown 内置的链接序列化器直接写 `[text](href "title")`，对 href
 * 不做任何转义：带空格或圆括号的地址（`a b`、`a)b`）会生成解析不回来的
 * Markdown，链接 mark 在下次加载时静默丢失。这里覆盖它的 `markdown`
 * storage（tiptap-markdown 的 `getMarkdownSpec` 会用扩展自己的 storage 覆盖
 * 内置 spec，因此不需要动它的源码）。
 *
 * 目标含空格/圆括号/反斜杠/尖括号时按 CommonMark 包一层尖括号 `<...>`：
 * 角括号目标里空格与不平衡的圆括号都是合法字符，内部的 `\` `<` `>` 用反斜杠
 * 转义、换行按 URL 百分号编码兜底（目标里不允许裸换行）。其余地址原样写出，
 * 不引入无谓的编码。
 */

import { Link } from "@tiptap/extension-link";

/** 把 href 写成 CommonMark 链接目标（需要时用尖括号包裹）。 */
export function serializeLinkDestination(href: string): string {
    if (!href) return "";
    if (!/[\s()\\<>]/.test(href)) return href;
    const escaped = href.replace(/[\\<>]/g, "\\$&").replace(/\r?\n/g, "%0A");
    return `<${escaped}>`;
}

interface LinkMarkAttrs {
    href?: string | null;
    title?: string | null;
}

interface SerializerState {
    /** `prosemirror-markdown` 的标题引号逻辑（`"x"` / `'x'` / `(x)`）。 */
    quote: (text: string) => string;
}

export const NotebookLink = Link.extend({
    addStorage() {
        return {
            markdown: {
                serialize: {
                    open: () => "[",
                    close: (state: SerializerState, mark: { attrs: LinkMarkAttrs }) => {
                        const href = String(mark.attrs.href ?? "");
                        const title = mark.attrs.title
                            ? ` ${state.quote(String(mark.attrs.title))}`
                            : "";
                        return `](${serializeLinkDestination(href)}${title})`;
                    },
                },
            },
        };
    },
});
