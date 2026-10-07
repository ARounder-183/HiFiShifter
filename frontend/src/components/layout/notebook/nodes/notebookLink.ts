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
import type { Attribute, Attributes } from "@tiptap/core";

import { normalizeLinkHref } from "../notebookLinkUrl";

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
    /*
     * 渲染时补协议。
     *
     * 【为什么还要在这一层做一遍】写入端（`applyNotebookLink`）已经归一化过，
     * 但文档里还可能有**不是经那个入口**来的链接：源码视图里手写的
     * `[x](www.bilibili.com)`（走 markdown 解析，直接写 attrs）、以及本次修复
     * 之前存下的笔记。它们的 href 仍是相对的 —— 渲染成 `<a href="...">` 之后
     * 会被浏览器按应用 origin 解析，点开跳到 `tauri.localhost/www.bilibili.com`。
     *
     * 只在**渲染**这一层补，不改 attrs：作者手写的源码（以及回写出的 Markdown）
     * 保持原样，不会被应用悄悄改写。点开、复制、导出这几条读 DOM 的路径则拿到
     * 绝对地址。
     */
    addAttributes() {
        /*
         * `this.parent?.()` 在 `Link.extend()` 之后退化成 `{}`：TipTap 的
         * `ParentConfig` 无法还原被 extend 的泛型。这里按它**实际的**形状
         * （属性名 → 属性定义）收窄一次，而不是整段重写 href 的定义 ——
         * 重写会在 TipTap 改动默认值时悄悄分叉。
         */
        const parent = (this.parent?.() ?? {}) as Attributes;
        const href: Attribute = parent.href ?? { default: null };
        return {
            ...parent,
            href: {
                ...href,
                renderHTML: (attributes) => ({
                    href: normalizeLinkHref(String(attributes.href ?? "")),
                }),
            },
        };
    },
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
