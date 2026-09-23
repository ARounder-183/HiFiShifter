/*
 * `image` 节点（记事本版）。
 *
 * 在 TipTap 官方 `image` 之上做三件事：
 *
 * 1. **宽度只保留一份真源**。文档内 `src` 不含 `#w=`，宽度是节点的 `width`
 *    属性；序列化成 Markdown 时才拼回 `src#w=<px>`（见 `assetRef`）。
 *    官方的 markdown 序列化不认 width，所以这里覆盖它的 `serialize`。
 * 2. **允许 `data:` 图片**（`allowBase64`）："内嵌"存储模式与网页粘贴都会
 *    产生 data URI。
 * 3. **自定义 NodeView**：`hifi-asset://` 与本地相对路径都要经后端取字节再
 *    转 blob URL，还要有"附件缺失"占位、替代文本编辑、右键菜单与拖拽改宽。
 */

import Image, { type ImageOptions } from "@tiptap/extension-image";
import { ReactNodeViewRenderer } from "@tiptap/react";

import { assetWidthFromSrc, stripWidthFragment, withWidthFragment } from "../assetRef";
import { NotebookImageNodeView } from "../NotebookImageNodeView";

interface MarkdownState {
    write: (text: string) => void;
    esc: (text: string) => string;
}

export const NotebookImage = Image.extend({
    addOptions(): ImageOptions {
        return {
            ...(this.parent?.() as ImageOptions),
            // 内嵌模式（data URI）与从网页粘贴的图片都需要它。
            allowBase64: true,
            HTMLAttributes: { class: "hs-notebook-image" },
        };
    },

    addAttributes() {
        return {
            ...this.parent?.(),
            src: {
                default: null,
                // 正文里的 `#w=` 归 width 属性管，src 保持干净。
                parseHTML: (element) => stripWidthFragment(element.getAttribute("src") ?? "") || null,
            },
            width: {
                default: null,
                parseHTML: (element) => {
                    const explicit = element.getAttribute("width");
                    if (explicit) {
                        const parsed = Number.parseInt(explicit, 10);
                        if (Number.isFinite(parsed) && parsed > 0) return parsed;
                    }
                    // 兼容手写的 `![x](hifi-asset://id.webp#w=640)`。
                    return assetWidthFromSrc(element.getAttribute("src") ?? "") ?? null;
                },
            },
        };
    },

    addStorage() {
        return {
            markdown: {
                serialize(
                    state: MarkdownState,
                    node: { attrs: { src?: string; alt?: string; title?: string; width?: number | null } },
                ) {
                    const src = withWidthFragment(node.attrs.src ?? "", node.attrs.width);
                    const alt = state.esc(node.attrs.alt ?? "");
                    const title = node.attrs.title
                        ? ` "${String(node.attrs.title).replace(/"/g, '\\"')}"`
                        : "";
                    // 圆括号会截断链接目标，转义掉（与 prosemirror-markdown 默认行为一致）。
                    state.write(`![${alt}](${src.replace(/[()]/g, "\\$&")}${title})`);
                },
            },
        };
    },

    addNodeView() {
        return ReactNodeViewRenderer(NotebookImageNodeView);
    },
});
