/*
 * `hifiClipBlock` 节点：HiFiShifter 剪贴板载荷的暂存块。
 *
 * ## 设计要点
 *
 * 节点只保存**一段 Markdown 围栏正文文本**（attr `body`），而不是把
 * kind/title/clips/... 拆成十来个属性。三个好处：
 *
 * 1. **往返绝对无损**：序列化就是把 `body` 原样写回围栏，不需要逐字段
 *    重排；未来新增字段（比如载荷版本）不会被旧代码吃掉。
 * 2. **单一真源**：`body` 就是用户能在源码视图里看到、能直接改的那段文本，
 *    不存在"属性与文本不一致"的可能。
 * 3. **HTML 表示同构**：`renderHTML` 产出 `<div data-hifi-clip="<body>">`，
 *    markdown-it 的围栏渲染规则也产出同一个 div —— 因此
 *    "Markdown → HTML → ProseMirror" 与 "ProseMirror → HTML → 再解析"
 *    两条路径完全一致，不需要额外的 DOM 变换钩子。
 */

import { Node } from "@tiptap/core";

import { HIFI_CLIP_FENCE_LANG, parseHifiClipFenceBody } from "../hifiClipBlock";

export const HIFI_CLIP_ATTR = "data-hifi-clip";

type FenceRule = (
    tokens: unknown[],
    idx: number,
    options: unknown,
    env: unknown,
    self: { renderToken: (...args: unknown[]) => string },
) => string;

/**
 * 每个 markdown-it 实例的原始 fence 渲染函数。
 *
 * 【为什么需要缓存】`parse.setup` **每次解析都会被调用一次**。若每次都
 * `const original = rules.fence` 再包一层，包装会层层叠加 —— 解析 N 次就有
 * N 层函数调用，长文档下是实打实的性能问题。这里按实例只记一次原始实现。
 */
const originalFenceRules = new WeakMap<object, FenceRule | undefined>();

/**
 * 把围栏正文写进 HTML 属性值。
 *
 * 只用在 markdown-it 的渲染规则里：那段 HTML 会被 `innerHTML` 解析，属性里的
 * `"`/`&`/`<` 必须转义，换行也编码成字符引用（属性值里的裸换行虽然合法，
 * 但会被某些序列化器规范化掉，编码后最稳）。
 *
 * 注意 `renderHTML` 那条路径**不做转义** —— ProseMirror 用 `setAttribute`
 * 写入，属性值本来就是原文。两条路径最终都由 `getAttribute` 读到同一份原文。
 */
export function escapeFenceAttrForHtml(body: string): string {
    return body
        .replace(/&/g, "&amp;")
        .replace(/"/g, "&quot;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/\r\n|\r|\n/g, "&#10;");
}

export const HifiClipBlock = Node.create({
    name: "hifiClipBlock",

    // 块级原子节点：内部不可编辑，整体作为一张卡片存在。
    group: "block",
    atom: true,
    selectable: true,
    draggable: true,

    addAttributes() {
        return {
            body: {
                default: "",
                parseHTML: (element) => element.getAttribute(HIFI_CLIP_ATTR) ?? "",
                renderHTML: (attributes) => ({ [HIFI_CLIP_ATTR]: String(attributes.body ?? "") }),
            },
        };
    },

    parseHTML() {
        return [{ tag: `div[${HIFI_CLIP_ATTR}]` }];
    },

    renderHTML({ HTMLAttributes }) {
        return ["div", HTMLAttributes];
    },

    addStorage() {
        return {
            markdown: {
                /**
                 * 解析：把 `hifi-clip` 围栏渲染成带 data 属性的 div。
                 *
                 * 用渲染规则而不是 `updateDOM` 后处理：直接产出目标 DOM 比
                 * "先渲染成 pre/code 再替换"少一次遍历，也不依赖 DOM 结构细节。
                 */
                parse: {
                    setup(markdownit: {
                        renderer: { rules: Record<string, FenceRule | undefined> };
                    }) {
                        const renderer = markdownit.renderer;
                        if (!originalFenceRules.has(renderer as object)) {
                            originalFenceRules.set(renderer as object, renderer.rules.fence);
                        }
                        const defaultFence = originalFenceRules.get(renderer as object);
                        renderer.rules.fence = (tokens, idx, options, env, self) => {
                            const token = tokens[idx] as { info?: string; content?: string };
                            const info = (token.info ?? "").trim().toLowerCase();
                            if (info === HIFI_CLIP_FENCE_LANG) {
                                // 正文解析不出 id 时退化成普通代码块：宁可显示成
                                // 一段代码，也不要把用户写坏的内容吞掉。
                                const body = token.content ?? "";
                                if (parseHifiClipFenceBody(body)) {
                                    return `<div ${HIFI_CLIP_ATTR}="${escapeFenceAttrForHtml(body)}"></div>`;
                                }
                            }
                            return defaultFence
                                ? defaultFence(tokens, idx, options, env, self)
                                : self.renderToken(tokens, idx, options);
                        };
                    },
                },
                serialize(
                    state: { write: (text: string) => void; closeBlock: (node: unknown) => void },
                    node: { attrs: { body?: string } },
                ) {
                    const body = String(node.attrs.body ?? "").replace(/\s+$/, "");
                    state.write(["```" + HIFI_CLIP_FENCE_LANG, body, "```", ""].join("\n"));
                    state.closeBlock(node);
                },
            },
        };
    },
});
