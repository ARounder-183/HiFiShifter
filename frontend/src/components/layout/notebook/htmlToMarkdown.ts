/*
 * 粘贴富文本 → Markdown。
 *
 * 需求是"剪贴板里有富文本时，粘贴需正确转换为合适的 Markdown"。实现分两步：
 *
 * 1. **DOMPurify 消毒**：从 Word / 网页 / 邮件复制来的 HTML 里混着
 *    `<script>`、`onclick`、`style` 乃至 `javascript:` 链接。Turndown 自己
 *    会丢掉一部分，但"消毒"应当是显式的、单一职责的一步，而不是依赖转换器
 *    的副作用。
 * 2. **Turndown 转换**：HTML → Markdown。表格、任务列表、代码块由
 *    `turndown-plugin-gfm` 补齐（GFM 表格是本 app 的高频场景：从文档里粘
 *    参数对照表）。
 *
 * 结果再交给 TipTap 解析入库，因此最终落到正文里的一定是规范 Markdown。
 */

import DOMPurify from "dompurify";
import TurndownService from "turndown";
// @ts-expect-error 该包没有类型声明
import { gfm } from "turndown-plugin-gfm";

let service: TurndownService | null = null;

function getService(): TurndownService {
    if (service) return service;
    const instance = new TurndownService({
        headingStyle: "atx",
        hr: "---",
        bulletListMarker: "-",
        codeBlockStyle: "fenced",
        fence: "```",
        emDelimiter: "*",
        strongDelimiter: "**",
        linkStyle: "inlined",
        // 空块替换：块级元素之间保留一个空行。
        blankReplacement: (_content, node) =>
            (node as unknown as { isBlock?: boolean }).isBlock ? "\n\n" : "",
    });

    // 列表项：Turndown 默认写成 `-   内容`（三个空格），而本 app 自己的
    // 序列化器写 `- 内容`。两边保持一致，源码视图里就不会出现"同一篇笔记
    // 两种缩进风格"。嵌套缩进同样收成 2 空格。
    //
    // 【注册顺序】必须在本规则之后再 `use(gfm)`：`addRule` 是前插，
    // 后注册的规则优先命中 —— gfm 的 `taskListItems` 只匹配带 checkbox 的
    // `li`，让它优先处理任务项，本规则处理普通项。
    instance.addRule("listItem", {
        filter: "li",
        replacement: (content, node, options) => {
            const parent = node.parentNode as HTMLElement | null;
            const isOrdered = parent?.nodeName === "OL";
            const index =
                isOrdered && parent
                    ? Array.prototype.indexOf.call(parent.children, node) + 1
                    : null;
            const prefix = isOrdered ? `${index}. ` : `${options.bulletListMarker} `;
            const text = content.replace(/^\n+/, "").replace(/\n+$/, "\n").replace(/\n/gm, "\n  ");
            return prefix + text + (node.nextSibling && !/\n$/.test(text) ? "\n" : "");
        },
    });

    instance.use(gfm);

    // 刻意**不**用 `keep(["u", "sub", "sup", "mark"])` 保留行内 HTML：
    // 记事本的 Markdown 解析关掉了裸 HTML（`html: false`），保留下来的标签
    // 在保存后会被原样显示成文本（`<u>xxx</u>`），比丢掉下划线更糟。
    // 因此这些标记在粘贴时降级为纯文本，只保留 Markdown 能表达的语义。

    // 图片：保留 src/alt，丢弃尺寸类属性（尺寸由本 app 自己的 `#w=` 承载）。
    instance.addRule("image", {
        filter: "img",
        replacement: (_content, node, options) => {
            const element = node as unknown as HTMLImageElement;
            const src = element.getAttribute("src") ?? "";
            const alt = element.getAttribute("alt") ?? "";
            if (!src) return "";
            // 内嵌 data URI 的解码体积按 base64 长度估算，超过上限就丢弃：
            // 网页粘贴可能带几 MB 的内嵌图，而工程字节是"只增不删"的（撤销要
            // 能恢复），事后没有瘦身手段。占位用语言无关的 `[image]` —— 这个
            // 模块拿不到 i18n 上下文，本地化它反而要拖一条 t 依赖进来。
            const limit =
                (options as unknown as { maxDataImageBytes?: number }).maxDataImageBytes ??
                DEFAULT_MAX_DATA_IMAGE_BYTES;
            if (dataImageDecodedBytes(src) > limit) return "[image]";
            // 标题里有括号会截断链接语法，折叠掉。
            return `![${alt.replace(/[[\]()]/g, "")}](${src})`;
        },
    });

    service = instance;
    return instance;
}

/** 消毒一段来自外部的 HTML。 */
export function sanitizePastedHtml(html: string): string {
    return DOMPurify.sanitize(html, {
        // 只允许"文档内容"类标签；表单、脚本、iframe 全部丢弃。
        FORBID_TAGS: ["script", "style", "iframe", "form", "input", "button", "object", "embed"],
        FORBID_ATTR: ["style", "onerror", "onload", "onclick"],
        // 允许 data: 图片（截图/网页里常见内联图），其余协议由 DOMPurify 默认策略收口。
        ADD_DATA_URI_TAGS: ["img"],
    });
}

/** `data:` 图片的解码体积上限默认值（与 `maxImageBytes` 的默认值一致）。 */
export const DEFAULT_MAX_DATA_IMAGE_BYTES = 20 * 1024 * 1024;

export interface HtmlToMarkdownOptions {
    /**
     * 内嵌 `data:` 图片的解码体积上限（字节）；超限的图丢弃并留下 `[image]`
     * 占位。不传时用 `DEFAULT_MAX_DATA_IMAGE_BYTES`；粘贴路径传用户的
     * `maxImageBytes`，让两条大小纪律走同一个数。
     */
    maxDataImageBytes?: number;
}

/** 估算 data URI 解码后的字节数（base64 每 4 字符还原 3 字节）。 */
function dataImageDecodedBytes(src: string): number {
    const marker = ";base64,";
    const at = src.indexOf(marker);
    if (at < 0) return 0; // 非 base64 编码的 data URI 本就放不下大图，不设限。
    return Math.floor(((src.length - at - marker.length) * 3) / 4);
}

/** HTML → Markdown。失败时返回空串（调用方回退到纯文本粘贴）。 */
export function htmlToMarkdown(html: string, options?: HtmlToMarkdownOptions): string {
    if (!html.trim()) return "";
    try {
        const sanitized = sanitizePastedHtml(html);
        if (!sanitized.trim()) return "";
        const instance = getService();
        // turndown 是同步单次调用，把上限暂存在实例 options 上没有并发问题；
        // 未传时留空，规则内退回 `DEFAULT_MAX_DATA_IMAGE_BYTES`。
        (instance.options as unknown as { maxDataImageBytes?: number }).maxDataImageBytes =
            options?.maxDataImageBytes;
        return instance.turndown(sanitized).trim();
    } catch {
        return "";
    }
}
