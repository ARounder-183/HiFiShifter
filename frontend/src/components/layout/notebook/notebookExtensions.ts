/*
 * 记事本的 TipTap 扩展装配。
 *
 * ## 为什么 `html: false`
 *
 * `tiptap-markdown` 的 `html` 开关决定 markdown-it 是否放行裸 HTML。这里关掉
 * 它，理由有三：
 *
 * 1. **可预测**：裸 HTML 会原样显示为文本，用户立刻看到"这段没被解析"，
 *    而不是被悄悄解释成半成品结构；
 * 2. **零 XSS 面**：正文不经过任何 HTML 解释，编辑器的 schema 就是唯一白名单；
 * 3. **往返诚实**：不支持的结构在源码视图里保持原样，富文本视图只呈现它
 *    真正理解的部分 —— 保真度提示（见 `notebookFidelity`）会告诉用户哪里
 *    被规范化了。
 *
 * 代价是**表格**：`tiptap-markdown` 内置的表格序列化在遇到跨行/跨列或多段落
 * 单元格时会退化成裸 HTML 节点标记（`[markdownHTMLNode]`），在 html:false
 * 下等于内容消失。因此这里用 `NotebookTable` 覆盖它，把任意表格都写成 GFM
 * 表格（多段落用 `<br>` 连接，跨行跨列信息丢弃并在保真度提示里说明）。
 */

import { Markdown } from "tiptap-markdown";
import StarterKit from "@tiptap/starter-kit";
import { CharacterCount, Placeholder } from "@tiptap/extensions";
import { TaskItem, TaskList } from "@tiptap/extension-list";
import { Table, TableCell, TableHeader, TableRow } from "@tiptap/extension-table";

import { HifiClipBlock } from "./nodes/hifiClipBlock";
import { NotebookImage } from "./nodes/notebookImage";
import { NotebookLink, serializeLinkDestination } from "./nodes/notebookLink";

interface MarkdownState {
    write: (text: string) => void;
    ensureNewLine: () => void;
    closeBlock: (node: unknown) => void;
}

interface PmNode {
    type: { name: string };
    attrs: Record<string, unknown>;
    isText?: boolean;
    text?: string;
    marks?: Array<{ type: { name: string }; attrs: Record<string, unknown> }>;
    childCount: number;
    forEach: (fn: (child: PmNode, offset: number, index: number) => void) => void;
}

/**
 * 表格 → GFM。
 *
 * 与内置实现的区别：**永不退化成 HTML 节点标记**。内置实现在遇到跨行/跨列或
 * 多段落单元格时会走 HTML 兜底，而在 `html: false` 下那个兜底写出来的是一句
 * 无意义的占位文本 —— 等于把表格内容丢掉。
 *
 * 这里自己渲染单元格的行内内容（而不是复用 `state.renderInline`）：序列化器
 * 的状态机（`out` 缓冲 + `inlines` 栈 + `atBlank`）是为"顺序写整篇文档"设计
 * 的，把它的输出缓冲临时换成字符串再换回来，会让行内标记的裁剪偏移算错。
 * 单元格里只有行内内容（GFM 表格的固有限制），自己渲染既简单又确定。
 *
 * 跨行/跨列无法用 GFM 表达：丢弃结构、保留内容 —— 丢格式比丢内容好。
 */
const NotebookTable = Table.extend({
    addStorage() {
        return {
            markdown: {
                serialize(state: MarkdownState, node: PmNode) {
                    let rowIndex = 0;
                    node.forEach((row) => {
                        const cells: string[] = [];
                        row.forEach((cell) => {
                            cells.push(renderCell(cell));
                        });
                        state.write(`| ${cells.join(" | ")} |`);
                        state.ensureNewLine();
                        if (rowIndex === 0) {
                            state.write(`| ${cells.map(() => "---").join(" | ")} |`);
                            state.ensureNewLine();
                        }
                        rowIndex += 1;
                    });
                    state.closeBlock(node);
                },
            },
        };
    },
});

/** 单元格 → 一行 GFM 文本（多段落用 `<br>` 连接，管道转义）。 */
function renderCell(cell: PmNode): string {
    const blocks: string[] = [];
    cell.forEach((block) => {
        blocks.push(renderCellBlock(block));
    });
    return blocks.join("<br>").replace(/\|/g, "\\|").replace(/\r?\n/g, " ").trim();
}

/**
 * 单元格里的一个块。图片是块级节点，解析后作为单元格的**直接子块**存在
 * （它进不了段落），必须交给 `renderInlineNode` 写成行内图 —— 否则会被当成
 * "没有行内内容的块"丢成空串。
 */
function renderCellBlock(block: PmNode): string {
    if (block.type.name === "image") return renderInlineNode(block);
    return renderInlineContent(block);
}

/** 渲染一个块级节点的行内内容（文本 + 已知标记 + 图片 + 硬换行）。 */
function renderInlineContent(block: PmNode): string {
    let out = "";
    block.forEach((child) => {
        out += renderInlineNode(child);
    });
    return out;
}

function renderInlineNode(node: PmNode): string {
    if (node.type.name === "hardBreak") return "<br>";
    if (node.type.name === "image") {
        const src = String(node.attrs.src ?? "");
        const alt = String(node.attrs.alt ?? "");
        // 转义规则与主图片序列化器（nodes/notebookImage.ts）一致：src 的圆
        // 括号会截断目标，alt 的反斜杠/方括号会截断替代文本。
        return `![${escapeCellAlt(alt)}](${src.replace(/[()]/g, "\\$&")})`;
    }
    if (node.type.name === "text") {
        return applyMarks(node.text ?? "", node.marks ?? []);
    }
    // 单元格里出现块级子节点（列表等）时退化为纯文本拼接，不丢内容。
    let nested = "";
    node.forEach((child) => {
        nested += renderInlineNode(child);
    });
    return nested;
}

/** 单元格 alt 的转义（对应主序列化器里的 `state.esc`：反斜杠与方括号）。 */
function escapeCellAlt(alt: string): string {
    return alt.replace(/[\\[\]]/g, "\\$&");
}

function applyMarks(
    text: string,
    marks: Array<{ type: { name: string }; attrs: Record<string, unknown> }>,
): string {
    let out = text;
    // 顺序固定：链接在最外层（`[**x**](url)`），与常见 Markdown 习惯一致。
    for (const mark of marks) {
        switch (mark.type.name) {
            case "code":
                out = `\`${out}\``;
                break;
            case "bold":
                out = `**${out}**`;
                break;
            case "italic":
                out = `*${out}*`;
                break;
            case "strike":
                out = `~~${out}~~`;
                break;
            default:
                break;
        }
    }
    const link = marks.find((mark) => mark.type.name === "link");
    if (link) {
        const href = String(link.attrs.href ?? "");
        if (href) out = `[${out}](${serializeLinkDestination(href)})`;
    }
    return out;
}

export interface NotebookExtensionOptions {
    /** 空文档占位文案。 */
    placeholder: string;
    /** Markdown 输入规则（`## ` / `- ` / `> ` 即时转换）。 */
    markdownShortcuts: boolean;
    /** `/` 唤出插入菜单。 */
    slashCommands: boolean;
}

/**
 * 组装记事本编辑器用的扩展。
 *
 * `slashCommands` 目前由面板层实现（在编辑器 DOM 上挂 `/` 触发的浮层菜单），
 * 不占用扩展位 —— 见 `NotebookPanel` 的 `useSlashMenu`。
 */
/**
 * 组装记事本编辑器用的扩展。
 *
 * 【每个编辑器都要拿到**独立的扩展实例**】TipTap 不克隆传入的扩展：同一个
 * 实例被两个编辑器复用时，`extension.storage` 等按实例创建的状态会被共享
 * （分栏模式同时存在两个编辑器）。这里对模块级的三个自定义扩展统一
 * `.extend({})` 复制一份，其余扩展本来就由 `.configure()` 产生新实例。
 */
export function buildNotebookExtensions(options: NotebookExtensionOptions) {
    void options.slashCommands;
    const extensions = [
        StarterKit.configure({
            heading: { levels: [1, 2, 3, 4, 5, 6] },
            // 下划线在 Markdown 里没有对应语法（`_x_` 是斜体），保留它会让
            // 粘贴来的下划线内容在保存时无处安放，因此直接禁用。
            underline: false,
            // 内置 link 关掉，换成下面的 NotebookLink：同名扩展只保留一份，
            // 换这个版本是为了覆盖它的 Markdown 序列化（见 nodes/notebookLink）。
            link: false,
            // 输入规则由 Markdown 扩展自带一套更贴合 Markdown 语义的，关闭
            // StarterKit 的默认规则避免两套规则打架（例如 `1. ` 的序号处理）。
            ...(options.markdownShortcuts ? {} : { inputRules: false }),
        }),
        NotebookLink.extend({}).configure({
            openOnClick: false,
            autolink: true,
            // 粘贴时的链接识别由记事本自己的粘贴分流负责（见 notebookClipboard）。
            linkOnPaste: false,
            // 内部链接（`hifi://seek/…`、`hifi://clip/…`）必须进白名单，否则
            // Link 扩展的 URI 校验会**静默丢掉链接标记** —— 正文里的时间码会
            // 退化成一段普通文字。
            protocols: ["hifi"],
            HTMLAttributes: { rel: "noreferrer noopener", target: "_blank" },
        }),
        NotebookImage.extend({}).configure({ allowBase64: true }),
        NotebookTable.extend({}),
        TableRow.extend({}),
        TableHeader.extend({}),
        TableCell.extend({}),
        TaskList.extend({}),
        TaskItem.extend({}).configure({ nested: true }),
        HifiClipBlock.extend({}),
        Placeholder.extend({}).configure({ placeholder: options.placeholder }),
        CharacterCount.extend({}),
        Markdown.configure({
            html: false,
            tightLists: true,
            bulletListMarker: "-",
            linkify: false,
            breaks: false,
            transformPastedText: false,
            transformCopiedText: false,
        }),
    ];
    return extensions;
}
