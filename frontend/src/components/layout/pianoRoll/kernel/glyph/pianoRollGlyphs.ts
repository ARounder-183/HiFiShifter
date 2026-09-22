/**
 * 参数编辑器内核 · 字形渲染适配层
 *
 * 【主要内容】
 * 把参数编辑器需要的文字（键盘音名、数值轴刻度标签、画布中央提示文字）接到时间轴
 * 内核那套**已存在但未接线**的字形管线：光栅化器（离屏 Canvas2D → 字形图集）+
 * 布局器（按字符测量 + 截断）+ 四边形构建 + GL program。
 *
 * 【作用】
 * 阶段 2 要把所有 `fillText` 搬到 GL。文字从"每帧调用 `ctx.fillText`"变成"字形
 * 光栅化一次进图集、之后每帧只提交四边形"，因此播放帧不再重排文字。
 *
 * 【与 Canvas2D 的语义对齐（本模块最容易出错的地方）】
 * `fillText` 的垂直定位依赖 `ctx.textBaseline`，而字形管线用的是"槽位左上角"
 * 语义（光栅化时 `textBaseline = "top"`）。两者的换算关系是本模块的核心职责：
 *
 * - Canvas2D `textBaseline = "top"`：y 是**em 盒顶部**。
 * - Canvas2D `textBaseline = "middle"`：y 是**em 盒中线**。
 * - 字形管线：`originY` 是**槽位顶部**，槽位高 = `字号 × GLYPH_LINE_HEIGHT_RATIO`
 *   （1.2），而字形墨迹落在槽位顶部的 em 盒内（即 `[槽位顶, 槽位顶 + 字号]`）。
 *
 * 因此：
 * - 对齐 `"top"`    → `originY = y`
 * - 对齐 `"middle"` → `originY = y − 字号 / 2`（em 盒中线对齐到 y）
 *
 * 参数编辑器的全部 `fillText` 都用 `"middle"`，因此这两种已覆盖现有需求。
 *
 * 【与其他模块的关系】
 * - 上游：`pianoRollKernelHost` 在几何变化时构建文字四边形并上传。
 * - 横向：复用 `renderKernel/glyph/*` 与 `renderKernel/gl/glyphProgram`。
 * - 独立性：除测量 / 光栅化依赖 Canvas2D 外无 DOM 耦合；无 DOM 环境下
 *   `createPianoRollGlyphs` 返回 null，调用方跳过文字（不崩）。
 */

import type { Rgba } from "../../../renderKernel/instanceTypes";
import type { AtlasSlot } from "../../../renderKernel/glyph/glyphAtlas";
import {
    createGlyphLayout,
    type GlyphLayout,
    type GlyphRun,
} from "../../../renderKernel/glyph/glyphLayout";
import {
    createGlyphRasterizer,
    GLYPH_LINE_HEIGHT_RATIO,
    type GlyphRasterizer,
} from "../../../renderKernel/glyph/glyphRasterizer";
import { buildGlyphQuads, type GlyphQuad } from "../../../renderKernel/gl/glyphQuads";

/** 图集单页边长（物理像素）。 */
export const PIANO_ROLL_ATLAS_PAGE_SIZE_PX = 2048;

/** 图集最大页数。 */
export const PIANO_ROLL_ATLAS_MAX_PAGES = 2;

/**
 * 以 `middle` 基准绘制一行文字时，字形槽位在锚点**下方**延伸的高度（CSS px）。
 *
 * 【推导】由文件头的换算关系：`originY = y − 字号 / 2`（槽位顶部），槽位高 =
 * `字号 × GLYPH_LINE_HEIGHT_RATIO`，故槽位下缘 = `y − 字号/2 + 字号 × ratio`；
 * 相对锚点向下的溢出即 `字号 × (ratio − 0.5)`。
 *
 * 【用途】给"锚点正好落在绘图区下边缘"的文字预留画布高度。数值轴最下方那条刻度
 * 正是这种情况：`valueToY` 把值域下界映射到 `heightPx` 本身，于是标签的中线落在
 * 绘图区下边缘，下半截会被画布边界裁掉（观感是"最下面的刻度值被挡住了"）。
 *
 * @param fontSizePx 字号（CSS px）。
 * @returns 需要向下预留的高度（CSS px）；非法字号返回 0。
 */
export function glyphMiddleSlotDescentPx(fontSizePx: number): number {
    if (!Number.isFinite(fontSizePx) || fontSizePx <= 0) return 0;
    return Math.max(0, fontSizePx * GLYPH_LINE_HEIGHT_RATIO - fontSizePx / 2);
}

/** 数值轴刻度标签的字号（CSS px）。字形请求与画布预留高度必须同源。 */
export const AXIS_TICK_LABEL_FONT_SIZE_PX = 10;

/**
 * 数值轴画布在**绘图区**（`viewportHeightPx`）下方额外预留的高度（CSS px）。
 *
 * 【为什么留下它】以 `middle` 基准落笔的文字，其**槽位下缘**在锚点下方
 * `字号 × (1.2 − 0.5)` 处（见 `glyphMiddleSlotDescentPx`）；锚点正好压在绘图区
 * 下边缘时，那截槽位需要这点高度才画得完。
 *
 * 【当前没有任何标签依赖它】数值轴刻度标签已由 {@link axisTickLabelAnchorBounds}
 * 夹进绘图区内部；键盘音名的中线最多落在 `下边缘 − 半个键高`（键高不足 6px 时
 * 干脆不画），墨迹同样留在绘图区内。因此这份预留现在是**保险**：文字是位图槽位，
 * 槽位下缘一旦越出画布就会被硬裁，而"是否越界"取决于字号 / 行高比 / 字体墨迹三者
 * 的组合，留出这段比日后再踩一次坑便宜。
 *
 * 【约束】必须 ≤ `PARAM_EDITOR_BOTTOM_BAR_PX`（面板为自绘水平滚动条预留的行高，
 * 也就是纵轴列比滚动视口高出的那一条），否则画布会超出列被父层裁掉、问题复现。
 * 该约束由 `axisLabelMetrics.test.ts` 钉住。
 */
export const AXIS_TICK_LABEL_DESCENT_PX = Math.ceil(
    glyphMiddleSlotDescentPx(AXIS_TICK_LABEL_FONT_SIZE_PX),
);

/**
 * 刻度标签在绘图区两端额外内缩的余量（CSS px）。
 *
 * 【为什么不止"正好放得下"】下面的区间把标签的 em 盒约束在**绘图区内部**，
 * 而墨迹并非严格等于 em 盒（实测常见字体里，数字的墨迹比 em 盒顶低 0.4~1.6px，
 * 个别字体如 Meiryo 会下探到 em 盒底附近）。留 1px 余量让"贴着边缘的那一行"
 * 不至于恰好压在边界像素上；同时它也让本区间**不依赖**画布下方那点预留高度
 * （见 `AXIS_TICK_LABEL_DESCENT_PX`）——预留高度只够容下 em 盒的一半，一旦父层
 * 裁掉那一条（父容器 `overflow-hidden`、预留行小于常量、分数 DPR 取整），
 * 最下方标签就会缺半截。
 */
export const AXIS_TICK_LABEL_EDGE_MARGIN_PX = 1;

/**
 * 数值轴刻度标签锚点 y 的**安全区间**（视口坐标，`middle` 基准）。
 *
 * 【要解决的问题】刻度标签锚定在 `valueToY(值)` 上，而值域的两端恰好映射到绘图区
 * 的上下边缘（`y = 0` 与 `y = heightPx`）。以 `middle` 基准绘制的文字有一半在锚点
 * **上/下方**，于是：
 * - 最上面那条刻度（通常是视口上界）上半个字被画布上缘裁掉；
 * - 最下面那条刻度（通常是 `0` / dB 的 `-∞`）下半个字压在绘图区下边缘，只能靠
 *   `AXIS_TICK_LABEL_DESCENT_PX` 的下方预留来救 —— 一旦那条预留被裁或字体墨迹
 *   略低，就露出"缺半截"。
 *
 * 【做法】把锚点夹进"em 盒（±字号/2）连同 {@link AXIS_TICK_LABEL_EDGE_MARGIN_PX}
 * 余量都落在绘图区内"的区间。两端各最多内缩 `字号/2 + 1`（本字号 6 CSS px），
 * 对 5~12 条刻度的密度而言远小于刻度间距，不会与其他标签重叠。
 *
 * 【为什么夹锚点而不是加高画布】画布下方最多只能多出 `PARAM_EDITOR_BOTTOM_BAR_PX`
 * （8px），上方则**完全没有**可扩展空间（轴列顶端就是角框）。夹锚点是唯一在两个
 * 方向都成立的做法，且只动文字、不动任何几何（刻度线位置保持不变）。
 *
 * @param fontSizePx 标签字号（CSS px）。
 * @param viewportHeightPx 绘图区高度（CSS px）。
 * @returns 锚点 y 的上下限（闭区间）；入参非法时返回退化区间（调用方原样使用）。
 */
export function axisTickLabelAnchorBounds(
    fontSizePx: number,
    viewportHeightPx: number,
): { readonly minY: number; readonly maxY: number } {
    const half = fontSizePx / 2;
    if (!Number.isFinite(fontSizePx) || fontSizePx <= 0) {
        return { minY: 0, maxY: Number.POSITIVE_INFINITY };
    }
    if (!Number.isFinite(viewportHeightPx) || viewportHeightPx <= 0) {
        return { minY: 0, maxY: Number.POSITIVE_INFINITY };
    }
    const minY = half;
    // `Math.max(minY, …)` 兜住"绘图区比一个字还矮"的退化情形：此时宁可让标签落在
    // 顶端（仍可读），也不要让区间反转（夹取会得到不可预期的值）。
    const maxY = Math.max(minY, viewportHeightPx - half - AXIS_TICK_LABEL_EDGE_MARGIN_PX);
    return { minY, maxY };
}

/** 文本的水平对齐方式（对应 Canvas2D 的 `ctx.textAlign`）。 */
export type TextAlign = "left" | "center" | "right";

/**
 * 文本的垂直基准（对应 Canvas2D 的 `ctx.textBaseline`）。
 *
 * 特殊说明：只覆盖本面板**实际用到**的两种。参数编辑器的全部 `fillText` 都用
 * `"middle"`（键盘音名、刻度标签、画布中央提示文字），此处额外保留 `"top"` 以便
 * 后续图层使用。**不支持 `"alphabetic"`**：它需要按字体族实测 ascent 比例，而字形
 * 槽位是"顶部对齐 em 盒"，实测比例要经 `TextMetrics` 取得、且随字体族变化；
 * 在没有实际调用点的情况下实现它属于过早设计。
 */
export type TextBaseline = "top" | "middle";

/** 一段待绘制的文字。 */
export interface TextRequest {
    /** 文本内容。 */
    readonly text: string;
    /** 字体标识（CSS font 简写，如 `bold 9px sans-serif`）。 */
    readonly fontKey: string;
    /** 文本锚点 x（视口坐标 CSS px）。 */
    readonly x: number;
    /** 文本锚点 y（视口坐标 CSS px，语义由 `baseline` 决定）。 */
    readonly y: number;
    /** 水平对齐；缺省 `"left"`。 */
    readonly align?: TextAlign;
    /** 垂直基准；缺省 `"alphabetic"`（与 Canvas2D 默认一致）。 */
    readonly baseline?: TextBaseline;
    /** 颜色。 */
    readonly rgba: Rgba;
}

/** 字形渲染器。 */
export interface PianoRollGlyphs {
    /**
     * 布局并构建文字四边形。
     *
     * @param requests 待绘制文字（视口坐标）。
     * @returns 四边形序列（供 `glyphProgram.render`）；无 DOM / 无文字时为空数组。
     */
    build(requests: readonly TextRequest[]): GlyphQuad[];
    /** 图集单页边长（物理像素），必须与 `glyphProgram.uploadAtlas` 一致。 */
    atlasPageSizePx(): number;
    /** 取出并清空"有变更的图集页"（供调用方上传纹理）。 */
    consumeDirtyPages(): readonly number[];
    /** 读取某页的像素数据（供上传）。 */
    readPage(page: number): Uint8ClampedArray | null;
    /** 释放离屏画布等资源。 */
    dispose(): void;
}

/**
 * 解析字体标识中的字号（CSS px）。
 *
 * 特殊说明：与 `glyphRasterizer.parseFontSizePx` 同一口径——**不锚定行首**，
 * 因此 `bold 9px …` 也能正确取到 9。本模块需要字号来做垂直基准换算。
 *
 * @param fontKey 字体标识。
 * @returns 字号；解析失败回退 12（与光栅化器一致）。
 */
function fontSizeOf(fontKey: string): number {
    const match = /(\d+(?:\.\d+)?)px/.exec(fontKey);
    return match ? Number(match[1]) : 12;
}

/**
 * 创建参数编辑器的字形渲染器。
 *
 * 流程：建离屏光栅化器（失败即返回 null）→ 建布局器（复用光栅化器的测量）→
 * 返回 build / 图集访问 / 释放。
 *
 * 特殊说明 1：无 DOM 环境（node 单测）返回 null，调用方据此跳过文字绘制。
 *
 * 特殊说明 2：`build()` 里对 `align` 的处理是**先布局再平移**——字形管线按"文本
 * 起点 + 逐字符偏移"组织，没有对齐概念，因此居中 / 右对齐要靠起点回退：
 * `center` 退回 `width/2`、`right` 退回 `width`。回退量用的是**实际布局宽度**
 * （可能因截断而小于请求宽度），与 Canvas2D 的行为一致。
 *
 * @param options 构造参数（dpr 必填，图集尺寸可选）。
 * @returns 渲染器；WebGL2 / Canvas2D 不可用时为 null。
 */
export function createPianoRollGlyphs(options: {
    readonly dpr: number;
    readonly pageSizePx?: number;
    readonly maxPages?: number;
}): PianoRollGlyphs | null {
    const pageSize = options.pageSizePx ?? PIANO_ROLL_ATLAS_PAGE_SIZE_PX;
    const rasterizer: GlyphRasterizer | null = createGlyphRasterizer({
        pageSizePx: pageSize,
        maxPages: options.maxPages ?? PIANO_ROLL_ATLAS_MAX_PAGES,
        dpr: options.dpr,
    });
    if (rasterizer === null) return null;

    const layout: GlyphLayout = createGlyphLayout((text, fontKey) =>
        rasterizer.measure(text, fontKey),
    );

    /** 槽位查询：同一 `(字符, 字体)` 复用；缺字形时返回 null 由管线跳过该字符。 */
    const slotFor = (char: string, fontKey: string): AtlasSlot | null =>
        rasterizer.acquire(char, fontKey);

    return {
        atlasPageSizePx: () => pageSize,
        consumeDirtyPages: () => rasterizer.consumeDirtyPages(),
        readPage: (page) => rasterizer.readPage(page),
        dispose: () => rasterizer.dispose(),

        build(requests) {
            const quads: GlyphQuad[] = [];
            for (const request of requests) {
                if (request.text.length === 0) continue;
                const fontSize = fontSizeOf(request.fontKey);
                const run: GlyphRun = layout.layout(request.text, request.fontKey, Infinity);
                if (run.glyphs.length === 0) continue;

                // 水平对齐：按**实际布局宽度**回退起点
                const align = request.align ?? "left";
                const originX =
                    align === "center"
                        ? request.x - run.width / 2
                        : align === "right"
                          ? request.x - run.width
                          : request.x;

                // 垂直基准 → 槽位顶部（见文件头的换算说明）。
                // 槽位顶部即 em 盒顶部，故 middle 需回退半个字号。
                const baseline = request.baseline ?? "middle";
                const originY = baseline === "top" ? request.y : request.y - fontSize / 2;

                quads.push(
                    ...buildGlyphQuads({
                        glyphs: run.glyphs,
                        originX,
                        originY,
                        // 契约：高度必须等于 字号 × 行高比例（见 glyphQuads 说明）
                        heightPx: fontSize * GLYPH_LINE_HEIGHT_RATIO,
                        atlasPageSizePx: pageSize,
                        resolveSlot: (char) => slotFor(char, request.fontKey),
                        rgba: request.rgba,
                    }),
                );
            }
            return quads;
        },
    };
}
