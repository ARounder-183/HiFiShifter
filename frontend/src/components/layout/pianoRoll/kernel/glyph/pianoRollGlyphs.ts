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
