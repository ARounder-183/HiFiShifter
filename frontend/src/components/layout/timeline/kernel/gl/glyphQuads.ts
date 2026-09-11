/**
 * 时间轴渲染内核 · 字形四边形构建
 *
 * 【主要内容】
 * 把「已布局的字形序列」（`glyphLayout` 的输出）与「图集槽位」组合成可实例化绘制的
 * 纹理四边形：内容坐标位置 + 归一化 uv + 颜色。
 *
 * 【作用】
 * 单 WebGL2 渲染器没有 `fillText`，文字必须由纹理四边形拼出。本模块是「排版结果 →
 * GPU 实例」的纯转换层：不查表、不分配图集，槽位查询由调用方注入（因此可在无 DOM
 * 环境单测）。
 *
 * 【与其他模块的关系】
 * - 上游：`glyph/glyphLayout` 产出字形序列；`glyph/glyphRasterizer` 提供槽位查询。
 * - 下游：`gl/glyphProgram` 把四边形写入实例缓冲并绘制。
 * - 独立性：纯函数，不依赖 DOM / WebGL。
 *
 * 【设计约束】
 * 1. uv 由槽位坐标除以**页边长**得到（图集坐标与纹理坐标的唯一换算点）。
 * 2. 槽位缺失（字形尚未光栅化 / 图集已满被淘汰）时**跳过该字形**：宁可少画一个字，
 *    也不要为它分配占位纹理（会导致采样到别的字形，出现乱码）。
 */

import type { AtlasSlot } from "../glyph/glyphAtlas";
import type { LayoutGlyph } from "../glyph/glyphLayout";
import type { Rgba } from "../scene/instanceTypes";

/** 一个字形四边形（内容坐标 + 归一化 uv）。 */
export interface GlyphQuad {
    /** 左缘 x（内容坐标 CSS px）。 */
    readonly x: number;
    /** 上缘 y（内容坐标 CSS px）。 */
    readonly y: number;
    /** 宽度（CSS px）。 */
    readonly w: number;
    /** 高度（CSS px）。 */
    readonly h: number;
    /** 纹理 u 起点（0..1）。 */
    readonly u0: number;
    /** 纹理 v 起点（0..1）。 */
    readonly v0: number;
    /** 纹理 u 终点（0..1）。 */
    readonly u1: number;
    /** 纹理 v 终点（0..1）。 */
    readonly v1: number;
    /** 颜色（含 alpha）。 */
    readonly rgba: Rgba;
}

/** 字形四边形构建参数。 */
export interface GlyphQuadArgs {
    /** 已布局的字形序列（相对文本起点的 x 偏移）。 */
    readonly glyphs: readonly LayoutGlyph[];
    /** 文本起点 x（内容坐标 CSS px）。 */
    readonly originX: number;
    /** 文本顶部 y（内容坐标 CSS px）。 */
    readonly originY: number;
    /**
     * 字形四边形高度（CSS px）。
     *
     * 特殊说明：**必须等于 `字号 × GLYPH_LINE_HEIGHT_RATIO`**（光栅化槽位高度按该
     * 系数生成，uv 覆盖整个槽位）。不等时字形会被纵向拉伸 / 压扁——观感是"发虚"
     * 而非"错位"，很难归因，因此这条等式是跨模块契约。
     */
    readonly heightPx: number;
    /** 图集单页边长（与槽位坐标同单位）。 */
    readonly atlasPageSizePx: number;
    /**
     * 槽位查询。
     *
     * @param char 字符。
     * @returns 槽位；返回 null 时跳过该字形（见文件头约束 2）。
     */
    readonly resolveSlot: (char: string) => AtlasSlot | null;
    /** 颜色。 */
    readonly rgba: Rgba;
}

/**
 * 构建字形四边形序列。
 *
 * 流程：逐字形查询槽位 → 缺失则跳过 → 组合位置（文本起点 + 字形 x 偏移）、尺寸
 * （字形宽度 × 统一高度）与 uv（槽位 ÷ 页边长）。
 *
 * 特殊说明：四边形宽度取**布局宽度**而非槽位宽度：布局宽度来自 `measureText`
 * （含字形左右边距），槽位宽度是光栅化时的取整结果，两者相差不到 1 物理像素；
 * 用布局宽度才能保证多个字形的间距与浏览器排版一致。
 *
 * @param args 构建参数。
 * @returns 字形四边形数组；顺序与输入一致（缺失字形被跳过）。
 */
export function buildGlyphQuads(args: GlyphQuadArgs): GlyphQuad[] {
    const pageSize =
        Number.isFinite(args.atlasPageSizePx) && args.atlasPageSizePx > 0
            ? args.atlasPageSizePx
            : 1;
    const out: GlyphQuad[] = [];
    for (const glyph of args.glyphs) {
        const slot = args.resolveSlot(glyph.char);
        if (slot === null) continue;
        out.push({
            x: args.originX + glyph.x,
            y: args.originY,
            w: glyph.width,
            h: args.heightPx,
            u0: slot.x / pageSize,
            v0: slot.y / pageSize,
            u1: (slot.x + slot.w) / pageSize,
            v1: (slot.y + slot.h) / pageSize,
            rgba: args.rgba,
        });
    }
    return out;
}
