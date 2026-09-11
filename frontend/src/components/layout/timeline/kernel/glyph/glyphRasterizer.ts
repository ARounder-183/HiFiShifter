/**
 * 时间轴渲染内核 · 字形光栅化（离屏 Canvas2D → 图集）
 *
 * 【主要内容】
 * 用离屏 Canvas2D 把单个字符按 `dpr` 放大绘制到图集的指定槽位，并提供文本测量
 * （供 `glyphLayout` 使用）与页像素读取（供纹理上传）。字形以**白色**绘制，
 * 颜色由渲染端实例决定（见 `gl/glyphProgram` 的文件头约束 2）。
 *
 * 【作用】
 * 单 WebGL2 渲染器没有 `fillText`，文字质量只能靠「Canvas2D 光栅化 + 纹理采样」
 * 保住：把浏览器已验证的字形渲染结果搬进图集，从而与既有实现的文字观感一致。
 *
 * 【与其他模块的关系】
 * - 上游：内核渲染器在装配阶段创建本实例；`glyph/glyphLayout` 用 `measure` 注入测量。
 * - 下游：`gl/glyphProgram.uploadAtlas` 消费 `readPage` 的像素数据。
 * - 依赖：`glyph/glyphAtlas` 提供槽位分配。
 *
 * 【设计约束】
 * 1. **按 dpr 光栅化**：dpr 变化后必须整体重建（字形位图与物理像素一一对应，
 *    换 dpr 后旧图集会被采样成模糊字形）。调用方负责在 dpr 变化时丢弃本实例。
 * 2. 缓存 key 为 `(字符, 字体)`，与 `glyphLayout` 的测量缓存键空间一致。
 * 3. 图集满（`allocate` 返回 null）时 `acquire` 返回 null，调用方应跳过该字形
 *    并停止继续申请——本模块不做淘汰（淘汰会破坏已上传的 uv）。
 */

import { createGlyphAtlas, type AtlasSlot } from "./glyphAtlas";

/**
 * 行高系数：槽位高度 = 字号 × 该系数。
 *
 * 取 1.2（常规排版行高）：字形上下的升部 / 降部都在槽位内，避免被截断。
 */
const LINE_HEIGHT_RATIO = 1.2;

/** 光栅化器构造参数。 */
export interface GlyphRasterizerOptions {
    /** 单页边长（物理像素）。 */
    pageSizePx: number;
    /** 最大页数。 */
    maxPages: number;
    /** 设备像素比（字形按此放大光栅化）。 */
    dpr: number;
}

/** 字形光栅化器。 */
export interface GlyphRasterizer {
    /**
     * 测量文本宽度（CSS px）。
     *
     * @param text 文本。
     * @param fontKey 字体标识（`"<size>px <family>"`）。
     * @returns 宽度（CSS px）；无 DOM 环境下由调用方保证不会走到这里。
     */
    measure(text: string, fontKey: string): number;
    /**
     * 取得字形槽位（不存在则光栅化并写入图集）。
     *
     * @param char 单个字符。
     * @param fontKey 字体标识。
     * @returns 槽位；图集已满时为 null。
     */
    acquire(char: string, fontKey: string): AtlasSlot | null;
    /**
     * 读取指定页的像素数据（RGBA）。
     *
     * @param page 页索引。
     * @returns 像素数据；页不存在时为 null。
     */
    readPage(page: number): Uint8ClampedArray | null;
    /** 页边长（物理像素）。 */
    pageSizePx(): number;
    /** 消费自上次调用以来的脏页索引（用于按页增量上传纹理）。 */
    consumeDirtyPages(): number[];
    /** 释放离屏画布（置空尺寸，让浏览器尽快回收）。 */
    dispose(): void;
}

/**
 * 解析字体标识中的字号（CSS px）。
 *
 * @param fontKey 字体标识（`"<size>px <family>"`）。
 * @returns 字号；解析失败回退 12。
 */
function parseFontSizePx(fontKey: string): number {
    const match = /^(\d+(?:\.\d+)?)px/.exec(fontKey.trim());
    return match ? Number(match[1]) : 12;
}

/**
 * 按比例缩放字体标识中的字号。
 *
 * @param fontKey 字体标识。
 * @param scale 缩放比例（通常为 dpr）。
 * @returns 缩放后的字体标识；解析失败时原样返回。
 */
function scaleFontKey(fontKey: string, scale: number): string {
    const match = /^(\d+(?:\.\d+)?)px\s+(.*)$/.exec(fontKey.trim());
    if (!match) return fontKey;
    return `${Number(match[1]) * scale}px ${match[2]}`;
}

/**
 * 创建字形光栅化器。
 *
 * 流程：建图集分配器与测量画布 → `acquire` 时按 `(字符, 字体)` 查缓存，
 * 未命中则申请槽位、按 dpr 光栅化到对应页并标记脏页。
 *
 * 特殊说明：无 DOM 环境（node 单测）返回 null——调用方据此跳过文字渲染而不是崩溃。
 *
 * @param options 构造参数。
 * @returns 光栅化器；无 DOM 时为 null。
 */
export function createGlyphRasterizer(options: GlyphRasterizerOptions): GlyphRasterizer | null {
    if (typeof document === "undefined") return null;

    const pageSize = Math.max(16, Math.floor(options.pageSizePx));
    const dpr = Number.isFinite(options.dpr) && options.dpr > 0 ? options.dpr : 1;
    const atlas = createGlyphAtlas({
        pageSizePx: pageSize,
        paddingPx: 1,
        maxPages: options.maxPages,
    });

    const pageContexts: CanvasRenderingContext2D[] = [];
    const dirtyPages = new Set<number>();
    const slotsByFont = new Map<string, Map<string, AtlasSlot>>();

    const measureCanvas = document.createElement("canvas");
    measureCanvas.width = 1;
    measureCanvas.height = 1;
    const measureCtxRaw = measureCanvas.getContext("2d");
    if (!measureCtxRaw) return null;
    // 绑定到非空常量：TS 的 null 收窄不会穿透到下方嵌套函数，显式定型避免重复判空。
    const measureCtx: CanvasRenderingContext2D = measureCtxRaw;

    /**
     * 确保指定页存在并返回其 2D 上下文。
     *
     * @param page 页索引。
     * @returns 上下文；创建失败时为 null。
     */
    function ensurePage(page: number): CanvasRenderingContext2D | null {
        while (pageContexts.length <= page) {
            const canvas = document.createElement("canvas");
            canvas.width = pageSize;
            canvas.height = pageSize;
            const ctx = canvas.getContext("2d");
            if (!ctx) return null;
            pageContexts.push(ctx);
        }
        return pageContexts[page];
    }

    /** 测量文本宽度（CSS px）。 */
    function measure(text: string, fontKey: string): number {
        measureCtx.font = fontKey;
        return measureCtx.measureText(text).width;
    }

    return {
        measure,

        acquire(char, fontKey) {
            let byChar = slotsByFont.get(fontKey);
            if (byChar === undefined) {
                byChar = new Map<string, AtlasSlot>();
                slotsByFont.set(fontKey, byChar);
            }
            const cached = byChar.get(char);
            if (cached !== undefined) return cached;

            const fontSizeCss = parseFontSizePx(fontKey);
            const slotWidth = Math.max(1, Math.ceil(measure(char, fontKey) * dpr));
            const slotHeight = Math.max(1, Math.ceil(fontSizeCss * LINE_HEIGHT_RATIO * dpr));
            const slot = atlas.allocate(slotWidth, slotHeight);
            if (slot === null) return null;

            const ctx = ensurePage(slot.page);
            if (ctx === null) return null;
            ctx.font = scaleFontKey(fontKey, dpr);
            // 基线取 top：槽位坐标即字形左上角，与四边形的位置语义一致。
            ctx.textBaseline = "top";
            ctx.fillStyle = "#ffffff";
            ctx.fillText(char, slot.x, slot.y);

            byChar.set(char, slot);
            dirtyPages.add(slot.page);
            return slot;
        },

        readPage(page) {
            const ctx = pageContexts[page];
            if (ctx === undefined) return null;
            return ctx.getImageData(0, 0, pageSize, pageSize).data;
        },

        pageSizePx() {
            return pageSize;
        },

        consumeDirtyPages() {
            const pages = Array.from(dirtyPages);
            dirtyPages.clear();
            return pages;
        },

        dispose() {
            for (const ctx of pageContexts) {
                const canvas = ctx.canvas;
                canvas.width = 0;
                canvas.height = 0;
            }
            pageContexts.length = 0;
            slotsByFont.clear();
            dirtyPages.clear();
        },
    };
}
