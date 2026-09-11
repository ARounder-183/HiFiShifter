/**
 * 时间轴渲染内核 · 字形图集分配器（shelf packing，多页）
 *
 * 【主要内容】
 * 把「宽 × 高」矩形装箱到一页或多页固定边长的纹理空间中，返回可用的页索引与
 * 矩形位置。采用 shelf packing（货架装箱）：每页维护若干条等高货架，货架内沿 x
 * 依次放置；当前货架放不下就起新货架；整页放不下就开新页。
 *
 * 【作用】
 * 单 WebGL2 渲染器没有 `fillText`，字形位图必须预渲染到纹理图集再由四边形采样。
 * 图集分配是纯装箱问题，与 WebGL / Canvas 无关，因此单独成模块以便完整单测。
 *
 * 【与其他模块的关系】
 * - 上游：`glyphRasterizer` 在需要某个 `(字符, 字体)` 的位图时，先向本模块申请槽位，
 *   再把位图写入对应页的纹理区域。
 * - 下游：`glyphProgram` 用槽位算出纹理坐标（uv）组装实例。
 * - 独立性：纯逻辑，不依赖 DOM / WebGL / React；单位由调用方统一（本模块只做算术）。
 *
 * 【设计约束】
 * 1. **不做淘汰**：本模块只负责装箱；`(字符, 字体) → 槽位` 的映射与 LRU 淘汰由
 *    调用方（字形缓存）负责——装箱算法与缓存策略分离，便于各自单测。
 * 2. 槽位之间保留 `paddingPx` 间隔：相邻字形若紧贴，纹理双线性采样会在边缘串色。
 * 3. 分配是**只增不减**的：已分配的槽位在本模块生命周期内保持有效（调用方换页
 *    或重建图集时必须整体丢弃本实例）。
 */

/**
 * 图集构造参数。
 *
 * 特殊说明：`pageSizePx` 指**单页可用的正方形边长**；调用方需保证它不超过
 * `MAX_TEXTURE_SIZE`（WebGL 上限，常见 2048 / 4096 / 8192）。
 */
export interface GlyphAtlasOptions {
    /** 单页边长（与调用方的像素单位一致）。 */
    pageSizePx: number;
    /** 槽位之间的最小间隔，防止纹理采样串色。 */
    paddingPx: number;
    /** 最大页数；超过后 `allocate` 返回 null，由调用方决定降级策略。 */
    maxPages: number;
}

/** 一次分配得到的槽位。 */
export interface AtlasSlot {
    /** 页索引（0 起），对应调用方持有的第几张纹理。 */
    readonly page: number;
    /** 槽位左上角 x（页内坐标）。 */
    readonly x: number;
    /** 槽位左上角 y（页内坐标）。 */
    readonly y: number;
    /** 槽位宽度（等于请求宽度，不含 padding）。 */
    readonly w: number;
    /** 槽位高度（等于请求高度，不含 padding）。 */
    readonly h: number;
}

/** 字形图集分配器。 */
export interface GlyphAtlas {
    /**
     * 申请一个矩形槽位。
     *
     * 流程：非法尺寸 / 超出单页边长直接拒绝 → 逐页尝试（当前货架 → 新货架）→
     * 全部失败则开新页（未超 `maxPages` 时）。
     *
     * @param w 请求宽度（> 0）。
     * @param h 请求高度（> 0）。
     * @returns 槽位；无法放下（尺寸非法、超出单页、或页数已达上限）时返回 null。
     */
    allocate(w: number, h: number): AtlasSlot | null;
    /** 已创建的页数（含尚未写入任何槽位的空页，正常情况下不会出现空页）。 */
    pageCount(): number;
}

/** 一条货架：等高矩形沿 x 依次放置。 */
interface Shelf {
    /** 货架顶边 y。 */
    y: number;
    /** 货架高度（由首个放入的矩形决定）。 */
    height: number;
    /** 下一个可用 x（含 padding 后的游标）。 */
    cursorX: number;
}

/** 一页的内部状态。 */
interface Page {
    shelves: Shelf[];
    /** 下一条新货架的 y（= 已有货架底部 + padding）。 */
    nextShelfY: number;
}

/**
 * 读取一个正整数参数，非法值回退到默认。
 *
 * @param value 候选值。
 * @param fallback 回退值（调用方保证合法）。
 * @returns 有限且 >= 1 的整数。
 */
function resolvePositiveInt(value: number, fallback: number): number {
    return Number.isFinite(value) && value >= 1 ? Math.floor(value) : fallback;
}

/**
 * 创建图集分配器。
 *
 * 流程：解析参数（非法值回退）→ 惰性建页（首次 allocate 才建第 0 页）→
 * 分配时先试当前页的当前货架，再试当前页新货架，最后尝试新页。
 *
 * 特殊说明：`paddingPx` 只加在槽位**之间**（横向由 `cursorX` 累加、纵向由
 * `nextShelfY` 累加），页边缘不预留——纹理坐标按实际槽位计算，无需边缘留白。
 *
 * @param options 构造参数。
 * @returns 分配器实例；分配只增不减，换页 / 重建需整体丢弃本实例。
 */
export function createGlyphAtlas(options: GlyphAtlasOptions): GlyphAtlas {
    const pageSize = resolvePositiveInt(options.pageSizePx, 1);
    const padding = Number.isFinite(options.paddingPx) && options.paddingPx > 0
        ? Math.floor(options.paddingPx)
        : 0;
    const maxPages = resolvePositiveInt(options.maxPages, 1);

    const pages: Page[] = [];

    /** 建一个空页。 */
    function createPage(): Page {
        return { shelves: [], nextShelfY: 0 };
    }

    /**
     * 尝试把矩形放进指定页。
     *
     * 流程：先试当前货架（高度足够且 x 未越界）→ 失败则尝试在页尾起新货架。
     *
     * @param page 目标页。
     * @param pageIndex 目标页索引（写入槽位）。
     * @param w 宽度。
     * @param h 高度。
     * @returns 槽位或 null（本页放不下）。
     */
    function tryAllocateInPage(page: Page, pageIndex: number, w: number, h: number): AtlasSlot | null {
        const current = page.shelves[page.shelves.length - 1];
        if (current !== undefined && current.height >= h && current.cursorX + w <= pageSize) {
            const slot: AtlasSlot = { page: pageIndex, x: current.cursorX, y: current.y, w, h };
            current.cursorX += w + padding;
            return slot;
        }
        // 当前货架放不下（高度不够或 x 越界）：在页尾起新货架。
        if (page.nextShelfY + h <= pageSize) {
            const shelf: Shelf = { y: page.nextShelfY, height: h, cursorX: w + padding };
            page.shelves.push(shelf);
            page.nextShelfY += h + padding;
            return { page: pageIndex, x: 0, y: shelf.y, w, h };
        }
        return null;
    }

    return {
        allocate(w, h) {
            if (!Number.isFinite(w) || !Number.isFinite(h) || w <= 0 || h <= 0) return null;
            const width = Math.ceil(w);
            const height = Math.ceil(h);
            // 单页都放不下的矩形不可能装箱成功，直接拒绝（避免无意义地开页）。
            if (width > pageSize || height > pageSize) return null;

            for (let index = 0; index < pages.length; index += 1) {
                const slot = tryAllocateInPage(pages[index], index, width, height);
                if (slot !== null) return slot;
            }

            if (pages.length >= maxPages) return null;
            const page = createPage();
            pages.push(page);
            return tryAllocateInPage(page, pages.length - 1, width, height);
        },

        pageCount() {
            return pages.length;
        },
    };
}
