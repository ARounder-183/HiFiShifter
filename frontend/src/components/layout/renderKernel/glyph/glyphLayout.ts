/**
 * 时间轴渲染内核 · 字形布局（文本切分与测量缓存）
 *
 * 【主要内容】
 * 把一段文本切分为「逐字符 + x 偏移 + 宽度」的字形序列，供 glyph-quad 实例化渲染
 * 使用；超宽时截断并追加省略号。测量结果按 `(字符, 字体)` 缓存，避免每帧对同一
 * 批文本重复调用 `measureText`。
 *
 * 【作用】
 * 单 WebGL2 渲染器没有 `fillText`，文字必须由字形四边形拼出——因此需要一个
 * 与 Canvas2D 排版结果等价的布局层：宽度来源仍是注入的 `measure`（生产环境为
 * 离屏 Canvas2D 的 `measureText`），从而保持与既有实现的文字宽度逐像素一致。
 *
 * 【与其他模块的关系】
 * - 上游：场景构建（clip 名称、增益/速率、徽标标签、标尺刻度、轨道头文字）提供文本；
 *   渲染器装配阶段注入 `measure`。
 * - 下游：`glyphProgram` 把字形序列 + 图集分配结果转成实例；`glyphRasterizer`
 *   按同一 `(字符, 字体)` 键缓存字形位图（两者键空间一致，便于共用失效策略）。
 * - 独立性：纯逻辑，不依赖 DOM / Canvas / React；测量函数由调用方注入。
 *
 * 【强制约束】
 * 1. 字符切分按 Unicode 码点（`for...of`），不处理字素簇组合（emoji ZWJ、变音符号）：
 *    clip 名称与 UI 文案不依赖组合字形，拆开渲染与浏览器排版差异可忽略（YAGNI）。
 * 2. 布局结果宽度恒 <= maxWidthPx；截断时末尾字形必为省略号（除非连省略号都放不下）。
 * 3. 缓存 key 为 `(字符, 字体)`，字体变化不会读到旧值；调用方可整体丢弃布局器
 *    实现失效（dpr 变化时字体 key 应包含 dpr，见 glyphRasterizer 的约定）。
 */

/** 截断标记。用单字符省略号（U+2026）而非三个点：宽度更小且与既有 UI 一致。 */
export const ELLIPSIS_CHAR = "…";

/**
 * 文本测量函数。
 *
 * 契约：返回文本在指定字体下的宽度（CSS px）；生产实现为离屏 Canvas2D 的
 * `measureText(text).width`，同一 `(text, fontKey)` 必须返回稳定值（内核按字符缓存）。
 */
export type TextMeasure = (text: string, fontKey: string) => number;

/** 一个已布局的字形。 */
export interface LayoutGlyph {
    /** 字符（截断时末位为 `ELLIPSIS_CHAR`）。 */
    readonly char: string;
    /** 相对文本起点的 x 偏移（CSS px）。 */
    readonly x: number;
    /** 字符前进宽度（CSS px）。 */
    readonly width: number;
}

/** 一段文本的布局结果。 */
export interface GlyphRun {
    /** 字形序列（按 x 递增）。 */
    readonly glyphs: readonly LayoutGlyph[];
    /** 实际占用宽度（CSS px），恒 <= maxWidthPx。 */
    readonly width: number;
    /** 是否发生截断（末尾为省略号）。 */
    readonly truncated: boolean;
}

/** 字形布局器。 */
export interface GlyphLayout {
    /**
     * 布局一段文本。
     *
     * @param text 原始文本。
     * @param fontKey 字体标识（生产环境为 `"<fontSize>px <family>"`，可含 dpr）。
     * @param maxWidthPx 可用宽度上限（CSS px）。`Infinity` 表示不限制宽度（完整布局）；
     *                   `NaN` / 负值 / `-Infinity` 按 0 处理（无可绘制空间）。
     * @returns 布局结果；宽度恒不超过上限。
     */
    layout(text: string, fontKey: string, maxWidthPx: number): GlyphRun;
    /** 当前缓存的 `(字符, 字体)` 测量条目数（诊断 / 测试用）。 */
    measureCacheSize(): number;
}

/**
 * 创建字形布局器。
 *
 * 流程：
 * 1. 建立 `fontKey → (char → width)` 两级缓存（避免字符串拼接 key 的开销）；
 * 2. `layout()` 逐字符累加宽度，超限即停止并回退到「省略号能放下」的位置；
 * 3. 截断时追加省略号；若连省略号都放不下（上限小于省略号宽度）则返回空序列。
 *
 * 特殊说明：测量结果做非负有限值归一化——脏字体 / 未加载字体会让 `measureText`
 * 返回 0 或 NaN，直接参与累加会产生 NaN 偏移，污染整批实例数据。
 *
 * @param measure 文本测量函数（生产注入离屏 Canvas2D 实现，测试注入桩）。
 * @returns 布局器实例（自带测量缓存，可长期持有）。
 */
export function createGlyphLayout(measure: TextMeasure): GlyphLayout {
    const measureCache = new Map<string, Map<string, number>>();

    /**
     * 测量单个字符（带缓存与归一化）。
     *
     * @param char 单个字符（Unicode 码点）。
     * @param fontKey 字体标识。
     * @returns 字符宽度（CSS px，有限且 >= 0）。
     */
    function measureChar(char: string, fontKey: string): number {
        let byChar = measureCache.get(fontKey);
        if (byChar === undefined) {
            byChar = new Map<string, number>();
            measureCache.set(fontKey, byChar);
        }
        const cached = byChar.get(char);
        if (cached !== undefined) return cached;
        const raw = measure(char, fontKey);
        const width = Number.isFinite(raw) && raw > 0 ? raw : 0;
        byChar.set(char, width);
        return width;
    }

    return {
        layout(text, fontKey, maxWidthPx) {
            // `Infinity` 是调用方表达「不限制宽度」的合法值（例如容器宽度尚未量出），
            // 必须走完整布局；只有 NaN / 负值 / -Infinity 才视为「无可绘制空间」。
            const limit =
                maxWidthPx === Number.POSITIVE_INFINITY
                    ? Number.POSITIVE_INFINITY
                    : Number.isFinite(maxWidthPx)
                      ? Math.max(0, maxWidthPx)
                      : 0;
            const glyphs: LayoutGlyph[] = [];
            let x = 0;
            let truncated = false;

            for (const char of text) {
                const width = measureChar(char, fontKey);
                if (x + width > limit) {
                    truncated = true;
                    break;
                }
                glyphs.push({ char, x, width });
                x += width;
            }

            if (truncated) {
                // 回退：把末尾字形逐个弹出，直到省略号能放下为止。
                const ellipsisWidth = measureChar(ELLIPSIS_CHAR, fontKey);
                while (glyphs.length > 0 && x + ellipsisWidth > limit) {
                    const removed = glyphs.pop();
                    if (removed === undefined) break;
                    x = removed.x;
                }
                // 上限小于省略号本身时不再强塞（宁可空白，也不越界绘制）。
                if (ellipsisWidth <= limit) {
                    glyphs.push({ char: ELLIPSIS_CHAR, x, width: ellipsisWidth });
                    x += ellipsisWidth;
                }
            }

            return { glyphs, width: x, truncated };
        },

        measureCacheSize() {
            let total = 0;
            for (const byChar of measureCache.values()) total += byChar.size;
            return total;
        },
    };
}
