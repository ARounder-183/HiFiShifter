/**
 * 参数编辑器内核 · 钢琴键盘轴实例构建（纯函数）
 *
 * 【主要内容】
 * 把左侧键盘轴（白键 / 黑键 / 黑键右缘渐变 / 键分隔线）计算为**视口坐标**下的
 * 矩形实例，供 WebGL2 实例化渲染上传。
 *
 * 【作用】
 * 键盘轴是最大的静态图层之一（最多 60 个键 × 若干图元），而它**只依赖 pitchView**
 * ——播放与横向滚动都不会改变它。阶段 2 把它搬上 GL 后，播放帧不必再重绘这几百个
 * 图元。
 *
 * 【与 render.ts 的对应关系】
 * - 白键 / 黑键底色 → `render.ts:445-461`
 * - 黑键右缘渐变   → `render.ts:455-460`
 * - 键分隔线       → `render.ts:488-495`
 *
 * 【坐标约定】全部是**视口坐标**：键盘轴不参与横向滚动，`valueToY` 直接给出视口 y，
 * 因此 GL 侧的 `u_viewOrigin` 必须传 (0, 0)（与网格层同一约定）。
 *
 * 【与其他模块的关系】
 * - 上游：`pianoRollKernelHost` 在几何变化时调用。
 * - 下游：实例缓冲（`writeFlatInstance`）→ WebGL2。
 * - 独立性：纯函数，不依赖 DOM / WebGL / React。
 */

import type { Rgba } from "../../../renderKernel/instanceTypes";

/**
 * 一个键盘轴矩形实例。
 *
 * 特殊说明：`x` / `w` 是**绝对视口 x**（键盘轴自身从 0 起算，但渐变条从
 * `axisWidthPx * 0.62` 起），不是相对某个原点的偏移。
 */
export interface KeyboardInstance {
    readonly x: number;
    readonly y: number;
    readonly w: number;
    readonly h: number;
    readonly rgba: Rgba;
}

/** 键盘轴构建参数。 */
export interface KeyboardArgs {
    /** 当前音高视口（值域中心与跨度）。 */
    readonly view: { readonly center: number; readonly span: number };
    /** 绝对音高下界（MIDI）。 */
    readonly absMin: number;
    /** 绝对音高上界（MIDI）。 */
    readonly absMax: number;
    /** 视口高度（CSS px）。 */
    readonly heightPx: number;
    /** 键盘轴宽度（CSS px，= `AXIS_W`）。 */
    readonly axisWidthPx: number;
    /** 设备像素比。 */
    readonly dpr: number;
    /** 音高值 → 视口 y 的投影。 */
    readonly valueToY: (midi: number, heightPx: number) => number;
    /** 是否为黑键。 */
    readonly isBlackKey: (midi: number) => boolean;
    /** 白键底色。 */
    readonly whiteKeyRgba: Rgba;
    /** 黑键底色。 */
    readonly blackKeyRgba: Rgba;
    /** 黑键右缘渐变的深色端（不透明度由调用方按 step 递减）。 */
    readonly blackKeyGradientRgba: Rgba;
    /** C 键分隔线（强）。 */
    readonly cSeparatorRgba: Rgba;
    /** 其余键分隔线（弱）。 */
    readonly keySeparatorRgba: Rgba;
    /** 键盘轴右缘分隔线（Canvas2D 路径里先画、随后被琴键盖住）。 */
    readonly axisBorderRgba: Rgba;
}

/** 黑键宽度占轴宽的比例（与 `render.ts:454` 一致）。 */
const BLACK_KEY_WIDTH_RATIO = 0.72;

/** 黑键右缘渐变的起点比例（与 `render.ts:456` 一致）。 */
const GRADIENT_START_RATIO = 0.62;

/**
 * 生成黑键右缘渐变的矩形实例（**按设备像素列**精确取覆盖率）。
 *
 * 【为什么不用固定段数近似】渐变只有 5.6 CSS px 宽（`0.62w → 0.72w`，w=56）。
 * 最初用 8 段等宽矩形、按段中点取 alpha，实测**右边界**偏差达 60 个色阶：
 * Canvas2D 的 `fillRect` 对边界做覆盖率抗锯齿（末列只被部分覆盖，alpha 按比例
 * 衰减），而 GL 矩形是硬边、该列拿到整段 alpha。
 *
 * 改为**每个设备像素列一个矩形**，alpha 取该列与渐变区交集内的线性均值——因为
 * alpha 沿 x 线性，交集内的均值恰等于交集中点处的值，故这是精确解而非近似。
 * 代价可接受：5.6 CSS px 在 dpr=2 下约 12 个矩形，每个黑键至多 12 个实例。
 *
 * @param args 渐变几何与颜色。
 * @returns 覆盖渐变的矩形实例；区间非法时为空数组。
 */
function buildGradientInstances(args: {
    readonly startX: number;
    readonly endX: number;
    readonly top: number;
    readonly heightPx: number;
    readonly peakRgba: Rgba;
    readonly dpr: number;
}): KeyboardInstance[] {
    const { startX, endX, top, heightPx, peakRgba, dpr } = args;
    if (!Number.isFinite(startX) || !Number.isFinite(endX) || endX <= startX) return [];
    if (!(heightPx > 0)) return [];

    const deviceStart = startX * dpr;
    const deviceEnd = endX * dpr;
    const firstColumn = Math.floor(deviceStart);
    const lastColumn = Math.ceil(deviceEnd) - 1;
    const deviceSpan = deviceEnd - deviceStart;

    const out: KeyboardInstance[] = [];
    for (let column = firstColumn; column <= lastColumn; column += 1) {
        const overlapStart = Math.max(column, deviceStart);
        const overlapEnd = Math.min(column + 1, deviceEnd);
        // 退化列（浮点残差产生的零覆盖）不可见，且会污染调用方的结构判定。
        if (overlapEnd - overlapStart <= 1e-9) continue;
        // alpha 沿 x 线性：交集内的均值 == 交集中点处的值（精确，非近似）。
        const midpoint = (overlapStart + overlapEnd) / 2;
        const t = (midpoint - deviceStart) / deviceSpan;
        // **再乘该列的覆盖率**：边界列只被渐变区部分覆盖（例如黑键右缘列只覆盖
        // 64%），而 GL 矩形是硬边、会整列着色。Canvas2D 的 fillRect 会按覆盖率
        // 衰减 alpha——不乘这一项，边界列会偏暗（实测该列 94 vs 目标 104）。
        const coverage = (overlapEnd - overlapStart) / 1;
        out.push({
            x: column / dpr,
            y: top,
            w: 1 / dpr,
            h: heightPx,
            rgba: [peakRgba[0], peakRgba[1], peakRgba[2], peakRgba[3] * t * coverage],
        });
    }
    return out;
}

/**
 * 构建键盘轴矩形实例。
 *
 * 流程：钳制值域 → 求可见半音区间 `[floor(min), ceil(max))`（**不含右端点**，
 * 与 `render.ts:435` 的 `< endMidi` 一致）→ 逐键产出白/黑键底色、黑键渐变分段与
 * 键分隔线。
 *
 * 特殊说明 1：分隔线的 y 用 `top + 0.5`（**0.5 个 CSS 像素**，不是半个设备像素），
 * 与 `render.ts:492` 逐字一致。这是 dpr=1 时代的整数对齐写法，在分数 DPR 下并非
 * 严格的物理像素对齐——但它就是现有行为，迁移必须原样保留（"顺手修正"会改变像素）。
 *
 * 特殊说明 2：键高有 `Math.max(1, …)` 下限（`render.ts:440`）。极小的键仍要保留
 * 1px 可见体，否则缩放到底部时键盘会出现空洞。
 *
 * 特殊说明 3：分隔线用**矩形**逼近 `stroke`。Canvas2D 的 `lineWidth = 1` 以路径为
 * 中心，覆盖 `[y-0.5, y+0.5]`；`lineWidth = 0.5` 覆盖 `[y-0.25, y+0.25]`。矩形
 * 用上缘语义，故上缘 = 中心 − 半宽。这与网格层处理的是同一个语义差。
 *
 * @param args 构建参数。
 * @returns 实例序列（绘制顺序：底色 → 渐变 → 分隔线）；参数非法时返回空数组。
 */
export function buildKeyboardInstances(args: KeyboardArgs): KeyboardInstance[] {
    const { view, absMin, absMax, heightPx, axisWidthPx, dpr, valueToY, isBlackKey } = args;
    if (!Number.isFinite(view.span) || !Number.isFinite(view.center)) return [];
    if (!Number.isFinite(heightPx) || !Number.isFinite(axisWidthPx)) return [];
    if (!Number.isFinite(dpr) || dpr <= 0) return [];

    const range = absMax - absMin;
    if (!Number.isFinite(range) || range <= 0) return [];

    const span = Math.min(Math.max(view.span, 1e-6), range);
    const min = Math.min(Math.max(view.center - span / 2, absMin), absMax - span);
    const max = min + span;
    // 与 render.ts:434-435 一致：闭区间起点、开区间终点。
    const startMidi = Math.min(Math.max(Math.floor(min), absMin), absMax);
    const endMidi = Math.min(Math.max(Math.ceil(max), absMin), absMax);

    const items: KeyboardInstance[] = [];
    // 右缘分隔线**必须最先画**（在琴键之下）：Canvas2D 路径就是先 stroke 这条线、
    // 再用不透明的白/黑键把它盖住，因此 pitch 参数下它实际不可见。若由上层画布
    // 单独绘制，它会浮在琴键之上——实测浅色主题右缘出现一条 229 的竖线。
    // 也不能省略：非音高参数（无琴键）时它就是唯一可见的轴线。
    items.push(
        ...buildCoverageLine({
            centerY: axisWidthPx - 0.5,
            lineWidth: 1,
            rgba: args.axisBorderRgba,
            widthPx: axisWidthPx,
            dpr,
            horizontal: false,
            lengthPx: heightPx,
        }),
    );
    const gradientStartX = axisWidthPx * GRADIENT_START_RATIO;
    const gradientWidth = axisWidthPx * (BLACK_KEY_WIDTH_RATIO - GRADIENT_START_RATIO);

    for (let midi = startMidi; midi < endMidi; midi += 1) {
        const y0 = valueToY(midi, heightPx);
        const y1 = valueToY(midi + 1, heightPx);
        const top = Math.min(y0, y1);
        const bottom = Math.max(y0, y1);
        const keyH = Math.max(1, bottom - top);
        const black = isBlackKey(midi);

        if (!black) {
            // 白键铺满轴宽：轴宽通常是整数、dpr 为整数时边界恰在设备像素上，
            // 不产生边界列（buildSpanRects 会退化为单矩形）。
            items.push(
                ...buildRectInstances({
                    x0: 0,
                    x1: axisWidthPx,
                    y0: top,
                    y1: top + keyH,
                    rgba: args.whiteKeyRgba,
                    dpr,
                }),
            );
        } else {
            items.push(
                ...buildRectInstances({
                    x0: 0,
                    x1: axisWidthPx * BLACK_KEY_WIDTH_RATIO,
                    y0: top,
                    y1: top + keyH,
                    rgba: args.blackKeyRgba,
                    dpr,
                }),
            );
            // 右缘渐变：按设备像素列精确取覆盖率（见 buildGradientInstances）。
            items.push(
                ...buildGradientInstances({
                    startX: gradientStartX,
                    endX: gradientStartX + gradientWidth,
                    top,
                    heightPx: keyH,
                    peakRgba: args.blackKeyGradientRgba,
                    dpr,
                }),
            );
        }

        // 分隔线：C 用 1px 强线，其余 0.5px 弱线（render.ts:489-490）。
        const isC = ((midi % 12) + 12) % 12 === 0;
        const lineWidth = isC ? 1 : 0.5;
        // stroke 以中心对齐：上缘 = (top + 0.5) − lineWidth/2。
        const centerY = top + 0.5;
        const rgba = isC ? args.cSeparatorRgba : args.keySeparatorRgba;
        items.push(
            ...buildCoverageLine({
                centerY,
                lineWidth,
                rgba,
                widthPx: axisWidthPx,
                dpr,
            }),
        );
    }
    return items;
}

/**
 * 把一条**亚像素宽**的描边线展开为「整数设备像素矩形 + 覆盖率 alpha」。
 *
 * 【为什么必须这样做】Canvas2D 的 `stroke` 对非整数设备宽度的线做**覆盖率抗锯齿**：
 * 一条 0.5 CSS px（= 1 设备 px）宽、中心落在 `k + 0.5` 设备像素的线，会跨越
 * 两个设备行、每行各覆盖 50%，于是被摊成**两个半强度行**。
 * 而 GL 的 `INSTANCE_MODE_FLAT` 是硬边矩形、边缘不做抗锯齿——直接把 0.5 CSS px
 * 当高度画，只会落在一个设备行上、且是全强度。实测差异：Canvas2D 在设备行
 * 706/707 各得 alpha 0.10，GL 只在 707 得满 alpha 0.20。
 *
 * 解决办法是**显式复现覆盖率**：把线按设备像素栅格切开，每个被覆盖的设备行一个
 * 矩形，alpha 乘以该行的覆盖率。这样既不依赖着色器抗锯齿，也与 Canvas2D 的
 * 混合结果逐像素一致。
 *
 * 特殊说明 1：先按 `lineWidth <= 1` 判定是否需要展开。宽度 >= 1 CSS px 且对齐到
 * 整数设备像素时，覆盖率恒为 1，展开会退化为原矩形（保留原路径以免无谓分配）。
 *
 * 特殊说明 2：覆盖率按"与设备行区间的交集长度"计算，然后 alpha 乘以覆盖率。
 * 这是 Canvas2D 的实际模型（源覆盖 × 源 alpha），不是近似。
 *
 * @param args 线的几何与颜色。
 * @returns 覆盖该线的矩形实例（可能是 1 个或多个）；无有效覆盖时为空数组。
 */
function buildCoverageLine(args: {
    readonly centerY: number;
    readonly lineWidth: number;
    readonly rgba: Rgba;
    readonly widthPx: number;
    readonly dpr: number;
    /**
     * 线方向：`true`（缺省）为横线，`false` 为竖线。
     *
     * 特殊说明：竖线用于键盘轴右缘——它的"覆盖展开"发生在 **x 方向**，长度则沿
     * y 铺满整列。两种方向的展开逻辑相同，只是交换轴。
     */
    readonly horizontal?: boolean;
    /** 竖线的长度（CSS px，即轴列高）；横线时忽略。 */
    readonly lengthPx?: number;
}): KeyboardInstance[] {
    const { centerY, lineWidth, rgba, widthPx, dpr, horizontal = true } = args;
    if (!(lineWidth > 0) || !Number.isFinite(centerY)) return [];

    const deviceStart = (centerY - lineWidth / 2) * dpr;
    const deviceEnd = (centerY + lineWidth / 2) * dpr;
    const firstRow = Math.floor(deviceStart);
    const lastRow = Math.ceil(deviceEnd) - 1;

    const out: KeyboardInstance[] = [];
    for (let row = firstRow; row <= lastRow; row += 1) {
        // 与设备行 [row, row+1) 的交集长度即覆盖率。
        const coverage = Math.min(row + 1, deviceEnd) - Math.max(row, deviceStart);
        if (coverage <= 0) continue;
        out.push(
            horizontal
                ? {
                      x: 0,
                      y: row / dpr,
                      w: widthPx,
                      h: 1 / dpr,
                      rgba: [rgba[0], rgba[1], rgba[2], rgba[3] * coverage],
                  }
                : {
                      x: row / dpr,
                      y: 0,
                      w: 1 / dpr,
                      h: args.lengthPx ?? widthPx,
                      rgba: [rgba[0], rgba[1], rgba[2], rgba[3] * coverage],
                  },
        );
    }
    return out;
}

/**
 * 生成一个水平区间 `[x0, x1]` 的矩形实例，并在**边界列**上做覆盖率修正。
 *
 * 【为什么需要】GL 的 `INSTANCE_MODE_FLAT` 是硬边矩形、不做抗锯齿：当区间边界
 * 落在设备像素内部时（例如黑键宽 `0.72 × 56 = 40.32` CSS px，在 dpr=2 下右缘落在
 * 设备列 80 内部 64% 处），该列会被**整列**着色，而 Canvas2D 会按覆盖率只着 64%
 * 的色。实测这一列偏差达 60 个色阶（黑键右缘列：Canvas2D 104、GL 44）。
 *
 * 解决办法与分隔线一致：把边界列单独拆出来、alpha 乘以覆盖率。为控制实例数，
 * 只拆**边界**两列，中间部分仍用一个矩形（黑键整体因此至多 3 个矩形）。
 *
 * 特殊说明：区间边界恰好落在整数设备像素上时（例如白键 `[0, 56]` 在 dpr=2 下为
 * `[0, 112]`），不产生额外的边界行，输出退化为单个矩形——白键不受影响。
 *
 * @param args 区间、纵向范围与颜色。
 * @returns 覆盖该区间的矩形实例；区间非法时为空数组。
 */
function buildSpanRects(args: {
    readonly x0: number;
    readonly x1: number;
    readonly y: number;
    readonly heightPx: number;
    readonly rgba: Rgba;
    readonly dpr: number;
}): KeyboardInstance[] {
    const { x0, x1, y, heightPx, rgba, dpr } = args;
    if (!Number.isFinite(x0) || !Number.isFinite(x1) || x1 <= x0) return [];
    if (!(heightPx > 0)) return [];

    const deviceStart = x0 * dpr;
    const deviceEnd = x1 * dpr;
    const wholeStart = Math.ceil(deviceStart - 1e-9);
    const wholeEnd = Math.floor(deviceEnd + 1e-9);
    const out: KeyboardInstance[] = [];
    const tone = (coverage: number): Rgba => [rgba[0], rgba[1], rgba[2], rgba[3] * coverage];

    // 左边界列（仅当区间起点不在整数设备像素上，且覆盖率**显著大于 0**）
    const EPS = 1e-9;
    if (wholeStart - deviceStart > EPS) {
        const coverage = Math.min(wholeStart, deviceEnd) - deviceStart;
        if (coverage > 0) {
            out.push({
                x: Math.floor(deviceStart) / dpr,
                y,
                w: 1 / dpr,
                h: heightPx,
                rgba: tone(coverage),
            });
        }
    }
    // 中间整列部分
    if (wholeEnd - wholeStart > EPS) {
        out.push({
            x: wholeStart / dpr,
            y,
            w: (wholeEnd - wholeStart) / dpr,
            h: heightPx,
            rgba,
        });
    }
    // 右边界列
    if (deviceEnd - wholeEnd > EPS) {
        const coverage = deviceEnd - Math.max(wholeEnd, deviceStart);
        if (coverage > 0) {
            out.push({
                x: wholeEnd / dpr,
                y,
                w: 1 / dpr,
                h: heightPx,
                rgba: tone(coverage),
            });
        }
    }
    return out;
}

/**
 * 生成一个矩形实例，并在**纵向（y）边界**上做覆盖率修正。
 *
 * 【为什么还需要纵向修正】横向修正解决了 x 方向的部分覆盖，但键体的上下边界同样
 * 会落在设备行内部：实测键 82 的 `top = 34.2917`（dpr=2 -> 设备 68.583），其
 * 上边界行 68 只被覆盖 41.7%。Canvas2D 的 `fillRect` 对该行按覆盖率混合，
 * 而 GL 硬边矩形会整行着色——表现为**每个八度的 C 分隔线上方少一行**
 * （实测 Canvas2D 有 718 行分隔线像素、GL 只有 708 行，恰好少 10 行）。
 *
 * 解决办法同横向：把上下边界行单独拆出、alpha 乘以覆盖率，中间整行部分用一个
 * 矩形。因此一个键体最多被拆成 3×3 = 9 个矩形（横向 3 × 纵向 3），
 * 但只有边界落在设备行内部时才产生额外的行/列。
 *
 * @param args 矩形范围、颜色与 dpr。
 * @returns 覆盖该矩形的实例；参数非法或区间为空时返回空数组。
 */
function buildRectInstances(args: {
    readonly x0: number;
    readonly x1: number;
    readonly y0: number;
    readonly y1: number;
    readonly rgba: Rgba;
    readonly dpr: number;
}): KeyboardInstance[] {
    const { x0, x1, y0, y1, rgba, dpr } = args;
    if (!Number.isFinite(x0) || !Number.isFinite(x1) || x1 <= x0) return [];
    if (!Number.isFinite(y0) || !Number.isFinite(y1) || y1 <= y0) return [];

    // 纵向：拆成 [起始边界行] + [整行部分] + [结束边界行]
    const devY0 = y0 * dpr;
    const devY1 = y1 * dpr;
    const wholeY0 = Math.ceil(devY0 - 1e-9);
    const wholeY1 = Math.floor(devY1 + 1e-9);
    const bands: { y0: number; y1: number; coverage: number }[] = [];
    // 边界带必须**有正的覆盖率**才产出：浮点残差会造出 coverage ≈ 0（甚至 1e-16）
    // 的退化带。这种带不可见，却会污染调用方对"每个键由哪些矩形组成"的判定
    // （实测在单测聚类里表现为"键数从 60 变成 67"）。
    const EPS = 1e-9;
    if (wholeY0 - devY0 > EPS) {
        bands.push({ y0: devY0, y1: wholeY0, coverage: wholeY0 - devY0 });
    }
    if (wholeY1 - wholeY0 > EPS) {
        bands.push({ y0: wholeY0, y1: wholeY1, coverage: 1 });
    }
    if (devY1 - wholeY1 > EPS) {
        bands.push({ y0: wholeY1, y1: devY1, coverage: devY1 - wholeY1 });
    }

    const out: KeyboardInstance[] = [];
    for (const band of bands) {
        // 【关键】部分覆盖的边界带必须**占满一个设备行**、用 alpha 表达覆盖率。
        //
        // 原因：GL 按**像素中心采样**光栅化，没有面积抗锯齿。若边界带比一个设备行还
        // 窄（实测黑键的上边界带只有 0.417 个设备行高），且不含任何像素中心，它会被
        // **整条丢弃**——表现为该行少了一层墨。Canvas2D 的 `fillRect` 用的是面积
        // 覆盖率，这层墨会按 41.7% 混进去。
        //
        // 因此把边界带对齐到它所落入的那个设备行、alpha 乘以覆盖率：既保证像素中心
        // 必然被覆盖，又让最终混合结果与 Canvas2D 的面积覆盖率一致。
        const isPartial = band.coverage < 1;
        const rowStart = Math.floor(band.y0 + 1e-9);
        const yTop = isPartial ? rowStart / dpr : band.y0 / dpr;
        const heightPx = isPartial ? 1 / dpr : (band.y1 - band.y0) / dpr;

        // 横向：复用同一套拆分逻辑（buildSpanRects），再把纵向覆盖率乘到 alpha 上。
        const rowRects = buildSpanRects({
            x0,
            x1,
            y: yTop,
            heightPx,
            rgba,
            dpr,
        });
        for (const rect of rowRects) {
            out.push(
                band.coverage === 1
                    ? rect
                    : {
                          ...rect,
                          rgba: [
                              rect.rgba[0],
                              rect.rgba[1],
                              rect.rgba[2],
                              rect.rgba[3] * band.coverage,
                          ],
                      },
            );
        }
    }
    return out;
}
