/**
 * 参数编辑器内核 · 网格实例构建（纯函数）
 *
 * 【主要内容】
 * 把参数编辑器的横向网格线（音高半音线 / 各类非音高参数的刻度线）、**钢琴背景的
 * 黑键行背景带**与**音阶高亮强调线**计算为**内容坐标**下的矩形实例，供 WebGL2
 * 实例化渲染直接上传。
 *
 * 【作用】
 * 阶段 2 要把静态图层搬上 GL，前提是几何能脱离 Canvas 与 DOM 独立计算、独立测试。
 * 本模块只做算术：值域 → 视口 y → 设备像素对齐 → 实例矩形。它与 `render.ts`
 * 的 Canvas2D 路径**必须逐值等价**（包括两种不同的半像素取向），否则迁移后网格
 * 会整体偏移半个设备像素——表现为线条"发虚"，很难归因。
 *
 * 【实例顺序即合成顺序】`INSTANCE_MODE_FLAT` 按缓冲顺序合成，后写的盖住先写的。
 * 音高分支的顺序固定为「黑键背景带 → 网格线 → 音阶强调线」（见
 * `buildPitchGridInstances` 特殊说明 3）——顺序错了不会报错，只会画错。
 *
 * 【与 render.ts 的对应关系】
 * - 音高分支   → 原 `render.ts` 的音高网格分支，**已随该分支删除而成为唯一实现**
 *   （Canvas2D 侧 `skipGrid` 恒为 true，那段代码不可达）。网格线本身逐值等价于
 *   旧实现的行循环，等价性由 `gridInstances.test.ts` 末节的组合比对守护。
 * - cents 分支 → `render.ts:741-766`（原行号；该文件已因删除音高分支而整体前移）
 * - degrees 分支 → 原 `render.ts:767-793`
 * - formant 分支 → 原 `render.ts:794-821`
 * - 钢琴背景 / 音阶高亮 → **无 Canvas2D 对应物**（新增能力；旧 Canvas2D 高亮
 *   只有单音阶与 Tempo Map 分段两条路径，后者未迁移，见下）
 *
 * 【已知限制】音阶高亮只支持**单一音阶**：GL 网格几何是视口坐标、不含时间轴，
 * 无法表达 Tempo Map 的分段音阶。详见 `buildPitchGridInstances` 特殊说明 5。
 *
 * 【为什么要显式防御 NaN / Infinity】原实现的 span 由调用方保证有限，因此直接进
 * `for` 循环。本模块跑在渲染热路径上，一旦上界为 `Infinity` 会**死循环卡死面板**，
 * 故对非有限 span 直接返回空数组。
 *
 * 【与其他模块的关系】
 * - 上游：`pianoRollKernelHost` 在几何变化时调用。
 * - 依赖：`../../utils` 的 `isBlackKey`（无依赖叶子模块，node 单测可安全引用）。
 * - 下游：实例缓冲（`writeFlatInstance`）→ WebGL2。
 * - 独立性：纯函数，不依赖 DOM / WebGL / React，可在 node 环境单测。
 */

import type { Rgba } from "../../../renderKernel/instanceTypes";
import { isBlackKey } from "../../utils";

/** 非音高参数的网格种类（决定步进与强线间隔）。 */
export type ValueGridKind = "cents" | "degrees" | "formantCents";

/**
 * 一条横向网格线（或背景带）的实例几何。
 *
 * 特殊说明（易错点）：横线的 `w` 是**横向范围**（横跨整个视口），`h` 是**线厚**。
 * 两者不可互换——写反会画出一条通高的竖条，而不是一条线。
 */
export interface GridInstance {
    /** 左缘 x（内容坐标 CSS px）。横线与背景带都从视口左缘开始，故恒为 0。 */
    readonly x: number;
    /** 上缘 y（内容坐标 CSS px，已按该层的半像素取向对齐）。 */
    readonly y: number;
    /** 横向范围（CSS px，= 视口宽）。 */
    readonly w: number;
    /** 线厚（CSS px，弱线 1/dpr、强线 2/dpr）；背景带则是**整行的键高**。 */
    readonly h: number;
    /** 颜色。 */
    readonly rgba: Rgba;
    /**
     * 该实例对应的参数值（供比例高亮与调试）。
     *
     * 【特殊约定：负数 = 黑键行背景带】背景带**不是**网格线，它对应的是一个半音的
     * **行体**而不是行心的线。为了在同一个数组里区分两类实例（调用方要按顺序合成、
     * 测试要按类型筛选），背景带把它的 midi 编码为 `-(midi + 1)` —— 因为 midi 恒
     * ≥ 0，取负后必然 < 0；`+1` 是为了让 midi = 0 也不与「值 0 的网格线」混淆。
     * 网格线则原样存 midi（≥ 0）。**不得**把它当作背景带的"行心值"使用：
     * 行体的 y 范围是 `[valueToY(midi+1), valueToY(midi)]`，与行心
     * `valueToY(midi+0.5)` 差半个键高。
     */
    readonly value: number;
}

/** 音高网格构建参数。 */
export interface PitchGridArgs {
    /** 当前音高视口（值域中心与跨度）。 */
    readonly view: { readonly center: number; readonly span: number };
    /** 绝对值域下界（MIDI）。 */
    readonly absMin: number;
    /** 绝对值域上界（MIDI）。 */
    readonly absMax: number;
    /** 视口高度（CSS px）。 */
    readonly heightPx: number;
    /** 视口宽度（CSS px）——横线的横向范围。 */
    readonly viewportWidthPx: number;
    /** 设备像素比。 */
    readonly dpr: number;
    /** 值 → 视口 y 的投影（与 Canvas2D 路径同一个函数，保证等价）。 */
    readonly valueToY: (value: number, heightPx: number) => number;
    /** C 音（pc === 0）的线色。 */
    readonly colorC: Rgba;
    /** 其余半音的线色。 */
    readonly colorOther: Rgba;
    /**
     * 黑键行背景带的颜色（钢琴背景）。
     *
     * 特殊说明 1：**仅黑键行**（`isBlackKey`）产带，白键行保持原背景——这是
     * REAPER / Logic / Ableton 的既有惯例，也是浅色主题下唯一不会糊成一片的做法。
     *
     * 特殊说明 2：背景带**先于所有网格线发射**。GL 的 `INSTANCE_MODE_FLAT` 按缓冲
     * 顺序合成，后写的盖住先写的；带子是整行高度的实心矩形，若排在线之后就会把
     * 网格线整段盖掉（而且"盖掉"在数值上完全合法，只能靠顺序断言发现）。
     *
     * 特殊说明 3：缺省（undefined）时**不产任何带**，行为与加本字段之前完全一致。
     */
    readonly blackKeyRowBandRgba?: Rgba | undefined;
    /**
     * 音阶高亮的音级集合（**音级** pitch class，0..11，不是 MIDI）。
     *
     * 特殊说明 1：空数组与缺省等价——都不产强调线。调用方在「音阶高亮关闭」或
     * 「没有工程音阶」时传空数组即可，无需再给一个开关参数。
     *
     * 特殊说明 2：判定用**音级**而不是 MIDI：同一音级的所有八度都要高亮。
     */
    readonly scaleNotes?: readonly number[] | undefined;
    /**
     * 音阶强调线的颜色。
     *
     * 特殊说明：与 {@link blackKeyRowBandRgba} 的相对顺序同样是**语义顺序**——
     * 强调线在背景带**之上**（背景是纹理、高亮是语义），因此它随所属网格线一起
     * 在对应该半音的位置发射，而不是提前到带子那一段。
     */
    readonly scaleHighlightRgba?: Rgba | undefined;
}

/** 非音高参数网格构建参数。 */
export interface ValueGridArgs {
    /** 参数种类（决定步进与强线间隔）。 */
    readonly kind: ValueGridKind;
    /** 当前视口（值域中心与跨度）。 */
    readonly view: { readonly center: number; readonly span: number };
    /** 视口高度（CSS px）。 */
    readonly heightPx: number;
    /** 视口宽度（CSS px）。 */
    readonly viewportWidthPx: number;
    /** 设备像素比。 */
    readonly dpr: number;
    /** 值 → 视口 y 的投影。 */
    readonly valueToY: (value: number, heightPx: number) => number;
    /** 强线颜色。 */
    readonly strongRgba: Rgba;
    /** 弱线颜色。 */
    readonly weakRgba: Rgba;
}

/** 各参数的步进与强线间隔（与 render.ts 的分支逐一对应）。 */
const VALUE_GRID_SPEC: Record<ValueGridKind, { step: number; strongMod: number }> = {
    cents: { step: 100, strongMod: 1200 },
    degrees: { step: 1, strongMod: 7 },
    formantCents: { step: 50, strongMod: 600 },
};

/**
 * 弱线 / 普通线的**描边中心**设备像素对齐。
 *
 * 与 `render.ts:440` 的 `hairlineY` 逐字一致（注意它加的是 0.5 个**设备**像素，
 * 即 `0.5/dpr` CSS 像素——不是 0.5 个 CSS 像素）。Canvas2D 的 `stroke()` 以该值
 * 为**线的中心**。
 *
 * @param cssY 视口 y（CSS px）。
 * @param dpr 设备像素比。
 * @returns 描边中心 y（CSS px）。
 */
function hairlineCenterY(cssY: number, dpr: number): number {
    return (Math.round(cssY * dpr) + 0.5) / dpr;
}

/**
 * 强线的**描边中心**设备像素对齐：只取整到物理像素，**不加**半像素。
 *
 * 与 `render.ts:752` 一致：强线宽 2 个物理像素，偶数宽度无需半像素偏移。
 * 两种取向并存是既有行为，迁移时必须原样保留（统一它们会改变像素）。
 *
 * @param cssY 视口 y（CSS px）。
 * @param dpr 设备像素比。
 * @returns 描边中心 y（CSS px）。
 */
function strongCenterY(cssY: number, dpr: number): number {
    return Math.round(cssY * dpr) / dpr;
}

/**
 * 把「描边中心 y」换算为 GL 实例矩形的**上缘 y**。
 *
 * 【这是一个必须显式处理的语义差，不是可选优化】
 * Canvas2D 的 `stroke()` 以给定 y 为线的**中心**，实际覆盖
 * `[y − thickness/2, y + thickness/2]`；而 GL 的实例矩形把 `y` 当作**上缘**，
 * 覆盖 `[y, y + thickness]`。直接把 Canvas2D 的中心值喂给矩形，整条线会**下移
 * 半个线厚**——在 dpr=2 的弱线上恰好是 1 个物理像素，表现为网格线与 Canvas2D
 * 版本整体错开一行（实测：Canvas2D 在设备行 740、GL 在 741）。
 *
 * 因此发射实例前必须减去半个线厚。两者都保持"对齐到设备像素栅格"的性质：
 * 中心落在 `k + 0.5` 个设备像素、厚度为 1 个设备像素时，上缘恰好落在整数
 * 设备像素 `k` 上。
 *
 * @param centerY 描边中心 y（CSS px，已按该层取向对齐）。
 * @param thicknessPx 线厚（CSS px）。
 * @returns 矩形上缘 y（CSS px）。
 */
function rectTopFromCenter(centerY: number, thicknessPx: number): number {
    return centerY - thicknessPx / 2;
}

/**
 * 生成一个**整行背景带**实例，并在纵向（y）边界上做设备像素覆盖率修正。
 *
 * 【为什么需要纵向修正】GL 的 `INSTANCE_MODE_FLAT` 是硬边矩形、不做抗锯齿，而背景
 * 带的上 / 下缘通常落在设备行内部（半音行高 = 视口高 / span，几乎不可能是整数设备
 * 行的倍数）。不修正时边界行会被**整行**着色，向相邻的白键行多渗一行 8% 的黑——
 * Canvas2D 的 `fillRect` 会按覆盖率只着该行的对应比例。
 *
 * 【与 `keyboardInstances.buildRectInstances` 的关系】那是同一问题的canonical 实现，
 * 但它**未导出**（`keyboardInstances.ts` 的私有函数），且返回 `KeyboardInstance`
 * （无 `value` 字段）。为一次调用而把那个模块的私有实现提升为公开 API 并不划算，
 * 因此这里按同一算法就地实现；两处口径必须一致，改动其一必须同步另一处。
 *
 * 特殊说明 1：区间为空 / 非有限 / 覆盖率 ≤ 0 的退化带一律丢弃——浮点残差会造出
 * `coverage ≈ 1e-16` 的不可见带，它会污染调用方与测试对"产出几个实例"的判定。
 *
 * 特殊说明 2：**不做横向覆盖率修正**。背景带的横向范围是 `[0, 视口宽]`，左缘恰在
 * 设备像素栅格上（x0 = 0）；只有右缘可能落在设备列内部，而那一列就是画布最右一列，
 * 溢出不可见。
 *
 * @param args 行体的纵向范围（已排序）、视口宽、颜色、dpr 与标签值。
 * @returns 覆盖该行体的实例；区间非法时返回空数组。
 */
function buildBandInstances(args: {
    readonly yTop: number;
    readonly yBottom: number;
    readonly widthPx: number;
    readonly rgba: Rgba;
    readonly dpr: number;
    readonly value: number;
}): GridInstance[] {
    const { yTop, yBottom, widthPx, rgba, dpr, value } = args;
    if (!Number.isFinite(yTop) || !Number.isFinite(yBottom) || yBottom <= yTop) return [];
    if (!Number.isFinite(widthPx) || widthPx <= 0) return [];
    if (!Number.isFinite(dpr) || dpr <= 0) return [];

    const deviceTop = yTop * dpr;
    const deviceBottom = yBottom * dpr;
    const wholeTop = Math.ceil(deviceTop - 1e-9);
    const wholeBottom = Math.floor(deviceBottom + 1e-9);

    // 拆成 [起始边界行] + [整行部分] + [结束边界行]（与 buildRectInstances 同构）。
    const EPS = 1e-9;
    const bands: { y0: number; y1: number; coverage: number }[] = [];
    if (wholeTop - deviceTop > EPS) {
        bands.push({ y0: deviceTop, y1: wholeTop, coverage: wholeTop - deviceTop });
    }
    if (wholeBottom - wholeTop > EPS) {
        bands.push({ y0: wholeTop, y1: wholeBottom, coverage: 1 });
    }
    if (deviceBottom - wholeBottom > EPS) {
        bands.push({ y0: wholeBottom, y1: deviceBottom, coverage: deviceBottom - wholeBottom });
    }

    const out: GridInstance[] = [];
    for (const band of bands) {
        // 部分覆盖的边界带必须**占满一个设备行**、用 alpha 表达覆盖率：GL 按像素
        // 中心采样，比一个设备行还窄的带可能不含任何像素中心而被整条丢弃。
        const isPartial = band.coverage < 1;
        const rowStart = Math.floor(band.y0 + 1e-9);
        const y = isPartial ? rowStart / dpr : band.y0 / dpr;
        const h = isPartial ? 1 / dpr : (band.y1 - band.y0) / dpr;
        out.push({
            x: 0,
            y,
            w: widthPx,
            h,
            rgba: isPartial
                ? [rgba[0], rgba[1], rgba[2], rgba[3] * band.coverage]
                : rgba,
            value,
        });
    }
    return out;
}

/**
 * 构建音高网格线实例（每个整数半音一条）。
 *
 * 流程：钳制 span 与值域 → 求可见半音区间 `[floor(min), ceil(max)]`（**含端点**）
 * → ① 先为每个**黑键**半音发射背景带（钢琴背景）→ ② 再逐半音算 y 发射网格线，
 * pc === 0 用 C 线色、其余用普通线色 → ③ 音阶音级在本行线之后**紧跟**一条 2 倍
 * 线厚的强调线。
 *
 * 特殊说明 1：区间含端点是刻意的（`render.ts:698` 用 `<=`），与键盘列使用的
 * `< endMidi` 存在一处不对称；这属于既有行为，不在迁移中"顺手修正"。
 *
 * 特殊说明 2：所有线都用弱线取向（`hairlineY`）——音高分支没有强线概念，
 * C 线只是**换色**、不换宽度与取向。音阶强调线沿用同一条中心线（只加粗，不换取向）。
 *
 * 特殊说明 3（**发射顺序即合成顺序**）：宿主的 `INSTANCE_MODE_FLAT` 按缓冲顺序
 * 合成，索引小的先画。因此顺序被固定为「背景带 → 网格线 → 强调线」：带子是整行
 * 实心矩形，排在线之后就**盖掉网格线**；强调线排在带子之前就会被带子压暗。
 * 这条顺序是不可见却致命的约束，改动前务必确认。
 *
 * 特殊说明 4（**背景带用行体、网格线用行心**）：网格线画在半音**行心**
 * `valueToY(midi + 0.5)`；背景带要铺满半音的**行体**
 * `[valueToY(midi + 1), valueToY(midi)]`。两者差半个键高——把行心当行体用，带子
 * 会只覆盖半行且整体错位半个键高。
 *
 * 特殊说明 5（音阶高亮的**已知限制**）：只支持**单一音阶**（工程音阶）。旧 Canvas2D
 * 路径还支持 Tempo Map 分段（不同时间段用不同音阶），但 GL 网格几何是**视口坐标、
 * 不含时间轴**——它不知道自己在哪个时间段上，无法表达分段。要恢复分段高亮需要把
 * 网格层也做成时间相关（或另开一层按段绘制），超出本次改动范围。详见
 * `PianoRollPanel.buildGridSpec` 的说明。
 *
 * @param args 构建参数。
 * @returns 实例序列（绘制顺序：黑键背景带 → 网格线 → 音阶强调线）；参数非法时返回空数组。
 */
export function buildPitchGridInstances(args: PitchGridArgs): GridInstance[] {
    const { view, absMin, absMax, heightPx, viewportWidthPx, dpr, valueToY } = args;
    if (!Number.isFinite(view.span) || !Number.isFinite(view.center)) return [];
    if (!Number.isFinite(heightPx) || !Number.isFinite(dpr) || dpr <= 0) return [];

    const range = absMax - absMin;
    if (!Number.isFinite(range) || range <= 0) return [];

    const span = Math.min(Math.max(view.span, 1e-6), range);
    const min = Math.min(Math.max(view.center - span / 2, absMin), absMax - span);
    const max = min + span;
    const startMidi = Math.min(Math.max(Math.floor(min), absMin), absMax);
    const endMidi = Math.min(Math.max(Math.ceil(max), absMin), absMax);

    const thickness = 1 / dpr;
    const items: GridInstance[] = [];

    // ── ① 黑键行背景带（钢琴背景）：必须先于所有网格线 ──────────────────
    //
    // 只在**提供了颜色**时产出：缺省（undefined）时行为与加本字段之前完全一致。
    if (args.blackKeyRowBandRgba !== undefined) {
        for (let midi = startMidi; midi <= endMidi; midi += 1) {
            if (!isBlackKey(midi)) continue;
            // 行体范围：valueToY 可能反向（值越大 y 越小），故取 min/max。
            const edgeA = valueToY(midi, heightPx);
            const edgeB = valueToY(midi + 1, heightPx);
            const top = Math.min(edgeA, edgeB);
            const bottom = Math.max(edgeA, edgeB);
            // 键高下限 1px（与键盘轴同一个 floor）：极小的键仍要保留可见体，
            // 否则缩放到极限时背景带会整片消失、留下空洞。
            const keyH = Math.max(1, bottom - top);
            items.push(
                ...buildBandInstances({
                    yTop: top,
                    yBottom: top + keyH,
                    widthPx: viewportWidthPx,
                    rgba: args.blackKeyRowBandRgba,
                    dpr,
                    // 负值编码 = 背景带（见 GridInstance.value 的约定）。
                    value: -(midi + 1),
                }),
            );
        }
    }

    // ── ② 网格线（+ ③ 紧跟其后的音阶强调线）───────────────────────────
    // 音级集合与强调线颜色必须**同时**可用才画强调线：只有一个时该图层无意义
    // （没有颜色无法画、没有音级不知道画哪行）。两个局部变量收窄类型，避免在
    // 行循环里反复做可选性判断。
    const scaleNoteSet: Set<number> | null =
        args.scaleNotes !== undefined && args.scaleNotes.length > 0
            ? new Set(args.scaleNotes)
            : null;
    const highlightRgba: Rgba | null =
        scaleNoteSet !== null && args.scaleHighlightRgba !== undefined
            ? args.scaleHighlightRgba
            : null;
    const highlightThickness = thickness * 2;

    for (let midi = startMidi; midi <= endMidi; midi += 1) {
        // 描边中心 → 矩形上缘（见 rectTopFromCenter 说明：差半个线厚）。
        const centerY = hairlineCenterY(valueToY(midi + 0.5, heightPx), dpr);
        const pc = ((midi % 12) + 12) % 12;
        items.push({
            x: 0,
            y: rectTopFromCenter(centerY, thickness),
            w: viewportWidthPx,
            h: thickness,
            rgba: pc === 0 ? args.colorC : args.colorOther,
            value: midi,
        });

        // 音阶强调线：紧跟本行的普通线 → 压在该线**与所有背景带**之上。沿用同一个
        // 描边中心（只加粗、不换取向），与 Canvas2D 的 `lineWidth = 2` 同中心。
        if (highlightRgba !== null && scaleNoteSet !== null && scaleNoteSet.has(pc)) {
            items.push({
                x: 0,
                y: rectTopFromCenter(centerY, highlightThickness),
                w: viewportWidthPx,
                h: highlightThickness,
                rgba: highlightRgba,
                value: midi,
            });
        }
    }
    return items;
}

/**
 * 构建非音高参数的刻度线实例。
 *
 * 流程：按 `kind` 取步进与强线间隔 → 求可见值区间内的步进点 → 强线用强线取向
 * 与 2 倍线厚，弱线用弱线取向与 1 倍线厚 → 描边中心换算为矩形上缘。
 *
 * 特殊说明：`span` 按 `render.ts:743` 的 `Math.max(1e-6, …)` 取下限，因此
 * `span = 0` 会退化为**单行**（域为 `[center, center]`），而不是空数组。
 * 非有限 span 直接返回空数组（见文件头：防死循环）。
 *
 * @param args 构建参数。
 * @returns 刻度线实例（按值升序）；参数非法时返回空数组。
 */
export function buildValueGridInstances(args: ValueGridArgs): GridInstance[] {
    const { kind, view, heightPx, viewportWidthPx, dpr, valueToY } = args;
    if (!Number.isFinite(view.span) || !Number.isFinite(view.center)) return [];
    if (!Number.isFinite(heightPx) || !Number.isFinite(dpr) || dpr <= 0) return [];

    const spec = VALUE_GRID_SPEC[kind];
    const span = Math.max(1e-6, view.span);
    const vMin = view.center - span / 2;
    const vMax = view.center + span / 2;
    const start = Math.ceil(vMin / spec.step) * spec.step;

    const weakThickness = 1 / dpr;
    const strongThickness = 2 / dpr;
    const items: GridInstance[] = [];
    for (let v = start; v <= vMax + spec.step * 0.01; v += spec.step) {
        const isStrong = Math.round(v) % spec.strongMod === 0;
        const thickness = isStrong ? strongThickness : weakThickness;
        // 描边中心 → 矩形上缘（见 rectTopFromCenter 说明：差半个线厚）。
        const centerY = isStrong
            ? strongCenterY(valueToY(v, heightPx), dpr)
            : hairlineCenterY(valueToY(v, heightPx), dpr);
        items.push({
            x: 0,
            y: rectTopFromCenter(centerY, thickness),
            w: viewportWidthPx,
            h: thickness,
            rgba: isStrong ? args.strongRgba : args.weakRgba,
            value: v,
        });
    }
    return items;
}
