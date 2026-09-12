/**
 * 参数编辑器内核 · GL 场景层的**实时视口解析**与内容签名。
 *
 * 【主要内容】
 * 1. `resolveLiveGridView`：由内核的竖向真值（`scrollTop`）与面板提供的跨度 /
 *    值域边界，算出**本帧**的视口中心与跨度；
 * 2. `gridGeometrySignature` / `keyboardGeometrySignature`：网格与键盘 / 数值轴
 *    几何的**内容签名**（只有签名变化才重建几何）。
 *
 * 【作用：为什么要从内核解析视口，而不是直接用面板写下的快照】
 * 竖向滚动的真值在 `ScrollKernel`（`scrollTop`），而**竖轴没有逐帧回写面板的通道**
 * （横向有：宿主每帧 `onFrame` → 面板 `scrollLeftRef`）。面板写进
 * `PianoRollGridSpec.view` 的 `center` 只在 **React render** 时刷新，而拖竖向
 * 滚动条 / 中键竖向平移 / 键盘翻页都**不进 React**（内核直接写 `scrollTop`）。
 * 于是快照会长期停留在旧值，签名也就永远不变 —— 几何不重建。
 *
 * 实测（1920×1200 @dpr2、pitch span=24）：内核中心 72 → 79.29（内容位移约
 * 250px），GL 画布上的横线中心逐像素完全相同；同一场景下旧实现 Canvas2D 路径的
 * 横线移动了 10px。用户看到的现象是「拖竖向滚动条，网格/键盘纹丝不动」，
 * 即报告里的「经常拖不动」。
 *
 * 【为什么签名必须含**实时**中心而不是快照中心】
 * 签名是"要不要重建几何"的唯一判据。用快照就等于用了一个在竖向滚动期间恒定的量，
 * 相当于把重建条件退化成"只有 React 重渲染才重建"。含实时中心后，竖向滚动会改
 * 变签名 → 重建 → 线条随之内移。
 *
 * 【与其他模块的关系】
 * - 上游：`pianoRollKernelHost` 在帧提交时读取内核视口并调用本模块。
 * - 下游：`gridInstances` / `keyboardInstances` / `axisMarkInstances` 按解析出的
 *   视口构建几何。
 * - 换算复用：中心 ↔ 像素的映射走 `scroll/verticalValueScroll`（既有实现），
 *   不在这里另写一份比值公式——那会让人手感在两个渲染模式下分叉。
 * - 独立性：纯函数，不依赖 DOM / React / WebGL，可直接单测。
 */

import { centerFromKernelScrollTop } from "../scroll/verticalValueScroll";

/** 实时视口的解析入参。 */
export interface LiveGridViewArgs {
    /** 该参数的绝对值域下界。 */
    readonly absMin: number;
    /** 该参数的绝对值域上界。 */
    readonly absMax: number;
    /** 视口可见的值跨度（由面板决定：缩放不进内核）。 */
    readonly span: number;
    /** 内核当前的竖向滚动位置（CSS px，0..1600）。 */
    readonly scrollTop: number;
}

/** 解析出的视口（与 `PianoRollGridSpec.view` 同形）。 */
export interface LiveGridView {
    readonly center: number;
    readonly span: number;
}

/**
 * 由内核竖向位置解析本帧视口。
 *
 * 流程：值域边界 + 跨度 + 内核 `scrollTop` → `centerFromKernelScrollTop` → 视口。
 *
 * 特殊说明 1：`center` 会被既有映射钳制到**可动中心域**
 * `[min + span/2, max − span/2]`，这是刻意的有损行为（与旧实现一致，见
 * `verticalValueScroll` 的说明）。因此两端越界时中心停在边界，不会漂到域外。
 *
 * 特殊说明 2：`span` **原样透传**。它描述"能看到多少值"，由面板的缩放手势决定，
 * 内核不参与；在解析里夹取会掩盖面板侧的越界 bug。
 *
 * @param args 值域边界、跨度与内核竖向位置。
 * @returns 本帧视口（`center` 已钳制到合法区间）。
 */
export function resolveLiveGridView(args: LiveGridViewArgs): LiveGridView {
    return {
        center: centerFromKernelScrollTop({
            min: args.absMin,
            max: args.absMax,
            span: args.span,
            scrollTop: args.scrollTop,
        }),
        span: args.span,
    };
}

/** 网格几何签名的入参（与 `PianoRollGridSpec` 的几何相关字段一一对应）。 */
export interface GridGeometrySignatureArgs {
    readonly kind: string;
    readonly view: LiveGridView;
    readonly absMin: number;
    readonly absMax: number;
    readonly viewportWidthPx: number;
    readonly viewportHeightPx: number;
    readonly dpr: number;
    readonly strongRgba: readonly number[];
    readonly weakRgba: readonly number[];
}

/**
 * 网格几何的内容签名。
 *
 * 必须包含**全部决定几何的输入**：种类、视口（中心 + 跨度）、值域边界、视口尺寸
 * （决定横线的横向范围与竖线的可见区间）、dpr（决定线条按物理像素的取向与厚度）、
 * 两条颜色（决定实例的 rgba）。
 *
 * 漏项的代价是"该输入变化后几何不更新"：颜色漏了 → 切主题不生效；视口尺寸漏了 →
 * 缩放窗口后网格不铺满。因此这里**宁可多编**，代价只是偶尔多一次重建。
 *
 * 特殊说明：必须是**纯字符串**且对同一输入稳定（本函数不含时间戳 / 随机量），
 * 否则每帧都会重建几何，滚动帧预算会被直接吃掉。
 *
 * @param args 见 `GridGeometrySignatureArgs`。
 * @returns 内容签名；对同一几何输入恒等。
 */
export function gridGeometrySignature(args: GridGeometrySignatureArgs): string {
    return [
        args.kind,
        args.view.center,
        args.view.span,
        args.absMin,
        args.absMax,
        args.viewportWidthPx,
        args.viewportHeightPx,
        args.dpr,
        args.strongRgba.join(","),
        args.weakRgba.join(","),
    ].join("|");
}

/** 键盘 / 数值轴几何签名的入参（与 `PianoRollGridSpec` 的轴列字段对应）。 */
export interface KeyboardGeometrySignatureArgs {
    readonly kind: string;
    readonly view: LiveGridView;
    readonly absMin: number;
    readonly absMax: number;
    readonly viewportHeightPx: number;
    readonly axisWidthPx: number;
    readonly dpr: number;
    readonly strongRgba: readonly number[];
    readonly weakRgba: readonly number[];
    readonly whiteKeyRgba?: readonly number[] | undefined;
    readonly blackKeyRgba?: readonly number[] | undefined;
    readonly blackKeyGradientRgba?: readonly number[] | undefined;
    readonly cSeparatorRgba?: readonly number[] | undefined;
    readonly keySeparatorRgba?: readonly number[] | undefined;
    readonly axisBorderRgba?: readonly number[] | undefined;
}

/**
 * 键盘 / 数值轴几何的内容签名。
 *
 * 特殊说明 1：**非 pitch 参数返回空串**。此时轴列画的是数值刻度线而不是键盘，
 * 调用方据空串清空几何；若返回值非空会让上一位参数的键盘残留（视觉上"切参数后
 * 钢琴键还在"）。
 *
 * 特殊说明 2：键盘配色全部可选且参与签名——缺省与"显式给出透明色"是两种不同的
 * 几何输入，不能混为一谈。
 *
 * 特殊说明 3：**不含**横向视口：轴列几何是视口坐标（x 从轴列左缘起算），与
 * `scrollLeft` 无关。把它编进来会造成无谓重建。
 *
 * @param args 见 `KeyboardGeometrySignatureArgs`。
 * @returns 内容签名；非 pitch 参数为空串。
 */
export function keyboardGeometrySignature(args: KeyboardGeometrySignatureArgs): string {
    if (args.kind !== "pitch") return "";
    return [
        args.view.center,
        args.view.span,
        args.absMin,
        args.absMax,
        args.viewportHeightPx,
        args.axisWidthPx,
        args.dpr,
        args.strongRgba.join(","),
        args.weakRgba.join(","),
        args.whiteKeyRgba?.join(",") ?? "",
        args.blackKeyRgba?.join(",") ?? "",
        args.blackKeyGradientRgba?.join(",") ?? "",
        args.cSeparatorRgba?.join(",") ?? "",
        args.keySeparatorRgba?.join(",") ?? "",
        args.axisBorderRgba?.join(",") ?? "",
    ].join("|");
}
