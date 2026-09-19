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

/** 「实时跨度」的取值入参。 */
export interface LiveSpanArgs {
    /**
     * 值域镜像里的跨度（面板在 `syncVerticalScrollbarForViewport` 里**就地刷新**）。
     *
     * 这是首选来源：竖向缩放改的是面板 ref，而该镜像与滚动位置在同一处刷新，
     * 因此它与"本帧"同时效。
     */
    readonly domainSpan: number;
    /**
     * 网格快照里的跨度（`PianoRollGridSpec.view.span`，React render 期写下）。
     *
     * 仅作兜底：它不是实时的（见 `resolveLiveSpan` 说明），但镜像不可用时
     * （未初始化 / 非有限值）总得有个值，否则几何会按 NaN 枚举而整层消失。
     */
    readonly snapshotSpan: number;
}

/**
 * 取本帧的**实时值域跨度**。
 *
 * 【为什么需要这个函数——`span` 与 `center` 走的是两条不同的过期路径】
 * `center` 的真值在内核（`scrollTop`），用快照会"滚动时不重建几何"。
 * `span` 的真值在面板的 ref 里，而**竖向缩放改 ref 后直接 `invalidate()`，不触发
 * React 渲染**（见 `setPitchView`）——于是 `PianoRollGridSpec.view.span` 这个 render
 * 期快照会停在旧值上。几何按实时跨度枚举半音、却只覆盖旧窗口，屏幕留下空白带
 * （实测 dpr 2：实时 span 24 → 42.5 而快照与实例数都不变；旧实现同手势最大空白带
 * 398px，内核模式曾出现同量级空白）。
 *
 * 【为什么优先用 `domainSpan`】面板在 `syncVerticalScrollbarForViewport` 里就地刷新
 * `valueDomain`（竖向缩放 / 滚动 / 切参数都经过那里），它与本帧同时效；两者在构建期
 * 同源（都取自 `getCurrentViewportForScrollbar`），因此读它不会引入新的分叉。
 *
 * 特殊说明：`domainSpan` 非有限值或非正时退回快照值——宁可偶尔用到略旧的跨度，
 * 也不能让 NaN 传进几何枚举（那会让整层实例消失，比"少画几行"严重得多）。
 *
 * @param args 见 `LiveSpanArgs`。
 * @returns 用于几何的跨度；两者都不可用时返回 `domainSpan` 原值（由调用方处理）。
 */
export function resolveLiveSpan(args: LiveSpanArgs): number {
    const domain = args.domainSpan;
    if (Number.isFinite(domain) && domain > 0) return domain;
    const snapshot = args.snapshotSpan;
    if (Number.isFinite(snapshot) && snapshot > 0) return snapshot;
    return domain;
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
    /** 黑键行背景带颜色（钢琴背景）；缺省表示不画背景带。 */
    readonly blackKeyRowBandRgba?: readonly number[] | undefined;
    /**
     * 音阶高亮的音级集合（pitch class 0..11）。
     *
     * 特殊说明：必须是**集合内容**而不是数组引用——签名是字符串，`join` 后比较
     * 内容；同一组音级的两个不同数组应产生同一个签名（否则每次 React 渲染都会
     * 重建几何）。顺序不同（`[0,4,7]` vs `[7,4,0]`）会得到不同签名，代价只是
     * 偶尔多一次重建，不影响正确性。
     */
    readonly scaleNotes?: readonly number[] | undefined;
    /** 音阶强调线颜色；缺省表示不画强调线。 */
    readonly scaleHighlightRgba?: readonly number[] | undefined;
    /**
     * Tempo Map 分段音阶（已投影到视口 x）。
     *
     * 必须入签名：段边界是**时间锚定**的，滚动 / 缩放会改变投影后的 x 范围——
     * 不入签名就会出现"滚动了但分段高亮没跟着动"。
     */
    readonly scaleSegments?:
        | readonly {
              readonly x0: number;
              readonly x1: number;
              readonly notes: readonly number[];
          }[]
        | undefined;
}

/**
 * 网格几何的内容签名。
 *
 * 必须包含**全部决定几何的输入**：种类、视口（中心 + 跨度）、值域边界、视口尺寸
 * （决定横线的横向范围与竖线的可见区间）、dpr（决定线条按物理像素的取向与厚度）、
 * 以及**每一种颜色与每一组音级集合**——两条网格线颜色（`strongRgba` / `weakRgba`）、
 * 黑键行背景带颜色（`blackKeyRowBandRgba`）、音阶高亮音级（`scaleNotes`）与强调线
 * 颜色（`scaleHighlightRgba`）。
 *
 * 漏项的代价是"该输入变化后几何不更新"：颜色漏了 → 切主题不生效；音级集合漏了 →
 * 开 / 关音阶高亮画面不动；背景带颜色漏了 → 主题切换后带子仍是旧色。因此这里
 * **宁可多编**，代价只是偶尔多一次重建。
 *
 * 特殊说明 1：可选颜色 / 集合按 `?? ""` 参与——缺省与"显式给出"是两种不同的几何
 * 输入，不能混为一谈（缺省表示不画该图层）。
 *
 * 特殊说明 2：必须是**纯字符串**且对同一输入稳定（本函数不含时间戳 / 随机量），
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
        args.blackKeyRowBandRgba?.join(",") ?? "",
        args.scaleNotes?.join(",") ?? "",
        // 分段音阶：段边界（投影后的视口 x）与各段音级都要入签名。
        args.scaleSegments
            ?.map((segment) => `${segment.x0},${segment.x1},${segment.notes.join(".")}`)
            .join(";") ?? "",
        args.scaleHighlightRgba?.join(",") ?? "",
    ].join("|");
}

/** 键盘 / 数值轴几何签名的入参（与 `PianoRollGridSpec` 的轴列字段对应）。 */
export interface KeyboardGeometrySignatureArgs {
    readonly kind: string;
    /**
     * 参数标识（如 `"dyn"` / `"volume"`）。仅数值轴签名使用：同 kind 的两个参数
     * （值域、标签格式可能不同）必须产出不同签名，否则切换时轴几何不重建。
     * pitch 分支不需要它。
     */
    readonly paramName?: string | undefined;
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
 * 特殊说明 1：**非 pitch 参数返回数值轴的签名，而不是空串**。
 *
 * 早期实现为了"非 pitch 时清空键盘"在非 pitch 分支返回 `""`，但这同时丢掉了
 * 区分各种非 pitch 参数的能力：`volume → dyn` 这类**非 pitch 之间的切换**会得到
 * 两个相同的空签名，于是轴几何永不重建，上一参数的刻度被原样 repaint ——
 * 表现为"动态参数显示成了音量的标尺"。
 *
 * 现在非 pitch 分支带上 `kind` 与参数标识：既与 pitch 签名天然不同（pitch 分支
 * 有自己独立的前缀），也能在各非 pitch 参数之间互相区分。清空逻辑由
 * `rebuildKeyboardGeometry` / `rebuildAxisMarkGeometry` 各自负责，不再依赖
 * 空串这个信号。
 *
 * 特殊说明 2：键盘配色全部可选且参与签名——缺省与"显式给出透明色"是两种不同的
 * 几何输入，不能混为一谈。
 *
 * 特殊说明 3：**不含**横向视口：轴列几何是视口坐标（x 从轴列左缘起算），与
 * `scrollLeft` 无关。把它编进来会造成无谓重建。
 *
 * @param args 见 `KeyboardGeometrySignatureArgs`。
 * @returns 内容签名；pitch 为键盘签名，其余为数值轴签名。
 */
export function keyboardGeometrySignature(args: KeyboardGeometrySignatureArgs): string {
    if (args.kind !== "pitch") {
        // 数值轴：kind 决定刻度种类（cents / degrees / formantCents / level /
        // fallback），paramName 兜住"同 kind 但不同参数"的情况（例如 volume 与
        // dyn 都是数值轴，但值域与标签不同）。
        return [
            "value",
            args.kind,
            args.paramName ?? "",
            args.view.center,
            args.view.span,
            args.absMin,
            args.absMax,
            args.viewportHeightPx,
            args.axisWidthPx,
            args.dpr,
        ].join("|");
    }
    return [
        "pitch",
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
