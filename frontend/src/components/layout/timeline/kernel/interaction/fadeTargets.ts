/**
 * 时间轴渲染内核 · 淡变命中目标
 *
 * 【主要内容】
 * 判定点是否落在某个 clip 的淡变命中区内（包络线小块 / 区域边缘竖线），并给出
 * 命中的是哪一侧（淡入 / 淡出）。
 *
 * 【作用】
 * 旧实现的 `FadeHitLayer`（在 `ClipItem` 内）为**每个 clip** 提供「画线即控件」：
 * 沿包络线任意位置都能抓住并拖动调长度。内核原先只有**角部小方块**
 * （`fade-in-corner` / `fade-out-corner`）能抓——长淡变的曲线中段完全抓不到，
 * 用户只能去顶角那个小方块上碰运气。
 *
 * 【与其他模块的关系】
 * - 几何来源：`timeline/fadeHitTargets` 的 `buildFadeHitTargets`（它保证与绘制端
 *   `drawFadeCurveStroke` 逐像素一致——这是「看到的 = 可点的」的前提）。
 * - 两个消费方：内核的通用命中（本模块）与重叠区解析（`overlapControls`，它会
 *   额外把命中区裁剪到重叠区）。两者共用同一套几何，避免两份实现漂移。
 * - 独立性：纯函数，无 DOM / React 依赖。
 *
 * 【坐标系】
 * `clipLeftPx` / `contentX` 都是**内容坐标**（与 `buildFadeHitTargets` 的入参一致），
 * `localY` 以**行顶**为 0（函数内部换算为 body 内的相对位置）。
 */

import { CLIP_BODY_PADDING_Y, CLIP_HEADER_HEIGHT } from "../../constants";
import { buildFadeHitTargets } from "../../fadeHitTargets";

/** 淡变命中所需的 clip 字段（全部可选：未设过淡变的 clip 缺省即为 0）。 */
export interface FadeTargetClip {
    readonly fadeInSec?: number;
    /** 自动交叉淡化长度（秒）：> 0 时覆盖手动值（与绘制端一致）。 */
    readonly autoFadeInSec?: number;
    readonly fadeInShape?: number;
    readonly fadeInDir?: number;
    readonly fadeOutSec?: number;
    readonly autoFadeOutSec?: number;
    readonly fadeOutShape?: number;
    readonly fadeOutDir?: number;
}

/**
 * 淡变有效长度：自动交叉淡化（> 0 时）覆盖手动淡变。
 *
 * 与绘制端 / 旧实现 `OverlapEditLayer.effectiveFadeInSec` 同一规则——自动交叉淡化
 * 是重叠区的派生结果，用户看不到也改不了"手动值"，因此它必须赢。
 *
 * @param manual 手动淡变长度（秒）。
 * @param auto 自动交叉淡化长度（秒）。
 * @returns 生效的淡变长度（秒）；两者都无效时为 0。
 */
export function effectiveFadeSec(manual: number | undefined, auto: number | undefined): number {
    const autoSec = typeof auto === "number" && Number.isFinite(auto) && auto > 0 ? auto : 0;
    if (autoSec > 0) return autoSec;
    return typeof manual === "number" && Number.isFinite(manual) && manual > 0 ? manual : 0;
}

/** 命中参数。 */
export interface ClipFadeTargetArgs {
    readonly clip: FadeTargetClip;
    /** clip 左缘的内容坐标（CSS px）。 */
    readonly clipLeftPx: number;
    /** clip 的像素宽度。 */
    readonly clipWidthPx: number;
    /** 指针的内容坐标 x。 */
    readonly contentX: number;
    /** 指针在该行内的局部 y（0 = 行顶）。 */
    readonly localY: number;
    readonly pxPerSec: number;
    readonly rowHeight: number;
    /** 可选的 x 范围裁剪（重叠区解析传入重叠区边界；通用命中不传）。 */
    readonly clipXFrom?: number;
    readonly clipXTo?: number;
}

/**
 * 淡变命中结果。
 *
 * `kind` 必须保留「包络线本体 / 区域边缘竖线」的区分：**双击重置曲率只对本体生效**
 * （旧实现 `OverlapEditLayer` 的 `zone.line` 与 `FadeHitLayer` 的 `isLine` 都以此为
 * 门槛）。少了它，用户在边缘竖线上双击会意外重置曲率。
 */
export interface ClipFadeTargetHit {
    readonly side: "in" | "out";
    readonly kind: "line" | "edge";
}

/**
 * 判定点是否落在该 clip 的淡变命中区内。
 *
 * @param args 见 `ClipFadeTargetArgs`。
 * @returns 命中的一侧与命中类型；未命中为 null。
 */
export function hitClipFadeTarget(args: ClipFadeTargetArgs): ClipFadeTargetHit | null {
    const pxPerSec = Number.isFinite(args.pxPerSec) ? Math.max(1e-9, args.pxPerSec) : 1e-9;
    const rowHeight = Number.isFinite(args.rowHeight) ? Math.max(1, args.rowHeight) : 1;
    const bodyTop = CLIP_HEADER_HEIGHT;
    const bodyHeight = Math.max(1, rowHeight - CLIP_BODY_PADDING_Y - CLIP_HEADER_HEIGHT);
    const clipWidthPx = Math.max(1, args.clipWidthPx);

    const fadeInSec = effectiveFadeSec(args.clip.fadeInSec, args.clip.autoFadeInSec);
    const fadeOutSec = effectiveFadeSec(args.clip.fadeOutSec, args.clip.autoFadeOutSec);
    if (fadeInSec <= 0 && fadeOutSec <= 0) return null;

    // 两侧一起算：同一段空间可能同时属于两条包络线（交叉淡化），交给
    // buildFadeHitTargets 的返回顺序决定（淡入在前、淡出在后，后者覆盖）。
    const targets = buildFadeHitTargets({
        clipLeftPx: args.clipLeftPx,
        clipWidthPx,
        bodyTop,
        bodyHeight,
        fadeInPx: fadeInSec * pxPerSec,
        fadeOutPx: fadeOutSec * pxPerSec,
        fadeInShape: args.clip.fadeInShape ?? 0,
        fadeInDir: args.clip.fadeInDir ?? 0,
        fadeOutShape: args.clip.fadeOutShape ?? 0,
        fadeOutDir: args.clip.fadeOutDir ?? 0,
        clipXFrom: args.clipXFrom,
        clipXTo: args.clipXTo,
    });

    let hit: ClipFadeTargetHit | null = null;
    for (const target of targets) {
        if (
            args.contentX >= target.left &&
            args.contentX <= target.left + target.width &&
            args.localY >= target.top &&
            args.localY <= target.top + target.height
        ) {
            hit = {
                side: target.type === "fade_in" ? "in" : "out",
                kind: target.kind,
            };
        }
    }
    return hit;
}
