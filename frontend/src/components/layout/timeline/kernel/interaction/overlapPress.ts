import type { ClipHitRegion } from "./hitTest";

/**
 * 按下目标解析所需的最小命中形状。
 *
 * clip 是控件归属；foregroundClip 只表示 body 的视觉最上层。重叠区控件
 * 命中不得把 clip 换成 foregroundClip，否则 A 的淡出线会去拖 B。
 */
export interface OverlapPressHit {
    readonly kind: string;
    readonly region?: ClipHitRegion;
    readonly fromOverlapRegion?: boolean;
    readonly clip?: { readonly id: string };
    readonly foregroundClip?: { readonly id: string };
}

export interface OverlapPressResolution<T extends OverlapPressHit> {
    /** 原样保留的命中：控件命中时 clip 仍是控件归属。 */
    readonly hit: T;
    /** 本次按下应选中的 Clip；body 用前景，控件用控件归属。 */
    readonly selectionClipId: string | null;
    /** 是否为重叠区编辑控件（边缘 / 淡变线 / 交叉点抓手）。 */
    readonly controlOwner: boolean;
}

export function isOverlapEditRegion(region: ClipHitRegion | undefined): boolean {
    return (
        region === "left-edge" ||
        region === "right-edge" ||
        region === "fade-in-corner" ||
        region === "fade-out-corner" ||
        region === "crossfade-grip"
    );
}

/**
 * 解析一次左键按下的选择目标，同时保留控件归属。
 *
 * body 命中选中视觉最上层（foregroundClip）；重叠区控件命中选中控件所属
 * Clip，使“拖谁就选谁”。返回值中的 hit 永远原样返回，手势状态机继续按
 * 原始 region 升级为 trim / fade / crossfade-grip。
 */
export function resolveOverlapPressTarget<T extends OverlapPressHit>(
    hit: T,
): OverlapPressResolution<T> {
    if (hit.kind !== "clip" || hit.clip === undefined) {
        return { hit, selectionClipId: null, controlOwner: false };
    }
    const controlOwner = hit.fromOverlapRegion === true && isOverlapEditRegion(hit.region);
    return {
        hit,
        selectionClipId: controlOwner ? hit.clip.id : (hit.foregroundClip?.id ?? hit.clip.id),
        controlOwner,
    };
}
