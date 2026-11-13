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

/** 按下时的修饰键快照。 */
export interface OverlapPressModifiers {
    readonly altKey: boolean;
    readonly ctrlKey: boolean;
    readonly metaKey: boolean;
    readonly shiftKey: boolean;
}

export interface OverlapPressResolution<T extends OverlapPressHit> {
    /** 原样保留的命中：控件命中时 clip 仍是控件归属。 */
    readonly hit: T;
    /**
     * 未拖动时选中的 Clip。
     *
     * 无修饰键的重叠区控件单击保留既有的「选前景」手感；带编辑修饰键时
     * 直接选控件归属，避免修饰键编辑与普通选择互相打架。
     */
    readonly selectionClipId: string | null;
    /** 超过拖拽阈值后实际编辑的控件归属；非重叠控件为 null。 */
    readonly controlClipId: string | null;
    /** 是否为重叠区编辑控件（边缘 / 淡变线 / 交叉点抓手）。 */
    readonly controlOwner: boolean;
    /** 是否为无修饰键的重叠区控件单击（单击选前景，拖动才转向控件归属）。 */
    readonly plainControlClick: boolean;
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

function hasEditModifier(modifiers: OverlapPressModifiers): boolean {
    return modifiers.altKey || modifiers.ctrlKey || modifiers.metaKey || modifiers.shiftKey;
}

/**
 * 解析一次左键按下的选择目标，同时保留控件归属。
 *
 * body 命中选中视觉最上层（foregroundClip）；重叠区控件命中在无修饰键时
 * 也先选前景，只有超过拖拽阈值后才改选控件所属 Clip，并继续按原始 region
 * 升级为 trim / fade / crossfade-grip。返回值中的 hit 永远原样返回，因此
 * 控件归属不会被选择目标覆盖。
 */
export function resolveOverlapPressTarget<T extends OverlapPressHit>(
    hit: T,
    modifiers: OverlapPressModifiers,
): OverlapPressResolution<T> {
    if (hit.kind !== "clip" || hit.clip === undefined) {
        return {
            hit,
            selectionClipId: null,
            controlClipId: null,
            controlOwner: false,
            plainControlClick: false,
        };
    }
    const controlOwner = hit.fromOverlapRegion === true && isOverlapEditRegion(hit.region);
    if (!controlOwner) {
        return {
            hit,
            selectionClipId: hit.foregroundClip?.id ?? hit.clip.id,
            controlClipId: null,
            controlOwner: false,
            plainControlClick: false,
        };
    }
    const plainControlClick = !hasEditModifier(modifiers);
    return {
        hit,
        selectionClipId: plainControlClick ? (hit.foregroundClip?.id ?? hit.clip.id) : hit.clip.id,
        controlClipId: hit.clip.id,
        controlOwner: true,
        plainControlClick,
    };
}
