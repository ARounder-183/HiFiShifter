/**
 * fadeContextMenuBus — 淡变包络上下文菜单的全局事件总线。
 *
 * 右键目标分散在三处渲染层（ClipItem 角部手柄、FadeHitLayer 命中块、
 * OverlapEditLayer 重叠区/抓手），而菜单宿主（FadeContextMenuHost）挂在
 * TimelinePanel 上层。用 prop 链下传会穿透六层组件；这里用一个极小的
 * CustomEvent 总线：发送方带"请求打开"载荷 dispatch，宿主监听后弹菜单。
 *
 * 载荷几何：primary = 右键命中的那侧包络；secondary = 抓手/交叉点时的
 * 对侧包络（无则 null）。
 */

import type { FadeContextSide } from "./FadeContextMenu";

export type FadeContextMenuRequest = {
    clientX: number;
    clientY: number;
    primary: FadeContextSide;
    secondary: FadeContextSide | null;
};

export const FADE_CONTEXT_MENU_OPEN_EVENT = "hs-fade-context-menu-open";

export function requestOpenFadeContextMenu(request: FadeContextMenuRequest): void {
    if (typeof window === "undefined") return;
    window.dispatchEvent(new CustomEvent(FADE_CONTEXT_MENU_OPEN_EVENT, { detail: request }));
}

export function onFadeContextMenuRequest(
    handler: (request: FadeContextMenuRequest) => void,
): () => void {
    if (typeof window === "undefined") return () => undefined;
    const listener = (event: Event) => {
        const detail = (event as CustomEvent).detail as FadeContextMenuRequest | undefined;
        if (detail) handler(detail);
    };
    window.addEventListener(FADE_CONTEXT_MENU_OPEN_EVENT, listener);
    return () => window.removeEventListener(FADE_CONTEXT_MENU_OPEN_EVENT, listener);
}

// ── 双击重置曲率 ────────────────────────────────────────────────

export type FadeCurvatureResetRequest = {
    sides: Array<{ clipId: string; isOut: boolean }>;
};

export const FADE_CURVATURE_RESET_EVENT = "hs-fade-curvature-reset";

/** 双击包络线/交叉点后：请求把指定侧的曲率重置为该形状默认值。 */
export function requestResetFadeCurvature(request: FadeCurvatureResetRequest): void {
    if (typeof window === "undefined") return;
    window.dispatchEvent(new CustomEvent(FADE_CURVATURE_RESET_EVENT, { detail: request }));
}

export function onFadeCurvatureReset(
    handler: (request: FadeCurvatureResetRequest) => void,
): () => void {
    if (typeof window === "undefined") return () => undefined;
    const listener = (event: Event) => {
        const detail = (event as CustomEvent).detail as FadeCurvatureResetRequest | undefined;
        if (detail) handler(detail);
    };
    window.addEventListener(FADE_CURVATURE_RESET_EVENT, listener);
    return () => window.removeEventListener(FADE_CURVATURE_RESET_EVENT, listener);
}

/**
 * 原生连击判定：PointerEvent 继承 MouseEvent，detail 即同一位置的
 * 连续按下次数（浏览器自身维护双击检测）。≥2 表示双击中的第二次按下。
 */
export function isNativeMultiClick(event: PointerEvent): boolean {
    const detail = (event as unknown as { detail?: number }).detail;
    return typeof detail === "number" && detail >= 2;
}
