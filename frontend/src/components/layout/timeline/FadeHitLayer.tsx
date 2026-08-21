/**
 * FadeHitLayer — 淡入淡出"画线即控件"的 DOM 命中层。
 *
 * 由 buildFadeHitTargets 生成的两类命中区域：
 *   1. 包络线命中块（沿包络线采样的小方块）；
 *   2. 区域边缘竖线命中条（淡化区域最外侧竖线）。
 *
 * 命中区域完全透明、不做任何悬停高亮——光标悬停仅体现为 resize 光标，
 * 画面保持就是"看哪条线就抓哪条线"。未覆盖到的区域不拦截事件，
 * 可自然穿透到 clip body（拖拽移动/选择）。
 */
import React from "react";
import { buildFadeHitTargets } from "./fadeHitTargets";
import type { FadeCurveType } from "./paths";

export const FadeHitLayer = React.memo(function FadeHitLayer({
    clipLeftPx,
    clipWidthPx,
    bodyTop,
    bodyHeight,
    fadeInPx,
    fadeOutPx,
    fadeInCurve,
    fadeOutCurve,
    clipXFrom,
    clipXTo,
    zIndex = 40,
    onFadeInPointerDown,
    onFadeOutPointerDown,
}: {
    clipLeftPx: number;
    clipWidthPx: number;
    bodyTop: number;
    bodyHeight: number;
    fadeInPx: number;
    fadeOutPx: number;
    fadeInCurve: FadeCurveType;
    fadeOutCurve: FadeCurveType;
    clipXFrom?: number;
    clipXTo?: number;
    zIndex?: number;
    onFadeInPointerDown: (e: React.PointerEvent<HTMLDivElement>) => void;
    onFadeOutPointerDown: (e: React.PointerEvent<HTMLDivElement>) => void;
}) {
    const targets = buildFadeHitTargets({
        clipLeftPx,
        clipWidthPx,
        bodyTop,
        bodyHeight,
        fadeInPx,
        fadeOutPx,
        fadeInCurve,
        fadeOutCurve,
        clipXFrom,
        clipXTo,
    });

    if (targets.length === 0) return null;

    return (
        <>
            {targets.map((target, index) => (
                <div
                    key={`${target.kind}-${target.type}-${index}`}
                    className="absolute"
                    style={{
                        left: target.left,
                        top: target.top,
                        width: target.width,
                        height: target.height,
                        zIndex,
                        cursor: target.type === "fade_in" ? "nwse-resize" : "nesw-resize",
                    }}
                    data-hs-fade-hit={target.type}
                    onPointerDown={(e) => {
                        if (e.button !== 0) return;
                        if (target.type === "fade_in") onFadeInPointerDown(e);
                        else onFadeOutPointerDown(e);
                    }}
                />
            ))}
        </>
    );
});
