export type AnchoredHorizontalZoomArgs = {
    currentScale: number;
    factor: number;
    minScale: number;
    maxScale: number;
    scrollLeft: number;
    viewportWidth: number;
    anchorSec: number;
    contentSec?: number;
};

export type AnchoredHorizontalZoomResult = {
    nextScale: number;
    nextScrollLeft: number;
    anchorX: number;
};

function clamp(value: number, min: number, max: number): number {
    return Math.min(max, Math.max(min, value));
}

export function computeAnchoredHorizontalZoom(
    args: AnchoredHorizontalZoomArgs,
): AnchoredHorizontalZoomResult | null {
    const {
        currentScale,
        factor,
        minScale,
        maxScale,
        scrollLeft,
        viewportWidth,
        anchorSec,
        contentSec,
    } = args;

    if (
        !Number.isFinite(currentScale) ||
        !Number.isFinite(factor) ||
        !Number.isFinite(minScale) ||
        !Number.isFinite(maxScale) ||
        !Number.isFinite(scrollLeft) ||
        !Number.isFinite(viewportWidth) ||
        !Number.isFinite(anchorSec) ||
        factor <= 0
    ) {
        return null;
    }

    const nextScale = clamp(currentScale * factor, minScale, maxScale);
    if (Math.abs(nextScale - currentScale) <= 1e-9) {
        return null;
    }

    const width = Math.max(1, viewportWidth);
    let anchorX = anchorSec * currentScale - scrollLeft;
    if (anchorX < 0 || anchorX > width) {
        const centeredScrollLeft = Math.max(0, anchorSec * currentScale - width / 2);
        anchorX = anchorSec * currentScale - centeredScrollLeft;
    }
    anchorX = clamp(anchorX, 0, width);

    const rawNextScrollLeft = anchorSec * nextScale - anchorX;
    const maxScroll = Number.isFinite(contentSec)
        ? Math.max(0, (contentSec ?? 0) * nextScale - width)
        : Number.POSITIVE_INFINITY;

    return {
        nextScale,
        nextScrollLeft: clamp(rawNextScrollLeft, 0, maxScroll),
        anchorX,
    };
}
