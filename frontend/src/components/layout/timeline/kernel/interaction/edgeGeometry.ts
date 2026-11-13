/**
 * Clip 边缘命中带的共享几何。
 *
 * 命中带只向 Clip 内部延伸，宽度不超过 Clip 宽度的 1/3；极短 Clip 仍保留
 * 中间的可拖动 body。通用命中与重叠区控件必须共用这份计算，否则同一条边缘
 * 会在两条路径上得到不同宽度。
 */
export function inwardEdgeWidthPx(clipWidthPx: number, rawEdgeWidthPx: number): number {
    const width = Math.max(1, clipWidthPx);
    return Math.min(Math.max(0, rawEdgeWidthPx), width / 3);
}
