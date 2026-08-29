export const DEFAULT_PX_PER_BEAT = 75;
export const MIN_PX_PER_BEAT = 8;
export const MAX_PX_PER_BEAT = 2000;

// 以秒为单位的缩放常量（pxPerSec = pxPerBeat / secPerBeat = pxPerBeat * bpm / 60）
// DEFAULT_PX_PER_SEC = 240 对应 120 BPM 时 pxPerBeat = 240 * (60/120) = 120
export const DEFAULT_PX_PER_SEC = 150;
export const MIN_PX_PER_SEC = 4;
export const MAX_PX_PER_SEC = 8000;

export const DEFAULT_ROW_HEIGHT = 96;
export const MIN_ROW_HEIGHT = 80;
export const MAX_ROW_HEIGHT = 192;
export const TRACK_ADD_ROW_HEIGHT = 32;

export const CLIP_HEADER_HEIGHT = 18;
export const CLIP_BODY_PADDING_Y = 2;

// ── 淡化角部手柄几何（左右边缘的垂直所有权切分）────────────────────
//
// 每条左/右边缘按高度分成两段、归属两种手势（几何级切分，非 z 竞争）：
//   y ∈ [0, FADE_CORNER_RESERVE_PX)      → 淡入/淡出角部拖拽控件；
//   y ∈ [FADE_CORNER_RESERVE_PX, 底部]   → ClipEdgeHandles 裁短/延长/拉伸。
/** 角部横帽：从边缘向内的宽度；高度恰为 header 带，覆盖"真边角"。 */
export const FADE_CORNER_CAP_WIDTH_PX = 22;
export const FADE_CORNER_CAP_HEIGHT_PX = CLIP_HEADER_HEIGHT;
/** 边缘上部竖条宽度（骑在边缘线上，位于横帽正下方直至保留区下沿）。 */
export const FADE_CORNER_EDGE_WIDTH_PX = 6;

/** 淡化角控件在边缘上保留的顶部带**上限**（px）。 */
export const FADE_CORNER_RESERVE_MAX_PX = 34;
/**
 * 裁短/拉伸至少要拿到的边缘高度占比。
 *
 * 保留区按此比例收缩，保证短 Clip 上 trim 手势仍有充裕的纵向空间。
 */
export const FADE_CORNER_TRIM_MIN_RATIO = 0.62;

/**
 * 计算某条边缘让给淡化角控件的顶部高度（px）。
 *
 * 【为什么不是固定值】原先固定 48px，而典型 Clip 高度只有
 * `rowHeight - CLIP_BODY_PADDING_Y`（80–96 行高 → 74–90px），角控件因此吃掉
 * 53%–65% 的边缘；用户想裁短 Clip、在边缘偏上位置按下时，命中的却是淡化
 * 控件——即"拉边界被判定成渐变"。
 *
 * 【策略】按比例收缩并加上限：trim 至少拿到 `FADE_CORNER_TRIM_MIN_RATIO` 的
 * 边缘高度，Clip 越高保留区越接近上限。保留区不会小于横帽本身，否则真边角
 * 就没有落点。
 */
export function fadeCornerReservePx(clipHeightPx: number): number {
    const height = Number.isFinite(clipHeightPx) ? Math.max(0, clipHeightPx) : 0;
    const byRatio = height * (1 - FADE_CORNER_TRIM_MIN_RATIO);
    return Math.min(
        FADE_CORNER_RESERVE_MAX_PX,
        Math.max(FADE_CORNER_CAP_HEIGHT_PX, byRatio),
    );
}

/**
 * 拖拽“落到新轨道”时的哨兵 trackId（moveClipTrack 用它标记待创建轨道）。
 * 放在轻量 constants 中，供渲染层等无 Redux 依赖的模块引用。
 */
export const NEW_TRACK_SENTINEL = "__hs_new_track__";

/** SnapOffset 三角视觉边长（px）。 */
export const SNAP_OFFSET_HANDLE_SIZE_PX = 9;
/** SnapOffset 命中区高度（px，相对行底部条带）。 */
export const SNAP_OFFSET_HIT_HEIGHT_PX = 12;
/**
 * SnapOffset 三角的 x（px，相对 Clip 左缘）= 偏移 × 缩放，**不做宽度
 * 回退钳制** —— 三角左侧竖直边必须严格对齐偏移实际值（与波形内橙色
 * 竖虚线同一 x）；越出 Clip 末尾的部分由绘制端按 Clip 矩形裁剪。
 */
export function snapOffsetHandleXPx(snapOffsetSec: number | undefined, pxPerSec: number): number {
    const offset = Number(snapOffsetSec);
    return Number.isFinite(offset) && offset > 0 ? offset * pxPerSec : 0;
}
