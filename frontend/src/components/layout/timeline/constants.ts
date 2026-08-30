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
/** 角部横帽：从边缘向内的宽度；位于 body 区顶部（header 之下）。 */
export const FADE_CORNER_CAP_WIDTH_PX = 22;
export const FADE_CORNER_CAP_HEIGHT_PX = 14;
/** 边缘上部竖条宽度（骑在边缘线上，位于横帽正下方直至保留区下沿）。 */
export const FADE_CORNER_EDGE_WIDTH_PX = 6;

/**
 * 淡化角控件保留区 = body 高度的 1/3（按轨道高度换算，随行高缩放）。
 *
 * 历史实现按 `body × 0.38` 且**封顶 34px**：行高超过 ~96px 后拖拽区恒为
 * 34px，不随轨道高度缩放（"看着是定长"）。现改为 body/3、无全局封顶：
 * 行高 80→20px、96→25px、120→33px、192→57px；裁短区始终保住 2/3 ≥
 * 之前的 62% 下限，任意行高下都成立。
 *
 * 角控件不得覆盖 header：header 上有旋钮 / badge / 名称等交互控件，
 * 角控件压在上面会让 header 无法点击。退化矮 body（< 3×横帽高）时
 * 回退为横帽高度本身（真边角必须有落点）。
 *
 * @param bodyHeightPx body 高度（= clip 高 - header 高）。
 */
export function fadeCornerReservePx(bodyHeightPx: number): number {
    const bodyH = Number.isFinite(bodyHeightPx) ? Math.max(0, bodyHeightPx) : 0;
    return Math.max(FADE_CORNER_CAP_HEIGHT_PX, Math.round(bodyH / 3));
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
