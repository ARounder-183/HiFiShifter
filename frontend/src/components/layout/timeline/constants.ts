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
//   y ∈ [0, FADE_CORNER_RESERVE_PX)      → 淡入/淡出角部拖拽控件（= 横帽）；
//   y ∈ [FADE_CORNER_RESERVE_PX, 底部]   → 裁短 / 延长 / 拉伸。
/** 角部横帽：从边缘向内的宽度；位于 body 区顶部（header 之下）。 */
export const FADE_CORNER_CAP_WIDTH_PX = 22;
export const FADE_CORNER_CAP_HEIGHT_PX = 14;

/**
 * 淡化角控件保留区：**恒等于横帽高度**，不随音频块高度缩放。
 *
 * 于是"渐变角命中区"就是横帽本身 —— 手柄即横帽，少一个概念。
 *
 * 【为什么是定值而不是 body 的比例】这条规则被改过三轮，每次的教训都留下：
 *
 * - 回归①：原先固定 **48px**，在典型 Clip 高度（74–90px）上吃掉 53%–65% 的边缘，
 *   用户想裁短却在边缘偏上按下时命中的是淡化控件（"拉边界被判定成渐变"）。
 * - 回归②：改为 `body × 0.38` 并**封顶 34px**，行高 >96px 后拖拽区不再随轨道高度
 *   缩放（"看着是定长"）；当时的结论是"必须随高度增长、不得封顶"，遂改成 `body/3`。
 * - 回归③（Issue 141，用户实测）：`body/3` 的比例虽然恒定，**绝对高度却随行高线性
 *   放大** —— 行高 80→20px、96→25px、120→33px、192→57px，变化 2.85 倍。行高 192 时
 *   57px 已接近音频块自身高度的三分之一，用户在边缘中部按下想裁短、命中却是渐变。
 *   参考实现 REAPER 用的正是**贴着 body 顶角的小手柄**，不是比例区。
 *
 * 因此回归②的结论被 Issue 141 推翻：把"高度比例化"当目标，换来的是绝对尺寸过大与
 * 与裁短手势争地。定值让裁短区在任意行高下都拿到 76%–92% 的边缘（`body/3` 时恒为
 * 67%），回归①的意图被**加强**而非削弱。
 *
 * 角控件不得覆盖 header：header 上有旋钮 / badge / 名称等交互控件，压在上面会让
 * header 无法点击。退化矮 body 时按 `body / 2` 收敛，保证真边角仍有落点。
 *
 * @param bodyHeightPx body 高度（= clip 高 - header 高）。
 */
export function fadeCornerReservePx(bodyHeightPx: number): number {
    const bodyH = Number.isFinite(bodyHeightPx) ? Math.max(0, bodyHeightPx) : 0;
    return Math.min(FADE_CORNER_CAP_HEIGHT_PX, Math.max(1, Math.floor(bodyH / 2)));
}

/**
 * 渐变角手柄角标（贴 body 顶角的小三角）的边长（px）。
 *
 * 【为什么要有这个角标】渐变角命中区是本仓唯一**完全隐形**的可点区：淡变为 0 时
 * 屏幕上没有任何东西提示"这里能按"。本仓的核心不变式是"看到的 = 可点的"，这里
 * 曾经缺失，于是用户只能靠试错建立肌肉记忆——而 Issue 141 正是这种试错的产物
 * （误把边缘上部当裁短区）。参考实现 REAPER 也在 item 顶角画同样的角标。
 *
 * 7px 是"看得见"与"不喧宾夺主"之间的平衡：它只画在**该侧还没有淡变**时，
 * 已有淡变的那一侧由包络线本身充当提示，因此不会在 clip 上堆两套图形。
 */
export const FADE_CORNER_HANDLE_SIZE_PX = 7;

/**
 * 渐变角手柄角标的外接矩形（相对 **clip 左上角**，与 header 控件同一坐标系）。
 *
 * 契约：角标必须**整体落在** `hitTest` 判为 `fade-in/out-corner` 的横帽带内
 * （高 `fadeCornerReservePx`、距该侧边缘 `FADE_CORNER_CAP_WIDTH_PX` 内）。
 * 这条包含关系由契约测试逐行高比对，是"看到的 = 可点的"在本模块的落点。
 *
 * @param side "in" = 贴左缘（淡入角）；"out" = 贴右缘（淡出角）。
 * @param clipWidthPx clip 的像素宽度（角标不得宽过 clip 的一半）。
 * @param bodyTopPx body 顶边相对 clip 顶边的 y（= `CLIP_HEADER_HEIGHT`）。
 * @param bodyHeightPx body 高度（用于收敛保留区）。
 */
export function fadeCornerHandleBoxPx(args: {
    side: "in" | "out";
    clipWidthPx: number;
    bodyTopPx: number;
    bodyHeightPx: number;
}): { left: number; top: number; width: number; height: number } {
    const reserve = fadeCornerReservePx(args.bodyHeightPx);
    const size = Math.max(
        1,
        Math.min(
            FADE_CORNER_HANDLE_SIZE_PX,
            reserve,
            Math.max(1, Math.floor(Math.max(0, args.clipWidthPx) / 2)),
        ),
    );
    // 内缩 1px：角标不压在 clip 描边与 header/body 分隔线上。退化矮 body 下
    // 内缩收敛到 0，保证角标仍整体落在保留区内。
    const inset = Math.min(1, Math.max(0, reserve - size));
    return {
        left: args.side === "in" ? inset : Math.max(inset, args.clipWidthPx - size - inset),
        top: args.bodyTopPx + inset,
        width: size,
        height: size,
    };
}

/**
 * 渐变角命中带在该 clip 的该侧是否**真的可达**。
 *
 * 两种情况会让"画了角标"变成"画了却点不到"，都必须先判掉：
 *
 * 1. **纵向** —— 吸附偏移三角手柄的优先级高于渐变角（见 `hitTest` 的
 *    `ClipHitRegion` 注释：握把贴在行底，若不先判就会被淡变角抢走），它的命中带
 *    占 body 底部 `SNAP_OFFSET_HIT_HEIGHT_PX`。clip 矮到两条带重叠时，在横帽里
 *    按下拿到的是手柄。
 * 2. **横向（仅淡出侧）** —— 左右横帽带的宽度都固定为 `FADE_CORNER_CAP_WIDTH_PX`，
 *    而判定顺序让**左侧先赢**；clip 比一个横帽宽不了多少时，淡出侧的横帽整个落在
 *    淡入侧的判定范围内，于是淡出角不可达。角标必须整个落在左侧横帽带之外才算数。
 *
 * 受支持的行高（80–192）下纵向永远成立；只有退化矮 clip 会落到 false。
 *
 * @param side "in" = 贴左缘；"out" = 贴右缘。
 * @param clipWidthPx clip 的像素宽度（判横向遮挡）。
 * @param clipHeightPx clip 高度（= 行高 − `CLIP_BODY_PADDING_Y`）。
 * @param headerHeightPx header 高度。
 * @param bodyHeightPx body 高度（用于收敛保留区）。
 */
export function isFadeCornerBandReachable(args: {
    side: "in" | "out";
    clipWidthPx: number;
    clipHeightPx: number;
    headerHeightPx: number;
    bodyHeightPx: number;
}): boolean {
    const reserve = fadeCornerReservePx(args.bodyHeightPx);
    if (args.headerHeightPx + reserve > args.clipHeightPx - SNAP_OFFSET_HIT_HEIGHT_PX) {
        return false;
    }
    if (args.side === "out") {
        return args.clipWidthPx > FADE_CORNER_CAP_WIDTH_PX + FADE_CORNER_HANDLE_SIZE_PX;
    }
    return true;
}

/**
 * 拖拽“落到新轨道”时的哨兵 trackId（moveClipTrack 用它标记待创建轨道）。
 * 放在轻量 constants 中，供渲染层等无 Redux 依赖的模块引用。
 */
export const NEW_TRACK_SENTINEL = "__hs_new_track__";

/**
 * 音量旋钮竖直拖拽的换算：每像素多少 dB。
 *
 * 单一事实来源：旧实现 `useEditDrag` 与渲染内核的旋钮手势共用它——两处各写一份
 * 会让"拖同样的距离得到不同的增益"（且差异随行高/缩放看不出规律）。
 */
export const CLIP_GAIN_DRAG_DB_PER_PX = 0.25;

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

/**
 * 滚轮缩放的每步因子（>1 放大、<1 缩小）。
 *
 * 只有内核的滚轮分支用它（画布与标尺走同一条路径，见宿主 `dispatchWheel`）。
 * 放在这里而不是内核模块内部：它是**交互手感**参数，与其它时间轴常量放在一起，
 * 便于统一调整。
 */
export const WHEEL_ZOOM_IN_FACTOR = 1.1;
export const WHEEL_ZOOM_OUT_FACTOR = 0.9;
