/**
 * 列表拖拽排序的纯逻辑。
 *
 * 【为什么抽出来】"指针落在第几个位置"与"落点下标怎么换算"都是纯算术，但错了不会
 * 抛错 —— 只会表现为"拖到某处会跳一格"这类只有上手才发现的偏移。放在这里可单测。
 *
 * 【为什么需要 `reorderTargetIndex` 这一步】`reorderUserVibratoPresets` 的下标是
 * **移除被拖项之后**的数组下标；而指示线、指针位置说的是**当前顺序**里的插入位置。
 * 两者在被拖项之前/之后差一位，必须显式换算。
 */

/** 拖拽排序的位移阈值（CSS 像素）：不超过它就仍算"点击"，不进入拖拽。 */
export const REORDER_DRAG_THRESHOLD_PX = 4;

/**
 * 指针 Y 落在**当前顺序**里的插入位置（`0..rowCenters.length`）。
 *
 * 逐行比较中线：指针越过某行中线即落到它之前；全部越过后落在末尾。这正是列表拖拽
 * 的常规手感 —— "推过半行才换位"。
 */
export function reorderInsertionIndex(rowCenters: readonly number[], pointerY: number): number {
    for (let index = 0; index < rowCenters.length; index += 1) {
        if (pointerY < rowCenters[index]) return index;
    }
    return rowCenters.length;
}

/**
 * 当前顺序的插入位置 → 移除被拖项后的下标。
 *
 * 被拖项之前的位置不受移除影响；之后的位置整体前移一位，因此要减一。
 */
export function reorderTargetIndex(insertionIndex: number, fromIndex: number): number {
    return insertionIndex > fromIndex ? insertionIndex - 1 : insertionIndex;
}

/** 指针进入这个距边缘的范围内即开始自动滚动（CSS 像素）。 */
export const REORDER_AUTOSCROLL_EDGE_PX = 24;

/** 自动滚动每帧的最大步长（CSS 像素）：越深入边缘越快。 */
export const REORDER_AUTOSCROLL_MAX_STEP_PX = 12;

/**
 * 拖拽到边缘时的自动滚动步长（负 = 向上滚，正 = 向下滚，0 = 不滚）。
 *
 * 【为什么越靠边越快】恒定步长在"刚碰到边缘"时会猛地窜一屏；按深入程度线性加速，
 * 用户既能贴着边缘微调，也能压到底快速掠过。每帧取整并保底 1px，避免深入不足 1px
 * 时反复取整成 0、看起来"到了边缘却不动"。
 *
 * 【为什么视口过矮就不滚】视口高度不到两倍边缘带时，整块区域都算"边缘"，一进去就
 * 永远在滚 —— 那种尺寸下列表本身也没几条，不需要自动滚动。
 */
export function reorderAutoScrollDelta(input: {
    pointerY: number;
    viewportTop: number;
    viewportBottom: number;
    edgePx?: number;
    maxStepPx?: number;
}): number {
    const edge = Math.max(1, input.edgePx ?? REORDER_AUTOSCROLL_EDGE_PX);
    const maxStep = Math.max(1, input.maxStepPx ?? REORDER_AUTOSCROLL_MAX_STEP_PX);
    const height = input.viewportBottom - input.viewportTop;
    if (height <= edge * 2) return 0;

    if (input.pointerY < input.viewportTop + edge) {
        const depth = Math.min(1, (input.viewportTop + edge - input.pointerY) / edge);
        return -Math.max(1, Math.round(maxStep * depth));
    }
    if (input.pointerY > input.viewportBottom - edge) {
        const depth = Math.min(1, (input.pointerY - (input.viewportBottom - edge)) / edge);
        return Math.max(1, Math.round(maxStep * depth));
    }
    return 0;
}
