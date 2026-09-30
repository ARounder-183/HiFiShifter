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
