/**
 * stretchParams.ts — 拉伸后「参数线时域映射」的共享入口。
 *
 * 【主要内容】
 * 把一次（或一批）拉伸操作对应的参数线时域映射交给后端完成：
 * `stretchTrackLinkedParams(trackId, mappings, false)`。
 *
 * 【为什么单独成模块】
 * 这段逻辑原先只存在于旧实现的 `useEditDrag` 内。渲染内核的「Alt + 拖边缘 = 拉伸」
 * 需要**同一份**语义（锁定参数线时把曲线从旧时间范围映射到新范围）；若内核侧
 * 另写一遍，两条渲染路径在「锁定参数线」开启时会出现不同的曲线位置——而且差异
 * 只在参数编辑器里可见，极难归因。
 *
 * 【映射由后端完成的原因】
 * 后端会一次性处理 pitch（用户编辑过时）、tension 以及该根轨道上所有已存在的
 * 自动化曲线（volume / 气声 / 子轨道偏移等，无论参数是否在 UI 中激活或有数据）。
 * 早期前端实现只映射 pitch + tension，导致其余参数线在拉伸后遗留在旧位置。
 *
 * 【与其他模块的关系】
 * - 上游：`useEditDrag`（旧实现拖拽收尾）与 `TimelinePanel`（内核手势收尾）。
 * - 依赖：`services/api` 的 `paramsApi`。
 */

import { paramsApi } from "../../../../services/api";

/** 一次「旧范围 → 新范围」的参数线映射。 */
export interface StretchRangeMapping {
    readonly oldStartSec: number;
    readonly oldLengthSec: number;
    readonly newStartSec: number;
    readonly newLengthSec: number;
}

/**
 * 批量拉伸同一根轨道上的多个 clip 对应的参数线。
 *
 * 特殊说明：后端先写入全部新范围，再恢复未被任何新范围覆盖的旧范围片段——因此
 * 相邻 clip 不会互相擦掉刚写入的值（顺序敏感，不能在前端逐条串行调用）。
 *
 * @param trackId 根轨道 id。
 * @param mappings 映射列表；空列表直接返回。
 * @returns 完成后 resolve。
 */
export async function stretchTrackLinkedParams(
    trackId: string,
    mappings: readonly StretchRangeMapping[],
): Promise<void> {
    if (mappings.length === 0) return;
    await paramsApi.stretchTrackLinkedParams(trackId, [...mappings], false);
}

/**
 * 单个 clip 的参数线时域映射（范围未变化时直接返回，避免无意义的 IPC）。
 *
 * @param trackId 根轨道 id。
 * @param oldStartSec 拉伸前起点（秒）。
 * @param oldLengthSec 拉伸前长度（秒）。
 * @param newStartSec 拉伸后起点（秒）。
 * @param newLengthSec 拉伸后长度（秒）。
 * @returns 完成后 resolve。
 */
export async function stretchLinkedParams(
    trackId: string,
    oldStartSec: number,
    oldLengthSec: number,
    newStartSec: number,
    newLengthSec: number,
): Promise<void> {
    if (
        Math.abs(oldLengthSec - newLengthSec) < 1e-6 &&
        Math.abs(oldStartSec - newStartSec) < 1e-6
    ) {
        return;
    }
    await stretchTrackLinkedParams(trackId, [
        { oldStartSec, oldLengthSec, newStartSec, newLengthSec },
    ]);
}
