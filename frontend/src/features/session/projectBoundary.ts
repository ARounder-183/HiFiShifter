/**
 * 工程边界（时长）求解。
 *
 * 【主要内容】
 * 1. `getDynamicProjectSec`：由 clip 列表求最后一个 clip 的末端；
 * 2. `resolveScrollableProjectSec`：求**可滚域**使用的工程时长（权威值 与 clip 末端取大）。
 *
 * 【作用】
 * 时间轴与参数编辑器的水平内容宽 / 可滚上限都由工程时长推出。两处若各取一个
 * 来源，共享 `scrollLeft` 的同步就会因上限不同而被浏览器钳制、永久错位
 * （缺陷 7：打开「同步时间轴视图」后对不上并抽搐）。本模块是**唯一来源**。
 *
 * 【与其他模块的关系】
 * - 上游：`PianoRollPanel`（`dynamicProjectSec`）与 `TimelineKernelView`
 *   （喂给 `ScrollKernel` 的 `projectSec`）。
 * - 独立性：纯函数，无 React / DOM / Redux 依赖，可在 node 环境单测。
 */

import type { ClipInfo } from "./sessionTypes";

const EMPTY_PROJECT_BOUNDARY_SEC = 30;

/**
 * 由 clip 列表求出的工程边界（秒）——**仅覆盖到最后一个 clip 的末端**。
 *
 * 特殊说明：这是**下界**，不是权威工程时长。后端给的 `session.projectSec` 才是
 * 权威值（它可以比 clip 末端更长，例如工程尾留白）。求可滚域时必须用下面的
 * `resolveScrollableProjectSec`，单独用本函数会让两个面板的滚动上限分叉
 * （见该函数说明）。
 *
 * @param clips clip 列表。
 * @returns 最后一个 clip 的末端；空工程返回默认边界。
 */
export function getDynamicProjectSec(clips: ClipInfo[]): number {
    if (!Array.isArray(clips) || clips.length === 0) {
        return EMPTY_PROJECT_BOUNDARY_SEC;
    }

    let maxEndSec = 0;
    for (const clip of clips) {
        const startSec = Math.max(0, Number(clip.startSec) || 0);
        const lengthSec = Math.max(0, Number(clip.lengthSec) || 0);
        const endSec = startSec + lengthSec;
        if (endSec > maxEndSec) maxEndSec = endSec;
    }

    return Math.max(1, maxEndSec);
}

/**
 * 求「可滚域」使用的工程时长（秒）——**时间轴与参数编辑器必须共用本函数**。
 *
 * 流程：取后端权威值 `projectSec` 与 clip 末端的**较大者**，再兜底为非负有限值。
 *
 * 【为什么必须有这个函数（缺陷 7 的根因）】同步模式下两个面板共享同一个
 * `scrollLeft`，但它们的可滚域此前来自**两个不同的时长来源**：
 * - 时间轴内核用 `session.projectSec`（后端权威值，实测 120 s）；
 * - 参数编辑器用 `getDynamicProjectSec(clips)`（clip 末端，实测 59.5 s）。
 *
 * 于是内容宽一个 18000px、一个 8925px（@150px/s），可滚上限一个 18000、一个 9125。
 * 同步把时间轴的 `scrollLeft` 推给参数编辑器后，**超出部分被浏览器钳制**，两边从此
 * 永久错位；快速来回拖动时共享值在两端跳变，表现为用户报告的「完全对不上 + 抽搐」。
 *
 * 【为什么取 max 而不是直接用 projectSec】`getDynamicProjectSec` 覆盖
 * 「clip 暂时超出 projectSec」的乐观中间态（拖长 clip 时后端值还没回来）；
 * `projectSec` 覆盖「工程尾留白」。取两者较大者同时满足，且因为是**同一个纯函数**，
 * 两边的滚动上限在结构上不可能再分叉。
 *
 * @param projectSec 后端权威工程时长（秒）；非法时按 0 处理。
 * @param clips clip 列表（用于兜住乐观中间态）。
 * @returns 用于计算内容宽 / 可滚上限的工程时长（秒，>= 0）。
 */
export function resolveScrollableProjectSec(
    projectSec: number,
    clips: ClipInfo[],
): number {
    const authoritative = Number.isFinite(projectSec) ? Math.max(0, projectSec) : 0;
    const clipBoundary = getDynamicProjectSec(clips);
    return Math.max(authoritative, clipBoundary);
}
