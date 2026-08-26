import type { ClipInfo } from "../../../features/session/sessionTypes";

export interface TakeLaneLayout {
    takeId: string;
    index: number;
    count: number;
    /** 相对 Clip body 顶部的像素位置 */
    top: number;
    height: number;
    inactive: boolean;
}

export const MIN_TAKE_LANE_HEIGHT_PX = 14;

/** 计算一个 Clip 的全部波形 lane；空间不足或非多 Take 时返回 null。 */
export function resolveTakeLaneLayouts(
    clip: ClipInfo,
    showAllTakes: boolean,
    bodyHeightPx: number,
): TakeLaneLayout[] | null {
    if (!showAllTakes || !clip.sourcePath) return null;
    const takes = (clip.takes ?? []).filter((take) => Boolean(take.sourcePath));
    if (takes.length <= 1) return null;
    if (takes.length * MIN_TAKE_LANE_HEIGHT_PX > bodyHeightPx + 1e-6) return null;

    const laneHeight = Math.max(MIN_TAKE_LANE_HEIGHT_PX, Math.floor(bodyHeightPx / takes.length));
    return takes.map((take, index) => ({
        takeId: take.id,
        index,
        count: takes.length,
        top: index * laneHeight,
        height:
            index === takes.length - 1
                ? Math.max(MIN_TAKE_LANE_HEIGHT_PX, bodyHeightPx - index * laneHeight)
                : laneHeight,
        inactive: take.id !== clip.activeTakeId,
    }));
}

/** 根据事件相对 Clip body 的本地 Y 坐标命中 inactive take。 */
export function hitInactiveTakeLane(
    clip: ClipInfo,
    showAllTakes: boolean,
    bodyHeightPx: number,
    localY: number,
): TakeLaneLayout | null {
    const lanes = resolveTakeLaneLayouts(clip, showAllTakes, bodyHeightPx);
    if (!lanes) return null;
    // 先钳制到 body 内：负值（header 底边 1px 容差）不应命中 lane 0。
    // 再用半开区间 [top, top+height)：±1px 双侧容差会让相邻 lane 边界处
    // 同时命中两个区间，find() 恒取上面的 lane，与视觉分界线不符。
    const clampedY = Math.max(0, Math.min(bodyHeightPx - 1e-6, localY));
    const lane = lanes.find(
        (entry) => entry.inactive && clampedY >= entry.top && clampedY < entry.top + entry.height,
    );
    return lane ?? null;
}
