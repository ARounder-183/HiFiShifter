/**
 * 粘贴后播放光标定位。
 *
 * 需求：执行各种粘贴操作（时间线剪贴板 / REAPER 剪贴板 / VocalShifter 剪贴板）
 * 产生新 Clip 后，播放光标应跳转到"粘贴所产生的所有 Clip 中，结束位置
 * （start_sec + length_sec）最靠右的那个 Clip 的结束位置"。
 *
 * 该模块只做纯计算；transport 同步与 Redux 状态更新由各 paste thunk /
 * reducer 分别完成。
 */

/** 后端返回的 Clip 载荷中与本计算相关的最小字段集合（snake_case）。 */
export interface PasteEndClipLike {
    id?: string;
    start_sec?: number;
    length_sec?: number;
}

/**
 * 计算粘贴产生的所有新 Clip 中最靠右的结束位置。
 *
 * @param clips  粘贴响应中的完整 Clip 列表（snake_case 字段）
 * @param newClipIds 粘贴新建的 Clip id 列表
 * @returns 最右结束位置（秒）；无法确定（无新 Clip / 数据缺失）时返回 null。
 */
export function computePasteEndSec(
    clips: ReadonlyArray<PasteEndClipLike> | undefined,
    newClipIds: ReadonlyArray<string | undefined | null> | undefined,
): number | null {
    if (!Array.isArray(newClipIds) || newClipIds.length === 0) {
        return null;
    }
    const clipById = new Map<string, PasteEndClipLike>();
    for (const clip of Array.isArray(clips) ? clips : []) {
        if (clip && typeof clip.id === "string") {
            clipById.set(clip.id, clip);
        }
    }

    let maxEndSec: number | null = null;
    for (const clipId of newClipIds) {
        if (!clipId) continue;
        const clip = clipById.get(clipId);
        if (!clip) continue;
        const start = Number(clip.start_sec);
        const length = Number(clip.length_sec);
        if (!Number.isFinite(start) || !Number.isFinite(length)) continue;
        // 长度按非负处理，避免异常负值把光标拉到 Clip 起点之前。
        const endSec = start + Math.max(0, length);
        if (maxEndSec === null || endSec > maxEndSec) {
            maxEndSec = endSec;
        }
    }
    return maxEndSec;
}
