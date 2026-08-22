/**
 * 实时波纹编辑（自动跟进）预览辅助。
 *
 * 设计：后端在编辑提交时才是波纹的唯一真源（`commands/timeline.rs` →
 * `state.rs::ripple_shift_clips`）。拖拽过程中无法逐帧走 IPC（延迟过大、
 * 全量时间线载荷过重），因此前端用与本文件相同的算法做**乐观预览**：
 * 把“编辑点之后的后续剪辑”按统一平移量实时更新，让用户像 REAPER 一样
 * 在拖拽时就看到波纹效果；松手提交时由后端权威结果（`sessionSlice` 中
 * `moveClipRemote / moveClipsRemote / setClipStateRemote /
 * setClipsStateBulkRemote` 的 fulfilled 均以 `force` 应用）覆盖校正到
 * 与后端一致的精确位置。
 *
 * 上下两侧判定必须与后端保持一致，避免“预览→提交”时跳变：
 * - 排除被编辑集合；
 * - `track` 模式：只取被编辑剪辑**原轨道**上的后续剪辑；
 *   `all` 模式：所有轨道；`off`：无；
 * - 只取**初始起点 >= 波纹原点**（被编辑剪辑的最早起点）的后续剪辑；
 * - 平移量 = 右缘位移（对移动为 drag 位移量，对右缘重设为长度变化量）。
 */
import type { AppDispatch } from "../../app/store";
import { moveClipStart } from "./sessionSlice";
import type { SessionState } from "./sessionSlice";

export type RippleMode = "off" | "track" | "all";

/** clipId → 该 clip 在本次拖拽开始时的初始起点（秒）。 */
export type RippleFollowerMap = Record<string, number>;

/**
 * 在拖拽/编辑开始时快照“波纹跟随集”（放在任何乐观更新之前调用）。
 *
 * @returns 空对象 = 波纹关闭或没有跟随对象。
 */
export function buildRippleFollowers(
    clips: SessionState["clips"],
    editedIds: ReadonlySet<string>,
    origin: number,
    rippleMode: RippleMode,
    editedTracks: ReadonlySet<string>,
): RippleFollowerMap {
    if (rippleMode === "off") return {};
    const followers: RippleFollowerMap = {};
    for (const clip of clips) {
        if (editedIds.has(clip.id)) continue;
        if (rippleMode === "track" && !editedTracks.has(clip.trackId)) continue;
        const start = Number(clip.startSec ?? 0);
        if (!Number.isFinite(origin) || start + 1e-9 < origin) continue;
        followers[clip.id] = start;
    }
    return followers;
}

/**
 * 实时波纹预览：把跟随集按 `delta` 平移。
 *
 * `moveClipStart` 写入的是绝对位置，因此这里始终用“初始位置 + delta”计算，
 * 避免逐帧累加造成漂移；`delta = 0` 时等价于把跟随集恢复回初始位置。
 */
export function applyRippleFollowerShift(
    dispatch: AppDispatch,
    followers: RippleFollowerMap,
    delta: number,
): void {
    const ids = Object.keys(followers);
    if (ids.length === 0) return;
    for (const clipId of ids) {
        const start = followers[clipId];
        dispatch(moveClipStart({ clipId, startSec: Math.max(0, start + delta) }));
    }
}
