/**
 * 「在播放头处分割」的操作数解析 —— 全应用唯一一份。
 *
 * 【为什么必须是独立模块】这个判定有三个消费点，且分处三种执行时机：
 * - `useTimelineClipActions`（回调期，真正派发后端）；
 * - `useTimelineEventHandlers`（事件期，键盘 op）；
 * - `TimelinePanel`（渲染期，菜单项的 `disabled`）。
 * 三者此前各写一遍区间比较，并且**精度不一致**：菜单用闭区间
 * `>= start && <= end`，执行端用开区间 `> start + 1e-6 && < end - 1e-6`。
 * 于是播放头恰好落在 Clip 边缘时，菜单项是亮的、点下去却什么都不发生
 * （`eligibleIds` 为空，后端根本没被调用）。收成纯函数后，判定只有一份，
 * 可以在单测里直接锁住，也不会再漂移。
 */

/**
 * 分割留白（秒）。
 *
 * 必须与后端 `TimelineState::split_clip` 的 `1e-6` 边沿判定一致
 * （`split <= start + 1e-6 || split >= end - 1e-6` 视为不可分割），
 * 否则前端会送出一批注定被后端丢弃的 id。
 */
export const SPLIT_EDGE_EPSILON_SEC = 1e-6;

/** 播放头是否**严格落在** Clip 内部（可分割）。 */
export function isClipSplittableAtSec(
    clip: { startSec: number; lengthSec: number },
    splitSec: number,
): boolean {
    return (
        splitSec > clip.startSec + SPLIT_EDGE_EPSILON_SEC &&
        splitSec < clip.startSec + clip.lengthSec - SPLIT_EDGE_EPSILON_SEC
    );
}

export interface SplitTargets {
    ids: string[];
    /** `selection` = 有选区（走原判断逻辑）；`playhead` = 无选区，取播放头处的全部。 */
    source: "selection" | "playhead";
}

/**
 * 解析分割的操作数。
 *
 * - **有选区**（多选或单选）→ 原样返回选区内的存活 id（既有行为不变）；
 * - **无选区** → 返回时间线上**所有**在 `splitSec` 处可分割的 Clip id。
 *
 * 无选区时取全部（而不是"当前选中轨道"）：播放头是全局概念，按轨道裁剪
 * 会让"我按了分割，这条轨道为什么没动"成为新的困惑；且时间线的"轨道选中"
 * 同时承担参数编辑器的换轨语义，不适合兼任分割作用域。
 */
export function resolveSplitTargetsAtSec(args: {
    clips: ReadonlyArray<{ id: string; startSec: number; lengthSec: number }>;
    splitSec: number;
    selectedIds: readonly string[];
}): SplitTargets {
    const liveSelected = args.selectedIds.filter((id) =>
        args.clips.some((clip) => clip.id === id),
    );
    if (liveSelected.length > 0) {
        return { ids: [...liveSelected], source: "selection" };
    }
    return {
        ids: args.clips
            .filter((clip) => isClipSplittableAtSec(clip, args.splitSec))
            .map((clip) => clip.id),
        source: "playhead",
    };
}
