/**
 * 播放光标时间读数的**放置位置**。
 *
 * 【为什么这是一个用户偏好而不是布局属性】读数在工具栏、状态栏、轨道头三处都是
 * chrome，不属于任何一个面板；布局树只描述面板几何与标签页。把它塞进 DockLayout
 * 会让"换个位置"变成"改布局"，用户还得为此保存/重载一套布局。
 *
 * 【为什么有 `auto`】插件窗口常常只有 640×400：工具栏里一行读数会把传输组挤到换行，
 * 而状态栏始终存在、不占工具栏宽度。`auto` 让"默认放哪里"由产品决定，而不是写死在
 * 某个组件的自然宽度里（见 [resolvePlayheadTimePlacement]）。
 */
export type PlayheadTimePlacement = "auto" | "toolbar" | "statusBar" | "trackHeader" | "hidden";

/** 所有合法取值（顺序即菜单顺序）。 */
export const PLAYHEAD_TIME_PLACEMENTS: readonly PlayheadTimePlacement[] = [
    "auto",
    "toolbar",
    "statusBar",
    "trackHeader",
    "hidden",
];

/** 除 `auto` / `hidden` 之外，真正会渲染读数的那几个槽位。 */
export type PlayheadTimeSlot = "toolbar" | "statusBar" | "trackHeader";

export function isPlayheadTimePlacement(value: unknown): value is PlayheadTimePlacement {
    return (
        typeof value === "string" && (PLAYHEAD_TIME_PLACEMENTS as readonly string[]).includes(value)
    );
}

/**
 * 把 `auto` 解析成真正的槽位（或 `hidden`）。
 *
 * 【为什么 auto 落在状态栏】它始终存在、不占工具栏宽度、在插件 640×400 下也可见。
 * 未来若产品希望"独立 App 的 auto = 工具栏"，只改这一个函数即可 —— 这正是把它
 * 单独抽出来的理由：调用点不该各自知道"auto 到底是哪儿"。
 */
export function resolvePlayheadTimePlacement(
    placement: PlayheadTimePlacement,
): PlayheadTimeSlot | "hidden" {
    if (placement === "auto") return "statusBar";
    return placement;
}

/** 读数的本地化标签键（设置页与右键菜单共用）。 */
export type PlayheadTimePlacementLabelKey =
    | "playhead_time_placement_auto"
    | "playhead_time_placement_toolbar"
    | "playhead_time_placement_status_bar"
    | "playhead_time_placement_track_header"
    | "playhead_time_placement_hidden";

export function playheadTimePlacementLabelKey(
    placement: PlayheadTimePlacement,
): PlayheadTimePlacementLabelKey {
    switch (placement) {
        case "auto":
            return "playhead_time_placement_auto";
        case "toolbar":
            return "playhead_time_placement_toolbar";
        case "statusBar":
            return "playhead_time_placement_status_bar";
        case "trackHeader":
            return "playhead_time_placement_track_header";
        case "hidden":
            return "playhead_time_placement_hidden";
    }
}
