/**
 * 播放光标时间读数的**放置位置**。
 *
 * 【为什么这是一个用户偏好而不是布局属性】读数在工具栏、状态栏、轨道头三处都是
 * chrome，不属于任何一个面板；布局树只描述面板几何与标签页。把它塞进 DockLayout
 * 会让"换个位置"变成"改布局"，用户还得为此保存/重载一套布局。
 *
 * 【为什么有 `auto`】用户不该被迫先想清楚"我到底要放哪儿"再去看时间。`auto` 让
 * 应用按一条**优先级链**挑一处（见 [AUTO_PLACEMENT_PRIORITY]），链上的每一级都以
 * "那个槽位此刻真的挂载着"为条件 —— 因此它是一条真的会顺延的链，而不是把某一个
 * 槽位写死成默认值。
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

/** 各槽位此刻是否真的挂载（由槽位自己登记，见 `transport/playheadTimeSlots`）。 */
export type PlayheadTimeSlotAvailability = Readonly<Record<PlayheadTimeSlot, boolean>>;

/**
 * `auto` 的优先级链：**工具栏 → 状态栏 → 轨道头**。
 *
 * 【为什么工具栏排第一】传输控件就在工具栏里，读数紧挨着它才读得通（眼睛不用横跨
 * 整个窗口去对时间与播放状态）。
 *
 * 【为什么状态栏不能排第一】它在本窗口里**永远存在**。把它放在链首，这条链就永远
 * 走不到第二级 —— "自动"退化成"状态栏"的别名，用户选它和不选它看到的是同一处
 * （这正是它被报告的问题）。链首必须是**可能不在**的那一处，顺延才有意义。
 *
 * 【为什么轨道头排在最后】它是紧凑的从属读数（轨道列宽有限），适合作兜底，
 * 不适合作默认。
 */
export const AUTO_PLACEMENT_PRIORITY: readonly PlayheadTimeSlot[] = [
    "toolbar",
    "statusBar",
    "trackHeader",
];

export function isPlayheadTimePlacement(value: unknown): value is PlayheadTimePlacement {
    return (
        typeof value === "string" && (PLAYHEAD_TIME_PLACEMENTS as readonly string[]).includes(value)
    );
}

/**
 * 把放置策略解析成**此刻真正该渲染的槽位**（或 `hidden`）。
 *
 * `availability` 是各槽位的挂载事实：
 * - 省略 / `null` ⇒ 还没有测到任何槽位（首帧、单元测试、没有槽位的窗口）。
 *   此时按"链首"处理 —— 让读数第一帧就出现在它该在的地方，而不是先闪没再出现。
 * - 给了 ⇒ 严格按链顺延；一个槽位都没挂载时返回 `hidden`。
 *
 * 显式选择**不受** `availability` 影响：用户说"放轨道头"就是轨道头，没有轨道头
 * 时看不见是用户自己的选择（与 `hidden` 同一类结果），而不是被悄悄搬去别处。
 */
export function resolvePlayheadTimePlacement(
    placement: PlayheadTimePlacement,
    availability?: PlayheadTimeSlotAvailability | null,
): PlayheadTimeSlot | "hidden" {
    if (placement !== "auto") return placement;
    if (!availability) return AUTO_PLACEMENT_PRIORITY[0];
    return AUTO_PLACEMENT_PRIORITY.find((slot) => availability[slot]) ?? "hidden";
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
