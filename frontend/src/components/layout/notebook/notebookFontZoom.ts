/*
 * 记事本编辑区字号的滚轮缩放（Ctrl/⌘ + 滚轮）。
 *
 * 【为什么单独成文件】滚轮手势里唯一需要判断的是"这一步让字号变成多少"，而它
 * 有三个边界情形（向上到顶、向下到底、本次事件不产生缩放步），每个都对应一次
 * "要不要写设置 + 要不要落盘"。把它从面板组件里提出来就能直接测这些边界 ——
 * 面板要挂 TipTap 才能渲染，边界情形在那种测试里几乎不会被覆盖。
 *
 * 【为什么复用内核的滚轮判定】"Ctrl+滚轮 = 缩放"在本仓库已有实现（时间轴 /
 * 钢琴卷帘的画布缩放），而它踩过的两个坑在这里一模一样：
 *   - `deltaY < 0 ? 放大 : 缩小` 会把 `deltaY === 0`（纯横向手势、纵向噪声归零）
 *     判成缩小；
 *   - 没有幅度门限时，precision touchpad 的小幅变号增量会让方向逐事件翻转，
 *     表现为"剧烈抖动"而不是缩放。
 * 因此这里**不自己发明判定**：方向由内核的 `resolveWheelZoomStep`（主轴 + 死区
 * 累积）给出，本模块只把它的方向换算成字号，并按 `stepPolicy` 的 `pixels` 单位
 * 走一格（与设置对话框里的输入框、方向键同一份步长语义）。
 */

import { stepValue } from "../../../ui/stepPolicy";
import type { WheelZoomDirection } from "../timeline/kernel/input/wheelZoomIntent";
import { NOTEBOOK_FONT_SIZE_MAX, NOTEBOOK_FONT_SIZE_MIN } from "./notebookSettings";

/**
 * `deltaMode = 1`（行）换算成像素时的一行高度。
 *
 * 与内核滚轮输入层用的值一致（`timelineKernelHost` 的 `lineHeightPx: 16`）：
 * 这不是"字号"，而是把"滚了 3 行"折成像素去和死区比较的换算系数，因此取浏览器
 * 常规行高 16px，而不是当前字号 —— 字号小的时候一格滚轮会显得更"重"。
 */
export const NOTEBOOK_ZOOM_LINE_HEIGHT_PX = 16;

/**
 * 从 `current` 出发按一步缩放后的字号；本次不缩放或已到界时返回 `null`。
 *
 * 【方向约定】内核用 `-1 = 放大`、`1 = 缩小`、`0 = 本次不缩放`（与滚轮向上
 * 放大一致）；`stepPolicy` 用 `+1 = 加一格`。两个约定在这里换算一次，别处不必知道。
 *
 * 【为什么返回 null 而不是原值】调用方据此决定"不写设置、不落盘"。若返回原值，
 * 用户滚到 24 之后继续滚，每一格都会照样 dispatch 一次相同值并触发一次去抖落盘
 * —— 手势结束时那次保存纯属白费。
 */
export function nextFontSizeForZoomStep(
    current: number,
    direction: WheelZoomDirection,
): number | null {
    if (direction === 0) return null;
    const next = stepValue({
        value: current,
        // 内核的"放大"是 -1，而 stepPolicy 的"向上"是 +1。
        direction: direction < 0 ? 1 : -1,
        unit: "pixels",
        // 字号没有"精细一档"：`pixels` 的粗/精都是 1，所以这里恒为 false。
        fine: false,
        min: NOTEBOOK_FONT_SIZE_MIN,
        max: NOTEBOOK_FONT_SIZE_MAX,
    });
    return next === current ? null : next;
}
