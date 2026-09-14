/**
 * 时间轴渲染内核 · 双击 clip 的选区写入方式
 *
 * 【主要内容】
 * 判定一次「双击 clip」应当以哪种方式改写参数编辑器选区：`"replace"`（替换为
 * 本次范围）或 `"toggle"`（该块范围已被完整覆盖则挖掉，否则并入）。
 *
 * 【作用】
 * 参数编辑器的多区间选区有两个来源：**单击**（替换）与**按住
 * `modifier.clipRangeToParamSelection`（默认 Alt）双击**（并入 / 挖掉）。内核
 * 只负责识别手势与修饰键，真正的选区运算在 `PianoRollPanel` 侧。
 *
 * 【为什么要抽成纯函数】这条手势在历史上**被合并整个吞掉过一次**：它原本写在旧
 * 组件 `ClipItem` 里，而旧组件随后被「内核唯一路径」改造删除，手势随之消失——
 * 但接收端（`PianoRollPanel` 的 `add` / `toggle` 分支）还在，于是表现为
 * 「功能静默失效、也没有任何报错」。抽成纯函数 + 单测后，无论宿主怎么重写，
 * 「按 Alt 双击 = toggle」都有可执行的断言钉住。
 *
 * 【与其他模块的关系】
 * - 上游：`host/timelineKernelHost` 在双击分派处调用。
 * - 下游：`TimelinePanel` 把结果塞进 `hifi:editOp/selectClipParamRange` 的
 *   `mode` 字段；`PianoRollPanel` 据此调用 `addBeatRange` / `toggleBeatRange`。
 * - 独立性：只依赖 `isModifierActive`（纯函数）与 `Keybinding` 类型，无 DOM / React。
 *
 * 【为什么绑 Alt】单击已被「替换选区」占用；Ctrl 在时间轴上属于多选切换、Shift
 * 属于范围选择，而 Alt 在**点击**层面是空的（它的时间轴绑定都是拖拽：slip /
 * stretch / 淡变曲率），因此不与任何既有手势冲突。
 */

import { isModifierActive } from "../../../../../features/keybindings/keybindingsSlice";
import type { Keybinding } from "../../../../../features/keybindings/types";

/** 双击 clip 后参数编辑器选区的写入方式。 */
export type ClipDoubleClickMode = "replace" | "toggle";

/** 修饰键状态快照（结构上兼容 `isModifierActive` 的 event 参数）。 */
export interface DoubleClickModifiers {
    readonly ctrlKey: boolean;
    readonly shiftKey: boolean;
    readonly altKey: boolean;
    readonly metaKey?: boolean;
}

/**
 * 解析双击 clip 的选区写入方式。
 *
 * 流程：绑定为空 → 视为未请求（返回 `"replace"`，与不按修饰键同义）；否则用
 * 与其它修饰键判定**同一个** `isModifierActive`（保证「用户改绑」在内核各手势
 * 间表现一致，不会出现某个手势写死按键）。
 *
 * @param binding `modifier.clipRangeToParamSelection` 的当前绑定；缺省 / null
 *   表示该动作没有绑定。
 * @param event 本次 pointerdown 的修饰键状态。
 * @returns `"toggle"` = 按住该修饰键；`"replace"` = 其余一切情况（缺省行为，
 *   与旧实现的普通双击逐字一致）。
 */
export function resolveClipDoubleClickMode(
    binding: Keybinding | null | undefined,
    event: DoubleClickModifiers,
): ClipDoubleClickMode {
    if (binding == null) return "replace";
    return isModifierActive(binding, event) ? "toggle" : "replace";
}
