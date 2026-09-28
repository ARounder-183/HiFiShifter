/*
 * 浮窗标题栏的动作矩阵。
 *
 * 【为什么抽成纯函数】标题栏四枚动作（折叠 / 拆分 / 重停 / 关闭）的可见性各自
 * 由 `PanelDefinition` 的一条声明决定，此前直接散在 JSX 的条件渲染里 —— 于是
 * `dockable: false` 的外观设置面板仍然渲染着「重新停靠回主区域」，一键就能把
 * 面板塞回主区标签组，把拖拽层抑制掉的停靠通道整个绕过。把可见性收敛成一个
 * 可单测的函数，"声明 → 动作"的映射就只有这一处。
 */
import type { PanelDefinition } from "../../features/dock/panelRegistry";

/** 拆分动作的三态：可用 / 不可用（灰色解释占位）/ 不渲染。 */
export type FloatDetachAction = "available" | "unsupported" | "hidden";

export interface FloatTitleBarActions {
    /** 折叠为标题条。所有浮窗都有。 */
    collapse: boolean;
    /**
     * 拆分到独立窗口。注册且 `detachable` → 可用；注册但不可拆 → `unsupported`
     * （灰色占位 + 解释，与时间轴一致）；未注册 → 不渲染。
     */
    detach: FloatDetachAction;
    /**
     * 「重新停靠回主区域」。与**拖拽停靠**共用 `PanelDefinition.dockable`：
     * 拖不进去的面板也不该留一枚按钮绕道 dock 进去。
     */
    redock: boolean;
}

export function floatTitleBarActions(
    definition: PanelDefinition | undefined,
): FloatTitleBarActions {
    if (!definition) {
        // 未注册的面板（布局损坏时的兜底）：按可停靠的默认处理，但**不给**
        // 拆分动作 —— 没有声明就没有可信的解释文案，灰占位只会撒谎。
        return { collapse: true, detach: "hidden", redock: true };
    }
    return {
        collapse: true,
        detach: definition.detachable ? "available" : "unsupported",
        redock: definition.dockable ?? true,
    };
}
