/**
 * "本实例是不是 Transfer 实例"的**唯一**判据。
 *
 * 【为什么需要它，以及为什么有两个来源】Transfer 面板此前只有一条入口：窗口打开时自动
 * 浮出。而那个判定依赖"形态"，形态要等 `setActive` 之后 `ARA_BINDING_GRACE`（5 秒）才
 * 落定，bootstrap 又是**窗口创建那一刻**算的 —— 于是窗口开得早的会话里两个来源都还没有
 * 答案，入口就彻底消失了（用户实测：非 ARA 宿主里看不到任何 Transfer 入口）。
 *
 * 两个来源取**并集**，任一说"是 Transfer"就算：
 *   * bootstrap 的 `hostKind`：窗口创建时形态已经落定（用户稍后才开窗）；
 *   * 状态快照的 `mode`：后端**立刻**知道自己是哪一份会话（`session.is_transfer()`），
 *     跟着 250ms 的既有轮询到达。
 *
 * 【为什么"未知"时不显示】显示一个在 ARA 宿主里毫无意义的入口只会让人点开一个空面板。
 * 两个来源都在 ≤250ms 内到达，因此"未知"的窗口很短。
 */
import { useSyncExternalStore } from "react";

import { pluginBootstrap } from "../../services/hostDetect";
import { isPluginMode } from "../../services/hostCapabilities";
import { getPluginApplySnapshot, subscribePluginApply } from "../ara/pluginApplyStore";

export function isTransferInstance(): boolean {
    if (!isPluginMode()) return false;
    if (getPluginApplySnapshot().state?.mode === "transfer") return true;
    return pluginBootstrap()?.hostKind === "transfer";
}

/**
 * 订阅版：形态落定之后菜单项与面板会自动出现，不需要用户重开窗口。
 *
 * `useSyncExternalStore` 的判据是**返回值相等**，而这里返回布尔 —— 因此 250ms 一次的
 * 轮询不会让菜单栏重渲染（只有真正翻转的那一次会）。
 */
export function useTransferInstance(): boolean {
    return useSyncExternalStore(subscribePluginApply, isTransferInstance, isTransferInstance);
}
