/**
 * Transfer 入口的可见性，以及"本实例是不是 Transfer 实例"的判据。
 *
 * 【为什么入口与实例判定是两件事】入口必须**无条件**可用，实例判定则要等形态落定 ——
 * 把它们合成一个判据，就会出现"形态还没落定 ⇒ 入口不存在"（用户实测：非 ARA 宿主里
 * 看不到任何 Transfer 入口）。而形态要等 `setActive` 之后 `ARA_BINDING_GRACE`（5 秒）
 * 才落定，bootstrap 又是**窗口创建那一刻**算的 —— 窗口开得早的会话里根本没有答案。
 *
 * 因此：
 *   * [`showsTransferEntry`] —— 「视图」菜单项：**只要在插件里就显示**。宿主形态是
 *     独立 App 还是插件，与"形态落没落定"无关，永远知道答案。
 *   * [`isTransferInstance`] —— 面板的自动浮出：这是锦上添花，可以用"可能还不知道"的
 *     值，两个来源取并集（bootstrap 的 `hostKind` 与状态快照的 `mode`）。
 *
 * 【为什么在 ARA 宿主里也显示入口】Melodyne 的 Transfer 按钮同样是常驻的。让入口
 * 消失会让"这个宿主到底能不能用"变成一个谜；面板与 `transfer_start` 会如实说明
 * "宿主正在通过 ARA 供音，不需要 Transfer"（见 `transfer_status.araBound`）。
 */
import { useSyncExternalStore } from "react";

import { pluginBootstrap } from "../../services/hostDetect";
import { isPluginMode } from "../../services/hostCapabilities";
import { getPluginApplySnapshot, subscribePluginApply } from "../ara/pluginApplyStore";

/** 「视图」菜单里是否显示 Transfer 入口。插件形态下**恒真**（见文件头）。 */
export function showsTransferEntry(): boolean {
    return isPluginMode();
}

/**
 * 本实例是否走 Transfer 通路。
 *
 * 两个来源取并集，任一说"是 Transfer"就算：bootstrap 的 `hostKind`（窗口创建时形态
 * 已经落定）与状态快照的 `mode`（后端**立刻**知道自己是哪份会话，跟着 250ms 的既有
 * 轮询到达）。
 */
export function isTransferInstance(): boolean {
    if (!isPluginMode()) return false;
    if (getPluginApplySnapshot().state?.mode === "transfer") return true;
    return pluginBootstrap()?.hostKind === "transfer";
}

/**
 * 订阅版（形态落定之后自动浮出面板用）。
 *
 * `useSyncExternalStore` 的判据是**返回值相等**，而这里返回布尔 —— 因此 250ms 一次的
 * 轮询不会让订阅方重渲染（只有真正翻转的那一次会）。
 */
export function useTransferInstance(): boolean {
    return useSyncExternalStore(subscribePluginApply, isTransferInstance, isTransferInstance);
}
