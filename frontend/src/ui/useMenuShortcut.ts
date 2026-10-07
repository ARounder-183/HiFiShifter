/**
 * 菜单项右侧的快捷键文案 —— 全应用**唯一**的取法。
 *
 * 【为什么必须是共享的】"读当前绑定 → 格式化成可读文本"这三行此前在两处各写了一遍
 * （`EditContextMenu` 的私有 hook、`TrackList` 里的三个内联选择器），而菜单栏
 * （`MenuBar.shortcutLabel`）是第三份。三份实现意味着三条会各自漂移的路径：
 * 时间轴那几个改用 `AppContextMenu` 的菜单正是因为"顺手只填了 label"，把
 * `shortcut` 整条信息丢掉了 —— 同一张右键菜单里，剪辑菜单有快捷键、轨道区域菜单
 * 没有，用户看到的是同一种菜单的两种规格。
 *
 * 【契约】未绑定（`__none__`）返回 `undefined`，调用方据此**不渲染**那一列
 * （原语在 `shortcut` 为空时不占位）。这与 `MenuBar` 的约定一致：空绑定显示成
 * `—` 会被误读成"这个键就是短横线"。
 *
 * 【一个动作绑多个键时全部显示】用户既然绑了两个键，菜单里就得让他看见两个 ——
 * 只显示主绑定会让他以为备用键没生效。多个文本用 `;` 连接（见
 * `formatKeybindingList`）。
 */
import { useAppSelector } from "../app/hooks";
import { formatKeybindingList, selectKeybindings } from "../features/keybindings/keybindingsSlice";
import type { ActionId } from "../features/keybindings/types";

/**
 * 取某动作当前生效的快捷键文本。
 *
 * 走 `useAppSelector`，因此用户在「快捷键设置」里改绑后，所有菜单下次打开即是新值。
 *
 * @param actionId 动作 id。
 * @returns 展示用文本（多绑定以 `;` 连接）；未绑定时为 `undefined`。
 */
export function useMenuShortcut(actionId: ActionId): string | undefined {
    const bindings = useAppSelector((state) => selectKeybindings(state, actionId));
    return formatKeybindingList(bindings, "") || undefined;
}
