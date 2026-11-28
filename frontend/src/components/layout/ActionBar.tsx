/**
 * 桌面顶栏的动作栏（薄壳）。
 *
 * 【为什么这里只剩一行】工具栏的**控件定义**只有一份，住在
 * `toolbar/ToolbarStrip.tsx`：移动端的工具栏 sheet 渲染的是同一个组件，只是
 * `variant="sheet"`。历史上这里是 2000 行、只服务桌面 —— 于是移动端"没有工具栏"
 * 这件事在代码里根本看不出来（不是被藏起来，而是从来没被渲染）。
 *
 * 【为什么薄壳还要存在】`App.tsx` 与既有测试都按 `ActionBar` 装配桌面形态；
 * 把调用点全改成 `ToolbarStrip variant="desktop"` 只是把同一个事实写两遍。
 */
import { ToolbarStrip } from "./toolbar/ToolbarStrip";

export function ActionBar() {
    return <ToolbarStrip variant="desktop" />;
}
