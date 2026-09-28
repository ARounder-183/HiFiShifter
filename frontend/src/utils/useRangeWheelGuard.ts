/**
 * 裸 `range` 滑块的滚轮守卫（回调 ref）。
 *
 * 【它解决什么】React 的 `onWheel` 是 passive 监听，里面的 `preventDefault()`
 * 是空操作 —— 于是"滚轮调滑块"会连带滚动祖先容器，产生第二路视觉位移。
 * 滑块的值仍由 React `onWheel` 改，本守卫只负责在**原生层**阻止滚动。
 *
 * 【为什么它取代 `useWheelScrollGuard`】后者在 `useEffect(..., [selector])` 里
 * 读 `ref.current`，而条件挂载的元素（对话框内容、条件渲染的菜单）在 effect
 * 运行时还不存在，`ref.current` 是 null 且 effect 不会重跑 —— 监听器从未挂上。
 * 全仓 4 处调用因此**全是死代码**：滚轮既改了值、又滚了容器，正是用户报告的现象。
 *
 * 回调 ref 把"元素出现"本身当作挂载时机：一出现就挂、一移除就摘，
 * 与元素何时出现无关（`useNonPassiveWheel` 的头部注释已把这个区别讲清）。
 *
 * 【与 `AppSlider` 的关系】新代码直接用 `AppSlider`（内建滚轮与精细调整），
 * 不需要本 hook。它服务于尚未迁移的裸 `range`。
 */
import { useCallback } from "react";

import { useNonPassiveWheel } from "./useNonPassiveWheel";

/** 只有落在这些控件上的滚轮才拦截，容器内其他位置的滚轮照常滚动。 */
const RANGE_SELECTOR = 'input[type="range"]';

export function useRangeWheelGuard<E extends HTMLElement>(): (element: E | null) => void {
    const handler = useCallback((event: { target: EventTarget | null; preventDefault: () => void }) => {
        const target = event.target;
        if (!(target instanceof Element) || !target.closest(RANGE_SELECTOR)) return;
        event.preventDefault();
    }, []);
    return useNonPassiveWheel<E>(handler as never);
}
