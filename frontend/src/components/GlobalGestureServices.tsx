/**
 * 进程级全局手势基建：自愈式修饰键跟踪（淡化曲率等 `modifierOnly` 键位）。
 *
 * 【为什么单独一个文件】它原先定义在 `mount.tsx` 里。而 `mount.tsx` 导出的是
 * `mountApp`（一个普通函数，不是组件），同时定义组件就会触发
 * `react-refresh/only-export-components` —— 该规则要求"导出组件的文件只导出组件"，
 * 否则热更新无法按组件粒度替换。本组件本身没有任何需要与 `mountApp` 共享的闭包，
 * 拆出来比加一条豁免注释更干净。
 */

import { useEffect } from "react";

import { initModifierWatcher } from "./layout/timeline/hooks/modifierWatcher";

export function GlobalGestureServices() {
    useEffect(() => initModifierWatcher() ?? undefined, []);
    return null;
}
