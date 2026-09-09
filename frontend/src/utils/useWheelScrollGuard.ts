/**
 * 非被动滚轮守卫。
 *
 * React ≥17 把 `wheel` 以 passive 监听挂在根节点上：合成事件里的
 * `preventDefault()` 是 no-op（控制台告警 "Unable to preventDefault inside
 * passive event listener"），滚轮步进滑块的同时会滚动可滚动的祖先容器。
 *
 * 修复方式：值的步进仍由 React `onWheel` 完成；本 hook 在目标元素上挂
 * **原生非被动**监听，在原生层面阻止该事件的默认滚动行为。
 */
import { useEffect, useRef } from "react";
import type { RefObject } from "react";

export function useWheelScrollGuard<E extends HTMLElement>(
    /** 仅当事件目标命中该选择器时才阻止默认滚动；缺省 = 元素自身。 */
    selector?: string,
): RefObject<E | null> {
    const ref = useRef<E | null>(null);
    useEffect(() => {
        const el = ref.current;
        if (!el) return;
        const onWheel = (e: WheelEvent) => {
            if (selector) {
                const target = e.target;
                if (!(target instanceof Element) || !target.closest(selector)) return;
            }
            e.preventDefault();
        };
        el.addEventListener("wheel", onWheel, { passive: false });
        return () => el.removeEventListener("wheel", onWheel);
    }, [selector]);
    return ref;
}
