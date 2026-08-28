/**
 * 非被动滚轮监听。
 *
 * React 17+ 在 root 上把 `wheel` 注册为 passive 监听：合成事件里的
 * `event.preventDefault()` 是空操作（浏览器会打干预警告），导致"滚轮调值"
 * 的控件同时滚动可滚动容器。需要阻止滚动的地方必须用原生
 * `addEventListener("wheel", ..., { passive: false })`（与时间轴/
 * 钢琴卷帘的做法一致）。
 */

import { useEffect, useRef } from "react";
import type { WheelEvent as ReactWheelEvent } from "react";

export function useNonPassiveWheel<E extends HTMLElement>(
    handler: (event: ReactWheelEvent<E>) => void,
): React.RefObject<E | null> {
    const ref = useRef<E | null>(null);
    const handlerRef = useRef(handler);

    useEffect(() => {
        handlerRef.current = handler;
    });

    useEffect(() => {
        const element = ref.current;
        if (!element) return;
        const listener = (event: WheelEvent) => {
            handlerRef.current(event as unknown as ReactWheelEvent<E>);
        };
        element.addEventListener("wheel", listener, { passive: false });
        return () => {
            element.removeEventListener("wheel", listener);
        };
    }, []);

    return ref;
}
