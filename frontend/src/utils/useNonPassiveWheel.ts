/**
 * 非被动滚轮监听。
 *
 * React 17+ 在 root 上把 `wheel` 注册为 passive 监听：合成事件里的
 * `event.preventDefault()` 是空操作（浏览器会打干预警告），导致"滚轮调值"
 * 的控件同时滚动可滚动容器。需要阻止滚动的地方必须用原生
 * `addEventListener("wheel", ..., { passive: false })`（与时间轴/
 * 钢琴卷帘的做法一致）。
 *
 * 【为什么返回回调 ref 而不是 `RefObject`】曾经它返回一个普通 ref 对象，并在
 * `useEffect(..., [])` 里读 `ref.current` 挂监听。那条路径**只对无条件挂载的
 * 元素有效**：条件挂载的控件（如双击才出现的 Tempo Map 内联输入框）在 effect
 * 运行时还不存在，`ref.current` 是 null，而 effect 不会重跑 —— 监听器永远挂不上；
 * 调用方此时往往已经删掉了 React 的 `onWheel`，于是滚轮**完全没反应**。
 *
 * 回调 ref 把"元素出现"本身当作挂载时机：一出现就挂、一移除就摘，与元素何时出现
 * 无关。挂载逻辑抽成纯工厂 `createWheelAttacher`，便于直接单测（见同名测试）。
 */

import { useEffect, useRef, useState } from "react";
import type { WheelEvent as ReactWheelEvent } from "react";

/** 回调 ref 语义的挂载器：既可作为 `ref` 直接使用，也可显式 `dispose()`。 */
export interface WheelAttacher<E extends HTMLElement> {
    (element: E | null): void;
    /** 摘除当前监听器（组件卸载兜底）。幂等。 */
    dispose(): void;
}

/**
 * 创建挂载器（与 React 解耦的纯逻辑）。
 *
 * @param handler 事件处理器；由调用方保证它总是转发到最新的闭包。
 */
export function createWheelAttacher<E extends HTMLElement>(
    handler: (event: ReactWheelEvent<E>) => void,
): WheelAttacher<E> {
    let cleanup: (() => void) | null = null;
    const attach = ((element: E | null) => {
        // 幂等：React 替换元素时会先用 null 调用一次；这里先摘旧的再决定挂不挂。
        cleanup?.();
        cleanup = null;
        if (element === null) return;
        const listener = (event: WheelEvent) => {
            handler(event as unknown as ReactWheelEvent<E>);
        };
        element.addEventListener("wheel", listener, { passive: false });
        cleanup = () => element.removeEventListener("wheel", listener);
    }) as WheelAttacher<E>;
    attach.dispose = () => {
        cleanup?.();
        cleanup = null;
    };
    return attach;
}

/**
 * 非被动滚轮监听 hook。
 *
 * 【元素为什么存在 state 里而不是 ref 里】本仓库启用了 React Compiler 的引用
 * 规则：渲染期读写 `ref.current` 都会被判为违规，**惰性初始化函数（渲染期执行）
 * 里出现 `ref.current` 同样如此**。回调 ref + `useState` 是仓库既有写法
 * （见 `DockTabBar` 的 `setBarElement`）：既让"元素出现"成为 effect 的依赖，
 * 又不触碰引用规则。
 *
 * @returns 回调 ref（`setElement` 身份稳定，可直接传给 `ref=`）。
 */
export function useNonPassiveWheel<E extends HTMLElement>(
    handler: (event: ReactWheelEvent<E>) => void,
): (element: E | null) => void {
    const [element, setElement] = useState<E | null>(null);

    // handler 经 ref 转发：调用方每次渲染都会重建闭包，直接依赖它会让监听器
    // 反复摘挂。写入发生在 effect 里（不是渲染期），符合引用规则。
    const handlerRef = useRef(handler);
    useEffect(() => {
        handlerRef.current = handler;
    });

    // 元素出现（或换成另一个）时挂监听、移除时摘掉 —— 与元素何时出现无关。
    useEffect(() => {
        if (element === null) return;
        const attacher = createWheelAttacher<E>((event) => handlerRef.current(event));
        attacher(element);
        return () => attacher.dispose();
    }, [element]);

    return setElement;
}
