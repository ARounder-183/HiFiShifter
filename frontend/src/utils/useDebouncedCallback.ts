/**
 * 去抖回调。
 *
 * 【为什么需要】有些控件的变更处理会 dispatch 一个**同步 Tauri 命令**。Tauri 2.10 里
 * 只有 `async fn` 命令会进线程池 —— 普通 `fn` 命令的**命令体在 UI 线程上内联执行**
 * （`tauri-macros` 的 `ExecutionContext::Blocking` / `body_blocking`），因此任何后端耗时
 * 都会阻塞窗口消息泵。逐格调用这类命令（滚轮、连续点选）累计约 5s 饥饿即被判"未响应"。
 *
 * 帧合并（`useFrameCommit`）解决的是"同一帧内合并"；跨帧的连续操作还需要**去抖**：
 * 只有在操作停止后才真正下发一次。
 *
 * 【与 `useDebouncedPersist` 的区别】那个是 localStorage 专用（同步磁盘 I/O），
 * 且其文档已注明当前无使用者。本 hook 面向"下发命令"，并在卸载时**补发**
 * 最后一次 —— 用户改完就关窗是常见操作，丢掉那次改动等于改了不生效。
 */
import { useCallback, useEffect, useRef } from "react";

export interface DebouncedCallback<A extends unknown[]> {
    /** 记下本次参数并重置计时；只有停止调用 `delayMs` 后才真正执行一次。 */
    call: (...args: A) => void;
    /** 立即执行挂起的那次（若有）。 */
    flush: () => void;
    /** 丢弃挂起的那次，不执行。 */
    cancel: () => void;
}

export function useDebouncedCallback<A extends unknown[]>(
    callback: (...args: A) => void,
    delayMs: number,
): DebouncedCallback<A> {
    const callbackRef = useRef(callback);
    useEffect(() => {
        callbackRef.current = callback;
    });

    const pendingRef = useRef<{ args: A } | null>(null);
    const handleRef = useRef<ReturnType<typeof setTimeout> | null>(null);

    const run = useCallback(() => {
        if (handleRef.current !== null) {
            clearTimeout(handleRef.current);
            handleRef.current = null;
        }
        const pending = pendingRef.current;
        pendingRef.current = null;
        if (pending !== null) callbackRef.current(...pending.args);
    }, []);

    const call = useCallback(
        (...args: A) => {
            pendingRef.current = { args };
            if (handleRef.current !== null) clearTimeout(handleRef.current);
            handleRef.current = setTimeout(run, delayMs);
        },
        [run, delayMs],
    );

    const cancel = useCallback(() => {
        if (handleRef.current !== null) {
            clearTimeout(handleRef.current);
            handleRef.current = null;
        }
        pendingRef.current = null;
    }, []);

    /*
     * 卸载时补发：用户"改完就关窗"是常见操作，丢掉最后一次改动等于改了不生效。
     * 补发发生在 cleanup 里，此时父组件仍可接收 dispatch。
     */
    useEffect(() => run, [run]);

    return { call, flush: run, cancel };
}
