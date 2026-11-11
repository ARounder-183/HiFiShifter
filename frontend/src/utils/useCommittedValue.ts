/**
 * 稳定容器：把"最新值"在提交后同步给事件回调 / rAF / 外部订阅读取。
 *
 * 写入发生在 layout effect，渲染期既不读也不写 ref.current；读取方统一使用
 * box.current.value。这样不会把上一次渲染的 ref 值带进本次 UI，也不会触发
 * React Compiler 的 refs / immutability 诊断。
 */
import { useLayoutEffect, useRef } from "react";

export interface CommittedValue<T> {
    readonly current: { value: T };
}

export function useCommittedValue<T>(value: T): CommittedValue<T> {
    const box = useRef({ value });
    useLayoutEffect(() => {
        box.current.value = value;
    }, [value]);
    return box;
}
