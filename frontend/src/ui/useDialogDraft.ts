/**
 * 对话框草稿状态钩子。
 *
 * 【为什么单独成文件】`Dialog.tsx` 只导出组件，以便 React Fast Refresh
 * 在编辑时保持组件状态；导出一个普通函数会破坏那条规则
 * （react-refresh/only-export-components）。
 */
import { useEffect, useRef, useState } from "react";

/**
 * 草稿状态：对话框打开时从 props 播种本地状态，关闭后不保留。
 *
 * 【为什么需要它】审查发现 8 个近似相同的 `useEffect` 块，每块都带一句
 * 各自的 `eslint-disable` 注释，做同一件事："`open` 翻成 true 时把外部值
 * 拷进本地草稿"。另外还有几处用 `key` 重挂载整个组件来达到同样效果。
 *
 * 语义要点：**只在 `open` 由 false 变 true 时重新播种**。若依赖数组里
 * 放了 `initial`，用户编辑到一半时上游刷新会把草稿冲掉。
 *
 * @param open 对话框开关。
 * @param initial 打开瞬间读取的初始值。
 */
export function useDialogDraft<T>(open: boolean, initial: () => T): [T, (next: T) => void] {
    const [draft, setDraft] = useState<T>(initial);
    const wasOpen = useRef(open);
    useEffect(() => {
        if (open && !wasOpen.current) setDraft(initial());
        wasOpen.current = open;
        // `initial` 刻意不入依赖：播种只应在「打开」这一事件上发生一次。
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [open]);
    return [draft, setDraft];
}
