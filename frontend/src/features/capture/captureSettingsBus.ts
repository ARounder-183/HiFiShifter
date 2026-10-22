/**
 * 捕获设置窗口的开合状态（模块级，零依赖）。
 *
 * 【为什么需要一个总线而不是把 open 提到某个组件的 props 里】入口有三处，而它们
 * 分散在三棵互不相邻的子树里：传输栏上的捕获按钮（右键）、「视图」菜单、以及
 * 捕获出错时那个图标。用一个 8 行的模块级 store 比把 `open` 一路穿过 ActionBar /
 * MenuBar / App 三个组件再传回来要短得多，也不会让"打开设置"这件事变成三份实现。
 *
 * 【为什么不用 Redux】它不是工程状态，不参与撤销、不落盘、没有第二个读者需要派生
 * 值 —— 与 `fadeContextMenuBus` / `snapHighlight` 同一套取舍。
 */

let open = false;
const listeners = new Set<() => void>();

function emit(): void {
    for (const listener of listeners) listener();
}

/** 订阅开合状态（`useSyncExternalStore` 的 subscribe）。 */
export function subscribeCaptureSettings(listener: () => void): () => void {
    listeners.add(listener);
    return () => {
        listeners.delete(listener);
    };
}

/** 当前是否打开（`useSyncExternalStore` 的 getSnapshot）。 */
export function getCaptureSettingsOpen(): boolean {
    return open;
}

/** 设置开合状态（对话框自身的 onOpenChange 用它）。 */
export function setCaptureSettingsOpen(next: boolean): void {
    if (open === next) return;
    open = next;
    emit();
}

/** 从任意入口打开设置窗口。 */
export function openCaptureSettings(): void {
    setCaptureSettingsOpen(true);
}

/** 仅供测试重置。 */
export function resetCaptureSettingsForTests(): void {
    open = false;
    listeners.clear();
}
