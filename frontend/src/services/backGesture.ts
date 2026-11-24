/**
 * Android 的返回键 / 返回手势。
 *
 * 【为什么需要它】Tauri 的 `TauriActivity` 把 `handleBackNavigation` 关掉了
 * （它只在多页 WebView 里有意义），于是返回键直接走系统默认 —— `finish()`，
 * **整个应用退出**，哪怕用户正开着对话框、抽屉或快速搜索。移动端的约定是
 * "返回键先关掉最上面那一层"，而"最上面那一层是什么"只有 DOM 知道。
 *
 * 因此判定放在网页里：原生侧按返回键时执行 `window.__HFS_HANDLE_BACK__()`，
 * 返回 `"exit"` 才真的退出（见 `MainActivity.kt`）。原生侧不再维护第二份
 * "打开了什么"的栈 —— 那必然与真实界面分叉。
 *
 * ## 三层判定，顺序即优先级
 *
 * 1. **Radix 浮层**（对话框 / 菜单 / 下拉 / 快速搜索）：给它们发一次 Escape。
 *    这是它们的原生关闭手势，会走各自的 `onOpenChange` 与未保存护栏 ——
 *    比我们自己去找"关闭按钮在哪"可靠得多。
 * 2. **注册过的处理器**（抽屉这类自绘浮层）：见 `registerBackHandler`。
 * 3. 都没有 → 退出。
 *
 * 【为什么不做"再按一次退出"】应用已有自动备份与显式保存，而 Android 的返回键
 * 在根界面上就是"离开应用"。多一次按键会把这个最常用的手势变成两下，
 * 却换不来任何数据安全 —— 真正的护栏在保存路径上，不在返回键上。
 */

/** 自绘浮层注册的返回键处理器；返回 `true` 表示"我处理掉了"。 */
type BackHandler = () => boolean;

const handlers: BackHandler[] = [];

/**
 * 注册一个返回键处理器（后注册的优先）。
 *
 * 【为什么是栈而不是单个回调】抽屉、未来的全屏面板、编辑模式……都可能有自己的
 * 一层；单回调会让最后挂载的那个独占返回键，而先挂载的那个再也关不掉。
 */
export function registerBackHandler(handler: BackHandler): () => void {
    handlers.push(handler);
    return () => {
        const index = handlers.lastIndexOf(handler);
        if (index >= 0) handlers.splice(index, 1);
    };
}

/**
 * 需要"先关掉"的 Radix 浮层。
 *
 * `role="menu"` 覆盖下拉菜单与上下文菜单；`data-radix-popper-content-wrapper`
 * 覆盖没有 role 的浮层（Select 的弹层等）。它们都是 `position: fixed` 且挂在
 * `body` 下，只有打开时才存在。
 */
const OVERLAY_SELECTOR = [
    '[role="dialog"]',
    '[role="alertdialog"]',
    '[role="menu"]',
    '[role="listbox"]',
    "[data-radix-popper-content-wrapper]",
].join(", ");

/** 给最上层浮层发一次 Escape；没有浮层时返回 `false`。 */
function closeTopmostOverlay(): boolean {
    if (typeof document === "undefined") return false;
    const overlay = document.querySelector(OVERLAY_SELECTOR);
    if (!overlay) return false;
    /*
     * 在 `document` 上派发而不是在浮层元素上：Radix 的 `DismissableLayer`
     * 把 `keydown` 监听挂在 `ownerDocument` 上（portal 的宿主文档），
     * 派发到元素上需要它恰好在冒泡路径里；派发到 `document` 两种情况都覆盖。
     */
    document.dispatchEvent(
        new KeyboardEvent("keydown", {
            key: "Escape",
            code: "Escape",
            bubbles: true,
            cancelable: true,
        }),
    );
    return true;
}

/**
 * 装上返回键处理器。
 *
 * 【为什么挂在 `window` 上而不是模块内变量】原生侧通过
 * `evaluateJavascript` 执行 `window.__HFS_HANDLE_BACK__()` —— 只能看见全局。
 */
export function installBackGesture(): void {
    if (typeof window === "undefined") return;
    (window as unknown as { __HFS_HANDLE_BACK__?: () => string }).__HFS_HANDLE_BACK__ = () => {
        if (closeTopmostOverlay()) return "handled";
        for (let index = handlers.length - 1; index >= 0; index -= 1) {
            if (handlers[index]()) return "handled";
        }
        return "exit";
    };
}
