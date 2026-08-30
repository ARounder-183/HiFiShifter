/**
 * gestureFocusGuard — 全局「手势失焦取消」注册表。
 *
 * 时间线所有拖拽的收尾都只挂在 window pointerup/pointercancel 上。当用户
 * 在拖拽途中把焦点切走（Alt+Tab 到桌面 / 其他程序、最小化等）并在**窗口外**
 * 松开鼠标时，WebView2（Chromium）通常不会把 pointerup/pointercancel 派发
 * 回本窗口 → end() 永不执行 → 切回后拖拽状态卡死（dragRef 悬置、交互锁与
 * 后端 undo group 泄漏、吸附高亮与 body 光标冻结）。
 *
 * 本模块提供一个惰性挂载的全局 blur/visibilitychange 监听：每个手势在
 * 开始时把自己的事件无关收尾 `finish()` 注册进来（registerDragAbort），
 * 失焦时统一调用（幂等、逐个异常隔离）。触发后各手势的 finish() 走的是与
 * pointerup/pointercancel 完全相同的收尾路径（提交当前值、关闭 undo
 * group、归还交互锁、清理监听器），因此撤销栈不会被冻结。
 *
 * 先例：钢琴卷帘 usePianoRollInteractions（activePointerGestureEndRef +
 * window blur / visibilitychange 收尾）。
 */

/** 最小事件目标接口：浏览器 window/document 与 node 测试桩都满足。 */
type FocusTarget = {
    addEventListener(type: "blur" | "visibilitychange", listener: () => void): void;
    removeEventListener(type: "blur" | "visibilitychange", listener: () => void): void;
    visibilityState?: string;
};

const activeAborts = new Set<() => void>();

let listenersAttached = false;

function targetWindow(): FocusTarget | null {
    const g = globalThis as { window?: FocusTarget };
    return g.window ?? null;
}

function targetDocument(): FocusTarget | null {
    const g = globalThis as { document?: FocusTarget };
    return g.document ?? null;
}

function fireAll(): void {
    if (activeAborts.size === 0) return;
    // 先快照再遍历：finish() 每次执行都会注销自己，Set 在遍历中删除安全，
    // 但快照更稳妥（避免"注销又注册"的极端情形跳过本轮）。
    for (const abort of [...activeAborts]) {
        try {
            abort();
        } catch {
            // 单个手势收尾失败不影响其他手势与后续收尾。
        }
    }
}

function ensureListeners(): void {
    if (listenersAttached) return;
    listenersAttached = true;
    // 通过 globalThis 解析（而非直接引用 window/document）：node 测试环境
    // 没有全局 window，本模块仍可注册、由测试桩驱动触发。
    targetWindow()?.addEventListener("blur", fireAll);
    targetDocument()?.addEventListener("visibilitychange", () => {
        // Alt+Tab 通常只触发 blur；最小化/遮挡到不可见才触发 visibilitychange。
        if (targetDocument()?.visibilityState !== "visible") {
            fireAll();
        }
    });
}

/**
 * 注册一个手势的事件无关收尾（finish）。
 *
 * @param abort 幂等的收尾函数：内部必须自守卫（状态为空即返回），
 *              并且第一步注销自己（与 pointerup 路径共用同一 finish 时
 *              防止双触发）。
 * @returns 注销函数（finish 内部也应调用它）。
 */
export function registerDragAbort(abort: () => void): () => void {
    if (!listenersAttached) {
        ensureListeners();
    }
    activeAborts.add(abort);
    let removed = false;
    return () => {
        if (removed) return;
        removed = true;
        activeAborts.delete(abort);
    };
}