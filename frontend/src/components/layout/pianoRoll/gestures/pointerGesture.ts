/**
 * 一次指针手势的**注册 / 注销协议**。
 *
 * # 它解决的是什么
 *
 * 参数编辑器里有十余处「按下之后接管指针，直到松手」的手势。每一处都要做同样五件事：
 * 在 window 上装 pointermove / pointerup / pointercancel、setPointerCapture、把收尾登记
 * 给面板（失焦时会被调用）、收尾时把上面全部撤销、并且**只能收尾一次**。
 *
 * 手写这五件事已经出过一次事故：`710ff764` 把指针按下拆成两个模块时，自由绘制那一支
 * 整块丢失 —— 没有 pointermove、没有 pointerup、从不 commitStroke。当时没有任何测试
 * 驱动过完整的 down→move→up 序列（那条缺口现在由 freehandPointerDown.test.ts 与
 * gestureLifecycle.adoption.test.ts 补上）。
 *
 * 本模块把五件事收成一次调用：漏掉任何一件都不再可能，因为调用方**拿不到**那些
 * addEventListener。
 *
 * # 为什么 end 是幂等的
 *
 * 三条收尾路径会竞争：真实 pointerup、pointercancel、以及各手势自己的「按键已松开 /
 * 换工具 / 失焦」分支。它们都会调 end；重复执行会把提交跑两遍（例如同一笔画提交两次）。
 * 因此 active 标志在**进入**时就置位，后来者直接返回。
 *
 * # 为什么把注册函数本身传给面板
 *
 * `setActivePointerGestureEnd` / `clearActivePointerGestureEnd` 按**函数身份**配对
 * （见 usePianoRollInteractions：身份不匹配时不清除，避免后来的手势把前一个的登记抹掉）。
 * 因此这里固定一个 endFromFocus，注册与注销用的是同一个引用。
 */

/** 收尾原因：正常松手，还是 OS / 浏览器取消（触摸被打断、指针被抢）。 */
export type GestureEndReason = "up" | "cancel";

/** 随手势一起安装与注销的额外 window 监听（右键菜单守卫等）。 */
export interface GestureExtraListener {
    type: string;
    listener: EventListener;
    capture?: boolean;
}

export interface PointerGestureOptions {
    /** 本次手势的指针 id：其余指针的 move 一律忽略（掌压 / 第二根手指）。 */
    pointerId: number;
    /** 捕获指针的元素（画布）。缺省则不捕获 —— 例如手势本来就只在元素内。 */
    captureTarget?: Element | null;
    /** 指针移动。调用方自己决定哪些按键算数（左键 1 / 右键 2 / 笔杆键 32）。 */
    onMove: (event: globalThis.PointerEvent) => void;
    /**
     * 收尾。**监听与捕获在调用它之前就已经撤销**，因此这里可以安全地做提交。
     *
     * @param reason 正常松手 / 被取消。被取消时调用方应当回滚而不是提交。
     * @param event 触发收尾的事件；程序内主动收尾时为 null。
     */
    onEnd: (reason: GestureEndReason, event: globalThis.PointerEvent | null) => void;
    /** 额外的 window 监听，随手势一起安装与注销。 */
    extra?: readonly GestureExtraListener[];
    /** 把收尾登记给面板（窗口失焦 / 换工具时由面板调用）。 */
    registerEnd?: (end: () => void) => void;
    /** 撤销登记。必须传**同一个**引用（见模块头）。 */
    clearEnd?: (end: () => void) => void;
}

export interface PointerGesture {
    /** 主动收尾（幂等）。onMove 里发现按键已松开时调它。 */
    end(reason?: GestureEndReason, event?: globalThis.PointerEvent | null): void;
    /** 是否仍然生效（收尾后为 false）。 */
    readonly active: boolean;
}

/**
 * 接管一次指针手势，返回它的收尾句柄。
 *
 * 【为什么不在函数内部调 onMove / onEnd】它只做登记与注销。手势的业务逻辑全部在
 * 调用方的两个回调里 —— 这样搬一段手势代码不需要同时搬协议。
 */
export function startPointerGesture(options: PointerGestureOptions): PointerGesture {
    const {
        pointerId,
        captureTarget = null,
        onMove,
        onEnd,
        extra = [],
        registerEnd,
        clearEnd,
    } = options;
    let active = true;

    const onPointerMove = (event: Event) => {
        if (!active) return;
        const pointerEvent = event as globalThis.PointerEvent;
        if (pointerEvent.pointerId !== pointerId) return;
        onMove(pointerEvent);
    };

    const end = (reason: GestureEndReason = "up", event: globalThis.PointerEvent | null = null) => {
        if (!active) return;
        active = false;
        window.removeEventListener("pointermove", onPointerMove);
        window.removeEventListener("pointerup", onPointerUp);
        window.removeEventListener("pointercancel", onPointerCancel);
        for (const item of extra) {
            window.removeEventListener(item.type, item.listener, item.capture ?? false);
        }
        if (captureTarget !== null) {
            try {
                captureTarget.releasePointerCapture(pointerId);
            } catch {
                // 指针已经被浏览器释放（元素被卸载 / 指针已不存在）。不是错误。
            }
        }
        clearEnd?.(endFromFocus);
        onEnd(reason, event);
    };

    /** 面板登记的收尾入口。**引用必须恒定**，见模块头。 */
    const endFromFocus = () => end("up");

    const onPointerUp = (event: Event) => end("up", event as globalThis.PointerEvent);
    const onPointerCancel = (event: Event) => end("cancel", event as globalThis.PointerEvent);

    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", onPointerUp);
    window.addEventListener("pointercancel", onPointerCancel);
    for (const item of extra) {
        window.addEventListener(item.type, item.listener, item.capture ?? false);
    }
    registerEnd?.(endFromFocus);

    return {
        end,
        get active() {
            return active;
        },
    };
}
