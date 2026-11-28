/**
 * 长按 → 上下文菜单。
 *
 * 【为什么是"合成 contextmenu"而不是给 136 个处理器各加一条触摸分支】全仓有 136 处
 * `onContextMenu`（52 个文件），它们覆盖了片段、轨道、参数块、停靠标签、文件行…
 * 逐处改写意味着 136 次机会漏掉一处，而漏掉的症状是"这个对象在手机上不能右键"，
 * 没有任何报错。
 *
 * 合成事件的思路：长按触发时，向 `document.elementFromPoint(x, y)` **派发一个真正的
 * `contextmenu` 事件**。浏览器随后按正常的捕获/冒泡路径把它送到那个元素上，
 * 于是**每一个** `onContextMenu` 处理器都会像桌面右键一样被调用 —— 一行都不用改。
 *
 * 【为什么不是 `el.dispatchEvent`】React 17+ 把事件监听挂在根容器上，事件必须**从
 * 目标元素冒泡到根**才会被 React 接住。`elementFromPoint` 给出的是最深的目标，
 * 在它上面 `dispatchEvent`（`bubbles: true`）正好走完整条路径。
 *
 * 【为什么需要仲裁】见 `longPressArbiter`：长按与拖动重排的起始姿势完全相同。
 */

import { claimLongPress, releaseLongPress, resetLongPressArbiter } from "./longPressArbiter";
import { INPUT_PROFILES, SECONDARY_LONG_PRESS_MS } from "./inputProfile";
import { surfaceProfile } from "../services/surfaceProfileStore";

/** 长按判定的默认参数（来自 `inputProfile` 的触摸剖面，不另写一份数值）。 */
export const LONG_PRESS_MS = SECONDARY_LONG_PRESS_MS;
/** 手指在这个范围内抖动不算移动。 */
export const LONG_PRESS_MOVE_TOLERANCE_PX = INPUT_PROFILES.touch.startThresholdPx;

export interface LongPressPoint {
    clientX: number;
    clientY: number;
}

export interface LongPressTrackerConfig {
    /** 判定时长（毫秒）。 */
    longPressMs?: number;
    /** 位移容差（像素）。 */
    moveTolerancePx?: number;
    /** 触发回调。返回 `false` 表示"我不要这次长按"，仲裁随即释放。 */
    onLongPress: (point: LongPressPoint) => boolean | void;
    /** 取消回调（移动超限 / 第二指落下 / 系统打断）。 */
    onCancel?: () => void;
}

/** 长按追踪器：**不依赖 React / DOM 全局**，可在 node 环境完整单测。 */
export interface LongPressTracker {
    /** 一根手指按下。返回 `false` 表示这次不参与（例如已被别人占用）。 */
    start(pointerId: number, point: LongPressPoint, pointerType?: string | null): boolean;
    /** 手指移动。 */
    move(pointerId: number, point: LongPressPoint): void;
    /** 手指抬起。 */
    end(pointerId: number): void;
    /** 主动取消（第二指落下 / 滚动开始 / 系统打断）。 */
    cancel(): void;
    /** 是否仍在等待判定。 */
    readonly pending: boolean;
    /** 本次按压是否已经触发过长按。 */
    readonly fired: boolean;
}

/**
 * 创建一个长按追踪器。
 *
 * 【为什么是工厂而不是一个 React hook】两个消费方形态完全不同：画布（时间轴 / 钢琴
 * 卷帘）在原生指针事件里派发，不需要 React；而简单的按钮表面用 hook 更顺手。
 * 追踪器是两者的公共内核 —— 写两遍必然有一遍漏掉"第二指落下要取消"。
 *
 * @param schedule 定时器注入（测试用假定时器）。
 */
export function createLongPressTracker(
    config: LongPressTrackerConfig,
    schedule: {
        setTimeout: (fn: () => void, ms: number) => number;
        clearTimeout: (handle: number) => void;
    } = {
        setTimeout: (fn, ms) => window.setTimeout(fn, ms),
        clearTimeout: (handle) => window.clearTimeout(handle),
    },
): LongPressTracker {
    const longPressMs = config.longPressMs ?? LONG_PRESS_MS;
    const tolerance = config.moveTolerancePx ?? LONG_PRESS_MOVE_TOLERANCE_PX;

    let timer: number | null = null;
    let activePointerId: number | null = null;
    let origin: LongPressPoint | null = null;
    let fired = false;

    const clear = () => {
        if (timer !== null) {
            schedule.clearTimeout(timer);
            timer = null;
        }
    };

    const reset = () => {
        clear();
        activePointerId = null;
        origin = null;
        fired = false;
    };

    return {
        get pending() {
            return timer !== null;
        },
        get fired() {
            return fired;
        },
        start(pointerId, point, pointerType) {
            // 只处理触摸。鼠标与笔本来就有右键 / 笔杆键，再合成一次会得到**两个**
            // contextmenu（原生的 + 我们合成的），菜单会开两次。
            if (pointerType !== undefined && pointerType !== "touch") return false;
            if (!claimLongPress("menu")) return false;
            reset();
            activePointerId = pointerId;
            origin = { ...point };
            timer = schedule.setTimeout(() => {
                timer = null;
                if (activePointerId === null || origin === null) return;
                fired = true;
                const accepted = config.onLongPress({ ...origin });
                if (accepted === false) {
                    fired = false;
                    releaseLongPress("menu");
                }
            }, longPressMs);
            return true;
        },
        move(pointerId, point) {
            if (pointerId !== activePointerId || origin === null) return;
            if (timer === null) return; // 已触发：后续移动不再影响判定
            const distance = Math.hypot(
                point.clientX - origin.clientX,
                point.clientY - origin.clientY,
            );
            if (distance > tolerance) {
                // 移动超限 = 用户想拖，不是想长按。释放仲裁，让拖动层接手。
                config.onCancel?.();
                releaseLongPress("menu");
                reset();
            }
        },
        end(pointerId) {
            if (pointerId !== activePointerId) return;
            // 抬起时若还没触发，就是一次普通点击 —— 释放仲裁，别挡住后续手势。
            releaseLongPress("menu");
            reset();
        },
        cancel() {
            config.onCancel?.();
            releaseLongPress("menu");
            reset();
        },
    };
}

/**
 * 当前形态是否应当合成上下文菜单。
 *
 * 【为什么判据是"没有右键"而不是"是移动端"】桌面触摸屏（Windows 平板）同样没有右键
 * 可用，而移动端接上鼠标后右键是存在的。能力表已经回答了这件事，不必再猜一次。
 */
export function shouldSynthesizeContextMenu(): boolean {
    const profile = surfaceProfile();
    return profile.input.secondary_button === false && profile.input.multi_touch;
}

/** 合成事件上打的标记，用来忽略"我们自己派发的"那一次。 */
const SYNTHETIC_FLAG = "__hfsSyntheticContextMenu";

/**
 * 在指定屏幕坐标上合成一次 `contextmenu`。
 *
 * @returns 是否真的派发出去了（坐标落在 `document` 之外时为 `false`）。
 */
export function synthesizeContextMenu(clientX: number, clientY: number): boolean {
    if (typeof document === "undefined") return false;
    const target = document.elementFromPoint(clientX, clientY);
    if (!target) return false;
    const event = new MouseEvent("contextmenu", {
        bubbles: true,
        cancelable: true,
        composed: true,
        clientX,
        clientY,
        // `button: 2` = 右键。有些处理器会读它来区分左右键菜单。
        button: 2,
        buttons: 2,
    });
    Object.defineProperty(event, SYNTHETIC_FLAG, { value: true });
    target.dispatchEvent(event);
    return true;
}

/** 这个 `contextmenu` 事件是不是我们自己合成的。 */
export function isSyntheticContextMenu(event: Event): boolean {
    return (event as Event & { [SYNTHETIC_FLAG]?: boolean })[SYNTHETIC_FLAG] === true;
}

/**
 * 在 `document` 上安装全局的"长按 → contextmenu"。
 *
 * 【为什么装在 document 而不是逐个交互面】见文件头：136 个处理器已经就位，缺的只是
 * "触摸上也会有一次 contextmenu"。装在 document 上，**今天存在的和明天新增的**
 * `onContextMenu` 都会自动生效 —— 这是唯一不会随功能增加而退化的做法。
 *
 * @returns 卸载函数。
 */
export function installTouchContextMenu(): () => void {
    if (typeof document === "undefined") return () => undefined;
    if (!shouldSynthesizeContextMenu()) return () => undefined;

    const tracker = createLongPressTracker({
        onLongPress: ({ clientX, clientY }) => {
            // 轻微震动是移动端"菜单要出来了"的标准反馈。WebView 不支持时静默跳过。
            try {
                navigator.vibrate?.(10);
            } catch {
                // 某些 WebView 在没有用户激活时会抛 —— 反馈失败不影响功能。
            }
            return synthesizeContextMenu(clientX, clientY);
        },
    });

    /** 当前按下的指针集合：第二根手指落下即取消长按（那是捏合，不是长按）。 */
    const downPointers = new Set<number>();

    const onPointerDown = (event: PointerEvent) => {
        downPointers.add(event.pointerId);
        if (downPointers.size > 1) {
            tracker.cancel();
            return;
        }
        tracker.start(
            event.pointerId,
            { clientX: event.clientX, clientY: event.clientY },
            event.pointerType,
        );
    };
    const onPointerMove = (event: PointerEvent) => {
        tracker.move(event.pointerId, { clientX: event.clientX, clientY: event.clientY });
    };
    const onPointerUp = (event: PointerEvent) => {
        downPointers.delete(event.pointerId);
        tracker.end(event.pointerId);
    };
    const onPointerCancel = (event: PointerEvent) => {
        downPointers.delete(event.pointerId);
        tracker.cancel();
    };
    /**
     * 原生 contextmenu：Android WebView 在某些元素上仍会自己派发一次。
     * 我们只做两件事：① 记下"原生已经给过了"，别让 400ms 后再来一次；
     * ② **不**阻止默认（长按选词、链接菜单是用户预期内的行为）。
     */
    const onNativeContextMenu = (event: Event) => {
        if (isSyntheticContextMenu(event)) return;
        tracker.cancel();
    };
    /** 页面开始滚动时取消：滚动与长按不可能同时是用户的意图。 */
    const onScroll = () => tracker.cancel();
    /**
     * 窗口失焦时取消。
     *
     * 【为什么必须有】用户在按住不放的中途 Alt+Tab 切走（或下拉通知栏、接电话），
     * WebView 不会把 `pointerup` 送回本窗口 —— 计时器照常到点，于是**在别的应用
     * 前面**弹出一个上下文菜单。这不只是残留：菜单里的动作会作用在用户已经看不见的
     * 对象上。
     *
     * 这是 `utils/gestureFocusGuard` 那条门禁要求的同一件事（"装了 pointermove 的
     * 手势必须处理失焦"），本模块自己接，不登记进豁免清单。
     */
    const onFocusLost = () => tracker.cancel();
    const onVisibilityChange = () => {
        if (document.visibilityState === "hidden") tracker.cancel();
    };

    document.addEventListener("pointerdown", onPointerDown, { capture: true, passive: true });
    document.addEventListener("pointermove", onPointerMove, { capture: true, passive: true });
    document.addEventListener("pointerup", onPointerUp, { capture: true, passive: true });
    document.addEventListener("pointercancel", onPointerCancel, { capture: true, passive: true });
    document.addEventListener("contextmenu", onNativeContextMenu, { capture: true, passive: true });
    window.addEventListener("scroll", onScroll, { capture: true, passive: true });
    window.addEventListener("blur", onFocusLost);
    document.addEventListener("visibilitychange", onVisibilityChange);

    return () => {
        document.removeEventListener("pointerdown", onPointerDown, { capture: true });
        document.removeEventListener("pointermove", onPointerMove, { capture: true });
        document.removeEventListener("pointerup", onPointerUp, { capture: true });
        document.removeEventListener("pointercancel", onPointerCancel, { capture: true });
        document.removeEventListener("contextmenu", onNativeContextMenu, { capture: true });
        window.removeEventListener("scroll", onScroll, { capture: true });
        window.removeEventListener("blur", onFocusLost);
        document.removeEventListener("visibilitychange", onVisibilityChange);
        // 卸载时清空仲裁：否则下一次挂载会继承一个"被占用"的单槽。
        resetLongPressArbiter();
    };
}
