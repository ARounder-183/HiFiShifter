/**
 * modifierWatcher — 全局修饰键按下状态的自愈式跟踪器。
 *
 * ## 为什么需要它
 * 拖拽手势链路里，事件对象有两种来源：
 * 1. 真实的浏览器 PointerEvent（window 级监听收到）；
 * 2. 各发起组件在"延迟起手"后**伪造合成的 React.PointerEvent**
 *   （为了把 dragStartClientX / 交叉配对 id 等私有通道带给 startEditDrag）。
 *
 * 若在拖拽每一帧里直接读 `event.altKey` 之类字段，合成事件必须逐字段
 * 齐全才不会静默失效——这是典型的脆弱耦合（历史上合成事件漏带修饰键
 * 字段会导致"修饰键完全不起作用"）。键盘焦点切换到其它窗口再切回时，
 * 单纯依赖 keydown/keyup 也可能丢失状态。
 *
 * ## 设计
 * - 全局唯一实例，App 启动时调用 {@link initModifierWatcher}；
 * - 监听 window 的 keydown / keyup / blur 维护按位状态；
 * - 同时暴露 {@link ModifierWatchController.refreshFromEvent}：任何持有真实
 *   原生事件的代码都可以顺手喂给它做自愈校正；
 * - 查询接口 {@link ModifierWatchController.isKeybindingActive} 采用
 *   "按键状态 ∪ 实时事件标志" 的并集语义 + 子集匹配（与 keybindingsSlice
 *   isModifierActive 同一套 required 判定）——两个信号源任一命中即算按下，
 *   彻底消除单一来源的漏报。
 */

import {
    getModifierFlags,
    isNoneBinding,
} from "../../../../features/keybindings/keybindingsSlice";
import type { Keybinding } from "../../../../features/keybindings/types";

export type ModifierName = "ctrl" | "shift" | "alt";

type EventLike = {
    ctrlKey?: boolean;
    shiftKey?: boolean;
    altKey?: boolean;
    metaKey?: boolean;
};

export class ModifierWatchController {
    private pressed = new Set<ModifierName>();
    private initialized = false;

    /** 幂等初始化；SSR/测试环境无 window 时安全跳过。 */
    init(target: Pick<Window, "addEventListener" | "removeEventListener">): () => void {
        this.initialized = true;
        const onKeyDown = (event: Event) => this.refreshFromEvent(event as unknown as EventLike);
        const onKeyUp = (event: Event) => this.refreshFromEvent(event as unknown as EventLike);
        const onBlurOrCancel = () => this.pressed.clear();
        target.addEventListener("keydown", onKeyDown);
        target.addEventListener("keyup", onKeyUp);
        target.addEventListener("blur", onBlurOrCancel);
        target.addEventListener("pointercancel", onBlurOrCancel);
        return () => {
            target.removeEventListener("keydown", onKeyDown);
            target.removeEventListener("keyup", onKeyUp);
            target.removeEventListener("blur", onBlurOrCancel);
            target.removeEventListener("pointercancel", onBlurOrCancel);
        };
    }

    get ready(): boolean {
        return this.initialized;
    }

    /**
     * 用任意携带修饰键位的原生事件刷新状态。
     * PointerEvent / KeyboardEvent 都可以 —— 硬件修饰键会反映在所有这类
     * 事件上，因此 pointermove 流也能持续自愈（例如用户按住 Alt 时窗口
     * 失焦再恢复、或首个 keydown 被别的层拦截的情况）。
     */
    refreshFromEvent(event: EventLike | null | undefined): void {
        if (!event) return;
        // macOS 主修饰键 (⌘/meta) 与 ctrl 统一归一化为 "ctrl"，与
        // platform.isPrimaryModifierDown 的语义保持一致。
        if (event.ctrlKey || event.metaKey) this.pressed.add("ctrl");
        else this.pressed.delete("ctrl");
        if (event.shiftKey) this.pressed.add("shift");
        else this.pressed.delete("shift");
        if (event.altKey) this.pressed.add("alt");
        else this.pressed.delete("alt");
    }

    snapshot(): { ctrl: boolean; shift: boolean; alt: boolean } {
        return {
            ctrl: this.pressed.has("ctrl"),
            shift: this.pressed.has("shift"),
            alt: this.pressed.has("alt"),
        };
    }

    clear(): void {
        this.pressed.clear();
    }

    /**
     * 判定某个 modifierOnly 键位绑定当前是否处于按下状态。
     *
     * @param kb        键位绑定（支持用户重映射）
     * @param liveEvent 可选的实时原生事件（第二个自愈信号源）；两者取并集
     */
    isKeybindingActive(kb: Keybinding, liveEvent?: EventLike | null): boolean {
        if (isNoneBinding(kb)) return false;
        const required = getModifierFlags(kb);
        if (!required.ctrl && !required.shift && !required.alt) return false;

        const watched = this.snapshot();
        const pressedCtrl = Boolean(watched.ctrl || liveEvent?.ctrlKey || liveEvent?.metaKey);
        const pressedShift = Boolean(watched.shift || liveEvent?.shiftKey);
        const pressedAlt = Boolean(watched.alt || liveEvent?.altKey);

        return (
            (!required.ctrl || pressedCtrl) &&
            (!required.shift || pressedShift) &&
            (!required.alt || pressedAlt)
        );
    }
}

/** 进程级共享单例。 */
export const modifierWatcher = new ModifierWatchController();

let bootstrapDone = false;

/**
 * 在应用启动时挂接全局监听。重复调用幂等，返回清理函数（首次调用有效）。
 */
export function initModifierWatcher(): (() => void) | null {
    if (bootstrapDone || typeof window === "undefined") return null;
    bootstrapDone = true;
    return modifierWatcher.init(window);
}
