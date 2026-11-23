/**
 * 修饰键行：把"按住 Ctrl 再拖"搬到屏幕上。
 *
 * 【为什么这是移动端**能不能用**的分水岭】全仓 29 条纯修饰键动作（`modifier.*`）
 * 覆盖了 slip 编辑、复制拖动、范围选择、不吸附、精细调整、参数形变这六类操作 ——
 * 它们是这个编辑器区别于"剪一段音频"的核心能力。触摸屏上没有第二个输入通道可以
 * "按住"，因此不做这一行，移动端就只剩"选、拖、删"三件事。
 *
 * 【为什么是两态（一次性 / 锁定）】见 `features/keybindings/modifierState.ts`。
 * 交互：**点按** = 一次性，**长按** = 锁定。这两个手势在芯片上是同一块区域，
 * 因此长按必须明显长于点按的抖动（500ms，且位移超过 8px 即取消）。
 *
 * 【为什么横向滚动而不是换行】29 条芯片换行会吃掉半个屏幕高度，而它们的使用场景
 * 是"我要拖了，先选一个修饰"—— 一行、可滑、按使用频率排序即可。
 */
import { useCallback, useEffect, useRef, useSyncExternalStore } from "react";

import type { ActionId } from "../features/keybindings/types";
import {
    consumeOnceModifiers,
    modifierLatch,
    subscribeModifierState,
    toggleModifierLock,
    toggleModifierOnce,
    type ModifierLatch,
} from "../features/keybindings/modifierState";
import { cx } from "../ui/cx";

/** 长按锁定修饰键的判定时长（毫秒）。 */
const MODIFIER_LOCK_LONG_PRESS_MS = 500;
/** 位移超过这个值即视为"用户在滚动这一行"，取消长按。 */
const MODIFIER_LOCK_MOVE_TOLERANCE_PX = 8;

export interface ModifierRailProps {
    chips: Array<{ id: ActionId; label: string }>;
}

/**
 * 「一次性」修饰的消费时机：**下一次在修饰键行之外按下**。
 *
 * 【为什么不能在手势结束时消费】用户点一下芯片（这是一次 `pointerup`），如果按
 * "手势结束即消费"来清，那次点按本身就会立刻把它清掉 —— 芯片看起来按了没反应。
 *
 * 【为什么是"按下"而不是"松开"】修饰键的语义是"**接下来这一下**带上我"。
 * 按下那一刻用户已经决定了这次交互，此时消费才与物理键盘的体验一致。
 *
 * 【为什么排除行内按下】用户在行内连续点两个芯片（例如"精细"+"不吸附"）时，
 * 第二次按下不该把第一个清掉。
 */
function useConsumeOnceOnOutsidePointerDown(enabled: boolean): void {
    useEffect(() => {
        if (!enabled) return;
        const onPointerDown = (event: PointerEvent) => {
            const target = event.target;
            if (target instanceof Element && target.closest("[data-modifier-rail]")) return;
            consumeOnceModifiers();
        };
        document.addEventListener("pointerdown", onPointerDown, { capture: true });
        return () => document.removeEventListener("pointerdown", onPointerDown, { capture: true });
    }, [enabled]);
}

export function ModifierRail({ chips }: ModifierRailProps) {
    const activeKey = useSyncExternalStore(
        subscribeModifierState,
        // 快照必须是**稳定字符串**：`useSyncExternalStore` 每次渲染都比较返回值，
        // 返回新对象会无限重渲染（React 会直接抛 "getSnapshot should be cached"）。
        () => chips.map((chip) => `${chip.id}:${modifierLatch(chip.id)}`).join("|"),
        () => "",
    );

    // 有任何芯片处于「一次性」时才开始监听；空闲时零开销。
    useConsumeOnceOnOutsidePointerDown(activeKey.includes(":once"));

    return (
        <div
            data-modifier-rail="1"
            className="hs-scroll-gutter flex shrink-0 items-center gap-1 overflow-x-auto px-1"
            // 这一行本身可横向滚动：`pan-x` 把纵向手势留给外层，避免与时间轴抢手势。
            style={{ touchAction: "pan-x" }}
            role="toolbar"
            aria-label="Modifier"
        >
            {chips.map((chip) => (
                <ModifierChip key={chip.id} id={chip.id} label={chip.label} activeKey={activeKey} />
            ))}
        </div>
    );
}

function ModifierChip({
    id,
    label,
    activeKey,
}: {
    id: ActionId;
    label: string;
    activeKey: string;
}) {
    // `activeKey` 只用于触发重渲染；真正的状态从 store 同步读（避免闭包过期）。
    void activeKey;
    const latch: ModifierLatch = modifierLatch(id);
    const timerRef = useRef<number | null>(null);
    const originRef = useRef<{ x: number; y: number } | null>(null);
    const longPressedRef = useRef(false);

    const cancelTimer = useCallback(() => {
        if (timerRef.current !== null) {
            window.clearTimeout(timerRef.current);
            timerRef.current = null;
        }
    }, []);

    useEffect(() => cancelTimer, [cancelTimer]);

    const onPointerDown = (event: React.PointerEvent<HTMLButtonElement>) => {
        originRef.current = { x: event.clientX, y: event.clientY };
        longPressedRef.current = false;
        cancelTimer();
        timerRef.current = window.setTimeout(() => {
            longPressedRef.current = true;
            toggleModifierLock(id);
        }, MODIFIER_LOCK_LONG_PRESS_MS);
    };

    const onPointerMove = (event: React.PointerEvent<HTMLButtonElement>) => {
        const origin = originRef.current;
        if (!origin) return;
        if (
            Math.hypot(event.clientX - origin.x, event.clientY - origin.y) >
            MODIFIER_LOCK_MOVE_TOLERANCE_PX
        ) {
            cancelTimer();
        }
    };

    const onPointerUp = () => {
        const wasLongPress = longPressedRef.current;
        cancelTimer();
        originRef.current = null;
        longPressedRef.current = false;
        // 长按已经处理过锁定，抬起时不能再补一次点按（否则"锁定"会立刻被解除）。
        if (wasLongPress) return;
        toggleModifierOnce(id);
    };

    const onPointerCancel = () => {
        cancelTimer();
        originRef.current = null;
        longPressedRef.current = false;
    };

    return (
        <button
            type="button"
            data-modifier-id={id}
            data-modifier-latch={latch}
            aria-pressed={latch !== "off"}
            className={cx(
                "hs-modifier-chip shrink-0 whitespace-nowrap rounded px-2",
                "border text-qt-xs transition-colors",
                latch === "locked"
                    ? "border-qt-highlight bg-qt-highlight/25 text-qt-text"
                    : latch === "once"
                      ? "border-qt-highlight/60 bg-qt-highlight/10 text-qt-text"
                      : "border-qt-border bg-qt-base text-qt-text-muted",
            )}
            style={{
                height: "var(--qt-ctl-sm)",
                // 芯片自身不消费纵向手势：整行是可横滑的。
                touchAction: "pan-x",
            }}
            onPointerDown={onPointerDown}
            onPointerMove={onPointerMove}
            onPointerUp={onPointerUp}
            onPointerCancel={onPointerCancel}
            onContextMenu={(event) => event.preventDefault()}
        >
            {label}
            {latch === "locked" ? <span aria-hidden> 🔒</span> : null}
        </button>
    );
}
