/**
 * 虚拟修饰键：把"按住修饰键"这件事从键盘搬到屏幕上。
 *
 * 【为什么必须有它】全仓有 29 条纯修饰键动作（`modifier.*`）与 160 处修饰键读取。
 * 它们在桌面上的表达是"按住 Ctrl 再拖"，而触摸屏上**没有第二个输入通道**可以按住。
 * 没有等价物意味着：slip 编辑、复制拖动、范围选择、不吸附、精细调整这五类操作在
 * 移动端**完全不可达** —— 不是"难用"，是"不存在"。
 *
 * 【为什么是 store 而不是 React state】读取点大量在事件处理器与手势代码里（不是组件
 * 渲染路径），它们需要**同步**读。`useSyncExternalStore` 只用在需要重渲染的
 * `ModifierRail` 上。
 *
 * 【为什么分成"一次性"与"锁定"两态】两者都是真实需求：
 * - **一次性**（点一下 = 下一次手势生效，随后自动复位）：与物理修饰键的语义一致，
 *   不会在用户忘记关掉时把后续所有拖动都变成 slip 编辑；
 * - **锁定**（长按 = 一直生效，再点取消）：连续做十次范围选择时，每次都要先点一下
 *   是灾难。
 *
 * 【为什么按 `ActionId` 索引而不是 ctrl/shift/alt】修饰键的**物理身份**与**语义**在
 * 本仓早已解耦：`modifier.clipSlipEdit` 默认绑 Ctrl，但用户可以把它改绑到 Alt。
 * 若这里按 `ctrl` 索引，用户在设置里改键之后屏幕上的芯片就会指错动作。
 */

import type { ActionId } from "./types";

/** 修饰动作的生效状态。 */
export type ModifierLatch = "off" | "once" | "locked";

const state = new Map<ActionId, ModifierLatch>();
const listeners = new Set<() => void>();

function emit(): void {
    for (const listener of listeners) listener();
}

/** 订阅（`useSyncExternalStore` 的形状）。 */
export function subscribeModifierState(listener: () => void): () => void {
    listeners.add(listener);
    return () => {
        listeners.delete(listener);
    };
}

/** 某条修饰动作当前的状态。 */
export function modifierLatch(actionId: ActionId): ModifierLatch {
    return state.get(actionId) ?? "off";
}

/** 是否有任何修饰动作处于生效状态。 */
export function hasActiveModifier(): boolean {
    for (const value of state.values()) if (value !== "off") return true;
    return false;
}

/** 当前生效的修饰动作（快照，供手势层一次性读取）。 */
export function activeModifierIds(): ActionId[] {
    const result: ActionId[] = [];
    for (const [id, value] of state) if (value !== "off") result.push(id);
    return result;
}

/**
 * 点按：`off → once → off`；已锁定时点按解除锁定。
 *
 * 【为什么点按不进入 locked】见文件头：一次性语义与物理修饰键一致，且不会留下
 * "为什么拖什么都变成 slip"这种找不到原因的故障。
 */
export function toggleModifierOnce(actionId: ActionId): void {
    const current = modifierLatch(actionId);
    if (current === "off") state.set(actionId, "once");
    else state.delete(actionId);
    emit();
}

/** 长按：锁定 / 解除锁定。 */
export function toggleModifierLock(actionId: ActionId): void {
    const current = modifierLatch(actionId);
    if (current === "locked") state.delete(actionId);
    else state.set(actionId, "locked");
    emit();
}

/**
 * 手势结束时消费一次性修饰。
 *
 * 【为什么必须由手势层显式调用】"一次性"的复位时机只有手势层知道：`pointerup`
 * 之后才该清。若放在 store 里用计时器，用户慢慢拖到一半时它就被清掉了。
 *
 * 锁定态不受影响。
 */
export function consumeOnceModifiers(): void {
    let changed = false;
    for (const [id, value] of state) {
        if (value === "once") {
            state.delete(id);
            changed = true;
        }
    }
    if (changed) emit();
}

/** 清空全部状态（切换工程 / 关闭抽屉 / 测试）。 */
export function clearModifierState(): void {
    if (state.size === 0) return;
    state.clear();
    emit();
}

/** 事件上的四个修饰键字段（`PointerEvent` / `WheelEvent` / `KeyboardEvent` 都满足）。 */
export interface ModifierView {
    ctrlKey: boolean;
    shiftKey: boolean;
    altKey: boolean;
    metaKey?: boolean;
}

/**
 * 把「物理修饰键 ∪ 虚拟锁存」合并成一份可读的修饰键视图。
 *
 * 【这是全仓唯一允许回答「修饰键生效了吗」的地方】桌面上的答案是「用户按着
 * Ctrl」，移动端的答案是「用户点了那枚芯片」。两者在**所有下游代码**里必须是
 * 同一件事 —— 否则 slip 编辑 / 复制拖动 / 边缘拉伸 / 不吸附 / 精细调整 / 参数
 * 形变 / 多选 / 块创建这八类操作在手机上就是不存在（此前正是如此：29 枚芯片
 * 只会自己高亮，`virtualModifierFlags()` 全仓零消费者）。
 *
 * 【桌面恒等】没有虚拟锁存时**原样返回入参**（同一个对象引用）。这不是省一次
 * 分配，而是零回归的锚点：桌面路径拿到的对象与改动前逐字段相同，任何依赖
 * 引用相等的代码也不会被扰动。
 *
 * 【为什么不是给事件对象塞字段】原生 `PointerEvent` 的 `ctrlKey` 是只读
 * getter，写不进去；React 合成事件虽然可写，但内核宿主用的是**原生监听器**。
 * 两套事件对象、两种行为 = 又一个「只在某条路径上坏」的坑。显式传值是唯一
 * 不会分叉的做法。
 */
export function modifierView(e: ModifierView): ModifierView {
    if (!hasActiveModifier()) return e;
    // 虚拟修饰键统一映射到 `ctrl`：语义是「这个修饰生效了」，而不是「用户按下
    // 了 Ctrl」（见 `virtualModifierFlags` 的映射表说明）。
    return {
        ctrlKey: true,
        shiftKey: e.shiftKey,
        altKey: e.altKey,
        metaKey: e.metaKey,
    };
}

/**
 * 把虚拟修饰键合成为"物理修饰键"的形状，供既有读取点复用。
 *
 * 【为什么不改那 160 处读取点】它们读的是 `event.ctrlKey` 这类**事件字段**，而事件
 * 是浏览器给的。要改就得给每个调用点注入一个"修饰键来源"，那是 160 处高风险改动，
 * 换来的只是"少一次适配"。反过来，在**事件派发的最外层**把虚拟修饰键写进事件对象，
 * 调用点一行都不用改。
 *
 * 具体的合成发生在 `utils/useLongPress.ts` 与手势层（它们本来就是触摸的入口）。
 *
 * 【映射表为什么是这张】修饰动作在默认键位里绑的是 Ctrl / Shift / Alt 三档
 * （见 `defaultKeybindings.ts` 的 `modifier.*` 段）。屏幕上的芯片不区分物理键，
 * 因此统一合成到 `ctrl`：语义是"这个修饰生效了"，而不是"用户按下了 Ctrl"。
 */
export function virtualModifierFlags(): {
    ctrl: boolean;
    shift: boolean;
    alt: boolean;
    meta: boolean;
} {
    return { ctrl: hasActiveModifier(), shift: false, alt: false, meta: false };
}
