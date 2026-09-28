/**
 * 视口总线**重放**的纯判定（从 `usePianoRollInteractions` 抽出）。
 *
 * ## 重放是干什么的
 *
 * 各手势的落点由「实时内核视口 + 指针位置」决定。拖拽期间用滚轮滚动 / 缩放时，
 * 视口变了而指针没动；不重放的话被拖对象会停在旧落点上、与光标脱开，直到用户再动
 * 一下鼠标才归位（用户报告的"拖拽中滚动 / 缩放导致偏移"）。因此视口提交后按最近
 * 一次指针位置补发一个 `pointermove`。
 *
 * ## 为什么必须把判定独立出来
 *
 * 重放是**合成**事件，它必须"看起来像"一次真实输入，否则手势会做出错误判断。
 * 这里踩过两个坑，都表现为「用绘制工具画 `音量` / `动态` 时按下那一点在、拖拽轨迹
 * 完全不画」：
 *
 * 1. **按键状态撒谎**：重放曾沿用"最近一次 `pointermove` 事件"的 `buttons`。而按下
 *    之后、第一次真实移动之前，那个"最近一次"是**按下之前的悬停事件**
 *    （`buttons === 0`）。各手势的移动处理器把 `buttons` 当作"键还按着吗"的判据
 *    （如 `(ev.buttons & 1) !== 1 → onUp()`），于是重放被判成"用户松手了"：笔画当场
 *    收尾、监听器全部移除，只剩按下那一点被提交。**按键状态只能取"此刻"，不能取
 *    某个事件自带的**（见 {@link ViewportReplayInput.pressedButtons}）。
 * 2. **把"重绘"当成"视口变化"**：重放挂在总线的 `subscribe()` 上，而
 *    `invalidate()`（强制重绘，投影不变）同样会 paint 所有订阅者。于是任何强制重绘
 *    都可能注入一次重放 —— 绘制 `音量` / `动态` 时**每帧**一次的波形重绘必然注入，
 *    并构成 `invalidate → 重放 → 移动 → invalidate` 的自激环。
 *
 * 两条判定都收口在这里并由单测钉住（见 viewportReplay.test.ts）：这类问题不会报错，
 * 只会"偶尔画不出来"，靠肉眼回归守不住。
 */

import { warnDev } from "./devWarn";

/**
 * 重放所需的指针快照。
 *
 * `PointerEvent` 在结构上是它的超集，因此调用方可以直接把事件对象传进来。
 */
export interface PointerMoveSnapshot {
    clientX: number;
    clientY: number;
    pointerId: number;
    pointerType: string;
    isPrimary: boolean;
    shiftKey: boolean;
    ctrlKey: boolean;
    altKey: boolean;
    metaKey: boolean;
    /**
     * 该事件**自带**的按键位掩码。
     *
     * ⚠ 仅供诊断（告警里回报"那个陈旧事件声称按着什么"）。重放载荷里的 `buttons`
     * 一律取 {@link ViewportReplayInput.pressedButtons} —— 见模块头说明的坑 1。
     */
    buttons: number;
}

/** 视口投影（`pianoRollViewportBus` 每次 paint 给出的那一份）。 */
export interface ViewportProjection {
    scrollLeftPx: number;
    pxPerSec: number;
    viewportWidthPx: number;
}

export interface ViewportReplayInput {
    /** 本次 paint 的投影。 */
    projection: ViewportProjection;
    /** 上一次 paint 的投影；`null` = 还没有基准（视为"变了"，与既有签名比较约定一致）。 */
    previousProjection: ViewportProjection | null;
    /** 左键拖拽进行中。沿用既有闸门，**不扩大**重放范围（右键拖拽不在内）。 */
    dragging: boolean;
    /** **此刻**真正按下的按键位掩码（0 = 没有任何键按下）。 */
    pressedButtons: number;
    /** 最近一次指针移动；`null` = 还没有任何位置可重放。 */
    lastMove: PointerMoveSnapshot | null;
}

/** 不重放的原因（调用方据此决定要不要打开发告警）。 */
export type ViewportReplaySkipReason =
    | "projection-unchanged"
    | "not-dragging"
    | "pressed-buttons-lost"
    | "no-last-move";

/** 合成事件载荷（直接交给 `new PointerEvent("pointermove", payload)`）。 */
export interface ViewportReplayPayload {
    bubbles: true;
    cancelable: true;
    clientX: number;
    clientY: number;
    pointerId: number;
    pointerType: string;
    isPrimary: boolean;
    /** 恒为**此刻**的按键掩码（见模块头说明的坑 1）。 */
    buttons: number;
    /** 合成事件不是"某个键按下"的那一刻；按规范移动事件的 `button` 为 -1。 */
    button: -1;
    shiftKey: boolean;
    ctrlKey: boolean;
    altKey: boolean;
    metaKey: boolean;
}

export type ViewportReplayDecision =
    | { replay: true; payload: ViewportReplayPayload }
    | { replay: false; reason: ViewportReplaySkipReason };

/**
 * 两份投影是否完全相同（重放去重的判据）。
 *
 * `null`（没有基准）**不算**相同 —— 与 `mainCanvasSignature` 的约定一致：
 * 无法比较时宁可多画一次 / 多放一次，也不要漏掉一次真实变化。
 */
export function isSameViewportProjection(
    previous: ViewportProjection | null,
    next: ViewportProjection,
): boolean {
    return (
        previous !== null &&
        previous.scrollLeftPx === next.scrollLeftPx &&
        previous.pxPerSec === next.pxPerSec &&
        previous.viewportWidthPx === next.viewportWidthPx
    );
}

/**
 * 决定这次总线 paint 要不要重放，以及重放什么。
 *
 * 判定顺序即优先级：**自相矛盾的输入最先拦下**（它是那个"轨迹不画"的成因，无论
 * 投影有没有变都必须不重放、并且要能被诊断看见），然后才是"这次本来就不需要重放"
 * 的几种情况。
 */
export function resolveViewportReplay(input: ViewportReplayInput): ViewportReplayDecision {
    // ① 拖拽中却没有任何按键按下 = 事件序列被破坏。此时**必须不重放**：载荷会声明
    //    "没有键按下"，而各手势会把它读成"用户松手了"并当场收尾（正是坑 1）。
    //    宁可少一次重放（用户再动一下鼠标即可归位），也不能破坏手势。
    if (input.dragging && input.pressedButtons === 0) {
        return { replay: false, reason: "pressed-buttons-lost" };
    }
    // ② 强制重绘不是视口变化：`invalidate()` 的语义是"内容变了，按同一投影重画
    //    一次"，此时光标与落点都没变，重放毫无意义（见坑 2）。
    if (isSameViewportProjection(input.previousProjection, input.projection)) {
        return { replay: false, reason: "projection-unchanged" };
    }
    if (!input.dragging) return { replay: false, reason: "not-dragging" };
    const last = input.lastMove;
    if (last === null) return { replay: false, reason: "no-last-move" };
    return {
        replay: true,
        payload: {
            bubbles: true,
            cancelable: true,
            clientX: last.clientX,
            clientY: last.clientY,
            pointerId: last.pointerId,
            pointerType: last.pointerType,
            isPrimary: last.isPrimary,
            // ★ 按键状态取"此刻"，**不取** `last.buttons`（见模块头说明的坑 1）。
            buttons: input.pressedButtons,
            button: -1,
            shiftKey: last.shiftKey,
            ctrlKey: last.ctrlKey,
            altKey: last.altKey,
            metaKey: last.metaKey,
        },
    };
}

/**
 * 开发模式告警：拖拽中却没有观测到任何按下的按键。
 *
 * 这是"事件序列被破坏"的信号。重放因此被**跳过**（安全降级：用户再动一下鼠标即可
 * 归位），而它历史上是**另一种**处理方式 —— 沿用最近一次事件的按键状态，把
 * `buttons === 0` 注入手势，各手势据此判定"用户松手了"并当场收尾手势。
 * 绘制 `音量` / `动态` 时"按下那一点在、拖拽轨迹完全不画"就是这么来的。
 *
 * @param lastMoveButtons 最近一次指针移动自称的按键掩码（诊断用；`null` = 无移动）。
 */
export function warnViewportReplayButtonsLost(lastMoveButtons: number | null): void {
    warnDev(
        "[pianoRoll] 拖拽中却没有观测到按下的按键，本次视口重放已跳过" +
            "（重放载荷的按键状态取「此刻」而非最近一次事件，见 viewportReplay.ts）" +
            ` lastMoveButtons=${lastMoveButtons ?? "<none>"}`,
    );
}
