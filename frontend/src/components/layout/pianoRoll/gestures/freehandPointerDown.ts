/**
 * 自由绘制（含右键 / 笔杆键擦除）的指针按下。
 *
 * 【为什么单独成文件】它原本是 `canvasPointerDown.ts` 里 `if (usesVibratoPath) … else …`
 * 的 **else 分支**，263 行。上一轮把指针按下拆成「选择分支 / 颤音分支」两个模块时，
 * 这一支没有跟着搬走，而是**整块消失了** —— 于是自由绘制只剩落笔那一下：
 * 没有 `pointermove`、没有 `pointerup`、从不调用 `commitStroke`。
 * 用户看到的是"按住拖拽只有按下那一刻生效"。本文件把它原样搬回。
 *
 * 【与颤音分支的关系】两者互斥：`usesVibratoPath`（直线 / 颤音工具）走
 * `vibratoPointerDown`，其余（自由绘制 / 擦除）走这里。分支判断留在调用方，
 * 本文件只负责"按下之后这一笔怎么走"。
 *
 * 【为什么边缘滚屏要经 `onEdgeScrolledRef`】驱动在 `canvasPointerDown` 里构造时就闭包读了
 * 那个槽位；按值传进来会得到一份副本，写入到不了驱动的闭包。与颤音分支同一约定。
 */
import { startPointerGesture, type PointerGesture } from "./pointerGesture";
import { PEN_ERASER_BUTTONS_MASK, coalescedEventsOf } from "../../../../utils/penInput";
import {
    isChildPitchOffsetCentsParam,
    isChildPitchOffsetDegreesParam,
} from "../childPitchOffsetParams";
import { timeToFrame } from "../utils";
import type { StrokeMode } from "../types";
import type { PointerEvent as ReactPointerEvent } from "react";
import type { EdgeScrollDriver } from "../kernel/edgeScrollDriver";
import type { PointerDeps } from "./pointerDeps";

/** `handleCanvasPointerDown` 作用域里的局部量（类型手写：它们不属于 PointerDeps）。 */
type PointerDownLocals = {
    edgeScroll: EdgeScrollDriver;
    fp: number;
    isDrawMode: boolean;
    mode: StrokeMode;
    onEdgeScrolledRef: { current: ((clientX: number) => void) | null };
    penEraserDown: boolean;
    value: number;
};

/** 自由绘制起手所需的全部上下文。 */
type FreehandPointerDeps = Pick<
    PointerDeps,
    | "applyDenseToLiveEdit"
    | "applyPostStrokeSmoothing"
    | "clearActivePointerGestureEnd"
    | "commitStroke"
    | "createFineAdjustedPointerState"
    | "disposeFineAdjustedPointerState"
    | "dragDirection"
    | "editParam"
    | "getFineAdjustedPointerPosition"
    | "installDragDirectionKeyCycler"
    | "invalidate"
    | "isSnapToggleModifierHeld"
    | "liveEditActiveRef"
    | "liveEditOverrideRef"
    | "onCycleDragDirection"
    | "onPitchSnapGestureActiveChange"
    | "paramViewRef"
    | "pointerSec"
    | "pointerValue"
    | "requestWaveformRepaint"
    | "setActivePointerGestureEnd"
    | "setVibratoDragCaptureActive"
    | "snapDrawValue"
    | "strokeRef"
    | "vibratoStateRef"
> &
    PointerDownLocals;

/**
 * 处理自由绘制 / 擦除的指针按下，并接管本次手势直到松手。
 *
 * @param deps hook 作用域 + 回调局部量的上下文。
 * @param e 触发本次手势的指针事件。
 */
export function handleFreehandPointerDown(
    deps: FreehandPointerDeps,
    e: ReactPointerEvent<HTMLCanvasElement>,
): void {
    const {
        applyDenseToLiveEdit,
        applyPostStrokeSmoothing,
        clearActivePointerGestureEnd,
        commitStroke,
        createFineAdjustedPointerState,
        disposeFineAdjustedPointerState,
        dragDirection,
        editParam,
        edgeScroll,
        fp,
        getFineAdjustedPointerPosition,
        installDragDirectionKeyCycler,
        invalidate,
        isDrawMode,
        isSnapToggleModifierHeld,
        liveEditActiveRef,
        liveEditOverrideRef,
        mode,
        onCycleDragDirection,
        onEdgeScrolledRef,
        onPitchSnapGestureActiveChange,
        paramViewRef,
        penEraserDown,
        pointerSec,
        pointerValue,
        requestWaveformRepaint,
        setActivePointerGestureEnd,
        setVibratoDragCaptureActive,
        snapDrawValue,
        strokeRef,
        value,
        vibratoStateRef,
    } = deps;

    // Draw tool: freehand drawing with interpolation between points
    // 橡皮端（button 5）在 buttons 位掩码里是位 32，不是右键的位 2。
    const requiredButtonMask =
        mode === "restore" ? (penEraserDown ? PEN_ERASER_BUTTONS_MASK : 2) : 1;
    let currentDragDir: "free" | "x-only" = dragDirection === "x-only" ? "x-only" : "free";
    const canCycleDragDirection = e.button === 0;
    const finePointerState = createFineAdjustedPointerState(
        e.nativeEvent,
        e.currentTarget as HTMLCanvasElement,
    );
    // 最近一次处理的采样（含修饰键）。边缘自动滚屏之后要按**同一个**
    // 指针位置重跑一次：视图滚了、指针屏幕坐标没变，但它对应的帧号变了。
    // 不重跑，笔画就会在滚动处跳变（视图过去了、线没跟上）。
    let lastStrokeSample: {
        clientX: number;
        clientY: number;
        ctrlKey: boolean;
        shiftKey: boolean;
        altKey: boolean;
    } | null = null;
    // 绘制类工具启用**持续**边缘滚屏（指针停在边缘不动也继续滚），
    // 见 `kernel/edgeScrollDriver` 的模块头说明。
    onEdgeScrolledRef.current = (clientX) => {
        const base = lastStrokeSample;
        if (base === null) return;
        // 用"当前指针 x + 最后一次采样的其余字段"重跑：修饰键 / 拖动方向
        // 等手势状态必须与用户最后一次真实输入一致，而 x 取回调带过来的
        // 最新值（比队列里那份可能还新一个事件）。
        processStrokeEvent({ ...base, clientX });
        // 自由绘制的 `processStrokeEvent` 只调 `invalidate()`（主画布）——
        // 波形面的重绘一向由外层的合帧循环统一发出。这里绕过了那层循环，
        // 必须自己补一次，否则滚屏期间波形面按旧动态值显示。
        requestWaveformRepaint();
    };

    // 逐事件处理的主体从 onMove 拆出：onMove 只把事件压进待处理
    // 队列并请求 rAF，真正的 dense 构建 + live 写入 + 重绘合帧到
    // 每渲染帧一次。pen 采样率（133–266Hz+）高于渲染帧率，逐事件
    // 执行 O(n) dense 分配会让同一帧内多次重算；与选区拖拽的
    // schedulePreview 同一模式。
    const processStrokeEvent = (ev: {
        clientX: number;
        clientY: number;
        ctrlKey: boolean;
        shiftKey: boolean;
        altKey: boolean;
    }) => {
        const st = strokeRef.current;
        if (!st || st.pointerId !== e.pointerId) return;
        lastStrokeSample = {
            clientX: ev.clientX,
            clientY: ev.clientY,
            ctrlKey: ev.ctrlKey,
            shiftKey: ev.shiftKey,
            altKey: ev.altKey,
        };
        const adjusted = getFineAdjustedPointerPosition(finePointerState, ev);
        const last = st.points[st.points.length - 1];
        const f2 = timeToFrame(pointerSec(adjusted.clientX), fp);
        const yDragEnabled = currentDragDir !== "x-only";
        const rawV2Abs = yDragEnabled ? pointerValue(adjusted.clientY) : (last?.value ?? value);
        const rawV2 = rawV2Abs;
        const moveSnapToggleHeld = isSnapToggleModifierHeld(ev);
        const v2 = isDrawMode ? snapDrawValue(rawV2, moveSnapToggleHeld, f2) : rawV2;

        const pv2 = paramViewRef.current;
        if (last && last.frame === f2) {
            last.value = v2;
            if (pv2) {
                applyDenseToLiveEdit(pv2, f2, mode === "restore" ? null : [v2], f2, f2, mode);
            }
        } else if (last) {
            const a = { frame: last.frame, value: last.value };
            const b = { frame: f2, value: v2 };
            st.points.push(b);

            const minF = Math.min(a.frame, b.frame);
            const maxF = Math.max(a.frame, b.frame);

            let dense: number[] | null = null;
            if (mode !== "restore") {
                const len = maxF - minF + 1;
                dense = new Array<number>(len);
                const denom = b.frame - a.frame;
                for (let f = minF; f <= maxF; f += 1) {
                    const t = denom === 0 ? 1 : (f - a.frame) / denom;
                    dense[f - minF] = a.value + (b.value - a.value) * t;
                }
            }

            if (pv2) {
                applyDenseToLiveEdit(pv2, minF, dense, minF, maxF, mode);
            }
        }
        invalidate();
        // live 覆盖只改 ref，波形面收不到通知 —— 重绘请求由本帧的
        // 处理循环末尾统一发出（见 flushPendingMoves / finishUp）：
        // 同一帧内的全部 coalesced 采样只重绘一次。
    };

    // 待处理事件队列 + rAF 合帧状态。flush 后队列即清空；finishUp/
    // onCancel 里取消挂起帧（挂起的帧只能看到"队列已消费"的状态，
    // 不取消也不会重复处理，但显式取消避免收尾后还跑一次回调）。
    let pendingMoveEvents: globalThis.PointerEvent[] = [];
    let strokeRafId: number | null = null;
    const flushPendingMoves = () => {
        strokeRafId = null;
        const queue = pendingMoveEvents;
        pendingMoveEvents = [];
        let processed = false;
        for (const ev of queue) {
            // 同帧多点：getCoalescedEvents 展开高速运笔时一帧内
            // 积累的全部采样，轨迹逐点保留（不支持时回退单事件）。
            for (const sample of coalescedEventsOf(ev)) {
                processStrokeEvent(sample);
                processed = true;
            }
        }
        // 波形重绘放在采样循环**之外**：一帧内的全部采样合起来只请求
        // 一次重绘（`requestWaveformRepaint` 自身也做帧内合并，这里
        // 少的是每次采样的闸门判定与修订号自增）。轨迹点仍逐采样
        // 累积到 `st.points`，提交保真度不变。
        if (processed) requestWaveformRepaint();
    };

    /** 手势句柄（onMove 需要主动收尾，见 selection 分支的同名字段）。 */
    let gesture: PointerGesture | null = null;
    const onMove = (ev: globalThis.PointerEvent) => {
        if ((ev.buttons & requiredButtonMask) !== requiredButtonMask) {
            gesture?.end("up", ev);
            return;
        }
        const st = strokeRef.current;
        if (!st) return;
        // 边缘自动滚屏：`track` 会立刻滚一帧并维持 rAF 循环（指针停在
        // 边缘不动时也继续滚）。**不在这里读坐标做处理** —— 事件仍走
        // 下面的合帧队列，滚动引发的重算由 `onEdgeScrolled` 负责。
        edgeScroll.track(ev.clientX);
        pendingMoveEvents.push(ev);
        if (strokeRafId == null) {
            strokeRafId = requestAnimationFrame(flushPendingMoves);
        }
    };

    const cancelPendingStrokeFrame = () => {
        if (strokeRafId != null) {
            cancelAnimationFrame(strokeRafId);
            strokeRafId = null;
        }
        pendingMoveEvents = [];
    };

    /** 正常松手：冲刷挂起采样并提交。 */
    const finishUp = () => {
        const st = strokeRef.current;
        const isOwnStroke = Boolean(st && st.pointerId === e.pointerId);
        // 提交前先同步消费挂起的采样队列：processStrokeEvent 依赖
        // strokeRef 仍在位（其内部按 pointerId 自守卫），必须先
        // flush 再清空引用，否则最后一帧内的轨迹点会丢失
        // （快速运笔时笔尾缺一段 ≤16ms 的点）。
        if (strokeRafId != null) {
            cancelAnimationFrame(strokeRafId);
            strokeRafId = null;
        }
        const queued = pendingMoveEvents;
        pendingMoveEvents = [];
        let processedQueued = false;
        for (const ev of queued) {
            for (const sample of coalescedEventsOf(ev)) {
                processStrokeEvent(sample);
                processedQueued = true;
            }
        }
        // 末帧同样只请求一次（见 flushPendingMoves）。
        if (processedQueued) requestWaveformRepaint();
        if (isOwnStroke) {
            strokeRef.current = null;
            vibratoStateRef.current = null;
            setVibratoDragCaptureActive(false);
        }
        disposeDragDirKey();
        disposeFineAdjustedPointerState(finePointerState);
        // 停掉边缘自动滚屏：松手后视图绝不能继续滚（见 driver 的 `stop`）。
        onEdgeScrolledRef.current = null;
        edgeScroll.stop();
        invalidate();
        if (!isOwnStroke || !st) return;
        if (
            editParam === "pitch" ||
            isChildPitchOffsetCentsParam(editParam) ||
            isChildPitchOffsetDegreesParam(editParam)
        ) {
            onPitchSnapGestureActiveChange?.(false);
        }
        void (async () => {
            await commitStroke(st.points, st.mode);
            await applyPostStrokeSmoothing(st.points, st.mode);
        })();
    };

    /**
     * pointercancel 收尾：**回滚**而不是提交（与直线 / 颤音工具分支
     * 同一语义）。OS 主动取消意味着手势不可信，半截笔画不写后端。
     */
    /** pointercancel：回滚而不是提交（见下方注释）。 */
    const finishCancel = () => {
        const st = strokeRef.current;
        const isOwnStroke = Boolean(st && st.pointerId === e.pointerId);
        if (isOwnStroke) {
            strokeRef.current = null;
        }
        cancelPendingStrokeFrame();
        vibratoStateRef.current = null;
        setVibratoDragCaptureActive(false);
        disposeDragDirKey();
        disposeFineAdjustedPointerState(finePointerState);
        // 取消路径同样必须停掉自动滚屏（否则视图会一直滚下去）。
        onEdgeScrolledRef.current = null;
        edgeScroll.stop();
        liveEditOverrideRef.current = null;
        if (liveEditActiveRef) liveEditActiveRef.current = false;
        invalidate();
    };

    const onContextMenuDuringDraw = (ev: Event) => {
        ev.preventDefault();
        ev.stopImmediatePropagation();
    };
    const cycleFreehandDragDir = () => {
        if (!canCycleDragDirection) return;
        currentDragDir = currentDragDir === "free" ? "x-only" : "free";
        if (onCycleDragDirection) {
            onCycleDragDirection("draw");
        }
    };
    const onMouseDownDuringDraw = (event: Event) => {
        // 注册在 window 的 `mousedown` 上，因此类型是 Event；这里收窄回 MouseEvent。
        const ev = event as globalThis.MouseEvent;
        if (ev.button !== 2) return;
        if (!canCycleDragDirection) return;
        // 仅在左键拖拽进行中时，右键才切换拖拽方向。
        if ((ev.buttons & 1) !== 1) return;
        ev.preventDefault();
        ev.stopPropagation();
        cycleFreehandDragDir();
    };
    // 触控板替代：拖拽中按下快捷键与右键同义。
    const disposeDragDirKey = installDragDirectionKeyCycler(cycleFreehandDragDir);

    gesture = startPointerGesture({
        pointerId: e.pointerId,
        // 指针捕获由 canvasPointerDown 在分派之前统一做（这条分支是被调用方，
        // 不是入口），因此这里不接管捕获，也不释放它。
        onMove,
        onEnd: (reason) => {
            if (reason === "up") finishUp();
            else finishCancel();
        },
        extra: [
            { type: "contextmenu", listener: onContextMenuDuringDraw, capture: true },
            { type: "mousedown", listener: onMouseDownDuringDraw, capture: true },
        ],
        registerEnd: setActivePointerGestureEnd,
        clearEnd: clearActivePointerGestureEnd,
    });
}
