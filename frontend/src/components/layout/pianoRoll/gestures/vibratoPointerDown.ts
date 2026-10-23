import { startPointerGesture, type PointerGesture } from "./pointerGesture";
import {
    cycleVibratoChoice,
    vibratoCycleAnchorId,
} from "../../../../features/vibrato/vibratoPresetList";
import { vibratoSeedForPreset } from "../../../../features/vibrato/vibratoSeed";
import { PEN_ERASER_BUTTONS_MASK, coalescedEventsOf } from "../../../../utils/penInput";
import {
    isChildPitchOffsetCentsParam,
    isChildPitchOffsetDegreesParam,
} from "../childPitchOffsetParams";
import { timeToFrame } from "../utils";
import {
    SIDE_BUTTON_BACK_MASK,
    SIDE_BUTTON_FORWARD_MASK,
    createDragWorking,
    resolveVibratoMiddleClickReset,
    resolveVibratoSideButton,
} from "../vibratoDragAdjust";
import type { StrokeMode } from "../types";
import type { PointerEvent as ReactPointerEvent } from "react";
import type { VibratoPreset } from "../../../../features/vibrato/vibratoTypes";
import type { EdgeScrollDriver } from "../kernel/edgeScrollDriver";
import type { PointerDeps } from "./pointerDeps";

/** `handleCanvasPointerDown` 作用域里的局部量（类型手写：它们不属于 PointerDeps）。 */
type PointerDownLocals = {
    dragPreset: VibratoPreset;
    edgeScroll: EdgeScrollDriver;
    fp: number;
    frame: number;
    isDrawMode: boolean;
    isLineTool: boolean;
    mode: StrokeMode;
    onEdgeScrolledRef: { current: ((clientX: number) => void) | null };
    penEraserDown: boolean;
    snapToggleHeld: boolean;
    usesVibratoPath: boolean;
    value: number;
};

/** 颤音 / 直线工具起手所需的全部上下文。 */
type VibratoPointerDeps = Pick<
    PointerDeps,
    | "applyDenseToLiveEdit"
    | "applyPostStrokeSmoothing"
    | "applyVibratoChoice"
    | "applyVibratoReset"
    | "buildVibratoDense"
    | "clearActivePointerGestureEnd"
    | "commitStroke"
    | "createFineAdjustedPointerState"
    | "currentParamRange"
    | "disposeFineAdjustedPointerState"
    | "dragDirection"
    | "editParam"
    | "finishVibratoDrag"
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
    | "reportVibratoDragState"
    | "requestWaveformRepaint"
    | "resetLiveEditPreview"
    | "setActivePointerGestureEnd"
    | "setVibratoDragCaptureActive"
    | "snapDrawValue"
    | "strokeRef"
    | "vibratoFrequencyAdjustKb"
    | "vibratoHudAnchorRef"
    | "vibratoPresetCycleList"
    | "vibratoStateRef"
> &
    PointerDownLocals;

/**
 * 处理颤音 / 直线工具的指针按下。
 *
 * @param deps hook 作用域 + 回调局部量的上下文。
 * @param e 触发本次手势的指针事件。
 */
export function handleVibratoPointerDown(
    deps: VibratoPointerDeps,
    e: ReactPointerEvent<HTMLCanvasElement>,
): void {
    const {
        applyDenseToLiveEdit,
        applyPostStrokeSmoothing,
        applyVibratoChoice,
        applyVibratoReset,
        buildVibratoDense,
        clearActivePointerGestureEnd,
        commitStroke,
        createFineAdjustedPointerState,
        currentParamRange,
        disposeFineAdjustedPointerState,
        dragDirection,
        dragPreset,
        edgeScroll,
        editParam,
        finishVibratoDrag,
        fp,
        frame,
        getFineAdjustedPointerPosition,
        installDragDirectionKeyCycler,
        invalidate,
        isDrawMode,
        isLineTool,
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
        reportVibratoDragState,
        requestWaveformRepaint,
        resetLiveEditPreview,
        setActivePointerGestureEnd,
        setVibratoDragCaptureActive,
        snapDrawValue,
        snapToggleHeld,
        strokeRef,
        usesVibratoPath,
        value,
        vibratoFrequencyAdjustKb,
        vibratoHudAnchorRef,
        vibratoPresetCycleList,
        vibratoStateRef,
    } = deps;

    // Line tool: draw a straight line from start to current pointer
    const startFrame = frame;
    const startValue = value;
    // 橡皮端（button 5）在 buttons 位掩码里是位 32，不是右键的位 2。
    const requiredButtonMask =
        mode === "restore" ? (penEraserDown ? PEN_ERASER_BUTTONS_MASK : 2) : 1;
    let currentDragDir: "free" | "x-only" = dragDirection === "x-only" ? "x-only" : "free";
    const canCycleDragDirection = e.button === 0;

    {
        // 起手自起手预设：颤音工具取活动预设，直线工具固定取直线预设。
        // 深度 / 速率都是预设自带值 —— 预设切换已持久化，不存在
        // "上一次的调整"要续。
        const working = createDragWorking(dragPreset, editParam, currentParamRange);
        // 先记下锚点：用户可能一起手就滚轮 / 按方向键调参，那时还没有
        // 任何 pointermove，气泡得有个位置可放。
        vibratoHudAnchorRef.current = { clientX: e.clientX, clientY: e.clientY };
        vibratoStateRef.current = {
            pointerId: e.pointerId,
            startFrame,
            startValue,
            currentFrame: startFrame,
            currentValue: startValue,
            mode,
            // 手势从起手时的工具开始；此后由本手势自己的切换改写
            // （见 `VibratoDragState.tool` 的说明）。
            tool: isLineTool ? "line" : "vibrato",
            working,
            shiftHeld: snapToggleHeld,
            seed: vibratoSeedForPreset(working.preset),
        };
        setVibratoDragCaptureActive(true);
    }
    const finePointerState = createFineAdjustedPointerState(
        e.nativeEvent,
        e.currentTarget as HTMLCanvasElement,
    );

    // 直线 / 颤音工具是「起点 → 当前点」的端点式预览：每个事件都重算
    // 全线 dense，O(线长)。pen 高采样率下逐事件执行 = 同帧多次
    // 全线重建；与自由绘制分支同一 rAF 合帧模式（事件入队，一帧
    // 一消费，getCoalescedEvents 展开同帧采样）。端点式预览只需
    // 队列里**最后一个**采样点即可得到正确结果。
    let lineRafId: number | null = null;
    let pendingLineEvent: globalThis.PointerEvent | null = null;
    // 最近一次处理的采样：边缘自动滚屏之后按**同一个**指针位置重跑一次
    // 端点式预览（视图滚了，指针屏幕坐标没变，对应的帧号却变了）。
    let lastLineSample: {
        clientX: number;
        clientY: number;
        ctrlKey: boolean;
        shiftKey: boolean;
        altKey: boolean;
    } | null = null;
    // 直线 / 颤音同样启用**持续**边缘滚屏：端点必须跟着视图继续延伸。
    // 端点式预览每帧整段重算，因此这里重跑一次天然正确 ——
    // 没有自由绘制那种"补一笔"的语义问题。
    onEdgeScrolledRef.current = (clientX) => {
        const base = lastLineSample;
        if (base === null) return;
        processLineEvent({ ...base, clientX });
    };

    const processLineEvent = (ev: {
        clientX: number;
        clientY: number;
        ctrlKey: boolean;
        shiftKey: boolean;
        altKey: boolean;
    }) => {
        const st = strokeRef.current;
        if (!st || st.pointerId !== e.pointerId) return;
        lastLineSample = {
            clientX: ev.clientX,
            clientY: ev.clientY,
            ctrlKey: ev.ctrlKey,
            shiftKey: ev.shiftKey,
            altKey: ev.altKey,
        };
        const adjusted = getFineAdjustedPointerPosition(finePointerState, ev);
        const f2 = timeToFrame(pointerSec(adjusted.clientX), fp);
        const yDragEnabled = currentDragDir !== "x-only";
        const rawV2 = yDragEnabled ? pointerValue(adjusted.clientY) : value;
        const moveSnapToggleHeld = isSnapToggleModifierHeld(ev);
        const v2 = isDrawMode ? snapDrawValue(rawV2, moveSnapToggleHeld, f2) : rawV2;

        // Update stroke to only have start and current end
        st.points = [
            { frame: startFrame, value: startValue },
            { frame: f2, value: v2 },
        ];

        const pv2 = paramViewRef.current;
        if (pv2) {
            // 擦掉上一帧的直线预览，避免端点往回拖时留下残影。
            // 只还原上一帧写过的区间（旧实现是丢弃整份覆盖再整份拷贝，
            // 每帧一次 O(窗口) 拷贝）。
            resetLiveEditPreview(pv2);
            const minF = Math.min(startFrame, f2);
            const maxF = Math.max(startFrame, f2);
            if (mode === "restore") {
                applyDenseToLiveEdit(pv2, minF, null, minF, maxF, mode);
            } else {
                // 只有颤音内核一条路径：直线工具喂的是直线预设（深度 0），
                // 内核输出的就是两端之间的那条直线 —— 与"另写一条插值"相比
                // 少了第二份会与内核漂移的实现。
                const vib = vibratoStateRef.current;
                if (vib) {
                    vib.currentFrame = f2;
                    vib.currentValue = v2;
                    vib.shiftHeld = moveSnapToggleHeld;
                    const built = buildVibratoDense(
                        startFrame,
                        startValue,
                        f2,
                        v2,
                        vib.working,
                        moveSnapToggleHeld,
                        vib.seed,
                    );
                    applyDenseToLiveEdit(
                        pv2,
                        built.minF,
                        built.dense,
                        built.minF,
                        built.maxF,
                        mode,
                    );
                    reportVibratoDragState(adjusted.clientX, adjusted.clientY);
                }
            }
        }
        invalidate();
        // 绘制中：live 覆盖只改 ref，波形面需显式重绘才能按动态值变化。
        requestWaveformRepaint();
    };
    const flushPendingLine = () => {
        lineRafId = null;
        const ev = pendingLineEvent;
        pendingLineEvent = null;
        if (ev != null) processLineEvent(ev);
    };

    /** 手势句柄（onMove 需要主动收尾，见 selection 分支的同名字段）。 */
    let gesture: PointerGesture | null = null;
    const onMove = (ev: globalThis.PointerEvent) => {
        if ((ev.buttons & requiredButtonMask) !== requiredButtonMask) {
            gesture?.end("up", ev);
            return;
        }
        const st = strokeRef.current;
        if (!st || st.pointerId !== e.pointerId) return;
        // 边缘自动滚屏：立刻滚一帧并维持 rAF 循环（指针停在边缘不动时
        // 也继续滚）。滚动引发的端点重算由 `onEdgeScrolled` 负责。
        edgeScroll.track(ev.clientX);
        // 端点式预览：只保留最新采样（含同帧 coalesced 末点）。
        const coalesced = coalescedEventsOf(ev);
        pendingLineEvent = coalesced[coalesced.length - 1] ?? ev;
        if (lineRafId == null) {
            lineRafId = requestAnimationFrame(flushPendingLine);
        }
    };

    const cancelPendingLineFrame = () => {
        if (lineRafId != null) {
            cancelAnimationFrame(lineRafId);
            lineRafId = null;
        }
        pendingLineEvent = null;
    };

    /** 正常松手：冲刷挂起采样并提交。 */
    const finishUp = () => {
        const st = strokeRef.current;
        const isOwnStroke = Boolean(st && st.pointerId === e.pointerId);
        const vib = isOwnStroke ? vibratoStateRef.current : null;
        // 提交前先同步消费挂起的端点采样：processLineEvent 依赖
        // strokeRef 在位（按 pointerId 自守卫），先 flush 再清引用，
        // 保证 st.points 收到最后一个采样对应的端点。
        if (lineRafId != null) {
            cancelAnimationFrame(lineRafId);
            lineRafId = null;
        }
        const pending = pendingLineEvent;
        pendingLineEvent = null;
        if (pending != null) processLineEvent(pending);
        if (isOwnStroke) {
            strokeRef.current = null;
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
            if (usesVibratoPath && vib && st.mode === "draw") {
                const built = buildVibratoDense(
                    vib.startFrame,
                    vib.startValue,
                    vib.currentFrame,
                    vib.currentValue,
                    vib.working,
                    vib.shiftHeld,
                    vib.seed,
                );
                const densePoints = built.dense.map((valueAtFrame, idx) => ({
                    frame: built.minF + idx,
                    value: valueAtFrame,
                }));
                await commitStroke(densePoints, st.mode);
                await applyPostStrokeSmoothing(densePoints, st.mode);
            } else {
                await commitStroke(st.points, st.mode);
                await applyPostStrokeSmoothing(st.points, st.mode);
            }
        })();
        finishVibratoDrag();
        vibratoStateRef.current = null;
        setVibratoDragCaptureActive(false);
    };

    /**
     * pointercancel 收尾：**回滚**而不是提交。OS 主动取消（触摸
     * 接管 / 掌拒 / 笔离开感应区）意味着手势不可信，半截笔画
     * 不应写进后端 —— 后端从未收到数据，丢弃 live 预览层即可。
     */
    /** pointercancel：回滚而不是提交（见上方注释）。 */
    const finishCancel = () => {
        const st = strokeRef.current;
        const isOwnStroke = Boolean(st && st.pointerId === e.pointerId);
        if (isOwnStroke) {
            strokeRef.current = null;
        }
        cancelPendingLineFrame();
        vibratoStateRef.current = null;
        setVibratoDragCaptureActive(false);
        disposeDragDirKey();
        disposeFineAdjustedPointerState(finePointerState);
        // 取消路径同样必须停掉自动滚屏（否则视图会一直滚下去）。
        onEdgeScrolledRef.current = null;
        edgeScroll.stop();
        // 与 `finishUp` 同款收尾：HUD 气泡（深度 / 速率 / 预设）挂在面板状态上，
        // 只有 `onVibratoDragEnd` 会把它撤下。取消路径漏掉这一步，气泡会
        // 永久停在画面上 —— 取消本就"什么都没发生"，却留下一个读数框。
        //（`blur` 路径不受影响：它经 `activePointerGestureEnd` 走 `onUp`。）
        finishVibratoDrag();
        liveEditOverrideRef.current = null;
        if (liveEditActiveRef) liveEditActiveRef.current = false;
        invalidate();
    };

    const onContextMenuDuringDraw = (ev: Event) => {
        ev.preventDefault();
        ev.stopImmediatePropagation();
    };
    const cycleLineDragDir = () => {
        if (!canCycleDragDirection) return;
        currentDragDir = currentDragDir === "free" ? "x-only" : "free";
        if (onCycleDragDirection) {
            // 直线与颤音共用同一份拖动方向，因此两者都归到 `"vibrato"`。
            onCycleDragDirection(usesVibratoPath ? "vibrato" : "draw");
        }
    };
    const onMouseDownDuringDraw = (event: Event) => {
        // 注册在 window 的 `mousedown` 上，因此类型是 Event；这里收窄回 MouseEvent。
        const ev = event as globalThis.MouseEvent;
        // 鼠标侧键（X1 / X2）在颤音拖拽期间切换预设。
        //
        // 【为什么在这里而不是画布的 pointerdown】见
        // `onCanvasPointerDown` 顶部那段注释：指针捕获不重定向
        // `pointerdown`，命中测试按真实坐标走，画布收不到。这条
        // `window` + 捕获阶段的 `mousedown` 不依赖命中测试，与下面
        // "右键切拖动方向"用的是同一个入口。
        //
        // 【为什么判 `buttons` 位掩码而不是 `button`】`button` 是
        // 事件里"哪个键变了"（侧键是 3 / 4），此时左键仍在按住，因此
        // 用位掩码确认"这次确实是拖拽中按下"（左键位 1 仍在）。
        if (usesVibratoPath) {
            // 两个侧键**同时**按下 = 切到直线工具（并清掉两条
            // 调整记录）。`buttons` 里两位都在，说明两次按下有重叠。
            const bothSideButtons =
                (ev.buttons & SIDE_BUTTON_BACK_MASK) !== 0 &&
                (ev.buttons & SIDE_BUTTON_FORWARD_MASK) !== 0;
            const direction = resolveVibratoSideButton(ev.button);
            if (direction !== null || bothSideButtons) {
                if ((ev.buttons & 1) !== 1) return;
                ev.preventDefault();
                ev.stopPropagation();
                if (bothSideButtons) {
                    applyVibratoReset("line", ev.clientX, ev.clientY);
                    return;
                }
                // 走到这里 `direction` 必非 null（仅 bothSideButtons 分支
                // 才可能两者皆空，而那已在上面返回）。
                if (direction === null) return;
                const choice = cycleVibratoChoice(
                    vibratoPresetCycleList,
                    vibratoCycleAnchorId({
                        lineTool: vibratoStateRef.current?.tool === "line",
                        currentPresetId: vibratoStateRef.current?.working.preset.id ?? null,
                    }),
                    direction,
                );
                if (choice) {
                    applyVibratoChoice(choice, ev.clientX, ev.clientY);
                }
                return;
            }

            // 中键：按**当前滚轮所处的那一路**重置（默认是振幅；按住
            // 频率修饰键时是频率）。与滚轮共用同一套修饰键判定。
            if (ev.button === 1) {
                if ((ev.buttons & 1) !== 1) return;
                ev.preventDefault();
                ev.stopPropagation();
                applyVibratoReset(
                    resolveVibratoMiddleClickReset(ev, vibratoFrequencyAdjustKb),
                    ev.clientX,
                    ev.clientY,
                );
                return;
            }
        }

        if (ev.button !== 2) return;
        if (!canCycleDragDirection) return;
        // 仅在左键拖拽进行中时，右键才切换拖拽方向。
        if ((ev.buttons & 1) !== 1) return;
        ev.preventDefault();
        ev.stopPropagation();
        cycleLineDragDir();
    };
    // 触控板替代：拖拽中按下快捷键与右键同义。
    const disposeDragDirKey = installDragDirectionKeyCycler(cycleLineDragDir);

    gesture = startPointerGesture({
        pointerId: e.pointerId,
        // 指针捕获由 canvasPointerDown 统一做（这条分支是被调用方），见 freehand 分支。
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
