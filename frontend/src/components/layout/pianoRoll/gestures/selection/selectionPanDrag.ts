import { armRightDragContextMenuGuard } from "../../../../../utils/rightDragContextMenuGuard";
import { startPointerGesture, type PointerGesture } from "../pointerGesture";
import { clampSelectionShift, normalizeSelection, snapFrame } from "../../paramSelection";
import type { ParamSelection } from "../../paramSelection";
import type { PointerEvent as ReactPointerEvent } from "react";
import type { PointerDeps } from "../pointerDeps";

/** `handleSelectPointerDown` 作用域里的局部量。 */
type SelectLocals = {
    maxSelectableFrame: number;
    sel: ParamSelection;
    selectionFrameFromClientX: (clientX: number, allowAutoScroll: boolean) => number;
    shiftRangeIndex: number;
};

/** 本分支所需的全部上下文。 */
type Deps = Pick<
    PointerDeps,
    | "clearActivePointerGestureEnd"
    | "invalidate"
    | "onContextMenu"
    | "selectionRef"
    | "setActivePointerGestureEnd"
    | "setCanvasCursor"
    | "updateSelectionUi"
> &
    SelectLocals;

/**
 * 右键拖动平移选区。
 *
 * @param deps hook 作用域 + 局部量的上下文。
 * @param e 触发本次手势的指针事件。
 * @returns `true` = 已消费、调用方应停止；`false` = 未消费，调用方继续。
 */
export function handleSelectionPanDrag(
    deps: Deps,
    e: ReactPointerEvent<HTMLCanvasElement>,
): boolean {
    const {
        clearActivePointerGestureEnd,
        invalidate,
        maxSelectableFrame,
        onContextMenu,
        sel,
        selectionFrameFromClientX,
        selectionRef,
        setActivePointerGestureEnd,
        setCanvasCursor,
        shiftRangeIndex,
        updateSelectionUi,
    } = deps;

    // 平移量以「按下时指针所在帧」为基准，避免逐帧累加造成漂移。
    const grabFrame = selectionFrameFromClientX(e.clientX, false);
    // 快照：每帧都以它重建（只替换被搬动的那一段），归一化
    // （合并/排序）不会造成下标漂移。
    const baseSelection = sel;
    // 夹取只看**被搬动的那一段**的包围区间：其余段不参与，因此
    // 它们不会限制本次平移的可用范围。
    const startRange = sel[shiftRangeIndex];
    const pid = e.pointerId;

    // 右键手势的收尾菜单：拖拽期间一律吞掉（否则松手会弹出上下文
    // 菜单）；**没拖动**时按"右键点击"处理，手工打开菜单 —— 与
    // 曲线上右键拖拽（形变）那一段同一套处理，见那里的注释。
    let didDrag = false;
    const startClientX = e.clientX;
    const suppressContextMenu = (ev: Event) => {
        ev.preventDefault();
        ev.stopImmediatePropagation();
    };
    /**
     * 手势句柄。
     *
     * 【为什么是可变槽】onMove 在"右键已松开"时要主动收尾，而句柄只有在
     * startPointerGesture 返回之后才存在。槽位在事件到来前一定已赋值。
     */
    let gesture: PointerGesture | null = null;
    const onMove = (ev: globalThis.PointerEvent) => {
        if ((ev.buttons & 2) !== 2) {
            // 右键松在画布外：自己收尾（指针 id 已由协议过滤）。
            gesture?.end("up", ev);
            return;
        }
        // 3px 死区：区分「右键点击」与「右键拖拽」，手抖不该被判成
        // 拖拽（否则正常右键点击的菜单会被吞掉）。
        if (!didDrag && Math.abs(ev.clientX - startClientX) > 3) {
            didDrag = true;
            // 光标只在确认拖拽后才变（右键点击不该留下拖拽光标）。
            setCanvasCursor("grabbing");
            // 确认构成拖拽：显式武装全局守卫，松手若落在**另一个**
            // 表面（标尺 / 轨道头）上也能吞掉那个表面的菜单。
            armRightDragContextMenuGuard();
        }
        // 死区内不平移：右键**点击**（含手抖）不得挪动选区，只有
        // 越过死区才算"拖拽"（与多选追加 / 框选同一套语义）。
        if (!didDrag) return;
        const frame = selectionFrameFromClientX(ev.clientX, true);
        // 位移夹到"平移后被搬动那一段仍在 [0, 工程末端] 内"——
        // 段的长度不变，只让它停在两端（见 clampSelectionShift）。
        // 基准点永远是**按下时**的帧：死区内不累计偏移，越过死区后
        // 一次就对齐到真实位移（与多选追加 / 框选同一套死区语义）。
        // 位移先取整到整数帧，夹取与落位才是同一组数字。
        const delta = clampSelectionShift(
            [startRange],
            snapFrame(frame - grabFrame),
            0,
            maxSelectableFrame,
        );
        // 只替换被抓住的那一段，其余段原样保留（快照为基底，
        // 因此归一化的合并/排序不会造成下标漂移）。
        selectionRef.current = normalizeSelection(
            baseSelection.map((range, index) =>
                index === shiftRangeIndex
                    ? {
                          startFrame: range.startFrame + delta,
                          frameCount: range.frameCount,
                      }
                    : range,
            ),
        );
        updateSelectionUi(selectionRef.current);
        invalidate();
    };
    const onEnd = (reason: "up" | "cancel", ev: globalThis.PointerEvent | null) => {
        // 【为什么取消也要走这里】取消时没有"右键点击"语义（没有可信的指针位置），
        // 因此不打开菜单；其余收尾（重绘）两种原因相同。
        if (!didDrag && reason === "up") {
            // 右键点击（没有平移）：选区原样保留，交回菜单。
            if (onContextMenu && document.hasFocus() && ev) {
                onContextMenu(ev.clientX, ev.clientY);
            }
        }
        // 只重绘：选区平移已经实时生效，**不写撤销历史、不写后端**。
        invalidate();
    };

    gesture = startPointerGesture({
        pointerId: pid,
        captureTarget: e.currentTarget,
        onMove,
        onEnd,
        extra: [{ type: "contextmenu", listener: suppressContextMenu, capture: true }],
        registerEnd: setActivePointerGestureEnd,
        clearEnd: clearActivePointerGestureEnd,
    });
    return true;
}
