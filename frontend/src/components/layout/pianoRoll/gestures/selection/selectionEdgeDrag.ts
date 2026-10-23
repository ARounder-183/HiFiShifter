import { startPointerGesture, type PointerGesture } from "../pointerGesture";
import { SELECTION_EDGE_MIN_WIDTH_PX } from "../../kernel/gestureHitTest";
import {
    frameRangeEndCut,
    frameRangeStartCut,
    normalizeSelection,
    resizeFrameRangeEdge,
} from "../../paramSelection";
import type { ParamSelection } from "../../paramSelection";
import type { PointerEvent as ReactPointerEvent } from "react";
import type { PointerDeps } from "../pointerDeps";

/** `handleSelectPointerDown` 作用域里的局部量。 */
type SelectLocals = {
    sel: ParamSelection;
    selectionEdgeHit: { rangeIndex: number; edge: "left" | "right" };
    selectionFrameFromClientX: (clientX: number, allowAutoScroll: boolean) => number;
};

/** 本分支所需的全部上下文。 */
type Deps = Pick<
    PointerDeps,
    | "clearActivePointerGestureEnd"
    | "frameToViewportPx"
    | "invalidate"
    | "selectionRef"
    | "setActivePointerGestureEnd"
    | "setCanvasCursor"
    | "updateSelectionUi"
> &
    SelectLocals;

/**
 * 调整选区边缘。
 *
 * @param deps hook 作用域 + 局部量的上下文。
 * @param e 触发本次手势的指针事件。
 * @returns `true` = 已消费、调用方应停止；`false` = 未消费，调用方继续。
 */
export function handleSelectionEdgeDrag(
    deps: Deps,
    e: ReactPointerEvent<HTMLCanvasElement>,
): boolean {
    const {
        clearActivePointerGestureEnd,
        frameToViewportPx,
        invalidate,
        sel,
        selectionEdgeHit,
        selectionFrameFromClientX,
        selectionRef,
        setActivePointerGestureEnd,
        setCanvasCursor,
        updateSelectionUi,
    } = deps;

    const rangeIndex = selectionEdgeHit.rangeIndex;
    const edge = selectionEdgeHit.edge;
    // pointerdown 时的快照：拖拽期间每帧都以它重建（只替换被拖动的
    // 那一段）。以快照为基底而不是"当前选区"，是为了让
    // `normalizeSelection` 的合并/排序不会造成下标漂移 —— 交换左右
    // 也靠这一点保持稳定（固定端始终是"按下时的对侧边界"）。
    const baseSelection = sel;
    const pid = e.pointerId;
    setCanvasCursor("ew-resize");

    const applyEdgePointer = (clientX: number, allowAutoScroll: boolean) => {
        const pointerFrame = selectionFrameFromClientX(clientX, allowAutoScroll);
        selectionRef.current = normalizeSelection(
            baseSelection.map((range, index) =>
                index === rangeIndex ? resizeFrameRangeEdge(range, edge, pointerFrame) : range,
            ),
        );
        updateSelectionUi(selectionRef.current);
        invalidate();
    };

    /** 手势句柄（onMove 需要主动收尾，见 pan 分支的同名字段）。 */
    let gesture: PointerGesture | null = null;
    const onMove = (ev: globalThis.PointerEvent) => {
        if ((ev.buttons & 1) !== 1) {
            gesture?.end("up", ev);
            return;
        }
        applyEdgePointer(ev.clientX, true);
    };
    const onEnd = () => {
        // 收尾：把被拖到**不可见宽度**的段丢掉（与"单击不留零宽
        // 选区"同一条规则）。判据与边缘命中同源 —— 像素宽度不足
        // `SELECTION_EDGE_MIN_WIDTH_PX` 的段既看不到区域，也再也
        // 抓不到它的边缘；留着只会留下"有光标提示、没东西可拖"
        // 的状态。交换左右的过程中会瞬时经过零宽，那是正常的中途
        // 形态，只有松手时才算数。
        const current = selectionRef.current;
        if (current) {
            const visible = current.filter(
                (range) =>
                    Math.abs(
                        frameToViewportPx(frameRangeEndCut(range)) -
                            frameToViewportPx(frameRangeStartCut(range)),
                    ) >= SELECTION_EDGE_MIN_WIDTH_PX,
            );
            if (visible.length !== current.length) {
                selectionRef.current = normalizeSelection(visible);
                updateSelectionUi(selectionRef.current);
            }
        }
        invalidate();
    };

    gesture = startPointerGesture({
        pointerId: pid,
        captureTarget: e.currentTarget,
        onMove,
        onEnd,
        registerEnd: setActivePointerGestureEnd,
        clearEnd: clearActivePointerGestureEnd,
    });
    return true;
}
