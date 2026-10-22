/**
 * 画布指针移动（悬停预览 / 拖拽推进）。
 *
 * 【为什么单独成文件】它原本是 `usePianoRollInteractions` 里的一个 `useCallback`
 * （104 行）。这些回调只做一件事，却和另外几十个回调挤在同一个 hook 里。
 *
 * 【依赖怎么拿】上下文由调用方打包传入（`Pick<PointerDeps, …>`，字段清单由 AST
 * 扫描得出），函数体原样搬过来，开头解构回局部变量。
 */
import { isModifierActive } from "../../../../features/keybindings/keybindingsSlice";
import { getDrawPreviewValue } from "../paramValuePreviewLogic";
import type { PointerEvent as ReactPointerEvent } from "react";

import type { PointerDeps } from "./pointerDeps";

/** 本回调需要的上下文。 */
type Deps = Pick<
    PointerDeps,
    | "editParam"
    | "getCurveValueNearPointer"
    | "getDefaultCanvasCursor"
    | "hoverPreviewNearCurveRef"
    | "isEffectivePitchSnapActive"
    | "isPointerNearDraggableSelection"
    | "isPointerNearSelectionEdge"
    | "isPointerNearStretchSelectionEdge"
    | "lastPointerClientRef"
    | "morphDragRef"
    | "onParamValuePreviewChange"
    | "panRef"
    | "paramMultiSelectKb"
    | "paramStretchKb"
    | "paramValuePopupEnabled"
    | "pitchSnapToleranceCents"
    | "pitchSnapUnit"
    | "pointerSec"
    | "pointerValue"
    | "projectScale"
    | "scaleAtSec"
    | "setCanvasCursor"
    | "strokeRef"
    | "toolMode"
>;

/** 画布指针移动（悬停预览 / 拖拽推进）。 */
export function handleCanvasPointerMove(deps: Deps, e: ReactPointerEvent<HTMLCanvasElement>): void {
    const {
        editParam,
        getCurveValueNearPointer,
        getDefaultCanvasCursor,
        hoverPreviewNearCurveRef,
        isEffectivePitchSnapActive,
        isPointerNearDraggableSelection,
        isPointerNearSelectionEdge,
        isPointerNearStretchSelectionEdge,
        lastPointerClientRef,
        morphDragRef,
        onParamValuePreviewChange,
        panRef,
        paramMultiSelectKb,
        paramStretchKb,
        paramValuePopupEnabled,
        pitchSnapToleranceCents,
        pitchSnapUnit,
        pointerSec,
        pointerValue,
        projectScale,
        scaleAtSec,
        setCanvasCursor,
        strokeRef,
        toolMode,
    } = deps;

    lastPointerClientRef.current = { x: e.clientX, y: e.clientY };
    if (paramValuePopupEnabled) {
        const draggingLeft = Boolean(strokeRef.current) && (e.buttons & 1) === 1;
        if (draggingLeft) {
            const rawPreviewValue = pointerValue(e.clientY);
            const dragPreviewValue =
                toolMode === "draw"
                    ? getDrawPreviewValue({
                          editParam,
                          rawValue: rawPreviewValue,
                          effectiveSnap: isEffectivePitchSnapActive(e.nativeEvent),
                          pitchSnapUnit,
                          projectScale: scaleAtSec?.(pointerSec(e.clientX)) ?? projectScale,
                          pitchSnapToleranceCents,
                      })
                    : rawPreviewValue;
            onParamValuePreviewChange?.({
                clientX: e.clientX,
                clientY: e.clientY,
                value: dragPreviewValue,
            });
        } else {
            const nearCurveValue = getCurveValueNearPointer(
                e.clientX,
                e.clientY,
                e.nativeEvent.pointerType,
            );
            // 记录悬停激活状态：数据刷新（refreshParamValuePreview）
            // 只续显已激活的浮窗。
            hoverPreviewNearCurveRef.current = nearCurveValue != null;
            if (nearCurveValue == null) {
                onParamValuePreviewChange?.(null);
            } else {
                onParamValuePreviewChange?.({
                    clientX: e.clientX,
                    clientY: e.clientY,
                    value: nearCurveValue,
                });
            }
        }
    }

    // 拖拽中（平移/绘制/Alt 形变）跳过悬停链：指针捕获会把 move 重定向到
    // 画布，悬停命中测试会反过来覆盖拖拽路径设置的光标（grabbing 等）。
    if (panRef.current || strokeRef.current || morphDragRef.current) return;
    // 光标提示的顺序与 pointerdown 的手势优先级**逐条对应**（否则会出现
    // "显示可拉伸、按下却在拖曲线"的错位）：
    //   1. **已选中的参数线** → 抓取。参数线的命中范围最窄（10px 带 + 必须
    //      落在选区段内），而且它在画面里可能只有很短一截、很难瞄准；选区的
    //      边缘则是两条贯穿整个高度的竖线，用户想抓边缘时会自己去对着线抓。
    //      因此参数线的交互优先级**永远**高于选区自身（边缘 / 多选 / 平移）。
    //   2. Alt（显式修饰键）压住边缘 → 拉伸（参数线未命中时才轮到它）；
    //   3. 多选修饰键提示框选 / 切换段（同样让位给参数线，否则按住修饰键就
    //      再也拖不动选区内的曲线）；
    //   4. 无修饰键的边缘 → 调整边界。
    if (isPointerNearDraggableSelection(e.clientX, e.clientY, e.nativeEvent.pointerType)) {
        setCanvasCursor("grab");
        return;
    }
    if (isPointerNearStretchSelectionEdge(e)) {
        setCanvasCursor("ew-resize");
        return;
    }
    // 多选修饰键按下时提示"可框选/可切换段"（Alt 拉伸更具体，优先）
    if (
        toolMode === "select" &&
        !isModifierActive(paramStretchKb, e.nativeEvent) &&
        isModifierActive(paramMultiSelectKb, e.nativeEvent)
    ) {
        setCanvasCursor("crosshair");
        return;
    }
    if (isPointerNearSelectionEdge(e)) {
        setCanvasCursor("ew-resize");
        return;
    }
    setCanvasCursor(getDefaultCanvasCursor());
}
