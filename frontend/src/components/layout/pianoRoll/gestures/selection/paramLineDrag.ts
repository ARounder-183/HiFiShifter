import { profileFor, scaledHitRadius } from "../../../../../utils/inputProfile";
import {
    CURVE_HIT_RADIUS_PX,
    curvePointAtPointer,
    isPointerNearCurve,
} from "../../kernel/gestureHitTest";
import { framesToTime } from "../../utils";
import type { ParamSelection } from "../../paramSelection";
import type { PointerEvent as ReactPointerEvent } from "react";
import type { PointerDeps } from "../pointerDeps";
import { handleParamLineNearDrag } from "./paramLineNearDrag";

/** `handleSelectPointerDown` 作用域里的局部量。 */
type SelectLocals = {
    b: number;
    sel: ParamSelection;
};

/** 本分支所需的全部上下文。 */
type Deps = Pick<
    PointerDeps,
    | "applyDenseToLiveEdit"
    | "buildUploadSegments"
    | "bumpRefreshToken"
    | "canvasRef"
    | "clearActivePointerGestureEnd"
    | "createFineAdjustedPointerState"
    | "disposeFineAdjustedPointerState"
    | "dragDirection"
    | "edgeHalfSpanForIndices"
    | "editParam"
    | "ensureLiveEditBase"
    | "fetchCommitBaseSource"
    | "getFineAdjustedPointerPosition"
    | "installDragDirectionKeyCycler"
    | "invalidate"
    | "isEffectivePitchSnapActive"
    | "liveEditActiveRef"
    | "liveEditOverrideRef"
    | "onContextMenu"
    | "onCycleDragDirection"
    | "onParamCommitSucceeded"
    | "onParamValuePreviewChange"
    | "onPitchSnapGestureActiveChange"
    | "paramValuePopupEnabled"
    | "paramViewRef"
    | "pitchSnapUnit"
    | "pointerFrame"
    | "pointerSec"
    | "pointerValue"
    | "projectScale"
    | "requestWaveformRepaint"
    | "resetLiveEditPreview"
    | "rootTrackId"
    | "scaleAtSec"
    | "selectionRef"
    | "setActivePointerGestureEnd"
    | "setCanvasCursor"
    | "setParamView"
    | "updateSelectionUi"
    | "valueToY"
    | "viewSizeRef"
> &
    SelectLocals;

/**
 * 参数线拖动。
 *
 * @param deps hook 作用域 + 局部量的上下文。
 * @param e 触发本次手势的指针事件。
 * @returns `true` = 已消费、调用方应停止；`false` = 未消费，调用方继续。
 */
export function handleParamLineDrag(deps: Deps, e: ReactPointerEvent<HTMLCanvasElement>): boolean {
    const {
        applyDenseToLiveEdit,
        b,
        buildUploadSegments,
        bumpRefreshToken,
        canvasRef,
        clearActivePointerGestureEnd,
        createFineAdjustedPointerState,
        disposeFineAdjustedPointerState,
        dragDirection,
        edgeHalfSpanForIndices,
        editParam,
        ensureLiveEditBase,
        fetchCommitBaseSource,
        getFineAdjustedPointerPosition,
        installDragDirectionKeyCycler,
        invalidate,
        isEffectivePitchSnapActive,
        liveEditActiveRef,
        liveEditOverrideRef,
        onContextMenu,
        onCycleDragDirection,
        onParamCommitSucceeded,
        onParamValuePreviewChange,
        onPitchSnapGestureActiveChange,
        paramValuePopupEnabled,
        paramViewRef,
        pitchSnapUnit,
        pointerFrame,
        pointerSec,
        pointerValue,
        projectScale,
        requestWaveformRepaint,
        resetLiveEditPreview,
        rootTrackId,
        scaleAtSec,
        sel,
        selectionRef,
        setActivePointerGestureEnd,
        setCanvasCursor,
        setParamView,
        updateSelectionUi,
        valueToY,
        viewSizeRef,
    } = deps;

    // 判断鼠标是否在**参数线**附近（沿折线的纵向距离 < 10px）
    const pv = paramViewRef.current;
    if (pv && pv.edit.length > 0) {
        // 插值 + 投影 + pitch 的 +0.5 偏移 + 邻域判定全部抽到
        // `kernel/gestureHitTest`（纯函数，有单测），与绘制折线同源。
        //
        // 【为什么必须沿折线插值】参数线是连续的：采样点之间由直线
        // 相连，指针落在两点中间时，取"最近采样点的值"量到的是端点 y
        // —— 陡峭段上可以远超 10px，于是"光标明明压在线上，却拖不动
        // 参数线"（只有正好落在采样点上才抓得住）。
        const canvas = canvasRef.current;
        const rectH = canvas ? canvas.getBoundingClientRect().height : viewSizeRef.current.h || 1;
        const curvePoint = curvePointAtPointer({
            sec: framesToTime(b, pv.framePeriodMs),
            startFrame: pv.startFrame,
            stride: pv.stride,
            framePeriodMs: pv.framePeriodMs,
            values: pv.edit,
            param: editParam,
            valueToY: (v) => valueToY(editParam, v, rectH),
        });
        // 下面拖拽分支仍要用这两个量（选区帧范围换算 / 起点值），
        // 因此保留声明，只是命中判定改走纯函数。
        const fp = pv.framePeriodMs;
        const mouseVal = pointerValue(e.clientY);
        const near = isPointerNearCurve({
            pointerY: canvas ? e.clientY - canvas.getBoundingClientRect().top : 0,
            curveY: curvePoint?.y ?? Number.NaN,
            hitRadiusPx: scaledHitRadius(CURVE_HIT_RADIUS_PX, profileFor(e.nativeEvent)),
        });

        if (near) {
            if (
                handleParamLineNearDrag(
                    {
                        applyDenseToLiveEdit,
                        buildUploadSegments,
                        bumpRefreshToken,
                        clearActivePointerGestureEnd,
                        createFineAdjustedPointerState,
                        curvePoint,
                        disposeFineAdjustedPointerState,
                        dragDirection,
                        edgeHalfSpanForIndices,
                        editParam,
                        ensureLiveEditBase,
                        fetchCommitBaseSource,
                        fp,
                        getFineAdjustedPointerPosition,
                        installDragDirectionKeyCycler,
                        invalidate,
                        isEffectivePitchSnapActive,
                        liveEditActiveRef,
                        liveEditOverrideRef,
                        mouseVal,
                        onContextMenu,
                        onCycleDragDirection,
                        onParamCommitSucceeded,
                        onParamValuePreviewChange,
                        onPitchSnapGestureActiveChange,
                        paramValuePopupEnabled,
                        paramViewRef,
                        pitchSnapUnit,
                        pointerFrame,
                        pointerSec,
                        pointerValue,
                        projectScale,
                        pv,
                        requestWaveformRepaint,
                        resetLiveEditPreview,
                        rootTrackId,
                        scaleAtSec,
                        sel,
                        selectionRef,
                        setActivePointerGestureEnd,
                        setCanvasCursor,
                        setParamView,
                        updateSelectionUi,
                    },
                    e,
                )
            ) {
                return true;
            }
        }
    }

    return false;
}
