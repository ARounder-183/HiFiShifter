/**
 * 参数选区右键幅度手势的共享入口。
 *
 * 选择工具（参数线右键）与块编辑工具（块体右键）必须使用同一套数据变换、
 * 百分比映射、预览合帧与提交/撤销语义。底层实现保持在 rightDragMorph.ts，
 * 本文件只把「显式参数」适配成它已经验证过的依赖对象，避免复制算法。
 */
import { handleRightDragMorph } from "./rightDragMorph";
import type { ParamSelection } from "../../paramSelection";
import type { ParamViewSegment } from "../../types";
import type { PointerDeps } from "../pointerDeps";
import type { FullResWriteSegment } from "../../selectionEditData";
import type { PointerEvent as ReactPointerEvent } from "react";

type MorphDeps = Pick<
    PointerDeps,
    | "applyDenseToLiveEdit"
    | "buildUploadSegments"
    | "bumpRefreshToken"
    | "clearActivePointerGestureEnd"
    | "createFineAdjustedPointerState"
    | "disposeFineAdjustedPointerState"
    | "edgeHalfSpanForIndices"
    | "editParam"
    | "ensureLiveEditBase"
    | "fetchCommitBaseSource"
    | "getFineAdjustedPointerPosition"
    | "invalidate"
    | "liveEditActiveRef"
    | "liveEditOverrideRef"
    | "onContextMenu"
    | "onParamCommitSucceeded"
    | "onParamValuePreviewChange"
    | "paramValuePopupEnabled"
    | "paramViewRef"
    | "requestWaveformRepaint"
    | "rootTrackId"
    | "setActivePointerGestureEnd"
    | "setCanvasCursor"
    | "setParamView"
>;

export interface ParamSelectionMorphArgs extends MorphDeps {
    readonly pointerId: number;
    readonly clientX: number;
    readonly clientY: number;
    readonly nativeEvent: globalThis.PointerEvent;
    readonly captureTarget: HTMLCanvasElement;
    readonly selection: ParamSelection;
    readonly paramView: ParamViewSegment;
    readonly framePeriodMs: number;
    /** 提交成功后的额外收尾（块工具用来清 pitch 块 value）。 */
    readonly onCommitted?: () => void;
    /** 块工具：曲线与清 value 在同一事务中提交。 */
    readonly commitAmplitude?: (segments: readonly FullResWriteSegment[]) => Promise<void>;
}

/** 开始一次右键幅度手势；返回值表示事件是否被消费。 */
export function beginParamSelectionMorph(args: ParamSelectionMorphArgs): boolean {
    const syntheticEvent = {
        preventDefault: () => undefined,
        clientX: args.clientX,
        clientY: args.clientY,
        pointerId: args.pointerId,
        currentTarget: args.captureTarget,
        nativeEvent: args.nativeEvent,
    } as unknown as ReactPointerEvent<HTMLCanvasElement>;
    return handleRightDragMorph(
        {
            applyDenseToLiveEdit: args.applyDenseToLiveEdit,
            buildUploadSegments: args.buildUploadSegments,
            bumpRefreshToken: args.bumpRefreshToken,
            clearActivePointerGestureEnd: args.clearActivePointerGestureEnd,
            createFineAdjustedPointerState: args.createFineAdjustedPointerState,
            disposeFineAdjustedPointerState: args.disposeFineAdjustedPointerState,
            edgeHalfSpanForIndices: args.edgeHalfSpanForIndices,
            editParam: args.editParam,
            ensureLiveEditBase: args.ensureLiveEditBase,
            fetchCommitBaseSource: args.fetchCommitBaseSource,
            fp: args.framePeriodMs,
            getFineAdjustedPointerPosition: args.getFineAdjustedPointerPosition,
            invalidate: args.invalidate,
            liveEditActiveRef: args.liveEditActiveRef,
            liveEditOverrideRef: args.liveEditOverrideRef,
            onContextMenu: args.onContextMenu,
            onParamCommitSucceeded: () => {
                args.onParamCommitSucceeded();
                args.onCommitted?.();
            },
            onParamValuePreviewChange: args.onParamValuePreviewChange,
            paramValuePopupEnabled: args.paramValuePopupEnabled,
            paramViewRef: args.paramViewRef,
            pv: args.paramView,
            requestWaveformRepaint: args.requestWaveformRepaint,
            rootTrackId: args.rootTrackId,
            sel: args.selection,
            setActivePointerGestureEnd: args.setActivePointerGestureEnd,
            setCanvasCursor: args.setCanvasCursor,
            setParamView: args.setParamView,
            commitAmplitude: args.commitAmplitude,
        },
        syntheticEvent,
    );
}
