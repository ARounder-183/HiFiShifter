import { startPointerGesture, type PointerGesture } from "../pointerGesture";
import {
    scaleStepDeltaBetween,
    transposePitchByScaleSteps,
} from "../../../../../utils/musicalScales";
import {
    isChildFormantOffsetCentsParam,
    isChildPitchOffsetCentsParam,
    isChildPitchOffsetDegreesParam,
} from "../../childPitchOffsetParams";
import { isDynParam, shiftDynValueForDrag, shiftValueForDrag } from "../../paramRanges";
import { selectionToFrameSpans, shiftSelectionRanges, snapFrame } from "../../paramSelection";
import { editablePitchValue } from "../../paramSmoothing";
import { getSelectDragPreviewValue } from "../../paramValuePreviewLogic";
import {
    buildMultiRangeEditPlan,
    fetchFullResCurve,
    readPvRange,
    uploadFullResCurveSegments,
} from "../../selectionEditData";
import {
    PARAM_DRAG_DEAD_ZONE_PX,
    makePvValueSource,
    recordGestureParamSelectionStep,
} from "../pointerSupport";
import { handleRightDragMorph } from "./rightDragMorph";
import type { FrameSpan, ParamSelection } from "../../paramSelection";
import type { ParamViewSegment } from "../../types";
import type { PointerEvent as ReactPointerEvent } from "react";
import type { PointerDeps } from "../pointerDeps";

/** `handleParamLineDrag` 作用域里的局部量。 */
type ParamLineLocals = {
    curvePoint: { y: number; value: number } | null;
    fp: number;
    mouseVal: number;
    pv: ParamViewSegment;
    sel: ParamSelection;
};

/** 本会话所需的全部上下文。 */
type Deps = Pick<
    PointerDeps,
    | "applyDenseToLiveEdit"
    | "buildUploadSegments"
    | "bumpRefreshToken"
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
> &
    ParamLineLocals;

/**
 * 处理指针落在参数线附近时的左键拖动。
 *
 * @param deps hook 作用域 + 局部量的上下文。
 * @param e 触发本次手势的指针事件。
 * @returns `true` = 已消费、调用方应停止；`false` = 未消费，调用方继续。
 */
export function handleParamLineNearDrag(
    deps: Deps,
    e: ReactPointerEvent<HTMLCanvasElement>,
): boolean {
    const {
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
    } = deps;

    if (e.button === 2) {
        if (
            handleRightDragMorph(
                {
                    applyDenseToLiveEdit,
                    buildUploadSegments,
                    bumpRefreshToken,
                    clearActivePointerGestureEnd,
                    createFineAdjustedPointerState,
                    disposeFineAdjustedPointerState,
                    edgeHalfSpanForIndices,
                    editParam,
                    ensureLiveEditBase,
                    fetchCommitBaseSource,
                    fp,
                    getFineAdjustedPointerPosition,
                    invalidate,
                    liveEditActiveRef,
                    liveEditOverrideRef,
                    onContextMenu,
                    onParamCommitSucceeded,
                    onParamValuePreviewChange,
                    paramValuePopupEnabled,
                    paramViewRef,
                    pv,
                    requestWaveformRepaint,
                    rootTrackId,
                    sel,
                    setActivePointerGestureEnd,
                    setCanvasCursor,
                    setParamView,
                },
                e,
            )
        ) {
            return true;
        }
    }

    // 进入拖拽选中曲线模式（支持 X+Y 双向拖拽）
    // 多选区语义：所有段同步移动同一 Δx/Δy，断层保持不变。
    setCanvasCursor("grabbing");
    const startMouseVal = mouseVal;
    const startFrame = pointerFrame(e.clientX);
    // 点击死区：未越过阈值前不预览、不更新，收尾时按"点击"
    // 处理（不回写数据、不打撤销点，见 PARAM_DRAG_DEAD_ZONE_PX）。
    const startClientX = e.clientX;
    const startClientY = e.clientY;
    let didDrag = false;
    const pid = e.pointerId;
    (e.currentTarget as HTMLCanvasElement).setPointerCapture(pid);
    const finePointerState = createFineAdjustedPointerState(
        e.nativeEvent,
        e.currentTarget as HTMLCanvasElement,
    );

    // 选区帧闭区间（逐段，升序互斥）—— 注意这里**不**夹到 pv 的
    // 已加载窗口内。「全选」时选区覆盖整个工程，若按 pv 裁剪，
    // 后续只会改到显示范围内的参数，工程其余部分纹丝不动。
    const dragSpans: FrameSpan[] = selectionToFrameSpans(sel);
    if (dragSpans.length === 0) return true;

    // 取各段的全分辨率原始值。
    // pv 在低缩放下按画布宽度做了降采样，直接拿去变换再回写
    // 会把全分辨率曲线覆盖掉，所以变换的输入必须是 stride=1
    // 的数据。pv 已以 stride=1 覆盖时零开销直接切片，行为与
    // 改动前一致；否则分块向后端拉取（不阻塞拖动）。
    const origCurvePromises = dragSpans.map((span) =>
        fetchFullResCurve({
            trackId: rootTrackId ?? "",
            param: editParam,
            startFrame: span.startFrame,
            endFrame: span.endFrame,
            paramView: pv,
        }),
    );
    // 先用 pv 覆盖到的部分做即时预览；全分辨率数据到位后自动
    // 替换，之后的每一帧预览都基于无损数据。
    let origValuesPerRange: number[][] = dragSpans.map((span) =>
        readPvRange(pv, span.startFrame, span.endFrame),
    );
    let dragSettled = false;
    void Promise.all(origCurvePromises)
        .then((curves) => {
            if (dragSettled) return;
            origValuesPerRange = curves.map((curve) => curve.values);
        })
        .catch(() => {
            // 取数失败只降级预览保真度（继续用 pv 数据预览）；
            // 提交路径（onUp）有自己的 try/catch 兜底。
        });

    // 逐帧变换。拖动过程中 `lastValueDelta` / `lastScaleStepDelta`
    // 会被 onMove 更新，这里读到的是最新值。
    const transformDragValue = (orig: number, frame: number) => {
        if (useScaleDegreeTranspose && editParam === "pitch" && dragAnchorScale) {
            // 每个帧用其落地时刻的生效音阶做度数移调。
            const frameScale = scaleAtSec?.((frame * fp) / 1000) ?? dragAnchorScale;
            return orig === 0
                ? 0
                : transposePitchByScaleSteps(orig, lastScaleStepDelta, frameScale);
        }
        // 动态（比值域）走**锚点缩放**：系数由"被抓住
        // 那条线的值"导出，使锚点位移恰好等于指针位移
        // （跟手），其余点按比例缩放（倍率域相对关系保留），
        // 静音 0 × k 仍为 0。锚点贴地时该函数内部退回线性
        // 偏移。完整法则收在 paramRanges.shiftDynValueForDrag，
        // 预览与提交共用它。
        if (isDynParam(editParam)) {
            return shiftDynValueForDrag(orig, dynDragAnchor ?? Number.NaN, lastValueDelta);
        }
        // 其余参数：值域内**线性偏移** —— 纵轴是线性刻度，
        // `原值 + Δ` 才让被抓取的那一点停在光标下
        //（见 paramRanges.shiftValueForDrag）。
        return shiftValueForDrag(editParam, orig, lastValueDelta);
    };

    // Tempo Map 感知：度数差以拖动锚点帧（选区整体起点）的
    // 生效音阶计算 —— 多选区下仍以第一段起点为锚点，
    // 保证一次拖动只有一个锚定音阶。
    const dragAnchorFrame = dragSpans[0].startFrame;
    const dragAnchorScale = scaleAtSec?.((dragAnchorFrame * fp) / 1000) ?? projectScale;
    ensureLiveEditBase(pv);
    if (liveEditActiveRef) liveEditActiveRef.current = true;
    if (
        editParam === "pitch" ||
        isChildPitchOffsetCentsParam(editParam) ||
        isChildPitchOffsetDegreesParam(editParam) ||
        isChildFormantOffsetCentsParam(editParam)
    ) {
        onPitchSnapGestureActiveChange?.(true);
    }

    // 用闭包变量记录最新 X/Y 偏移量
    let lastValueDelta = 0;
    /**
     * 拖拽的**锚点值**：按下时指针所在帧上的曲线值
     * （即"被抓住的那条线"的位置）。动态的拖拽幅度系数
     * 由它导出（见 `paramRanges.dynDragScaleFactor`），
     * 因此取**按下那一刻**的值、拖动过程中不变 ——
     * 若跟着实时值走，系数会随拖动自身变化而自激。
     *
     * 命中判定已经算过这个值（`curvePoint`，用于 10px 邻域
     * 判定）；此处只是把它固定下来。取不到时为 null，
     * 动态将退回线性偏移。
     */
    const dynDragAnchor: number | null =
        isDynParam(editParam) && curvePoint !== null ? curvePoint.value : null;
    let lastScaleStepDelta = 0;
    let useScaleDegreeTranspose = false;
    let lastFrameDelta = 0; // 帧偏移（整数）
    // 使用闭包变量跟踪当前拖动方向（可通过右键切换）
    let currentDragDir = dragDirection ?? "y-only";

    // 预览重算 rAF 合帧：与右键下拖路径同模式。pointermove
    // 在高刷新率鼠标上可达 125–1000Hz，而每次预览是
    // O(选区帧数) 的 dense 构建 + 整份 live 拷贝 + React
    // setState（全面板重渲），长选区逐事件执行会明显卡顿；
    // 一帧至多重算一次，onMove 只更新标量偏移。
    let previewRafId: number | null = null;
    let previewQueued = false;
    const runPreviewStep = () => {
        const pvNow = paramViewRef.current;
        if (!pvNow) return;

        // X 向平移会把早先帧写在计划窗口之外的值留在 live
        // 覆盖里。不能整份重建（= null + ensure，每帧 O(窗口)
        // 拷贝，见 resetLiveEditPreview 文档）：按"本手势写过的
        // 区间并集"精确还原后再写本帧计划窗口。写入点
        // applyDenseToLiveEdit 内部会按需建基准。
        resetLiveEditPreview(pvNow, { keepWrittenRange: true });

        // 多段计划：覆盖「各段原位 ∪ 落地位 ± 边缘淡化」的
        // 合并窗口（逐帧索引）。预览从 pv 取上下文 —— pv 只是
        // 显示数据，不会回写；提交走同一计划的另一份 sourceAt。
        const pieces = buildMultiRangeEditPlan({
            ranges: dragSpans,
            frameDelta: lastFrameDelta,
            valuesAt: (index) => origValuesPerRange[index],
            sourceAt: makePvValueSource(pvNow),
            transformAt: (_index, _i, sourceValue, targetFrame) =>
                transformDragValue(sourceValue, targetFrame),
            edgeHalfSpanAt: (index) =>
                edgeHalfSpanForIndices(origValuesPerRange[index]?.length ?? 0, 1),
            isEditable: editParam === "pitch" ? editablePitchValue : undefined,
        });

        for (const piece of pieces) {
            applyDenseToLiveEdit(
                pvNow,
                piece.startFrame,
                piece.values,
                piece.startFrame,
                piece.endFrame,
                "draw",
            );
        }

        // 实时更新选区位置显示（所有段同步平移；纯平移不改变
        // 段间距，故不会产生新的重叠段）。选区的单位就是帧，
        // 因此**直接**用同一个帧位移，不再换算回拍。
        selectionRef.current = shiftSelectionRanges(sel, lastFrameDelta);
        updateSelectionUi(selectionRef.current);

        invalidate();
        // live 覆盖只改 ref：拖拽音量/动态时波形面需
        // 显式重绘（波形 = 可听结果，须实时跟随）。
        requestWaveformRepaint();
    };
    const schedulePreview = () => {
        previewQueued = true;
        if (previewRafId == null) {
            previewRafId = requestAnimationFrame(() => {
                previewRafId = null;
                if (!previewQueued) return;
                previewQueued = false;
                runPreviewStep();
            });
        }
    };

    /** 手势句柄（onMove 需要主动收尾，见 pan 分支的同名字段）。 */
    let gesture: PointerGesture | null = null;
    const onMove = (ev: globalThis.PointerEvent) => {
        // 掌侧误触防御：协议已按 pointerId 过滤，第二指的 move 到不了这里。
        if ((ev.buttons & 1) !== 1) {
            gesture?.end("up", ev);
            return;
        }
        // 点击死区：越过阈值之前一律按"还没开始拖"处理 ——
        // 不更新位移、不预览、不弹浮窗（因此画面上不会有任何
        // 变化），松手时也不会回写。
        if (!didDrag) {
            if (
                Math.abs(ev.clientX - startClientX) <= PARAM_DRAG_DEAD_ZONE_PX &&
                Math.abs(ev.clientY - startClientY) <= PARAM_DRAG_DEAD_ZONE_PX
            ) {
                return;
            }
            didDrag = true;
        }
        const adjusted = getFineAdjustedPointerPosition(finePointerState, ev);
        const currentVal = pointerValue(adjusted.clientY);
        let rawValueDelta = currentVal - startMouseVal;

        // 音高吸附：Toggle snap modifier (XOR with pitchSnapEnabled)
        const effectiveSnap = isEffectivePitchSnapActive(ev);
        const yDragEnabled = currentDragDir !== "x-only";
        if (effectiveSnap && editParam === "pitch" && yDragEnabled) {
            if (pitchSnapUnit === "scale" && dragAnchorScale) {
                useScaleDegreeTranspose = true;
                lastScaleStepDelta = scaleStepDeltaBetween(
                    startMouseVal,
                    currentVal,
                    dragAnchorScale,
                );
                rawValueDelta = 0;
            } else {
                useScaleDegreeTranspose = false;
                rawValueDelta = Math.round(rawValueDelta);
            }
        } else if (effectiveSnap && isChildPitchOffsetCentsParam(editParam) && yDragEnabled) {
            useScaleDegreeTranspose = false;
            rawValueDelta = Math.round(rawValueDelta / 100) * 100;
        } else if (effectiveSnap && isChildPitchOffsetDegreesParam(editParam) && yDragEnabled) {
            useScaleDegreeTranspose = false;
            rawValueDelta = Math.round(rawValueDelta);
        } else if (effectiveSnap && isChildFormantOffsetCentsParam(editParam) && yDragEnabled) {
            useScaleDegreeTranspose = false;
            rawValueDelta = Math.round(rawValueDelta / 50) * 50;
        } else {
            useScaleDegreeTranspose = false;
            if (!yDragEnabled) {
                lastScaleStepDelta = 0;
                rawValueDelta = 0;
            }
        }

        // 计算 X 方向帧偏移：指针帧坐标之差，取整到整数帧
        // （选区与数据都按整数帧落位，无需任何单位换算）。
        const currentFrame = pointerFrame(adjusted.clientX);
        const rawFrameDelta = snapFrame(currentFrame - startFrame);

        // 应用拖动方向限制
        lastValueDelta = yDragEnabled ? rawValueDelta : 0;
        lastFrameDelta = currentDragDir === "y-only" ? 0 : rawFrameDelta;

        // 悬停弹窗跟随原始事件坐标（廉价，不合帧）。
        if (paramValuePopupEnabled) {
            const previewCurrentVal = yDragEnabled ? currentVal : startMouseVal;
            onParamValuePreviewChange?.({
                clientX: ev.clientX,
                clientY: ev.clientY,
                value: getSelectDragPreviewValue({
                    editParam,
                    startValue: startMouseVal,
                    currentValue: previewCurrentVal,
                    fineScale: 1,
                    effectiveSnap,
                    pitchSnapUnit,
                    projectScale: scaleAtSec?.(pointerSec(ev.clientX)) ?? projectScale,
                }),
            });
        }

        // O(n) 预览重算合帧到下一渲染帧（见上方注释）。
        schedulePreview();
    };

    const onEnd = async () => {
        disposeDragDirKey();
        disposeFineAdjustedPointerState(finePointerState);
        // 取消挂起的预览帧：松手后由提交路径以全分辨率
        // 数据重建，预览层不得再覆盖提交结果。
        if (previewRafId != null) {
            cancelAnimationFrame(previewRafId);
            previewRafId = null;
        }
        previewQueued = false;
        // 标记手势结束，避免已发出的取数请求再覆写 origValues
        dragSettled = true;

        // 捕获阶段的 contextmenu/mousedown 抑制与临时 snap
        // 高亮是纯清理：必须放在所有早退路径（包括下面的
        // 单击死区 return）之前，否则一次普通点击就会把
        // window 级监听器永久留在窗口上——全局右键菜单从此
        // 失效，且残留的 mousedown 监听还会让下一次拖拽的
        // 右键换向被执行两次。
        window.removeEventListener("contextmenu", onContextMenuDuringDrag, true);
        window.removeEventListener("mousedown", onMouseDownDuringDrag, true);
        if (
            editParam === "pitch" ||
            isChildPitchOffsetCentsParam(editParam) ||
            isChildPitchOffsetDegreesParam(editParam)
        ) {
            onPitchSnapGestureActiveChange?.(false);
        }

        // 单击（未越过点击死区）：这次手势没有改动任何值 ——
        // 丢弃 live 覆盖层并复位 active 标记后直接收尾。
        // 走提交路径会把变换结果整段写回后端（此处与原始数据
        // 逐帧相同）并多打一个"按下毫无变化"的撤销点。
        if (!didDrag) {
            liveEditOverrideRef.current = null;
            if (liveEditActiveRef) {
                liveEditActiveRef.current = false;
            }
            if (paramValuePopupEnabled) {
                onParamValuePreviewChange?.(null);
            }
            setCanvasCursor("grab");
            invalidate();
            return;
        }

        // 提交拖拽结果到后端
        const pvNow = paramViewRef.current;
        if (pvNow && rootTrackId) {
            // await 链路上的任何 IPC 失败都必须复位
            // liveEdit 状态（否则预览覆盖层永久冻结、
            // 曲线刷新被永久搁置）。
            try {
                // 等待拖动开始时发起的全分辨率取数完成 —— 提交必须
                // 基于无损数据，不能拿降采样的 pv 值去覆盖后端。
                const fullResCurves = await Promise.all(origCurvePromises);
                origValuesPerRange = fullResCurves.map((curve) => curve.values);

                // 边缘平滑度：毫秒定标的过渡带半宽（帧），
                // 扩展写入窗口以包含各段边界外侧上下文。
                // 提交 dense 恒为 stride=1。
                const edgeHalfSpanForDragRange = (index: number) =>
                    edgeHalfSpanForIndices(origValuesPerRange[index]?.length ?? 0, 1);

                // 提交前把计划写入窗口（「各段原位 ∪ 落地位 ±
                // 边缘淡化」合并后）的全分辨率数据拉下来作为基底。
                // 这是「回写不失真」的关键：预览用的是可能降采样的
                // pv，提交必须用 stride=1 的真实曲线，否则会把
                // 降采样后的值写回后端、覆盖掉原始分辨率。
                const commit = await fetchCommitBaseSource({
                    trackId: rootTrackId,
                    param: editParam,
                    ranges: dragSpans,
                    frameDelta: lastFrameDelta,
                    edgeHalfSpanAt: edgeHalfSpanForDragRange,
                    paramView: pvNow,
                    withSentinel: isDynParam(editParam),
                });

                const pieces = buildMultiRangeEditPlan({
                    ranges: dragSpans,
                    frameDelta: lastFrameDelta,
                    valuesAt: (index) => origValuesPerRange[index],
                    sourceAt: commit.sourceAt,
                    transformAt: (_index, _i, sourceValue, targetFrame) =>
                        transformDragValue(sourceValue, targetFrame),
                    edgeHalfSpanAt: edgeHalfSpanForDragRange,
                    isEditable: editParam === "pitch" ? editablePitchValue : undefined,
                });

                if (pieces.length > 0) {
                    // 立即同步更新本地 paramView state（逐帧映射到 pv 的
                    // 采样栅格上；pv 只是显示，随后会被后端数据刷新）
                    const nextEdit = pvNow.edit.slice();
                    const pvStepUp = Math.max(1, Math.floor(pvNow.stride));
                    for (const piece of pieces) {
                        for (let i = 0; i < piece.values.length; i += 1) {
                            const globalIdx = Math.round(
                                (piece.startFrame + i - pvNow.startFrame) / pvStepUp,
                            );
                            if (globalIdx >= 0 && globalIdx < nextEdit.length) {
                                nextEdit[globalIdx] = piece.values[i];
                            }
                        }
                    }
                    setParamView({
                        ...pvNow,
                        edit: nextEdit,
                    });
                    // 选区拖拽不走 commitStroke，这里显式收尾：
                    // 保留覆盖层直到"提交之后取的快照"到达。
                    onParamCommitSucceeded();

                    // 确保选区位置最终正确（多段同步平移；帧位移直接复用）
                    selectionRef.current = shiftSelectionRanges(sel, lastFrameDelta);
                    updateSelectionUi(selectionRef.current);

                    // 分块回写：整次编辑（含所有段）只打一个撤销点，
                    // 块之间让出事件循环，超长工程也不会卡死界面。
                    // 上传失败同样复位 liveEdit（finally），不让
                    // 覆盖层冻结。
                    // 拖动前的选区快照：本手势把选区与数据
                    // 一起平移，因此撤销时选区也必须一起回来
                    // （与拉伸同一套登记，见其注释）。
                    const selectionBeforeDrag = sel ? sel.map((range) => ({ ...range })) : null;
                    const selectionAfterDrag = selectionRef.current
                        ? selectionRef.current.map((range) => ({
                              ...range,
                          }))
                        : null;
                    void (async () => {
                        let uploaded = false;
                        try {
                            await uploadFullResCurveSegments({
                                trackId: rootTrackId,
                                param: editParam,
                                segments: buildUploadSegments(editParam, pieces, commit.sentinelAt),
                            });
                            uploaded = true;
                        } catch (err) {
                            console.error("[pianoRoll] select-drag upload failed", err);
                        } finally {
                            if (liveEditActiveRef) liveEditActiveRef.current = false;
                            bumpRefreshToken();
                        }
                        if (!uploaded) return;
                        await recordGestureParamSelectionStep({
                            before: selectionBeforeDrag,
                            after: selectionAfterDrag,
                        });
                    })();
                }
            } catch (err) {
                console.error("[pianoRoll] select-drag commit failed", err);
                liveEditOverrideRef.current = null;
                if (liveEditActiveRef) liveEditActiveRef.current = false;
            }
        } else {
            if (liveEditActiveRef) liveEditActiveRef.current = false;
        }
        if (paramValuePopupEnabled) {
            onParamValuePreviewChange?.(null);
        }

        setCanvasCursor("grab");
        invalidate();
    };

    // 拖拽过程中右键点击切换拖动方向
    const onContextMenuDuringDrag = (ev: Event) => {
        ev.preventDefault();
        ev.stopImmediatePropagation();
    };
    const cycleSelectDragDir = () => {
        const order: Array<"free" | "x-only" | "y-only"> = ["free", "x-only", "y-only"];
        const idx = order.indexOf(currentDragDir);
        currentDragDir = order[(idx + 1) % order.length];
        // Also cycle the global setting
        if (onCycleDragDirection) onCycleDragDirection("select");
    };
    const onMouseDownDuringDrag = (event: Event) => {
        // 注册在 window 的 `mousedown` 上，因此类型是 Event；这里收窄回 MouseEvent。
        const ev = event as globalThis.MouseEvent;
        if (ev.button !== 2) return;
        // 仅在左键拖拽进行中时，右键才切换拖拽方向。
        if ((ev.buttons & 1) !== 1) return;
        ev.preventDefault();
        ev.stopPropagation();
        cycleSelectDragDir();
    };
    // 触控板替代：拖拽中按下快捷键与右键同义。
    const disposeDragDirKey = installDragDirectionKeyCycler(cycleSelectDragDir);

    gesture = startPointerGesture({
        pointerId: pid,
        captureTarget: e.currentTarget,
        onMove,
        onEnd: () => {
            void onEnd();
        },
        extra: [
            { type: "contextmenu", listener: onContextMenuDuringDrag, capture: true },
            { type: "mousedown", listener: onMouseDownDuringDrag, capture: true },
        ],
        registerEnd: setActivePointerGestureEnd,
        clearEnd: clearActivePointerGestureEnd,
    });
    return true;
}
