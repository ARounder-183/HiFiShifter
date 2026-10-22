import { armRightDragContextMenuGuard } from "../../../../../utils/rightDragContextMenuGuard";
import { isDynParam } from "../../paramRanges";
import { selectionToFrameSpans } from "../../paramSelection";
import { editablePitchValue } from "../../paramSmoothing";
import { formatRightDragMorphPercent } from "../../paramValuePreviewLogic";
import {
    buildMultiRangeEditPlan,
    fetchFullResCurve,
    readPvRange,
    uploadFullResCurveSegments,
} from "../../selectionEditData";
import {
    createSelectionAmplifier,
    rightDragUpScale,
    transformSelectionByRightDrag,
} from "../../selectionTransforms";
import { makePvValueSource } from "../pointerSupport";
import type { FrameSpan } from "../../paramSelection";
import type { ParamViewSegment } from "../../types";
import type { PointerEvent as ReactPointerEvent } from "react";
import type { ParamSelection } from "../../paramSelection";
import type { PointerDeps } from "../pointerDeps";

/** `handleParamLineDrag` 作用域里的局部量。 */
type ParamLineLocals = {
    fp: number;
    pv: ParamViewSegment;
    sel: ParamSelection;
};

/** 本通路所需的全部上下文。 */
type Deps = Pick<
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
> &
    ParamLineLocals;

/**
 * 处理参数线上的右键拖动（百分比形变）。
 *
 * @param deps hook 作用域 + 局部量的上下文。
 * @param e 触发本次手势的指针事件。
 * @returns `true` = 已消费、调用方应停止；`false` = 未消费，调用方继续。
 */
export function handleRightDragMorph(deps: Deps, e: ReactPointerEvent<HTMLCanvasElement>): boolean {
    const {
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
    } = deps;

    e.preventDefault();
    const startClientY = e.clientY;
    const pid = e.pointerId;
    (e.currentTarget as HTMLCanvasElement).setPointerCapture(pid);
    const finePointerState = createFineAdjustedPointerState(
        e.nativeEvent,
        e.currentTarget as HTMLCanvasElement,
    );

    setCanvasCursor("grabbing");
    if (paramValuePopupEnabled) {
        onParamValuePreviewChange?.({
            clientX: e.clientX,
            clientY: e.clientY,
            value: 0,
            displayText: formatRightDragMorphPercent(0),
        });
    }

    // 多选区：每段独立取数、独立变换（断层两侧互不影响）
    const rightDragSpans: FrameSpan[] = selectionToFrameSpans(sel);
    if (rightDragSpans.length === 0) return true;
    // 拖拽数据与选区移动拖拽同一模式：先用 pv 近似值立即
    // 预览，全分辨率到位后自动替换；提交必须等全分辨率
    // 数据 —— 绝不把降采样值当连续帧写回后端（旧实现在
    // stride>1 时会把 stride 间隔采样当连续帧写入：
    // 时间压缩 + 覆盖未选帧，已修复）。
    let origValuesPerRange: number[][] = rightDragSpans.map((span) =>
        readPvRange(pv, span.startFrame, span.endFrame),
    );
    let lastDy = 0;
    let didDrag = false;
    // 下拖预览的 rAF 合帧状态（见 onMove）。
    let previewRafId: number | null = null;
    let previewQueuedDy: number | null = null;

    ensureLiveEditBase(pv);
    if (liveEditActiveRef) liveEditActiveRef.current = true;

    const origCurvePromises = rightDragSpans.map((span) =>
        fetchFullResCurve({
            trackId: rootTrackId ?? "",
            param: editParam,
            startFrame: span.startFrame,
            endFrame: span.endFrame,
            paramView: pv,
        }),
    );
    let dragSettled = false;
    void Promise.all(origCurvePromises)
        .then((curves) => {
            if (dragSettled) return;
            origValuesPerRange = curves.map((curve) => curve.values);
        })
        .catch((err) => {
            // 取数失败只影响预览保真度：保持 pv 近似值，
            // 提交路径会再次尝试并有自己的失败处理。
            console.error("[pianoRoll] full-res curve fetch failed", err);
        });

    // 每段自己的残差放大器：趋势按该段值数组引用缓存，
    // 每次 move 只做逐帧缩放，不重复高斯卷积。
    const amplifierCache = new Map<number, { src: number[]; apply: (scale: number) => number[] }>();
    const transformRightDragRange = (index: number, dy: number): number[] => {
        const values = origValuesPerRange[index];
        if (!values || values.length === 0) return [];
        if (dy >= 0) {
            let cached = amplifierCache.get(index);
            if (!cached || cached.src !== values) {
                cached = {
                    src: values,
                    apply: createSelectionAmplifier(values, editParam, {
                        framePeriodMs: fp,
                    }).apply,
                };
                amplifierCache.set(index, cached);
            }
            return cached.apply(rightDragUpScale(dy));
        }
        return transformSelectionByRightDrag(values, editParam, dy, {
            framePeriodMs: fp,
        });
    };

    const edgeHalfSpanForRange = (index: number) =>
        edgeHalfSpanForIndices(origValuesPerRange[index]?.length ?? 0, 1);

    /**
     * 帧索引 dense 片段（values[k] ↔ startFrame + k）：
     * 每段值已含变换，边缘淡化在计划内完成。复用与选区移动
     * 完全相同的纯函数，任意 stride 下预览与提交一致。
     */
    const buildRightDragPlan = (pvNow: ParamViewSegment, dy: number) =>
        buildMultiRangeEditPlan({
            ranges: rightDragSpans,
            valuesAt: (index) => transformRightDragRange(index, dy),
            sourceAt: makePvValueSource(pvNow),
            edgeHalfSpanAt: edgeHalfSpanForRange,
            isEditable: editParam === "pitch" ? editablePitchValue : undefined,
        });

    const suppressContextMenu = (ev: Event) => {
        ev.preventDefault();
        ev.stopImmediatePropagation();
    };

    const onMove = (ev: globalThis.PointerEvent) => {
        if ((ev.buttons & 2) !== 2) {
            onUp(ev);
            return;
        }
        // 只接受本指针的 move（掌压拒识的 move 侧防线）。
        if (ev.pointerId !== pid) return;
        const adjusted = getFineAdjustedPointerPosition(finePointerState, ev);
        const dy = startClientY - adjusted.clientY;
        if (Math.abs(dy) >= 2) {
            didDrag = true;
            // 本次手势在自己的阈值处确认构成拖拽：
            // 显式武装收尾守卫。松手若发生在**另一个**
            // 表面（标尺 / 轨道头）上，那个表面的菜单
            // 会被吞掉（见守卫模块头注释）——本手势
            // 自己的 window 级抑制只覆盖画布路径。
            armRightDragContextMenuGuard();
        }
        lastDy = dy;

        // 悬停弹窗跟随原始事件坐标（廉价，不合帧）。
        if (paramValuePopupEnabled) {
            onParamValuePreviewChange?.({
                clientX: ev.clientX,
                clientY: ev.clientY,
                value: dy,
                displayText: formatRightDragMorphPercent(dy),
            });
        }

        // 下拖是 O(n·r) 高斯重卷积（r 随拖距增长）：
        // rAF 合帧，一帧至多重算一次 —— 与上拖的
        // "趋势缓存"路径保持同量级的每帧成本，长选区
        // 全选拖动不再随拖距变卡。
        previewQueuedDy = dy;
        if (previewRafId == null) {
            previewRafId = requestAnimationFrame(() => {
                previewRafId = null;
                const queued = previewQueuedDy;
                previewQueuedDy = null;
                if (queued === null) return;
                const pvNow = paramViewRef.current;
                if (!pvNow) return;
                // 上拖：残差放大（各段趋势已缓存，仅逐帧
                // 缩放）；下拖：高斯平滑。两者都基于各段
                // 原始值无状态重算，段间互不影响。
                const pieces = buildRightDragPlan(pvNow, queued);
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
                invalidate();
                // live 覆盖只改 ref：音量/动态拖拽时波形面需显式重绘。
                requestWaveformRepaint();
            });
        }
    };

    const onUp = async (ev?: globalThis.PointerEvent) => {
        window.removeEventListener("pointermove", onMove);
        window.removeEventListener("pointerup", onUp);
        window.removeEventListener("pointercancel", onUp);
        window.removeEventListener("contextmenu", suppressContextMenu, true);
        disposeFineAdjustedPointerState(finePointerState);
        clearActivePointerGestureEnd(onUp);
        if (previewRafId != null) {
            cancelAnimationFrame(previewRafId);
            previewRafId = null;
            previewQueuedDy = null;
        }

        if (!didDrag) {
            liveEditOverrideRef.current = null;
            if (liveEditActiveRef) {
                liveEditActiveRef.current = false;
            }
            setCanvasCursor("grab");
            if (onContextMenu && document.hasFocus() && ev) {
                onContextMenu(ev.clientX, ev.clientY);
            }
            invalidate();
            return;
        }

        // 释放后按下的 contextmenu 一律吞掉（右键拖拽
        // 的收尾手势）；先注册再走后续早退分支，避免
        // 异常退出时弹出原生菜单。
        const suppressOnce = (evt: globalThis.MouseEvent) => {
            evt.preventDefault();
            evt.stopPropagation();
            window.removeEventListener("contextmenu", suppressOnce, true);
        };
        window.addEventListener("contextmenu", suppressOnce, true);
        setTimeout(() => {
            window.removeEventListener("contextmenu", suppressOnce, true);
        }, 0);

        const pvNow = paramViewRef.current;
        if (!pvNow || !rootTrackId) {
            if (liveEditActiveRef) {
                liveEditActiveRef.current = false;
            }
            setCanvasCursor("grab");
            invalidate();
            return;
        }

        // 全分辨率无损提交（与选区移动拖拽同一范式）：
        // 等待拖动开始时发起的全分辨率取数，用最终 dy 在
        // 真实曲线上重算变换，帧索引重建（含边缘淡化）后
        // 分块回写 —— 预览用的 pv 近似值绝不写回后端。
        // await 链路上的任何 IPC 失败都必须复位
        // liveEdit 状态（否则预览覆盖层永久冻结、曲线
        // 刷新被永久搁置）。
        dragSettled = true;
        try {
            // 先刷新各段的全分辨率原始值（拖动开始时已发起）
            const fullResCurves = await Promise.all(origCurvePromises);
            origValuesPerRange = fullResCurves.map((curve) => curve.values);
            // 提交基准 = 计划写入窗口（含边缘淡化扩展）的
            // 全分辨率数据；变换与预览同源（transformRightDragRange），
            // 因此写回结果与用户看到的预览逐帧一致。
            const commit = await fetchCommitBaseSource({
                trackId: rootTrackId,
                param: editParam,
                ranges: rightDragSpans,
                frameDelta: 0,
                edgeHalfSpanAt: edgeHalfSpanForRange,
                paramView: pvNow,
                withSentinel: isDynParam(editParam),
            });
            const pieces = buildMultiRangeEditPlan({
                ranges: rightDragSpans,
                valuesAt: (index) => transformRightDragRange(index, lastDy),
                sourceAt: commit.sourceAt,
                edgeHalfSpanAt: edgeHalfSpanForRange,
                isEditable: editParam === "pitch" ? editablePitchValue : undefined,
            });

            // 立即同步本地 paramView state（逐帧映射到 pv 的
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
            setParamView({ ...pvNow, edit: nextEdit });
            // 同上：不立刻撤下覆盖层（见 onParamCommitSucceeded）。
            onParamCommitSucceeded();

            // 分块回写：整次编辑（含所有段）只打一个撤销点，
            // 块之间让出事件循环，超长工程也不会卡死界面。
            // 上传失败同样复位 liveEdit（finally），不让覆盖层冻结。
            void (async () => {
                try {
                    await uploadFullResCurveSegments({
                        trackId: rootTrackId,
                        param: editParam,
                        segments: buildUploadSegments(editParam, pieces, commit.sentinelAt),
                    });
                } catch (err) {
                    console.error("[pianoRoll] right-drag upload failed", err);
                } finally {
                    if (liveEditActiveRef) {
                        liveEditActiveRef.current = false;
                    }
                    bumpRefreshToken();
                }
            })();
        } catch (err) {
            console.error("[pianoRoll] right-drag commit failed", err);
            liveEditOverrideRef.current = null;
            if (liveEditActiveRef) {
                liveEditActiveRef.current = false;
            }
        }

        setCanvasCursor("grab");
        invalidate();
    };

    window.addEventListener("contextmenu", suppressContextMenu, true);
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    window.addEventListener("pointercancel", onUp);
    setActivePointerGestureEnd(onUp);
    return true;

    return false;
}
