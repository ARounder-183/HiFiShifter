import { secToViewportPx } from "../../../renderKernel/timelineAxis.js";
import { clamp } from "../../../timeline";
import { isDynParam } from "../../paramRanges";
import { snapFrame } from "../../paramSelection";
import {
    expandStrideSampledDense,
    restoreDynSentinelsFromParamView,
    uploadFullResCurveSegments,
} from "../../selectionEditData";
import type { ParamMorphOverlay, ParamViewSegment } from "../../types";
import type { PointerEvent as ReactPointerEvent } from "react";
import type { PointerDeps } from "../pointerDeps";

/** `handleSelectPointerDown` 作用域里的局部量。 */
type SelectLocals = {
    canvas: HTMLCanvasElement;
    existingMorph: ParamMorphOverlay[];
    pvForMorph: ParamViewSegment;
};

/** 本分支所需的全部上下文。 */
type Deps = Pick<
    PointerDeps,
    | "applyMorphOverlayPreview"
    | "axisFromRefs"
    | "buildMorphDense"
    | "bumpRefreshToken"
    | "clearActivePointerGestureEnd"
    | "createFineAdjustedPointerState"
    | "disposeFineAdjustedPointerState"
    | "editParam"
    | "ensureLiveEditBase"
    | "getFineAdjustedPointerPosition"
    | "liveEditActiveRef"
    | "liveEditOverrideRef"
    | "morphDragRef"
    | "morphModifierDownRef"
    | "morphOverlayRef"
    | "onParamCommitSucceeded"
    | "onParamValuePreviewChange"
    | "paramValuePopupEnabled"
    | "paramViewRef"
    | "pointerFrame"
    | "pointerValue"
    | "rootTrackId"
    | "setActivePointerGestureEnd"
    | "setCanvasCursor"
    | "setMorphOverlay"
    | "setParamView"
    | "valueToY"
    | "viewSizeRef"
> &
    SelectLocals;

/**
 * 处理形变覆盖层控制点上的指针按下。
 *
 * @param deps hook 作用域 + 局部量的上下文。
 * @param e 触发本次手势的指针事件。
 * @returns `true` = 已消费、调用方应停止；`false` = 未消费，调用方继续。
 */
export function handleMorphPointerDown(
    deps: Deps,
    e: ReactPointerEvent<HTMLCanvasElement>,
): boolean {
    const {
        applyMorphOverlayPreview,
        axisFromRefs,
        buildMorphDense,
        bumpRefreshToken,
        canvas,
        clearActivePointerGestureEnd,
        createFineAdjustedPointerState,
        disposeFineAdjustedPointerState,
        editParam,
        ensureLiveEditBase,
        existingMorph,
        getFineAdjustedPointerPosition,
        liveEditActiveRef,
        liveEditOverrideRef,
        morphDragRef,
        morphModifierDownRef,
        morphOverlayRef,
        onParamCommitSucceeded,
        onParamValuePreviewChange,
        paramValuePopupEnabled,
        paramViewRef,
        pointerFrame,
        pointerValue,
        pvForMorph,
        rootTrackId,
        setActivePointerGestureEnd,
        setCanvasCursor,
        setMorphOverlay,
        setParamView,
        valueToY,
        viewSizeRef,
    } = deps;

    const rect = canvas.getBoundingClientRect();
    const h = Math.max(1, rect.height || viewSizeRef.current.h || 1);
    const fp = Math.max(1e-6, pvForMorph.framePeriodMs);
    const stride = Math.max(1, pvForMorph.stride);
    // 多选区：控制线每段一条，命中取距离最近的一个控制点
    let hit: {
        overlayIndex: number;
        point: ParamMorphOverlay["points"][number];
    } | null = null;
    let hitDistance = Number.POSITIVE_INFINITY;
    for (let i = 0; i < existingMorph.length; i += 1) {
        for (const point of existingMorph[i].points) {
            const sec = (point.frame * fp) / 1000;
            const x = secToViewportPx(axisFromRefs(), sec);
            const mapped = editParam === "pitch" ? point.value + 0.5 : point.value;
            const y = valueToY(editParam, mapped, h);
            const dx = Math.abs(e.clientX - rect.left - x);
            const dy = Math.abs(e.clientY - rect.top - y);
            if (dx > 8 || dy > 8) continue;
            const distance = dx + dy;
            if (distance < hitDistance) {
                hitDistance = distance;
                hit = { overlayIndex: i, point };
            }
        }
    }

    if (hit && e.button === 0) {
        e.preventDefault();
        morphDragRef.current = {
            pointerId: e.pointerId,
            overlayIndex: hit.overlayIndex,
            pointKind: hit.point.kind,
        };
        setCanvasCursor("grabbing");
        ensureLiveEditBase(pvForMorph);
        if (liveEditActiveRef) liveEditActiveRef.current = true;
        (e.currentTarget as HTMLCanvasElement).setPointerCapture(e.pointerId);
        const finePointerState = createFineAdjustedPointerState(
            e.nativeEvent,
            e.currentTarget as HTMLCanvasElement,
        );

        if (paramValuePopupEnabled) {
            onParamValuePreviewChange?.({
                clientX: e.clientX,
                clientY: e.clientY,
                value: hit.point.value,
            });
        }

        // 预览重算合帧到下一渲染帧（与拉伸 / 选区拖拽的
        // schedulePreview 同一模式）：pointermove 在高刷鼠标 / pen
        // 上可达数百 Hz，而 setMorphOverlay（整面板重渲）与
        // applyMorphOverlayPreview（每条 overlay 一次 buildMorphDense）
        // 逐事件执行纯属浪费。ref 同步保持逐事件（下一次 move 与
        // onUp 都要读到最新控制点），只有 React 状态 / live 预览 /
        // 浮窗延后到 rAF。
        let previewRafId: number | null = null;
        let pendingOverlays: ParamMorphOverlay[] | null = null;
        let pendingPopup: {
            clientX: number;
            clientY: number;
            value: number;
        } | null = null;
        const flushPreview = () => {
            previewRafId = null;
            if (pendingOverlays != null) {
                const overlays = pendingOverlays;
                pendingOverlays = null;
                setMorphOverlay(overlays);
                applyMorphOverlayPreview(overlays);
            }
            if (pendingPopup != null) {
                onParamValuePreviewChange?.(pendingPopup);
                pendingPopup = null;
            }
        };

        const onMove = (ev: globalThis.PointerEvent) => {
            if ((ev.buttons & 1) !== 1) {
                onUp();
                return;
            }
            // 只接受本指针的 move：第二指针（掌压触摸等）的
            // buttons 恒为 1，不校验 pointerId 会驱动本次手势，
            // 松手还会把掌压驱动的曲线提交到后端。
            if (ev.pointerId !== e.pointerId) return;
            const drag = morphDragRef.current;
            const overlayNow = morphOverlayRef.current;
            const pvNow = paramViewRef.current;
            if (!drag || drag.pointerId !== e.pointerId || !overlayNow || !pvNow) {
                return;
            }
            const target = overlayNow[drag.overlayIndex];
            if (!target) return;
            const adjusted = getFineAdjustedPointerPosition(finePointerState, ev);

            const nextPoints = target.points.map((pt) => {
                if (pt.kind !== drag.pointKind) return pt;
                const newValue = pointerValue(adjusted.clientY);
                if (pt.kind === "left" || pt.kind === "right") {
                    return { ...pt, value: newValue };
                }
                const rawFrame = Math.max(0, snapFrame(pointerFrame(adjusted.clientX)));
                const clampedFrame = clamp(
                    rawFrame,
                    target.selectionStartFrame,
                    target.selectionEndFrame,
                );
                return {
                    ...pt,
                    frame: clampedFrame,
                    value: newValue,
                };
            });

            // 只替换被拖动的那条控制线（其余段的控制线保持不动）
            const nextOverlays: ParamMorphOverlay[] = overlayNow.map((overlay, index) =>
                index === drag.overlayIndex ? { ...overlay, points: nextPoints } : overlay,
            );
            morphOverlayRef.current = nextOverlays;
            pendingOverlays = nextOverlays;

            if (paramValuePopupEnabled) {
                const movedPoint = nextPoints.find((pt) => pt.kind === drag.pointKind);
                pendingPopup = {
                    clientX: ev.clientX,
                    clientY: ev.clientY,
                    value: movedPoint?.value ?? pointerValue(adjusted.clientY),
                };
            }

            if (previewRafId == null) {
                previewRafId = requestAnimationFrame(flushPreview);
            }
        };

        const onUp = () => {
            const drag = morphDragRef.current;
            const overlayNow = morphOverlayRef.current;
            const pvNow = paramViewRef.current;
            morphDragRef.current = null;
            window.removeEventListener("pointermove", onMove);
            window.removeEventListener("pointerup", onUp);
            window.removeEventListener("pointercancel", onUp);
            disposeFineAdjustedPointerState(finePointerState);
            clearActivePointerGestureEnd(onUp);
            // 最后一次 move 的预览若还在 rAF 队列里，必须在提交前
            // 同步冲刷：提交读 overlayNow / live 层，漏冲刷会把
            // 末段位移丢掉（与选区拖拽"松手即取消"不同——这里
            // 提交数据就来自挂起的预览状态）。
            if (previewRafId != null) {
                cancelAnimationFrame(previewRafId);
                flushPreview();
            }

            if (!drag || !overlayNow || !pvNow || !rootTrackId) {
                // 早退同样必须复位 liveEdit（否则曲线刷新被永久搁置，
                // 与右键拖拽路径的处理一致）。
                if (liveEditActiveRef) liveEditActiveRef.current = false;
                setCanvasCursor("default");
                return;
            }

            // 每段各自的控制线 → 逐段 dense；整次提交一个撤销点。
            // 回写必须逐帧：buildMorphDense 产出的是 pv 步距采样
            // （dense[k] ↔ startFrame + k×stride），把 stride 间隔
            // 采样当连续帧写回会造成时间压缩 + 覆盖未选帧
            // （stride=1 时原样返回，零开销）。
            const packedPerOverlay = overlayNow.map((overlay) => {
                const packed = buildMorphDense(overlay, stride);
                const values = expandStrideSampledDense(packed.dense, stride);
                return {
                    startFrame: packed.startFrame,
                    // dyn 哨兵保留（与拉伸 / 选区拖拽同一契约）：
                    // 未画帧写回哨兵，不物化成显式基线。
                    values: isDynParam(editParam)
                        ? restoreDynSentinelsFromParamView(values.slice(), packed.startFrame, pvNow)
                        : values,
                };
            });

            const nextEdit = pvNow.edit.slice();
            const pvStepUp = Math.max(1, Math.floor(pvNow.stride));
            for (const packed of packedPerOverlay) {
                for (let i = 0; i < packed.values.length; i += 1) {
                    const globalIdx = Math.round(
                        (packed.startFrame + i - pvNow.startFrame) / pvStepUp,
                    );
                    if (globalIdx >= 0 && globalIdx < nextEdit.length) {
                        nextEdit[globalIdx] = packed.values[i];
                    }
                }
            }
            setParamView({ ...pvNow, edit: nextEdit });

            void (async () => {
                try {
                    await uploadFullResCurveSegments({
                        trackId: rootTrackId,
                        param: editParam,
                        segments: packedPerOverlay,
                    });
                    // 提交成功：保留覆盖层让波形继续显示提交值，等
                    // 提交之后取的响度快照到达再撤下（消除松手闪屏）。
                    onParamCommitSucceeded();
                } catch (err) {
                    console.error("[pianoRoll] morph upload failed", err);
                    // IPC 失败必须撤下覆盖层，否则冻结的预览值会一直
                    // 盖在真实曲线上。
                    liveEditOverrideRef.current = null;
                } finally {
                    if (liveEditActiveRef) liveEditActiveRef.current = false;
                    bumpRefreshToken();
                }
            })();

            if (morphModifierDownRef.current) {
                // 保持调整点位置不变；baselineValues 是进入形变模式时一次性捕获的，
                // 每次拖拽提交都在同一基准线上施加"总偏移"，不重置。
                // 只有松开修饰键再重新按下时，才会调用 buildMorphOverlaysFromSelection 重置。
            } else {
                setMorphOverlay(null);
            }
            setCanvasCursor("default");
        };

        window.addEventListener("pointermove", onMove);
        window.addEventListener("pointerup", onUp);
        window.addEventListener("pointercancel", onUp);
        setActivePointerGestureEnd(onUp);
        return true;
    }

    return false;
}
