import { clamp } from "../../../timeline";
import { startPointerGesture, type PointerGesture } from "../pointerGesture";
import { isDynParam } from "../../paramRanges";
import {
    frameBoundLeftFromPointer,
    frameBoundRightFromPointer,
    frameRangeEnd,
    frameRangeFromFrames,
    normalizeSelection,
    snapFrame,
} from "../../paramSelection";
import {
    expandStrideSampledDense,
    restoreDynSentinelsFromParamView,
    uploadFullResCurve,
} from "../../selectionEditData";
import { PARAM_DRAG_DEAD_ZONE_PX, recordGestureParamSelectionStep } from "../pointerSupport";
import type { ParamSelection } from "../../paramSelection";
import type { ParamViewSegment } from "../../types";
import type { PointerEvent as ReactPointerEvent } from "react";
import type { PointerDeps } from "../pointerDeps";

/** `handleSelectPointerDown` 作用域里的局部量。 */
type SelectLocals = {
    sel: ParamSelection;
    stretchHit: { rangeIndex: number; edge: "left" | "right" };
};

/** 本分支所需的全部上下文。 */
type Deps = Pick<
    PointerDeps,
    | "applyDenseToLiveEdit"
    | "blendDenseEdges"
    | "bumpRefreshToken"
    | "clearActivePointerGestureEnd"
    | "createFineAdjustedPointerState"
    | "disposeFineAdjustedPointerState"
    | "edgeHalfSpanForIndices"
    | "edgeSmoothnessPercent"
    | "editParam"
    | "ensureLiveEditBase"
    | "getFineAdjustedPointerPosition"
    | "invalidate"
    | "liveEditActiveRef"
    | "liveEditOverrideRef"
    | "onParamCommitSucceeded"
    | "paramViewRef"
    | "pointerFrame"
    | "requestWaveformRepaint"
    | "rootTrackId"
    | "selectionRef"
    | "setActivePointerGestureEnd"
    | "setCanvasCursor"
    | "setParamView"
    | "updateSelectionUi"
> &
    SelectLocals;

/**
 * 拉伸选区边缘。
 *
 * @param deps hook 作用域 + 局部量的上下文。
 * @param e 触发本次手势的指针事件。
 * @returns `true` = 已消费、调用方应停止；`false` = 未消费，调用方继续。
 */
export function handleStretchSelectionEdge(
    deps: Deps,
    e: ReactPointerEvent<HTMLCanvasElement>,
): boolean {
    const {
        applyDenseToLiveEdit,
        blendDenseEdges,
        bumpRefreshToken,
        clearActivePointerGestureEnd,
        createFineAdjustedPointerState,
        disposeFineAdjustedPointerState,
        edgeHalfSpanForIndices,
        edgeSmoothnessPercent,
        editParam,
        ensureLiveEditBase,
        getFineAdjustedPointerPosition,
        invalidate,
        liveEditActiveRef,
        liveEditOverrideRef,
        onParamCommitSucceeded,
        paramViewRef,
        pointerFrame,
        requestWaveformRepaint,
        rootTrackId,
        sel,
        selectionRef,
        setActivePointerGestureEnd,
        setCanvasCursor,
        setParamView,
        stretchHit,
        updateSelectionUi,
    } = deps;

    // 被拉伸那一段的两条**数据边界**（整数帧）：左边界 = 首帧，
    // 右边界 = 末帧 + 1（半开区间右端）。拉伸就是把其中一条跟着手
    // 移走，另一条不动。指针位置按抓住的是哪条边界套对应规则
    // （见 `frameBoundLeftFromPointer` / `frameBoundRightFromPointer`）。
    const aBound = sel[stretchHit.rangeIndex].startFrame;
    const bBound = frameRangeEnd(sel[stretchHit.rangeIndex]);
    const edgeKind = stretchHit.edge;
    // pointerdown 时的选区快照：拉伸期间的选区改写都以它为基底
    // 重建（只替换被拉伸的那一段），因此归一化合并不会造成
    // 「越拉越偏」的下标漂移。
    const stretchBaseSelection = sel;
    // 本手势同时改曲线与选区：收尾时把这对选区快照登记到该手势
    // 打出的那个历史步骤上（后端持有，撤销恢复 `before`、重做恢复
    // `after`，见 recordGestureParamSelectionStep）。
    const selectionBeforeStretch = stretchBaseSelection.map((range) => ({
        ...range,
    }));
    let stretchStartBound = aBound;
    let stretchEndBound = bBound;

    const pv = paramViewRef.current;
    if (!pv || pv.edit.length === 0) return true;
    const stride = Math.max(1, pv.stride);
    // 源窗口 = 选区内**恰好那些帧**（`[aBound, bBound)`），逐帧取下标；
    // 不再有"拍 → 帧两端取整不对称"带来的多一帧。
    const oldStartBound = aBound;
    const oldEndBound = bBound;
    const oldStartIdx = clamp(
        Math.round((oldStartBound - pv.startFrame) / stride),
        0,
        pv.edit.length - 1,
    );
    const oldLastIdx = clamp(
        Math.round((oldEndBound - 1 - pv.startFrame) / stride),
        oldStartIdx,
        pv.edit.length - 1,
    );
    const oldValues = pv.edit.slice(oldStartIdx, oldLastIdx + 1);
    if (oldValues.length <= 0) return true;

    const pid = e.pointerId;
    (e.currentTarget as HTMLCanvasElement).setPointerCapture(pid);
    setCanvasCursor("ew-resize");
    ensureLiveEditBase(pv);
    if (liveEditActiveRef) liveEditActiveRef.current = true;
    // 点击死区（见 PARAM_DRAG_DEAD_ZONE_PX）：按住拉伸修饰键在边缘
    // 点一下同样不该回写等值数据、不该多打一个撤销点。
    const startClientX = e.clientX;
    const startClientY = e.clientY;
    let didDrag = false;
    const finePointerState = createFineAdjustedPointerState(
        e.nativeEvent,
        e.currentTarget as HTMLCanvasElement,
    );

    const buildDense = (
        pvNow: ParamViewSegment,
        nextStartBoundRaw: number,
        nextEndBoundRaw: number,
    ) => {
        // 数据边界归一为整数帧，并把归一后的值返回给调用方复用 ——
        // 于是"预览画出来的选区"与"提交写回去的选区"必然是同一组
        // 数字，不会差半帧。
        const nextStartBound = Math.max(0, snapFrame(nextStartBoundRaw));
        const nextEndBound = Math.max(nextStartBound, snapFrame(nextEndBoundRaw));
        const nextStartIdx = clamp(
            Math.round((nextStartBound - pvNow.startFrame) / stride),
            0,
            pvNow.edit.length - 1,
        );
        const nextLastIdx = clamp(
            Math.round((nextEndBound - 1 - pvNow.startFrame) / stride),
            nextStartIdx,
            pvNow.edit.length - 1,
        );
        const nextLen = nextLastIdx - nextStartIdx + 1;
        if (nextLen <= 0) return null;

        // 平滑度 → 毫秒定标的过渡带半宽（dense 索引）
        const edgeHalfSpanIdx = edgeHalfSpanForIndices(nextLen, stride);
        const extraEdgeFrames = Math.ceil(edgeHalfSpanIdx) * stride;
        const overallMinFrame = Math.max(
            0,
            Math.min(oldStartBound, nextStartBound) - extraEdgeFrames,
        );
        const overallMaxFrame = Math.max(oldEndBound, nextEndBound) + extraEdgeFrames;
        const overallLen = Math.floor((overallMaxFrame - overallMinFrame) / stride) + 1;
        const dense = new Array<number>(overallLen);
        for (let i = 0; i < overallLen; i += 1) {
            const frame = overallMinFrame + i * stride;
            const idx = Math.round((frame - pvNow.startFrame) / stride);
            dense[i] = idx >= 0 && idx < pvNow.edit.length ? pvNow.edit[idx] : 0;
        }
        const denseBefore = dense.slice();

        const newValues = new Array<number>(nextLen);
        for (let i = 0; i < nextLen; i += 1) {
            const t = nextLen > 1 ? i / (nextLen - 1) : 0;
            const srcF = t * (oldValues.length - 1);
            const lo = Math.floor(srcF);
            const hi = Math.min(lo + 1, oldValues.length - 1);
            const frac = srcF - lo;
            const loVal = Number(oldValues[lo] ?? 0);
            const hiVal = Number(oldValues[hi] ?? 0);
            if (editParam === "pitch" && loVal === 0 && hiVal === 0) {
                newValues[i] = 0;
            } else {
                newValues[i] = loVal + (hiVal - loVal) * frac;
            }
        }

        for (let i = 0; i < nextLen; i += 1) {
            const frame = nextStartBound + i * stride;
            const dIdx = Math.round((frame - overallMinFrame) / stride);
            if (dIdx >= 0 && dIdx < dense.length) {
                dense[dIdx] = newValues[i];
            }
        }

        const sampleOutsideValue = (srcFrame: number, fallback: number) => {
            const srcIdx = Math.round((srcFrame - pvNow.startFrame) / stride);
            if (srcIdx >= 0 && srcIdx < pvNow.edit.length) {
                return pvNow.edit[srcIdx];
            }
            return fallback;
        };

        const maxOutsideWindow = Math.max(1, Math.round(oldValues.length * 0.2));
        const smoothRatio = clamp(Number(edgeSmoothnessPercent) || 0, 0, 100) / 100;
        const outsideWindowLen = Math.max(1, Math.round(maxOutsideWindow * smoothRatio));

        // 缩短时，用原选区内侧一小段值回填被腾空区域。
        // smoothness=0 时 outsideWindowLen=1，相当于边缘值沿边界内侧延展。
        if (nextStartBound > oldStartBound) {
            const fillLen = Math.floor((nextStartBound - oldStartBound) / stride);
            for (let i = 0; i < fillLen; i += 1) {
                const targetFrame = oldStartBound + i * stride;
                const targetIdx = Math.round((targetFrame - overallMinFrame) / stride);
                const srcWindowPos =
                    fillLen > 1 ? Math.round((i / (fillLen - 1)) * (outsideWindowLen - 1)) : 0;
                const srcFrame = oldStartBound + srcWindowPos * stride;
                if (targetIdx >= 0 && targetIdx < dense.length) {
                    dense[targetIdx] = sampleOutsideValue(srcFrame, dense[targetIdx]);
                }
            }
        }
        if (nextEndBound < oldEndBound) {
            const fillLen = Math.floor((oldEndBound - nextEndBound) / stride);
            for (let i = 0; i < fillLen; i += 1) {
                const targetFrame = nextEndBound + (i + 1) * stride;
                const targetIdx = Math.round((targetFrame - overallMinFrame) / stride);
                const srcWindowPos =
                    fillLen > 1 ? Math.round((i / (fillLen - 1)) * (outsideWindowLen - 1)) : 0;
                // 源取**末帧往内**（`oldEndBound - 1`）：源窗口就是选区内
                // 那些帧，右切点本身不属于选区。
                const srcFrame = oldEndBound - 1 - srcWindowPos * stride;
                if (targetIdx >= 0 && targetIdx < dense.length) {
                    dense[targetIdx] = sampleOutsideValue(srcFrame, dense[targetIdx]);
                }
            }
        }

        const movedStartDenseIdx = Math.round((nextStartBound - overallMinFrame) / stride);
        blendDenseEdges(dense, denseBefore, movedStartDenseIdx, nextLen, edgeHalfSpanIdx);

        return {
            dense,
            overallMinFrame,
            overallMaxFrame,
            nextStartBound,
            nextEndBound,
        };
    };

    /** 拉伸后的选区至少保留一个采样点（`stride` 帧）。 */
    const minFrameSpan = Math.max(1, stride);

    // 预览重算 rAF 合帧：dense 重建 + 整份 live 拷贝 +
    // React setState 都是 O(n)，pointermove 在高刷鼠标上
    // 可达数百 Hz，逐事件执行长选区会卡顿；与左键选区
    // 拖动路径同模式，一帧至多重算一次。
    let previewRafId: number | null = null;
    let queuedCursorFrame: number | null = null;
    const runPreviewStep = () => {
        const cursorFrame = queuedCursorFrame;
        if (cursorFrame == null) return;
        queuedCursorFrame = null;
        const pvNow = paramViewRef.current;
        if (!pvNow) return;
        // 指针 → **数据边界**：按抓住的是哪条边界选规则（与框选同一套，
        // 见 `paramSelection` 的两条规则）。于是把左边界一路拖到工程
        // 最左端，起点就能落到**第 0 帧**。
        const cursorBound =
            edgeKind === "left"
                ? frameBoundLeftFromPointer(cursorFrame)
                : frameBoundRightFromPointer(cursorFrame);
        // 被抓住的那条边界跟着手走，另一条不动；两条边界之间至少留
        // 一个采样点（`minFrameSpan`）。
        //
        // 【起点为什么夹在 >= 0（而框选不夹）】拉伸的窗口**就是**数据
        // 窗口：重采样出来的值必须铺满它。工程起点之前没有数据，把起点
        // 放到负帧会让"选区说 [-3, 5)、实际只写了 [0, 5)"两下对不上。
        // 框选的选区是**视觉范围**，允许越过工程起点（数据侧再由
        // `selectionToFrameRanges` 截断到 0），所以那边不夹。
        const nextStartBound =
            edgeKind === "left" ? clamp(cursorBound, 0, bBound - minFrameSpan) : aBound;
        const nextEndBound =
            edgeKind === "right" ? Math.max(aBound + minFrameSpan, cursorBound) : bBound;
        const built = buildDense(pvNow, nextStartBound, nextEndBound);
        if (!built) return;
        // 用 buildDense **归一后**的边界：选区与曲线因此逐帧一致。
        stretchStartBound = built.nextStartBound;
        stretchEndBound = built.nextEndBound;
        applyDenseToLiveEdit(
            pvNow,
            built.overallMinFrame,
            built.dense,
            built.overallMinFrame,
            built.overallMaxFrame,
            "draw",
        );
        selectionRef.current = normalizeSelection(
            stretchBaseSelection.map((range, index) =>
                index === stretchHit.rangeIndex
                    ? frameRangeFromFrames(
                          built.nextStartBound,
                          built.nextEndBound - built.nextStartBound,
                      )
                    : range,
            ),
        );
        updateSelectionUi(selectionRef.current);
        invalidate();
        // live 覆盖只改 ref：音量/动态拖拽时波形面需显式重绘。
        requestWaveformRepaint();
    };

    /** 手势句柄（onMove 需要主动收尾，见 pan 分支的同名字段）。 */
    let gesture: PointerGesture | null = null;
    const onMove = (ev: globalThis.PointerEvent) => {
        if ((ev.buttons & 1) !== 1) {
            gesture?.end("up", ev);
            return;
        }
        // 点击死区：未越过阈值前不预览、不更新（画面上没有任何变化），
        // 松手时按"点击"处理 —— 不回写数据、不打撤销点。拉伸路径的
        // 收尾同样会整段回写并打检查点，没有位移时那是纯粹的噪音。
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
        queuedCursorFrame = pointerFrame(adjusted.clientX);
        if (previewRafId == null) {
            previewRafId = requestAnimationFrame(() => {
                previewRafId = null;
                runPreviewStep();
            });
        }
    };

    const onEnd = () => {
        disposeFineAdjustedPointerState(finePointerState);
        // 取消挂起的预览帧：松手后的提交路径负责重建。
        if (previewRafId != null) {
            cancelAnimationFrame(previewRafId);
            previewRafId = null;
        }
        // 提交前先同步消费挂起的游标：边界标量（stretchStartBound /
        // stretchEndBound）只在 runPreviewStep 里更新，若松手与越过
        // 死区的第一次 move 落在同一帧，取消 rAF 后标量仍是初始值，
        // buildDense 会原样复现输入数据还多打一个"毫无变化"的撤销点
        // （与自由绘制路径提交前先 flush 队列同一处理）。
        runPreviewStep();
        queuedCursorFrame = null;

        // 单击（未越过点击死区）：这次手势没有改动任何值 —— 丢弃
        // live 覆盖层并复位 active 标记后直接收尾。走提交路径会把
        // 与原始数据逐帧相同的变换结果整段写回，并多打一个"按下毫无
        // 变化"的撤销点。
        if (!didDrag) {
            liveEditOverrideRef.current = null;
            if (liveEditActiveRef) {
                liveEditActiveRef.current = false;
            }
            setCanvasCursor("default");
            invalidate();
            return;
        }

        const pvNow = paramViewRef.current;
        if (!pvNow || !rootTrackId) {
            // 早退同样必须复位 liveEdit（否则曲线刷新被永久搁置，
            // 与 !didDrag 路径的处理一致）。
            if (liveEditActiveRef) liveEditActiveRef.current = false;
            setCanvasCursor("default");
            return;
        }
        // 被拉伸那一段的最终边界（多选区只改这一段；帧口径）
        const nextStartBound = stretchStartBound;
        const nextEndBound = stretchEndBound;
        const built = buildDense(pvNow, nextStartBound, nextEndBound);
        if (!built) {
            setCanvasCursor("default");
            return;
        }

        // 提交前把选区钉到**本次提交用的那组边界**上：预览的最后一帧
        // 可能被 rAF 取消，若沿用 `selectionRef.current`，撤销恢复的
        // 选区就可能与真正写回去的数据差一帧。
        selectionRef.current = normalizeSelection(
            stretchBaseSelection.map((range, index) =>
                index === stretchHit.rangeIndex
                    ? frameRangeFromFrames(
                          built.nextStartBound,
                          built.nextEndBound - built.nextStartBound,
                      )
                    : range,
            ),
        );
        updateSelectionUi(selectionRef.current);

        const nextEdit = pvNow.edit.slice();
        for (let i = 0; i < built.dense.length; i += 1) {
            const frame = built.overallMinFrame + i * stride;
            const idx = Math.round((frame - pvNow.startFrame) / stride);
            if (idx >= 0 && idx < nextEdit.length) {
                nextEdit[idx] = built.dense[i];
            }
        }
        setParamView({ ...pvNow, edit: nextEdit });
        // 同上：不立刻撤下覆盖层（见 onParamCommitSucceeded）。
        onParamCommitSucceeded();

        // 全分辨率无损提交（与右拖 / 选区拖拽同一范式）：
        // buildDense 产生的是 pv 步距采样（dense[k] ↔
        // overallMinFrame + k×stride）；按渲染同款线性插值
        // 展开为逐帧值后分块回写 —— 绝不把 stride 间隔采样
        // 当连续帧写入（旧实现在 stride>1 时时间压缩 +
        // 覆盖未选帧，见 selectionEditData.expandStrideSampledDense）。
        const expanded = expandStrideSampledDense(built.dense, stride);
        // dyn 哨兵保留：拉伸是"读-变换-写回"，未画帧必须以哨兵
        // 写回（数据源 = pv 自带位图，见
        // restoreDynSentinelsFromParamView），否则被物化成显式
        // 基线，日后基线重分析时响度静默漂移。
        const expandedForUpload = isDynParam(editParam)
            ? restoreDynSentinelsFromParamView(expanded.slice(), built.overallMinFrame, pvNow)
            : expanded;
        // 拉伸后的选区（本轮手势的最终形态）：与拉伸前的快照
        // 一起登记为该历史步骤的选区。
        const selectionAfterStretch = selectionRef.current
            ? selectionRef.current.map((range) => ({ ...range }))
            : null;
        void (async () => {
            let committed = false;
            try {
                await uploadFullResCurve({
                    trackId: rootTrackId,
                    param: editParam,
                    startFrame: built.overallMinFrame,
                    values: expandedForUpload,
                });
                committed = true;
            } catch (err) {
                console.error("[pianoRoll] stretch-edge commit failed", err);
                // IPC 失败必须撤下覆盖层，否则冻结的预览值会一直
                // 盖在真实曲线上（与右拖 / 选区拖拽 / 形变同规）。
                liveEditOverrideRef.current = null;
            } finally {
                if (liveEditActiveRef) {
                    liveEditActiveRef.current = false;
                }
                bumpRefreshToken();
            }
            if (!committed) return;

            // 回写成功（首块已打检查点、位置已指向新步）之后，把本次
            // 拉伸的选区变化登记到**那一步**上：撤销恢复拉伸前、重做
            // 恢复拉伸后的选区。快照由后端持有，前端不做任何"我这是
            // 第几步"的推断（见 recordGestureParamSelectionStep）。
            await recordGestureParamSelectionStep({
                before: selectionBeforeStretch,
                after: selectionAfterStretch,
            });
        })();
        setCanvasCursor("default");
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
