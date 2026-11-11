/**
 * 参数块渲染规格（PianoRollParamBlockSpec）的唯一派生点。
 *
 * 【为什么是一个 hook 而不是纯函数】输入里有 9 个 ref（拖拽快照 / 伸缩手势 /
 * 视觉事务 / 值缓存 …）。它们由命令式路径写入，**不触发渲染** —— 所以必须由
 * paramBlockStructureTick 这个显式闸门把 memo 打开一次。把这段留在这里，而不是
 * 散回面板模型，是为了让「块几何从哪里来」只有一个答案。
 *
 * 【优先级顺序（不可调整）】
 *   1. 视觉事务（且 epoch 仍然有效）
 *   2. 边缘伸缩 / 新建预览
 *   3. 拖拽快照（且 key / 手势 / 提交三者允许）
 *   4. canonical（由 canonicalBlocks 现算）
 * 调换 2 与 3 会让「松手才跳到位、不跟手」复发。
 *
 * 【为什么 framePeriodMs / curveOwned 传值而不是传 ref】它们在渲染期被读一次，
 * 与 memo 求值同一时刻；传 ref 只会让依赖数组无法表达「这两个字段变了」。
 */
import { useLayoutEffect, useMemo, type MutableRefObject } from "react";
import {
    averageParamValueInSpan,
    projectParamBlockCreatePreviewFromBase,
    shouldUseParamBlockDragSnapshotForRender,
    type ParamBlock,
} from "../paramBlocks";
import { logicalBlockId } from "../paramBlockTypes";
import {
    isParamBlockVisualEpochCurrent,
    shouldUseParamBlockVisualTransaction,
    type ParamBlockEpoch,
    type ParamBlockVisualTransaction,
} from "../paramBlockVisualTransaction";
import { buildParamBlockRenderSpec, type ParamBlockSpecColors } from "../paramBlockRenderSpec";
import type { ParamViewSegment } from "../types";
import type { PianoRollParamBlockSpec } from "../kernel/host/pianoRollKernelData";
import type { ParamBlockRenderBlock } from "../paramBlockRenderSpec";
import type { ParamBlockDragCommit, ParamBlockDragSnapshot } from "../paramBlockDrag";
import type { ParamBlockGesturePlan } from "../paramBlockGesture";
import type { ParamName } from "../types";

/** 拖拽快照 → 渲染规格（面板的 `paramBlockSpecForDragSnapshot`）。 */
type BuildDragSnapshotSpec = (snapshot: ParamBlockDragSnapshot) => PianoRollParamBlockSpec | null;

/** 一次渲染需要读到的全部输入。 */
export interface ParamBlockRenderSpecInput {
    /** 块层是否应当绘制（`isParamBlockLayerVisible` 的派生结果）。 */
    readonly layerVisible: boolean;
    /** 显式打破 memo 的闸门（ref 写入不触发渲染，见模块头）。 */
    readonly structureTick: number;
    readonly epoch: ParamBlockEpoch;
    readonly visualTransactionRef: MutableRefObject<ParamBlockVisualTransaction | null>;
    readonly resizeRef: MutableRefObject<ParamBlockResizePreview | null>;
    readonly dragSnapshotRef: MutableRefObject<ParamBlockDragSnapshot | null>;
    readonly activeGestureTokenRef: MutableRefObject<number | null>;
    readonly dragCommitRef: MutableRefObject<ParamBlockDragCommit | null>;
    readonly valueCacheRef: MutableRefObject<Map<string, { readonly value: number }>>;
    /** 面板的 `latestParamBlockSpecRef`：由本 hook 在 layout effect 里写入。 */
    readonly latestSpecRef: MutableRefObject<PianoRollParamBlockSpec | null>;
    readonly canonicalBlocks: readonly ParamBlock[];
    readonly paramView: ParamViewSegment | null;
    readonly selectedIds: ReadonlySet<string>;
    readonly editParam: ParamName;
    readonly colors: ParamBlockSpecColors;
    readonly bounds: { readonly min: number; readonly max: number };
    readonly valueToY: (param: ParamName, value: number, heightPx: number) => number;
    readonly framePeriodMs: number;
    readonly curveOwned: boolean;
    readonly buildDragSnapshotSpec: BuildDragSnapshotSpec;
}
/**
 * 边缘伸缩 / 新建块的手势状态（面板的 `paramBlockResizeRef`）。
 *
 * 【为什么类型定义在这里】渲染规格的派生要读它；把它留在面板里会让「谁能读
 * 手势状态」这件事只能靠 import 方向表达。面板反过来 import 这个类型。
 */
export interface ParamBlockResizePreview {
    readonly blockId: string;
    readonly edge: "left" | "right";
    readonly pointerClientX: number;
    readonly startSec: number;
    readonly endSec: number;
    readonly neighborStartSec: number | null;
    readonly neighborEndSec: number | null;
    readonly resizeCandidates: readonly ParamBlock[];
    /** 手势起点的完整块几何；每帧从它重新投影，已删除块不会永久丢失。 */
    readonly baseBlockSpans: readonly {
        readonly id: string;
        readonly startSec: number;
        readonly endSec: number;
        readonly selected?: boolean;
    }[];
    /** 新建块预览的冻结块集；提交后 React 派生变化不能污染预览。 */
    readonly createBaseBlocks?: PianoRollParamBlockSpec["blocks"];
    readonly createdBlockId?: string;
    /** 手势起点 epoch；外部历史 / 布局变化后预览立即失效。 */
    readonly epoch?: ParamBlockEpoch;
    /** 手势起点的帧周期；提交与 minBlockSec 都按它折算，不受回包影响。 */
    readonly framePeriodMsBasis?: number;
    targetSec: number;
    targetFrame: number;
    /** 本帧完整伸缩计划；预览 / 宿主刷新 / 提交共用。 */
    resizePlan: ParamBlockGesturePlan | null;
    /** 手势起点左右块的代表值/高度；仅用于锁定预览，不提交。 */
    readonly displayCenterValue?: number | null;
    readonly displayHeightPx?: number | null;
    /** 新建块的实时纵向值（只用于 `__create__` 预览）。 */
    readonly value?: number | null;
}
/** 见模块头：参数块渲染规格的唯一派生点。 */
export function useParamBlockRenderSpec(
    args: ParamBlockRenderSpecInput,
): PianoRollParamBlockSpec | null {
    const {
        layerVisible,
        structureTick,
        epoch,
        visualTransactionRef,
        resizeRef,
        dragSnapshotRef,
        activeGestureTokenRef,
        dragCommitRef,
        valueCacheRef,
        latestSpecRef,
        canonicalBlocks,
        paramView,
        selectedIds,
        editParam,
        colors,
        bounds,
        valueToY,
        framePeriodMs,
        curveOwned,
        buildDragSnapshotSpec,
    } = args;
    const spec = useMemo<PianoRollParamBlockSpec | null>(() => {
        // Visibility applies before every gesture/commit preview branch too.
        if (!layerVisible) return null;
        // 拖动 ref 由命令式路径更新；tick 是让下一次 React memo 读取新 ref 的显式闸门。
        void structureTick;
        const activeTransaction = visualTransactionRef.current;
        if (
            activeTransaction !== null &&
            shouldUseParamBlockVisualTransaction(activeTransaction, epoch) &&
            isParamBlockVisualEpochCurrent({
                epoch: activeTransaction.epoch,
                current: epoch,
            })
        ) {
            return activeTransaction.spec;
        }
        // 不在渲染期清 ref：失效清理统一放在 layout effect（早于 paint），
        // 这里只回退到 canonical，绝不让旧事务再覆盖一帧。
        /*
         * 边缘伸缩 / 新建块的预览必须优先于拖拽快照。以前这里先检查
         * `dragSnapshotRef`，而它的 key 与 `paramView.key` 相同，
         * 于是伸缩的 targetSec 记录在 ref 里却一直用旧块几何 —— 表现为
         * “松手才跳到位，不跟手”。同一帧里，行为 ref 比旧快照更权威。
         */
        const resizing = resizeRef.current;
        if (
            resizing !== null &&
            paramView !== null &&
            (resizing.epoch === undefined ||
                isParamBlockVisualEpochCurrent({
                    epoch: resizing.epoch,
                    current: epoch,
                }))
        ) {
            const resizeStart =
                resizing.resizePlan?.targetSpan !== null &&
                resizing.resizePlan?.targetSpan !== undefined
                    ? (resizing.resizePlan.targetSpan.startFrame *
                          resizing.resizePlan.viewAfter.framePeriodMs) /
                      1000
                    : resizing.startSec;
            const resizeEnd =
                resizing.resizePlan?.targetSpan !== null &&
                resizing.resizePlan?.targetSpan !== undefined
                    ? (resizing.resizePlan.targetSpan.endFrameExclusive *
                          resizing.resizePlan.viewAfter.framePeriodMs) /
                      1000
                    : resizing.endSec;
            const existingFor = (id: string, userBlockId?: string) =>
                canonicalBlocks.find((item) => item.id === id) ??
                (userBlockId === undefined
                    ? undefined
                    : canonicalBlocks.find((item) => item.userBlockId === userBlockId));
            if (resizing.blockId === "__create__") {
                const value =
                    resizing.value ??
                    averageParamValueInSpan({
                        startSec: Math.min(resizeStart, resizeEnd),
                        endSec: Math.max(resizeStart, resizeEnd),
                        values: paramView.edit,
                        startFrame: paramView.startFrame,
                        stride: paramView.stride,
                        framePeriodMs: paramView.framePeriodMs,
                        isPitch: editParam === "pitch",
                        param: editParam,
                    }) ??
                    null;
                const createId = resizing.createdBlockId ?? "__create__";
                const previewBlocks = projectParamBlockCreatePreviewFromBase({
                    framePeriodMs: framePeriodMs,
                    baseBlocks:
                        resizing.createBaseBlocks ??
                        canonicalBlocks.map((block) => ({
                            id: block.id,
                            startSec: block.startSec,
                            endSec: block.endSec,
                            averageValue: block.averageValue,
                            logicalCenterValue: block.manualValue,
                            displayCenterSnapToSemitone: block.manualValue == null,
                            selected: false,
                            grouped: block.kind === "grouped",
                        })),
                    startSec: resizeStart,
                    endSec: resizeEnd,
                    createId,
                    value,
                });
                const blocks: ParamBlockRenderBlock[] = previewBlocks.map((raw) => {
                    const block = raw as ParamBlockRenderBlock & {
                        readonly grouped?: boolean;
                        readonly manualValue?: number | null;
                    };
                    const existing = existingFor(block.id, block.userBlockId);
                    const created = block.id === createId;
                    return {
                        id: block.id,
                        userBlockId: block.userBlockId,
                        startSec: block.startSec,
                        endSec: block.endSec,
                        averageValue: block.averageValue ?? value ?? null,
                        logicalCenterValue: existing?.manualValue ?? block.manualValue ?? null,
                        frameCenterValue: created ? value : undefined,
                        displayCenterValue: created ? value : null,
                        displayCenterSnapToSemitone: created
                            ? false
                            : existing?.manualValue == null,
                        selected: created,
                        grouped: block.grouped === true,
                        transitionSec: existing?.transitionSec,
                        centerPolicy: created ? "locked" : "live",
                    };
                });
                return buildParamBlockRenderSpec({
                    blocks,
                    param: editParam,
                    valueMin: bounds.min,
                    valueMax: bounds.max,
                    valueToY: (value, heightPx) => valueToY(editParam, value, heightPx),
                    colors,
                    curveOwned,
                });
            }
            const plan = resizing.resizePlan;
            const blocks: ParamBlockRenderBlock[] =
                plan === null
                    ? canonicalBlocks.map((block) => ({
                          id: block.id,
                          userBlockId: block.userBlockId,
                          startSec: block.startSec,
                          endSec: block.endSec,
                          averageValue: block.averageValue,
                          logicalCenterValue: block.manualValue,
                          displayCenterSnapToSemitone: block.manualValue == null,
                          selected:
                              resizing.baseBlockSpans.find((item) => item.id === block.id)
                                  ?.selected === true,
                          grouped: block.kind === "grouped",
                          transitionSec: block.transitionSec,
                          centerPolicy: "live",
                      }))
                    : plan.viewAfter.blocks.map((block) => {
                          const fp = plan.viewAfter.framePeriodMs;
                          const startSec = (block.startFrame * fp) / 1000;
                          const endSec = (block.endFrameExclusive * fp) / 1000;
                          const existing = existingFor(block.id, block.userBlockId);
                          const resized = block.id === resizing.blockId;
                          return {
                              id: block.id,
                              userBlockId: block.userBlockId,
                              startSec,
                              endSec,
                              averageValue:
                                  existing?.averageValue ??
                                  valueCacheRef.current.get(block.id)?.value ??
                                  canonicalBlocks.find(
                                      (item) => item.startSec < endSec && item.endSec > startSec,
                                  )?.averageValue ??
                                  null,
                              logicalCenterValue: existing?.manualValue,
                              displayCenterValue: resized
                                  ? (resizing.displayCenterValue ?? null)
                                  : null,
                              displayCenterSnapToSemitone: resized
                                  ? false
                                  : existing?.manualValue == null,
                              displayHeightPx: resized ? (resizing.displayHeightPx ?? null) : null,
                              selected:
                                  resized ||
                                  (resizing.baseBlockSpans.find((item) => item.id === block.id)
                                      ?.selected === true &&
                                      block.id !== resizing.blockId),
                              grouped: block.kind === "grouped",
                              transitionSec: existing?.transitionSec,
                              centerPolicy: resized ? "locked" : "live",
                          };
                      });
            return buildParamBlockRenderSpec({
                blocks,
                param: editParam,
                valueMin: bounds.min,
                valueMax: bounds.max,
                valueToY: (value, heightPx) => valueToY(editParam, value, heightPx),
                colors,
                curveOwned,
            });
        }
        const dragSnapshot = dragSnapshotRef.current;
        if (
            dragSnapshot !== null &&
            isParamBlockVisualEpochCurrent({
                epoch: dragSnapshot.epoch,
                current: epoch,
            }) &&
            shouldUseParamBlockDragSnapshotForRender({
                snapshotKey: dragSnapshot.key,
                paramViewKey: paramView?.key ?? null,
                gestureActive: activeGestureTokenRef.current !== null,
                commitHeld: dragCommitRef.current !== null,
            })
        ) {
            return buildDragSnapshotSpec(dragSnapshot);
        }
        if (canonicalBlocks.length === 0) return null;
        const blocks: ParamBlockRenderBlock[] = canonicalBlocks.map((block) => ({
            id: block.id,
            userBlockId: block.userBlockId,
            startSec: block.startSec,
            endSec: block.endSec,
            topPx: 0,
            heightPx: 0,
            averageValue: block.averageValue,
            logicalCenterValue: block.manualValue,
            displayCenterSnapToSemitone: block.manualValue == null,
            selected: selectedIds.has(logicalBlockId(block)),
            grouped: block.kind === "grouped",
            transitionSec: block.transitionSec,
            centerPolicy: "live",
        }));
        return buildParamBlockRenderSpec({
            blocks,
            param: editParam,
            valueMin: bounds.min,
            valueMax: bounds.max,
            valueToY: (value, heightPx) => valueToY(editParam, value, heightPx),
            colors,
            curveOwned,
        });
    }, [
        layerVisible,
        canonicalBlocks,
        structureTick,
        epoch,
        colors,
        framePeriodMs,
        selectedIds,
        valueToY,
        paramView,
        // `themeMode` 不在此列：主题色已由 colors（其 useMemo
        // 依赖 themeMode）承载，重复列入只会让主题切换多跑一次全量重算。
        bounds.min,
        bounds.max,
        editParam,
        buildDragSnapshotSpec,
        // 下列 7 项是**稳定的 ref 对象 / 渲染期快照**。ref 的身份永远不变，列进
        // 依赖数组只是为了让 exhaustive-deps 认可「memo 读过它们」；`curveOwned`
        // 是渲染期读到的布尔值，必须列入（否则面板换参数后曲线归属判断会滞后）。
        activeGestureTokenRef,
        curveOwned,
        dragCommitRef,
        dragSnapshotRef,
        resizeRef,
        valueCacheRef,
        visualTransactionRef,
    ]);

    useLayoutEffect(() => {
        latestSpecRef.current = spec;
    }, [latestSpecRef, spec]);
    return spec;
}
