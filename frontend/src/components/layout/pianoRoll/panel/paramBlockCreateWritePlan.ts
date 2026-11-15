/**
 * 新建块的**写入计划**（队列槽内的准备阶段）。
 *
 * 【为什么单独成文件】这段逻辑要读全分辨率曲线、算补丁、挑 authority —— 它与
 * 手势状态机无关，只依赖几个快照值。放在面板模型里会让那个文件继续膨胀，也会让
 * 「读基准到底发生在队列槽内还是槽外」变得难以审计。
 *
 * 【为什么读也在这里】读与写必须在同一个队列槽里，否则 `param_curve_revision`
 * 会在两者之间被别人推进，这次提交的 `expectedCurveRevision` 立刻过期 ——
 * 旧实现正是这样把「新建块」变成概率性失败的。冲突时写队列会**再调一次**
 * `prepare`，那就是重基。
 *
 * 【三条分支（与旧实现一一对应，不得合并）】
 * 1. 没有曲线窗口 ⇒ 纯布局提交（块必须落盘，否则是静默丢数据）；
 * 2. 有曲线但没有块图层（`layoutBasis === null`）⇒ 只写曲线（无 CAS），
 *    仍然进队列，因为它同样推进曲线代次；
 * 3. 正常 ⇒ 布局 + 曲线补丁 + authority 一次提交。
 */
import type { ParamBlockView } from "../paramBlockTypes";
import type { ParamBlockState } from "../paramBlocks";
import { shiftParamBlockCurveValuesToTarget } from "../paramBlocks";
import { ParamBlockCommitError } from "../paramBlockCommitError";
import { fetchFullResCurve } from "../selectionEditData";
import type { ParamBlockWritePreparation } from "../paramBlockWritePlan";
import type { ParamViewSegment } from "../types";

export interface ParamBlockCreateWritePlanInput {
    readonly state: ParamBlockState;
    readonly rootTrackId: string;
    readonly editParam: string;
    /** 手势起点捕获的曲线窗口；`null` = 还没有曲线。 */
    readonly curveBase: ParamViewSegment | null;
    /** 手势起点的块视图；`null` = 块图层尚未就绪。 */
    readonly layoutBasis: ParamBlockView | null;
    readonly span: { readonly startSec: number; readonly endSec: number };
    readonly value: number;
    readonly shape: string;
    /** 每次准备算出的逐帧片段（成功后抄进交接与视觉事务）。 */
    readonly onSegments?: (
        segments: readonly { readonly startFrame: number; readonly values: number[] }[],
    ) => void;
}

/** 见模块头。返回的函数会被写队列在槽内调用（冲突时不止一次）。 */
export function createParamBlockWritePlan(
    input: ParamBlockCreateWritePlanInput,
): () => Promise<ParamBlockWritePreparation> {
    return async () => {
        const { curveBase, layoutBasis } = input;
        if (curveBase === null) {
            return { state: input.state, checkpoint: true, patches: [] };
        }
        const fp = Math.max(1e-6, curveBase.framePeriodMs);
        const startFrame = Math.max(0, Math.floor((input.span.startSec * 1000) / fp));
        const endFrame = Math.max(
            startFrame,
            (layoutBasis ? Math.round : Math.ceil)((input.span.endSec * 1000) / fp) - 1,
        );
        const source = await fetchFullResCurve({
            trackId: input.rootTrackId,
            param: input.editParam,
            startFrame,
            endFrame,
            paramView: layoutBasis !== null ? null : curveBase,
            // 创建提交带 curve patches；即使 curveBase 覆盖全分辨率也必须回源拿
            // authority，否则后端必拒。
            forceAuthority: true,
        });
        const segments = [
            {
                startFrame: source.startFrame,
                values: shiftParamBlockCurveValuesToTarget({
                    values: source.values,
                    target: input.value,
                    param: input.editParam,
                    flatten: input.shape === "flat",
                    curveOwned: layoutBasis !== null,
                }),
            },
        ];
        input.onSegments?.(segments);
        if (layoutBasis === null) {
            return {
                state: input.state,
                checkpoint: true,
                curveOnly: { param: input.editParam, segments },
            };
        }
        if (source.authorityRevision === undefined) {
            throw new ParamBlockCommitError({
                kind: "rejected",
                code: "missing_curve_authority",
                detail: "param block create requires expectedCurveRevision",
            });
        }
        return {
            state: input.state,
            checkpoint: true,
            patches: segments.map((segment) => ({ ...segment, param: input.editParam })),
            authority: {
                expectedCurveRevision: source.authorityRevision,
                ...(source.geometryRevision === undefined
                    ? {}
                    : { expectedGeometryRevision: source.geometryRevision }),
                ...(source.geometrySignature === undefined
                    ? {}
                    : { expectedGeometrySignature: source.geometrySignature }),
            },
        };
    };
}
