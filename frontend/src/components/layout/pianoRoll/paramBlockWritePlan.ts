/**
 * 参数块写入的「计划」。
 *
 * 【为什么需要这一层】后端 `set_param_block_edits` 的 CAS 同时校验编辑集代次、
 * 曲线代次与几何签名。前端只要**先读基准、后写**，中间任何一个别的写入都会让基准
 * 过期。旧实现里这段读-写横跨两条互不相识的队列（`useParamBlockPersistence` 的
 * `queues` 与面板的 `paramBlockCommitQueueRef`），交错是结构性的，不是概率事件 ——
 * 用户看到的就是「各种编辑场景下仍有概率报参数块操作失败」。
 *
 * 现在「读基准 → 算补丁 → 写」整体在**同一个队列槽**里执行，而这个槽跑的就是
 * `prepare()`：它可以在任意时刻被再调一次，等于「重新读一遍世界、按同一个意图
 * 重算补丁」。于是冲突的正确修复（重基）不需要调用方再实现一份逻辑。
 *
 * 【它不做什么】不排队、不发 IPC、不认识 React。队列与 IPC 在
 * `useParamBlockPersistence.commitPlan` 里。
 *
 * 【违反边界的症状】
 * - 在 `prepare()` 之外读基准再塞进 `state` / `patches`：冲突重新变成常态；
 * - 让 `prepare()` 有跨次副作用（例如每次都 `+1` 某个计数）：重基会重复施加；
 * - 在 `prepare()` 里再调 `commitPlan`：自己等自己，写入永久挂起。
 */
import type { ParamBlockState } from "./paramBlocks";
import type { ParamBlockCommitKind } from "./paramBlockCommitCoordinator";
import type { ParamCurveCommitAuthority } from "./paramBlockCommitPolicy";

/** 一次写入携带的曲线补丁（帧域绝对值）。 */
export interface ParamBlockCurvePatch {
    readonly param: string;
    readonly startFrame: number;
    readonly values: readonly number[];
}

/** 一次写入的「准备阶段」结果。 */
export interface ParamBlockWritePreparation {
    /** 要落盘的编辑集（整集覆盖）。 */
    readonly state: ParamBlockState;
    /** 是否打撤销点；缺省 true。 */
    readonly checkpoint?: boolean;
    /** 曲线补丁；空 = 纯布局提交。 */
    readonly patches?: readonly ParamBlockCurvePatch[];
    /** 有补丁时**必须**提供，否则后端必然拒绝。 */
    readonly authority?: ParamCurveCommitAuthority | null;
    /**
     * 「只写曲线」的降级路径（`set_param_frames`，无 CAS）。
     *
     * 【什么时候用】新建块时**块图层尚未就绪**（没有 `paramBlockView`）但曲线
     * 已经加载：这时没有可写的编辑集，曲线却必须落盘。它仍然要进队列 ——
     * 它同样会推进 `param_curve_revision`，绕过队列就会暗算别人的 CAS 基准。
     */
    readonly curveOnly?: {
        readonly param: string;
        readonly segments: readonly {
            readonly startFrame: number;
            readonly values: readonly number[];
        }[];
    };
}

/** 一次块写入的完整意图。 */
export interface ParamBlockWritePlan {
    readonly kind: ParamBlockCommitKind;
    /** 队列键。缺省 = 当前根。 */
    readonly rootTrackId?: string | null;
    /** 在队列槽内执行；冲突重基时会**再调一次**。 */
    readonly prepare: () => Promise<ParamBlockWritePreparation>;
}
