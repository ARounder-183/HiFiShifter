/**
 * 参数块提交契约。
 *
 * 【为什么需要】
 * 后端 `set_param_block_edits` 对任何非空 `curve_patches` 都要求
 * `expected_curve_revision == 当前权威曲线 revision`，否则整笔事务返回
 * `Conflict: parameter curve changed`。前端曾把`expectedCurveRevision`写成
 * 可选位置参数，过渡手柄 / 幅度提交都漏传，于是“每次操作都失败”。本模块把
 * “有曲线补丁就必须有权威 revision”从注释约定升级成运行时守卫。
 *
 * 【调用关系】
 * - `useParamBlockPersistence.commit` 在入口调用 `assertCurveCommitAuthority`；
 * - 过渡、幅度、创建、块拖动等所有带 patches 的路径都从这里取类型；
 * - 无 patches 的纯布局提交仍允许不携带 authority。
 */

export interface ParamCurveCommitAuthority {
    /** 必须来自同一次全分辨率读数的 `authorityRevision`。 */
    readonly expectedCurveRevision: number;
    readonly expectedGeometryRevision?: number;
    readonly expectedGeometrySignature?: string;
}

/**
 * 契约守卫。
 *
 * 抛错而不是静默降级：没有权威 watermark 的 curve patch 提交在后端**必然**
 * 被拒，提前在前端抛出能保留调用栈，且不会把一个用户手势变成“随机失败”。
 */
export function assertCurveCommitAuthority(args: {
    readonly hasPatches: boolean;
    readonly authority?: ParamCurveCommitAuthority | null;
}): void {
    if (!args.hasPatches) return;
    const revision = args.authority?.expectedCurveRevision;
    if (typeof revision !== "number" || !Number.isFinite(revision)) {
        throw new Error("param block curve commit requires expectedCurveRevision");
    }
}
