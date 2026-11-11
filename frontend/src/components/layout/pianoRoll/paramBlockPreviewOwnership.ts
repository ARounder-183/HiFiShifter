/**
 * 参数块视觉预览的“自身提交”归属。
 *
 * 【问题】创建 / 拉伸 / 过渡在 pointerup 时先推进本地 editSet revision，再
 * 保留预览等权威曲线。epoch 校验把 revision 变化及 `ownedEditRevision` 对比，
 * 若只给其中一种 ref 打标记，另一些会被判成外部结构编辑并提前撤下预览，
 * 用户看到“先回弹旧值，再跳到新值”。
 *
 * 【做法】一个入口同时标记所有仍存活的预览；调用方只传 ref，不关心每种
 * 预览的具体类型。纯函数，便于单测钉住“三条路径都写到”。
 */

export interface PreviewOwnershipTarget {
    ownedEditRevision?: number | null;
}

export interface PreviewOwnershipRef<T extends PreviewOwnershipTarget> {
    current: T | null;
}

export function markParamBlockPreviewOwnedRevision(
    refs: {
        readonly visualTransaction: PreviewOwnershipRef<PreviewOwnershipTarget>;
        readonly dragSnapshot: PreviewOwnershipRef<PreviewOwnershipTarget>;
        readonly resize: PreviewOwnershipRef<PreviewOwnershipTarget>;
    },
    revision: number,
): void {
    if (!Number.isFinite(revision)) return;
    for (const ref of [refs.visualTransaction, refs.dragSnapshot, refs.resize]) {
        if (ref.current !== null) ref.current.ownedEditRevision = revision;
    }
}
