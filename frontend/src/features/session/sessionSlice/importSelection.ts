/**
 * 导入完成后的选择交接。
 *
 * 【为什么不直接 `selectedClipId = newClipIds[0]`】参数编辑器以 **root track**
 * 为作用域。导入时无条件选中新 Clip，会把 `selectedTrackId` 连带切到新 Clip
 * 所在的轨道组；用户刚编辑过的曲线从“当前轨道的参数”变成“另一条轨道的参数”
 * 被重新取回 —— 看起来就是参数被新 Clip 覆盖了。
 *
 * 本函数只负责选择：
 * - 新 Clip 一律进入多选，方便用户看到“导入了哪些”；
 * - 已有有效选中 Clip 时，保留它作为参数编辑器的锚点；
 * - 没有任何选中时，才把新 Clip 设为单选。
 *
 * 用户真正点击新 Clip 仍走 `setSelectedClip` / `selectClipRemote`，那时才切 root。
 */
export interface ImportSelectionState {
    selectedClipId: string | null;
    multiSelectedClipIds: string[];
    multiSelectionIntentional: boolean;
}

export function applyImportSelection(
    state: ImportSelectionState,
    newClipIds: readonly string[],
): void {
    if (newClipIds.length === 0) return;
    state.multiSelectedClipIds = [...newClipIds];
    state.multiSelectionIntentional = false;
    if (state.selectedClipId === null) {
        state.selectedClipId = newClipIds[0] ?? null;
    }
}
