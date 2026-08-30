export type TrackParentRef = {
    id: string;
    parentId?: string | null;
};

export function resolveRootTrackId(
    tracks: TrackParentRef[],
    selectedTrackId: string | null,
): string | null {
    const selected = selectedTrackId ?? tracks[0]?.id ?? null;
    if (!selected) return null;

    const byId = new Map(tracks.map((tr) => [tr.id, tr] as const));
    let cur = selected;
    let guard = 0;
    while (guard++ < 2048) {
        const tr = byId.get(cur);
        const parent = tr?.parentId ?? null;
        if (!parent) return cur;
        cur = parent;
    }

    return selected;
}

/**
 * 计算「在选中轨道正下方新增一条同级轨道」的放置参数。
 *
 * - 有选中轨道：新轨道继承选中轨道的 parentId（同层级），并插入到
 *   选中轨道同级列表的紧后一位（"紧跟着选中轨道的下方"）。
 *   选中根轨道时，新轨道作为根轨道紧跟其后（显示在其子树之后）。
 * - 无选中轨道：新轨道追加为根轨道列表末尾。
 *
 * @param tracks 需为后端返回的 DFS 显示顺序（同一父级下按 order 升序；
 *   同级轨道的相对顺序与 DFS 数组序一致）。
 * @returns 可直接传给 addTrackRemote / webApi.addTrackNested 的
 *   parentTrackId + index（后端按"同级内第 index 位"插入）。
 */
export function computeInsertBelowPlacement(
    tracks: TrackParentRef[],
    selectedTrackId: string | null,
): { parentTrackId: string | null; index: number } {
    const selected = tracks.find((t) => t.id === selectedTrackId);
    if (!selected) {
        const rootCount = tracks.filter((t) => !t.parentId).length;
        return { parentTrackId: null, index: rootCount };
    }
    const parentId = selected.parentId ?? null;
    // 同级轨道在 DFS 数组中保持升序出现（根轨道间虽隔着子树，但相对顺序
    // 不变；同一父级下的直接子轨道两两之间也没有其他父级的轨道）。数出
    // 选中轨道之前有几个同级轨道，插在其后一位。
    let before = 0;
    for (const t of tracks) {
        if (t.id === selected.id) break;
        if ((t.parentId ?? null) === parentId) before += 1;
    }
    return { parentTrackId: parentId, index: before + 1 };
}
