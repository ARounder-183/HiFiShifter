/**
 * copy 幽灵块的几何比较（从 `useTimelineKernelSeekDrag` 拆出的纯函数）。
 *
 * 【为什么要拆】⌘ 拖拽的每个预览帧都会重算同一组 ghost 几何；`setKernelGhost`
 * 挂在面板级 state 上，逐帧提交新数组会重渲染整个 TimelinePanel。几何未变就
 * 不 setState —— 判据单独成模块，主文件只保留一行调用（也守住超大文件预算）。
 */
export interface KernelGhostEntry {
    key: string;
    leftPx: number;
    widthPx: number;
    trackId: string;
}

/** 两组幽灵块几何是否逐项相同（含条目数与顺序）。 */
export function sameKernelGhostGeometry(
    previous: readonly KernelGhostEntry[] | null,
    next: readonly KernelGhostEntry[],
): boolean {
    return (
        previous !== null &&
        previous.length === next.length &&
        previous.every(
            (item, index) =>
                item.key === next[index].key &&
                item.leftPx === next[index].leftPx &&
                item.widthPx === next[index].widthPx &&
                item.trackId === next[index].trackId,
        )
    );
}
