/**
 * 时间轴渲染内核 · 拖拽落点解析（纯函数）
 *
 * 【主要内容】
 * 把内核手势回调给出的 `targetTrackId` 解析为提交落库所需的三元组：
 * 是否新建轨道、相对锚点的轨道偏移量、目标轨道下标。
 *
 * 【作用】
 * 内核拖拽的回调把落点作为 `targetTrackId` 传回面板（拖到全部轨道之下时是
 * `NEW_TRACK_SENTINEL` 哨兵）。**预览分支与提交分支都必须用同一份解析**——
 * 原实现里预览分支老老实实按 `args.targetTrackId` 算，提交分支却把
 * `dropToNewTrack` / `trackOffset` 写死成 `false` / `0`，于是「幽灵预览能到新轨道
 * 和其他轨道，落库却永远留在原轨」。两处各写一份必然分叉，本模块是唯一来源。
 *
 * 【与其他模块的关系】
 * - 上游：`TimelinePanel` 的 `handleKernelDragCommit`（copy 分支与 move 分支）。
 * - 下游：`copyClipsFromDrag` 依据 `dropToNewTrack` / `trackOffset` 解析目标轨。
 * - 独立性：纯函数，无 React / DOM / Redux 依赖，可在 node 环境单测。
 *
 * 【设计约束】
 * 1. 落点不在轨道列表中**且不是哨兵**时按"回落原轨"处理（`trackOffset = 0`）：
 *    这既覆盖手指/指针落在轨道区之外的情形，也覆盖重名 id 之类的异常输入。
 * 2. 锚点下标非法（< 0，例如该 clip 的轨道已被删除）时不跨轨，避免用一个
 *    无意义的差值把 clip 甩到别的轨道上。
 */

/** 解析入参。 */
export interface KernelDropTargetArgs {
    /** 内核回调给出的落点轨道 id；可能是 `newTrackSentinel`。 */
    readonly targetTrackId: string;
    /** 当前工程的轨道 id 列表（顺序 = 纵向排列顺序）。 */
    readonly trackIds: readonly string[];
    /** 拖拽锚点 clip 的轨道下标；非法时为 -1。 */
    readonly anchorTrackIndex: number;
    /** 「拖到全部轨道之下」的哨兵值。 */
    readonly newTrackSentinel: string;
}

/** 解析结果。 */
export interface KernelDropTarget {
    /** 是否落到新建轨道。 */
    readonly dropToNewTrack: boolean;
    /** 相对锚点轨道下标的偏移量（同轨为 0；哨兵落点为 0）。 */
    readonly trackOffset: number;
    /** 目标轨道下标；哨兵或解析失败时为 -1。 */
    readonly targetTrackIndex: number;
}

/**
 * 解析内核拖拽落点。
 *
 * 流程：哨兵判定 → 在 `trackIds` 中定位目标下标 → 与锚点下标相减得偏移量
 * （任一下标非法时偏移量归零）。
 *
 * @param args 落点、轨道列表与锚点下标。
 * @returns 新建标记、轨道偏移量与目标下标。
 */
export function resolveKernelDropTarget(args: KernelDropTargetArgs): KernelDropTarget {
    if (args.targetTrackId === args.newTrackSentinel) {
        return { dropToNewTrack: true, trackOffset: 0, targetTrackIndex: -1 };
    }
    const targetTrackIndex = args.trackIds.indexOf(args.targetTrackId);
    if (targetTrackIndex < 0 || args.anchorTrackIndex < 0) {
        return { dropToNewTrack: false, trackOffset: 0, targetTrackIndex };
    }
    return {
        dropToNewTrack: false,
        trackOffset: targetTrackIndex - args.anchorTrackIndex,
        targetTrackIndex,
    };
}
