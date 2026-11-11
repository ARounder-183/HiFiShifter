/**
 * 参数编辑器「音节块」的纯派生模型。
 *
 * 【为什么块不进工程文件】音节边界最终应由音高分析 / 分词结果给出；第一版先在
 * 前端从现有 Clip + MIDI 音符 / 曲线推导，保证工具、显示、多选、合并这些交互
 * 可以先被真实使用和验证。用户手动合并的选择保存在会话内的 `ParamBlockState`，
 * 不复制参数数组，也不改写 Clip 几何。
 *
 * 【坐标】一律用**工程秒**。参数曲线本身是帧域，但这里只做“这一块覆盖哪段时间”，
 * 绘制时再经 `TimelineAxis` 投影；块高则由当前参数曲线在成员范围内的均值求值。
 *
 * 【作用域】只在当前根轨道组内建块。跨组块没有唯一参数所有者，第一版直接拒绝。
 */

import {
    createUserBlock,
    deleteProjectedBlocks,
    groupProjectedBlocks,
    newParamBlockId,
    ParamBlockEditError,
    ParamBlockPlanError,
    planSplitProjectedBlock,
    restoreDeletedBlocks,
    setProjectedBlockTransition,
    ungroupUserBlock,
    type DeletedParamBlockSnapshot,
} from "./paramBlockEditing";
import { canSplitParamBlockAtFrame } from "./paramBlockGesture";
import { toFrame, type ParamBlockSourceClip } from "./paramBlockProjection";
import { getDrawPreviewValue } from "./paramValuePreviewLogic";
import { clampParamWriteValue, isDynParam, DYN_VALUE_MAX, DYN_VALUE_MIN } from "./paramRanges";
import type { ScaleLike } from "../../../utils/musicalScales";
import {
    EMPTY_PARAM_BLOCK_EDIT_SET,
    blockMatchesLogicalId,
    logicalBlockId,
    type ParamBlockEditSet,
    type ParamBlockView,
    type ProjectedParamBlock,
} from "./paramBlockTypes";

export type { ParamBlockSourceClip } from "./paramBlockProjection";
export { newParamBlockId };

export interface ParamBlockMember {
    readonly clipId: string;
    readonly takeId?: string;
    readonly syllableId?: string;
    /** 该成员对应的源域区间；块的唯一身份锚点。 */
    readonly sourceStartSec?: number;
    readonly sourceEndSec?: number;
    /** Loop 周期；非 Loop 为 0。 */
    readonly cycle?: number;
    readonly startSec: number;
    readonly endSec: number;
}

export interface ParamBlock {
    /** 派生视图块；只读，不是持久化权威。 */
    readonly projected?: ProjectedParamBlock;
    /** 多 run 用户块所属的父块 id；base 块没有。 */
    readonly userBlockId?: string;
    readonly id: string;
    readonly rootTrackId: string;
    readonly kind: "auto" | "grouped" | "user" | "manual";
    readonly members: readonly ParamBlockMember[];
    readonly startSec: number;
    readonly endSec: number;
    /**
     * 块的代表参数值（**密度众数中心**，不是算术均值）。
     *
     * 用块内出现最密集的参数区间代表块的高度位置；颤音、短滑音与少量离群点
     * 不会被拉平。字段名保留 `averageValue` 以兼容既有宿主 / 测试载荷。
     */
    readonly averageValue: number | null;
    /** 音高块显式保存的 MIDI 中心；其他参数必须从各自曲线派生位置。 */
    readonly manualValue?: number | null;
    /**
     * 块间圆滑宽度（秒）：向两侧各 blend 这么多秒，跨越相邻块边界做线性过渡。
     * 只对同一 Clip/Take 内**首尾相接**的块生效；缺省 0 = 硬切换。
     */
    readonly transitionSec?: number;
    /** 实时拖动 / 过渡预览时显式保持连续中心；缺省吸附到半音行心。 */
    readonly displayCenterSnapToSemitone?: boolean;
}

/** 块边界命中结果：像素热区优先，避免窄块只能点中 1px 边线。 */
export interface ParamBlockEdgeHit<T> {
    readonly block: T;
    readonly edge: "left" | "right";
}

/** 块上的指针反馈：决定光标与手柄可见度。 */
export type ParamBlockHoverKind = "body" | "left-edge" | "right-edge" | "transition";

export function paramBlockCursorForHover(kind: ParamBlockHoverKind | null): string {
    if (kind === "left-edge" || kind === "right-edge") return "ew-resize";
    if (kind === "transition") return "ew-resize";
    if (kind === "body") return "grab";
    return "default";
}

/**
 * 组的可见 run 可能不连续。组的外边缘只在首/末 run；内部 run 的左右边界
 * 是子块边界，不是组边界，不能拿来触发“整个组”的伸缩。
 */
export function paramBlockEdgeCandidates(blocks: readonly ParamBlock[]): ParamBlock[] {
    const grouped = new Map<string, ParamBlock[]>();
    for (const block of blocks) {
        if (block.kind !== "grouped") continue;
        const id = logicalBlockId(block);
        const runs = grouped.get(id);
        if (runs === undefined) grouped.set(id, [block]);
        else runs.push(block);
    }
    return blocks.filter((block) => {
        if (block.kind !== "grouped") return true;
        const runs = grouped.get(logicalBlockId(block));
        if (runs === undefined || runs.length <= 1) return true;
        const startSec = Math.min(...runs.map((run) => run.startSec));
        const endSec = Math.max(...runs.map((run) => run.endSec));
        return (
            Math.abs(block.startSec - startSec) <= 1e-6 || Math.abs(block.endSec - endSec) <= 1e-6
        );
    });
}

/**
 * 块的左右边缘命中。
 *
 * 只按时间容差判断时，`pxPerSec` 很小会把“离边缘几像素”算成很大的秒数，
 * 窄块几乎整块都变成缩放热区。这里统一用**视口像素**：先把块左右端点投影成
 * x，再看指针 x 与哪一侧最近，并限制在热区半宽内。
 */
export function hitTestParamBlockEdges<
    T extends {
        readonly id: string;
        readonly startSec: number;
        readonly endSec: number;
        readonly topPx: number;
        readonly heightPx: number;
    },
>(args: {
    readonly blocks: readonly T[];
    readonly clientX: number;
    readonly localY: number;
    readonly secToX: (sec: number) => number;
    /** 单侧热区半宽（CSS px）。 */
    readonly hitPx?: number;
    /** 纵向允许超出块矩形的容差（CSS px）。 */
    readonly slopPx?: number;
    /** 只在指定块的端点内命中；hover 可传当前块，pointerdown 再按各自优先级筛选。 */
    readonly blockId?: string;
}): ParamBlockEdgeHit<T> | null {
    const hitPx = Math.max(1, finite(args.hitPx ?? 6, 6));
    const slop = Math.max(0, finite(args.slopPx ?? 4, 4));
    let best: { block: T; edge: "left" | "right"; distance: number } | null = null;
    for (let index = args.blocks.length - 1; index >= 0; index -= 1) {
        const block = args.blocks[index];
        if (block === undefined) continue;
        if (args.blockId !== undefined && block.id !== args.blockId) continue;
        if (args.localY < block.topPx - slop || args.localY > block.topPx + block.heightPx + slop) {
            continue;
        }
        const leftX = args.secToX(Math.min(block.startSec, block.endSec));
        const rightX = args.secToX(Math.max(block.startSec, block.endSec));
        /*
         * 窄块不能整块都变成边缘热区：热区按块宽收缩到 1/3，保证块体仍可拖动。
         * 宽块仍用固定 8px 热区。
         */
        const effectiveHitPx = Math.min(hitPx, Math.max(2, Math.abs(rightX - leftX) / 3));
        for (const edge of ["left", "right"] as const) {
            const edgeX = edge === "left" ? leftX : rightX;
            const distance = Math.abs(args.clientX - edgeX);
            if (distance > effectiveHitPx) continue;
            if (best === null || distance < best.distance - 1e-9) {
                best = { block, edge, distance };
            }
        }
    }
    return best === null ? null : { block: best.block, edge: best.edge };
}

/**
 * 相接块过渡手柄的命中区。
 *
 * 手柄绘制在块右边界外侧上方，视觉上是“圆点 + 横线”。命中只覆盖这条
 * 水平短线附近，**不覆盖块右边缘的缩放热区**：两者在空间上分开，避免
 * 圆滑手柄把伸缩控件吃掉。
 */
export function hitTestParamBlockTransitionHandles<
    T extends {
        readonly id?: string;
        readonly topPx: number;
        readonly heightPx: number;
        readonly selected?: boolean;
        readonly hovered?: boolean;
        readonly transitionHandle?: {
            readonly sec: number;
            readonly leftId: string;
            readonly rightId: string;
            readonly widthSec: number;
        } | null;
    },
>(args: {
    readonly blocks: readonly T[];
    readonly clientX: number;
    readonly localY: number;
    readonly secToX: (sec: number) => number;
    readonly hitHalfWidthPx?: number;
    readonly hitHalfHeightPx?: number;
    /** 手柄圆点相对右边界的水平偏移（CSS px）；0 = 正上方对齐。 */
    readonly outsetPx?: number;
    /** 手柄圆点相对块顶的上抬量（CSS px，正值在块外）。 */
    readonly outsetYPx?: number;
}): {
    readonly leftId: string;
    readonly rightId: string;
    readonly sec: number;
    readonly widthSec: number;
    readonly topPx: number;
    readonly heightPx: number;
} | null {
    // 命中区比视觉圆点/横线略大，保证细指针也能抓住；仍与块右边缘分开。
    const halfW = Math.max(1, finite(args.hitHalfWidthPx ?? 12, 12));
    const halfH = Math.max(1, finite(args.hitHalfHeightPx ?? 7, 7));
    const outsetX = finite(
        args.outsetPx ?? PARAM_BLOCK_TRANSITION_HANDLE_OUTSET_X_PX,
        PARAM_BLOCK_TRANSITION_HANDLE_OUTSET_X_PX,
    );
    const outsetY = Math.max(
        0,
        finite(args.outsetYPx ?? PARAM_BLOCK_TRANSITION_HANDLE_OUTSET_Y_PX, 6),
    );
    let best: {
        leftId: string;
        rightId: string;
        sec: number;
        widthSec: number;
        topPx: number;
        heightPx: number;
        distance: number;
    } | null = null;
    const byId = new Map(args.blocks.map((block) => [block.id, block]));
    for (const block of args.blocks) {
        const handle = block.transitionHandle;
        if (handle === null || handle === undefined) continue;
        // 手柄只在块被选中 / 悬停时绘制；不可见的手柄不能抢命中。
        if (block.selected !== true && block.hovered !== true) continue;
        const x = args.secToX(handle.sec);
        const handleX = x + outsetX;
        const handleY = paramBlockTransitionHandleY(
            block.topPx,
            byId.get(handle.rightId)?.topPx,
            outsetY,
        );
        const dx = args.clientX - handleX;
        const dy = args.localY - handleY;
        if (Math.abs(dx) > halfW || Math.abs(dy) > halfH) continue;
        const distance = Math.hypot(dx, dy);
        if (best === null || distance < best.distance) {
            best = {
                leftId: handle.leftId,
                rightId: handle.rightId,
                sec: handle.sec,
                widthSec: handle.widthSec,
                topPx: block.topPx,
                heightPx: block.heightPx,
                distance,
            };
        }
    }
    if (best === null) return null;
    return {
        leftId: best.leftId,
        rightId: best.rightId,
        sec: best.sec,
        widthSec: best.widthSec,
        topPx: best.topPx,
        heightPx: best.heightPx,
    };
}

/** 块边缘延展的实时网格吸附 + 最小宽度 / 邻居边界约束。 */
export function resolveParamBlockResizeTarget(args: {
    readonly edge: "left" | "right";
    readonly startSec: number;
    readonly endSec: number;
    readonly rawTargetSec: number;
    readonly snapToGrid: (sec: number) => number;
    /** 原始指针边界：即使吸附后落在限制之外，也允许联动压缩邻居。 */
    readonly unclampedTargetSec?: number;
    readonly minBlockSec: number;
    readonly neighborStartSec?: number | null;
    readonly neighborEndSec?: number | null;
}): number {
    const minSec = Math.max(1e-6, finite(args.minBlockSec, MIN_AUTO_BLOCK_SEC));
    /*
     * `unclampedTargetSec` 是未受邻居限制的指针端点。吸附永远基于它，而不是
     * 上一帧已钳制的 target；否则越靠近邻居，吸附结果越会被拉回旧边界。
     */
    const raw =
        args.unclampedTargetSec !== undefined && Number.isFinite(args.unclampedTargetSec)
            ? args.unclampedTargetSec
            : args.rawTargetSec;
    const snapped = finite(args.snapToGrid(raw), raw);
    const plan = resolveParamBlockResizePlan({
        edge: args.edge,
        blockStartSec: args.startSec,
        blockEndSec: args.endSec,
        targetSec: snapped,
        neighbor:
            args.neighborStartSec != null && args.neighborEndSec != null
                ? {
                      startSec: args.neighborStartSec,
                      endSec: args.neighborEndSec,
                  }
                : null,
        minBlockSec: minSec,
    });
    return args.edge === "left" ? plan.block.startSec : plan.block.endSec;
}

export type ParamBlockDragCommitReleaseState =
    "waiting-for-curve" | "waiting-for-paint" | "released";

/**
 * 只有 active gesture / 等待提交交接的事务才能持有拖动快照。
 * 旧实现还在 paramViewKey === snapshotKey 时返回 true；撤销或 Clip 编辑后
 * 这个 key 可能保持不变，旧快照就会在无事务状态下继续覆盖 canonical。
 */
export function shouldUseParamBlockDragSnapshotForRender(args: {
    readonly snapshotKey: string;
    readonly paramViewKey: string | null;
    /** A live gesture owns its frozen window even if a background fetch rekeys paramView. */
    readonly gestureActive: boolean;
    readonly commitHeld: boolean;
}): boolean {
    void args.snapshotKey;
    void args.paramViewKey;
    return args.gestureActive || args.commitHeld;
}

/**
 * 普通块平移必须继续读取实时曲线中心；只有 Alt 音高缩放才允许锁住中心，
 * 因为缩放会改变块内起伏而不改变块应有的代表音高。
 */
function resolveParamBlockResizePlan(args: {
    readonly blockId?: string;
    readonly edge: "left" | "right";
    readonly blockStartSec: number;
    readonly blockEndSec: number;
    readonly targetSec: number;
    /** 同方向全部候选块；会按目标区间筛出真正受影响的块。 */
    readonly neighbors?: readonly (
        | ParamBlock
        | {
              readonly id?: string;
              readonly startSec: number;
              readonly endSec: number;
          }
    )[];
    /** 兼容单邻居调用。 */
    readonly neighbor?:
        | {
              readonly id?: string;
              readonly startSec: number;
              readonly endSec: number;
          }
        | ParamBlock
        | null;
    readonly minBlockSec: number;
}): ParamBlockResizePlan {
    const minSec = Math.max(1e-6, finite(args.minBlockSec, MIN_AUTO_BLOCK_SEC));
    const blockStart = Math.min(args.blockStartSec, args.blockEndSec);
    const blockEnd = Math.max(args.blockStartSec, args.blockEndSec);
    const target = finite(args.targetSec, args.edge === "left" ? blockStart : blockEnd);
    const candidates = [
        ...(args.neighbors ?? []),
        ...(args.neighbor === null || args.neighbor === undefined ? [] : [args.neighbor]),
    ]
        .filter(
            (candidate, index, all) =>
                all.findIndex(
                    (other) =>
                        "id" in other &&
                        "id" in candidate &&
                        other.id !== undefined &&
                        other.id === candidate.id,
                ) === index,
        )
        .filter((candidate) =>
            args.edge === "right"
                ? candidate.startSec >= blockEnd - 1e-6
                : candidate.endSec <= blockStart + 1e-6,
        )
        .sort((a, b) => (args.edge === "right" ? a.startSec - b.startSec : b.endSec - a.endSec));
    if (args.edge === "right") {
        const lower = blockStart + minSec;
        // 右侧没有硬上限：拖过整块时可继续向前吞并后续块。
        const blockNextEnd = Math.max(lower, target);
        const removeBlock = blockNextEnd <= lower + 1e-9;
        const neighbors: ParamBlockResizeNeighborPlan[] = [];
        if (!removeBlock) {
            for (const candidate of candidates) {
                if (blockNextEnd <= candidate.startSec + 1e-9) continue;
                const removed =
                    blockNextEnd >= candidate.endSec - 1e-9 ||
                    candidate.endSec - blockNextEnd < minSec - 1e-9;
                neighbors.push({
                    id:
                        "id" in candidate && candidate.id !== undefined
                            ? candidate.id
                            : `neighbor:${candidate.startSec}`,
                    startSec: removed ? candidate.startSec : blockNextEnd,
                    endSec: candidate.endSec,
                    remove: removed,
                });
            }
        }
        return {
            block: { startSec: blockStart, endSec: blockNextEnd },
            neighbors,
            removeBlock,
        };
    }
    const upper = blockEnd - minSec;
    const blockNextStart = Math.min(upper, target);
    const removeBlock = blockNextStart >= upper - 1e-9;
    const neighbors: ParamBlockResizeNeighborPlan[] = [];
    if (!removeBlock) {
        for (const candidate of candidates) {
            if (blockNextStart >= candidate.endSec - 1e-9) continue;
            const removed =
                blockNextStart <= candidate.startSec + 1e-9 ||
                blockNextStart - candidate.startSec < minSec - 1e-9;
            neighbors.push({
                id:
                    "id" in candidate && candidate.id !== undefined
                        ? candidate.id
                        : `neighbor:${candidate.startSec}`,
                startSec: candidate.startSec,
                endSec: removed ? candidate.endSec : blockNextStart,
                remove: removed,
            });
        }
    }
    return {
        block: { startSec: blockNextStart, endSec: blockEnd },
        neighbors,
        removeBlock,
    };
}

/** 创建块：按指针横向位移得到吸附后的区间，粘贴到网格线而不是松手才对齐。 */
export function resolveParamBlockCreateSpan(args: {
    readonly framePeriodMs?: number;
    readonly anchorSec: number;
    readonly pointerSec: number;
    readonly snapToGrid: (sec: number) => number;
    readonly snapEnabled: boolean;
    readonly minBlockSec: number;
}): { readonly startSec: number; readonly endSec: number } {
    if (args.framePeriodMs) {
        const fp = args.framePeriodMs / 1000;
        const snap = (sec: number) =>
            Math.max(0, Math.round((args.snapEnabled ? args.snapToGrid(sec) : sec) / fp));
        const anchor = snap(args.anchorSec);
        let target = snap(args.pointerSec);
        if (target === anchor) target = anchor + 1;
        return { startSec: Math.min(anchor, target) * fp, endSec: Math.max(anchor, target) * fp };
    }
    const minSec = Math.max(1e-6, finite(args.minBlockSec, MIN_AUTO_BLOCK_SEC));
    const snap = (sec: number): number =>
        args.snapEnabled ? finite(args.snapToGrid(sec), sec) : sec;
    const anchor = snap(args.anchorSec);
    let target = snap(args.pointerSec);
    if (Math.abs(target - anchor) < minSec) {
        target = anchor + (target < anchor ? -minSec : minSec);
    }
    const startSec = Math.min(anchor, target);
    const endSec = Math.max(anchor, target);
    return { startSec, endSec };
}

/**
 * 播放头处的块 —— 切分的**唯一操作数**。
 *
 * 块在时间上互不重叠，所以播放头最多落在一个块里：操作数天然唯一，不需要
 * 也不应该读"当前选中了哪个块"。解析不到就是"现在没有可切的块"，调用方
 * 必须静默返回（与音频块的分割 / 编组一致）。
 *
 * 只接受**严格内部**：正好落在边界上切出来会是 0 长块。
 */
export function resolveParamBlockSplitTarget(
    blocks: readonly ParamBlock[],
    splitSec: number,
    framePeriodMs: number,
): ParamBlock | null {
    if (!Number.isFinite(splitSec)) return null;
    const frame = toFrame(splitSec, framePeriodMs);
    return (
        blocks.find(
            (block) =>
                block.projected !== undefined &&
                frame > block.projected.startFrame &&
                frame < block.projected.endFrameExclusive,
        ) ?? null
    );
}

/**
 * 播放头处**可切**的块：位置解析 + 编组块排除。
 *
 * 菜单禁用态与快捷键动作必须共用这一份判据 —— 各写一遍的话，编组块会出现在
 * "菜单可点、点了没反应"的错位状态。编组块的几何是若干源域区间的联合，
 * 切开没有唯一解（当前内核已支持组递归切分）。
 */
export function resolveSplittableParamBlockAt(
    blocks: readonly ParamBlock[],
    splitSec: number,
    framePeriodMs: number,
    context?: {
        readonly clips: readonly ParamBlockSourceClip[];
        readonly edits: ParamBlockEditSet;
    },
): ParamBlock | null {
    const target = resolveParamBlockSplitTarget(blocks, splitSec, framePeriodMs);
    if (target === null) return null;
    if (context === undefined) return target;
    const frame = toFrame(splitSec, framePeriodMs);
    if (target.projected === undefined) return null;
    return canSplitParamBlockAtFrame({
        clips: context.clips,
        edits: context.edits,
        framePeriodMs,
        block: target.projected,
        frame,
    })
        ? target
        : null;
}

/**
 * 切点是否严格落在该块内部（与 `splitProjectedBlock` 的边沿判定同一口径）。
 *
 * 用途是**约束磁吸**：切分点允许吸附，但磁吸结果一旦离开目标块就作废 ——
 * 相邻块的边界是候选之一，吸上去会正好切在边界上。
 */
export function isStrictlyInsideParamBlock(
    block: ParamBlock,
    splitSec: number,
    framePeriodMs: number,
): boolean {
    const projected = block.projected;
    if (projected === undefined || !Number.isFinite(splitSec)) return false;
    const frame = toFrame(splitSec, framePeriodMs);
    return frame > projected.startFrame && frame < projected.endFrameExclusive;
}

/** 切开位置：只在像素阈值内吸附；开关与 XOR 由调用方的 snapSec 统一决定。 */
export function resolveParamBlockSplitSec(args: {
    readonly sec: number;
    readonly snapSec: (sec: number) => number;
    readonly enabled: boolean;
    readonly pxPerSec: number;
    readonly thresholdPx: number;
    readonly snapToggleHeld?: boolean;
}): number {
    if (!args.enabled || !Number.isFinite(args.sec)) return args.sec;
    const snapped = args.snapSec(args.sec);
    const distancePx = Math.abs(snapped - args.sec) * Math.max(1, args.pxPerSec);
    return distancePx <= Math.max(0, args.thresholdPx) ? snapped : args.sec;
}

/**
 * 参数块编辑状态。
 *
 * 【唯一权威是 `editSet`】用户只声明"哪些源域区间属于同一个块"。
 *
 * 【这里**没有** `view`】块几何是 `(clips, editSet, framePeriodMs)` 的纯函数
 * （`composeParamBlockView`）。把它存进状态就有两个写入者 —— 本地合成与后端
 * 回读 —— 而回读在手势期间拿到的是旧几何：于是块在新旧位置之间逐帧来回跳
 * （闪烁），切分 / 伸缩读到过期边界（"切分点离边缘太近"）。派生值不该被存下来。
 *
 * 【这里**也没有** `clips`】同一份 clip 快照存两处，合成用的与调用方传入的迟早
 * 分叉；几何一律现算，编辑函数一律由调用方显式传 `clips`。
 */
export interface ParamBlockState {
    readonly editSet: ParamBlockEditSet;
    readonly rootTrackId?: string;
    /** 帧周期（工程常量）；由后端回包播种，缺省 5ms。 */
    readonly framePeriodMs: number;
    /** 上一批被删除的块源域快照，供用户一键恢复；仅会话内有效。 */
    readonly lastDeletedBlocks?: readonly DeletedParamBlockSnapshot[];
}

export const EMPTY_PARAM_BLOCK_STATE: ParamBlockState = {
    editSet: EMPTY_PARAM_BLOCK_EDIT_SET,
    framePeriodMs: 5,
};

export type BlockGroupResult =
    | { readonly ok: true; readonly state: ParamBlockState }
    | { readonly ok: false; readonly reason: BlockGroupFailure };

export type BlockGroupFailure =
    | "empty"
    | "different_root"
    | "overlap"
    | "already_grouped"
    | "mixed_geometry"
    | "nothing_to_group";

export type ParamBlockEditFailureReason =
    | "not_found"
    | "too_short"
    | "too_close_to_edge"
    | "resize_blocked"
    | "no_members"
    | "multi_fragment_unsupported"
    /** 未分类异常。**不要**兜底成 `too_short`：那会把真错误伪装成"离边缘太近"。 */
    | "unknown";

export type ParamBlockEditResult =
    | { readonly ok: true; readonly state: ParamBlockState; readonly blockId: string }
    | { readonly ok: false; readonly reason: ParamBlockEditFailureReason };

export interface ParamBlockSplitInput {
    readonly state: ParamBlockState;
    readonly block: ParamBlock;
    readonly sec: number;
    /**
     * **与派生视图同源**的 clip 快照。
     *
     * 块是源域锚定的，切点要反投影回源域才能改写成员区间；用另一份 clips
     * （例如上一次提交留下的副本）会让边界判定与块几何分叉。
     */
    readonly clips: readonly ParamBlockSourceClip[];
}

export interface ParamBlockResizeNeighborPlan {
    readonly id: string;
    readonly startSec: number;
    readonly endSec: number;
    /** 被拖块已经完整越过该块，松手时应删除它。 */
    readonly remove: boolean;
}

export interface ParamBlockResizePlan {
    /** 被拖块的新区间。 */
    readonly block: { readonly startSec: number; readonly endSec: number };
    /** 被部分缩短或完整覆盖的右侧 / 左侧块，按拖动方向排列。 */
    readonly neighbors: readonly ParamBlockResizeNeighborPlan[];
    /** 当前块已缩到最小宽度以下，松手时应删除。 */
    readonly removeBlock: boolean;
}
export interface ParamBlockResizeInput {
    readonly state: ParamBlockState;
    readonly block: ParamBlock;
    readonly edge: "left" | "right";
    readonly targetSec: number;
    readonly clips: readonly ParamBlockSourceClip[];
    readonly targetSpan?: { readonly startSec: number; readonly endSec: number } | null;
    readonly neighborSpan?: { readonly startSec: number; readonly endSec: number } | null;
    readonly neighborSpans?: readonly {
        readonly id: string;
        readonly startSec: number;
        readonly endSec: number;
        readonly remove?: boolean;
    }[];
    readonly neighborBlocks?: readonly ParamBlock[];
    readonly neighborStartSec?: number | null;
    readonly neighborEndSec?: number | null;
    readonly neighbor?: ParamBlock | null;
    readonly minBlockSec?: number;
}

export interface ParamBlockCreateInput {
    readonly state: ParamBlockState;
    readonly clip?: ParamBlockSourceClip;
    readonly clips?: readonly ParamBlockSourceClip[];
    readonly startSec: number;
    readonly endSec: number;
    readonly value: number | null;
    readonly blockId?: string;
    readonly minBlockSec?: number;
}

/** 自动块的最小宽度（秒）。 */
const MIN_AUTO_BLOCK_SEC = 0.02;
export const PARAM_BLOCK_ROW_HEIGHT_MULTIPLIER = 1.2;
export const PARAM_BLOCK_TRANSITION_HANDLE_OUTSET_X_PX = 0;
export const PARAM_BLOCK_TRANSITION_HANDLE_OUTSET_Y_PX = 6;
export const PARAM_BLOCK_TRANSITION_HANDLE_LINE_LENGTH_PX = 8;

export function paramBlockTransitionHandleY(
    leftTopPx: number,
    rightTopPx?: number,
    outsetY = PARAM_BLOCK_TRANSITION_HANDLE_OUTSET_Y_PX,
): number {
    return Math.min(leftTopPx, rightTopPx ?? leftTopPx) - outsetY;
}
function finite(value: number, fallback: number): number {
    return Number.isFinite(value) ? value : fallback;
}

/** 派生视图块 → UI 块。UI 只读；任何编辑都必须回到 editSet。 */
function paramBlockFromProjected(
    rootTrackId: string,
    block: ProjectedParamBlock,
    framePeriodMs: number,
): ParamBlock {
    const toSecLocal = (frame: number) => (frame * Math.max(0.1, framePeriodMs)) / 1000;
    return {
        id: block.id,
        ...(block.userBlockId === undefined ? {} : { userBlockId: block.userBlockId }),
        rootTrackId,
        kind:
            block.kind === "base"
                ? "auto"
                : block.kind === "grouped"
                  ? "grouped"
                  : block.kind === "manual"
                    ? "manual"
                    : "user",
        members: block.members.map((member) => ({
            clipId: member.clipId,
            takeId: member.takeId,
            syllableId: member.syllableId,
            sourceStartSec: member.sourceStartSec,
            sourceEndSec: member.sourceEndSec,
            cycle: member.cycle,
            startSec: toSecLocal(member.startFrame),
            endSec: toSecLocal(member.endFrameExclusive),
        })),
        startSec: toSecLocal(block.startFrame),
        endSec: toSecLocal(block.endFrameExclusive),
        averageValue: null,
        manualValue: block.value ?? null,
        transitionSec:
            block.transitionFrames === undefined
                ? undefined
                : (block.transitionFrames * Math.max(0.1, framePeriodMs)) / 1000,
        projected: block,
    };
}

function withEdits(state: ParamBlockState, edits: ParamBlockEditSet): ParamBlockState {
    return { ...state, editSet: edits };
}

/**
 * 从**调用方现算的**派生视图生成面板块。
 *
 * 不接收 `ParamBlockState`：状态里没有几何，任何"从状态读块"的入口都会
 * 引诱调用方去缓存第二份视图。`view === null` 表示"还没有根轨道"，此时没有块。
 */
export function materializeParamBlocks(args: {
    readonly rootTrackId: string;
    readonly view: ParamBlockView | null;
}): ParamBlock[] {
    const view = args.view;
    if (view === null) return [];
    return view.blocks.map((block) =>
        paramBlockFromProjected(args.rootTrackId, block, view.framePeriodMs),
    );
}

export function sortParamBlocksByTime<
    T extends { readonly startSec: number; readonly endSec: number },
>(blocks: readonly T[]): T[] {
    return blocks.slice().sort((a, b) => a.startSec - b.startSec || a.endSec - b.endSec);
}

function blocksAreAdjacent(
    a: { readonly startSec: number; readonly endSec: number },
    b: { readonly startSec: number; readonly endSec: number },
): boolean {
    return Math.abs(a.endSec - b.startSec) <= 1e-6;
}

export function paramBlockTransitionHandles<
    T extends {
        readonly id: string;
        readonly startSec: number;
        readonly endSec: number;
        readonly transitionSec?: number;
    },
>(
    blocks: readonly T[],
    framePeriodMs = 5,
): Array<{
    readonly leftId: string;
    readonly rightId: string;
    readonly boundarySec: number;
    readonly widthSec: number;
    readonly sec: number;
    readonly transitionSec: number;
}> {
    const sorted = sortParamBlocksByTime(blocks);
    const out = [];
    for (let index = 0; index + 1 < sorted.length; index += 1) {
        const left = sorted[index];
        const right = sorted[index + 1];
        if (!blocksAreAdjacent(left, right)) continue;
        out.push({
            leftId: left.id,
            rightId: right.id,
            boundarySec: right.startSec,
            widthSec: left.transitionSec ?? 0,
            sec: right.startSec,
            transitionSec: left.transitionSec ?? 0,
        });
    }
    void framePeriodMs;
    return out;
}

function paramBlockRangesOverlap(
    ranges: readonly { readonly startSec: number; readonly endSec: number }[],
): boolean {
    const ordered = ranges.slice().sort((a, b) => a.startSec - b.startSec || a.endSec - b.endSec);
    for (let index = 1; index < ordered.length; index += 1) {
        if (ordered[index]!.startSec < ordered[index - 1]!.endSec - 1e-6) return true;
    }
    return false;
}

export function canGroupParamBlocks(
    blocks: readonly ParamBlock[],
    selectedIds: readonly string[],
): boolean {
    const identities = new Set<string>();
    for (const block of blocks) {
        if (selectedIds.some((id) => blockMatchesLogicalId(block, id))) {
            identities.add(logicalBlockId(block));
        }
    }
    if (identities.size < 2) return false;
    const selected = [...identities]
        .map((id) => {
            const runs = blocks.filter((block) => logicalBlockId(block) === id);
            return {
                id,
                runs,
                startSec: Math.min(...runs.map((block) => block.startSec)),
                hasRoot: runs.some((block) => block.members.length === 0),
            };
        })
        .sort((a, b) => a.startSec - b.startSec);
    if (
        selected.some((block) => block.runs[0]?.rootTrackId !== selected[0]?.runs[0]?.rootTrackId)
    ) {
        return false;
    }
    const hasRoot = selected.some((block) => block.hasRoot);
    const hasTake = selected.some((block) => !block.hasRoot);
    if (hasRoot && hasTake) return false;
    return !paramBlockRangesOverlap(selected.flatMap((block) => block.runs));
}

export function groupParamBlocks(args: {
    readonly state: ParamBlockState;
    readonly blocks: readonly ParamBlock[];
    readonly selectedBlockIds: readonly string[];
    /** 与派生视图同源的 clip 快照（编组要按它把源域区间并起来）。 */
    readonly clips: readonly ParamBlockSourceClip[];
}): BlockGroupResult {
    const identities = new Set<string>();
    for (const block of args.blocks) {
        if (args.selectedBlockIds.some((id) => blockMatchesLogicalId(block, id))) {
            identities.add(logicalBlockId(block));
        }
    }
    if (identities.size < 2) return { ok: false, reason: "empty" };
    const selected = args.blocks.filter((block) => identities.has(logicalBlockId(block)));
    if (selected.some((block) => block.rootTrackId !== selected[0]?.rootTrackId)) {
        return { ok: false, reason: "different_root" };
    }
    const hasRoot = selected.some((block) => block.members.length === 0);
    const hasTake = selected.some((block) => block.members.length > 0);
    if (hasRoot && hasTake) return { ok: false, reason: "mixed_geometry" };
    if (paramBlockRangesOverlap(selected)) {
        return { ok: false, reason: "overlap" };
    }
    const projected = selected
        .map((block) => block.projected)
        .filter((block): block is ProjectedParamBlock => block !== undefined);
    if (projected.length === 0) return { ok: false, reason: "nothing_to_group" };
    const framePeriodMs = args.state.framePeriodMs;
    try {
        const edits = groupProjectedBlocks(
            args.clips,
            args.state.editSet,
            projected,
            framePeriodMs,
        );
        return { ok: true, state: withEdits(args.state, edits) };
    } catch (error) {
        const message = error instanceof Error ? error.message : "";
        if (message.includes("overlap")) return { ok: false, reason: "overlap" };
        if (message.includes("already grouped")) return { ok: false, reason: "already_grouped" };
        if (message.includes("root and take")) return { ok: false, reason: "mixed_geometry" };
        return { ok: false, reason: "nothing_to_group" };
    }
}
export function ungroupParamBlocks(
    state: ParamBlockState,
    blockIds: readonly string[],
): ParamBlockState {
    let edits = state.editSet;
    for (const id of blockIds) edits = ungroupUserBlock(edits, id);
    return withEdits(state, edits);
}

function editFailureFromError(error: unknown): ParamBlockEditFailureReason {
    if (error instanceof ParamBlockEditError) {
        return error.code === "no_source" ? "no_members" : "too_short";
    }
    if (error instanceof ParamBlockPlanError) {
        switch (error.code) {
            case "split_too_close_to_edge":
                return "too_close_to_edge";
            case "split_at_block_edge":
            case "split_outside_block":
            case "resize_too_short":
                return "too_short";
            case "resize_blocked_by_other_source":
                return "resize_blocked";
            case "block_not_found":
            case "block_deleted":
                return "not_found";
            case "loop_span_crosses_cycle":
                return "multi_fragment_unsupported";
            default:
                return "unknown";
        }
    }
    const message = error instanceof Error ? error.message : "";
    if (message.includes("no source") || message.includes("no members")) return "no_members";
    if (message.includes("not found") || message.includes("missing")) return "not_found";
    return "unknown";
}

export function splitParamBlock(input: ParamBlockSplitInput): ParamBlockEditResult {
    const block = input.block.projected;
    if (block === undefined) return { ok: false, reason: "not_found" };
    const framePeriodMs = input.state.framePeriodMs;
    const frame = toFrame(input.sec, framePeriodMs);
    if (frame <= block.startFrame || frame >= block.endFrameExclusive) {
        return { ok: false, reason: "too_short" };
    }
    try {
        const plan = planSplitProjectedBlock(
            input.clips,
            input.state.editSet,
            block,
            frame,
            framePeriodMs,
        );
        return { ok: true, blockId: plan.rightBlockId, state: withEdits(input.state, plan.edits) };
    } catch (error) {
        return { ok: false, reason: editFailureFromError(error) };
    }
}

export function createParamBlock(input: ParamBlockCreateInput): ParamBlockEditResult {
    const clips = input.clips ?? (input.clip === undefined ? [] : [input.clip]);
    const framePeriodMs = input.state.framePeriodMs;
    try {
        const edits = createUserBlock(
            clips,
            input.state.editSet,
            toFrame(input.startSec, framePeriodMs),
            toFrame(input.endSec, framePeriodMs),
            input.value,
            framePeriodMs,
            input.blockId,
        );
        const actualId = input.blockId ?? edits.blocks[edits.blocks.length - 1]?.id ?? "created";
        return {
            ok: true,
            blockId: actualId,
            state: withEdits(input.state, edits),
        };
    } catch (error) {
        return { ok: false, reason: editFailureFromError(error) };
    }
}

export function deleteParamBlocks(args: {
    readonly state: ParamBlockState;
    readonly blocks: readonly ParamBlock[];
    readonly selectedBlockIds: readonly string[];
    /** 与派生视图同源的 clip 快照（删除要按它把源域区间拆开）。 */
    readonly clips: readonly ParamBlockSourceClip[];
}): ParamBlockEditResult {
    const identities = new Set<string>();
    for (const block of args.blocks) {
        if (args.selectedBlockIds.some((id) => blockMatchesLogicalId(block, id))) {
            identities.add(logicalBlockId(block));
        }
    }
    const selected = args.blocks.filter((block) => identities.has(logicalBlockId(block)));
    if (selected.length === 0) return { ok: false, reason: "not_found" };
    const projected = selected
        .map((block) => block.projected)
        .filter((block): block is ProjectedParamBlock => block !== undefined);
    if (projected.length === 0) return { ok: false, reason: "not_found" };
    const framePeriodMs = args.state.framePeriodMs;
    try {
        const result = deleteProjectedBlocks(
            args.clips,
            args.state.editSet,
            projected,
            framePeriodMs,
        );
        return {
            ok: true,
            blockId: selected[0]!.id,
            state: {
                ...withEdits(args.state, result.edits),
                lastDeletedBlocks: result.deleted.map((snapshot) => ({
                    ...snapshot,
                    rootTrackId: args.state.rootTrackId,
                })),
            },
        };
    } catch (error) {
        return { ok: false, reason: editFailureFromError(error) };
    }
}
export function restoreDeletedParamBlocks(args: {
    readonly state: ParamBlockState;
}): ParamBlockEditResult {
    const deleted = (args.state.lastDeletedBlocks ?? []).filter(
        (snapshot) =>
            snapshot.rootTrackId === undefined ||
            args.state.rootTrackId === undefined ||
            snapshot.rootTrackId === args.state.rootTrackId,
    );
    if (deleted.length === 0) return { ok: false, reason: "not_found" };
    try {
        const edits = restoreDeletedBlocks(args.state.editSet, deleted);
        return {
            ok: true,
            blockId: deleted[0].block.id,
            state: { ...withEdits(args.state, edits), lastDeletedBlocks: [] },
        };
    } catch (error) {
        return { ok: false, reason: editFailureFromError(error) };
    }
}

/**
 * 参数命令改变曲线后，pitch 块的持久化中心不再代表曲线重心 —— 必须清掉，
 * 让块高重新从新曲线派生（否则块会悬空在旧高度）。
 *
 * 只改 `value`，不动几何；调用方应把它当作**布局**变更提交（不进撤销栈）。
 */
export function clearParamBlockValues(
    state: ParamBlockState,
    blockIds?: readonly string[],
): ParamBlockState {
    const ids = blockIds === undefined ? null : new Set(blockIds);
    let changed = false;
    const blocks = state.editSet.blocks.map((block) => {
        if (block.value === undefined || block.value === null) return block;
        if (ids !== null && !ids.has(block.id)) return block;
        changed = true;
        return { ...block, value: null };
    });
    if (!changed) return state;
    return {
        ...state,
        editSet: { ...state.editSet, revision: state.editSet.revision + 1, blocks },
    };
}

/** 把拖拽得到的逻辑中心写回可持久化的用户块 value；自动块没有 value，保持原样。 */
export function setParamBlockLogicalCenters(args: {
    readonly state: ParamBlockState;
    readonly blocks: readonly ParamBlock[];
    readonly centers: readonly (number | null)[] | ReadonlyMap<string, number | null>;
}): ParamBlockState {
    let changed = false;
    /*
     * 多 run 块的 view id 是 `parent:0` / `parent:1`，editSet 里只有 `parent`。
     * 只按 view id 查会让 value 永远写不回去（拖动看着动了，落库没有）。这里同时
     * 按 parent id 与任意一个 run 的 id 查，并让同一块的所有成员共享一个逻辑中心。
     */
    const centerFor = (blockId: string, index: number): number | null => {
        if (args.centers instanceof Map) {
            const direct = args.centers.get(blockId);
            if (direct !== undefined) return direct;
            const run = args.blocks.find(
                (item) => item.userBlockId === blockId || item.id === blockId,
            );
            return run === undefined ? null : (args.centers.get(run.id) ?? null);
        }
        const list = args.centers as readonly (number | null)[];
        if (index >= 0 && list[index] !== undefined) return list[index] ?? null;
        const run = args.blocks.find((item) => item.userBlockId === blockId || item.id === blockId);
        if (run === undefined) return null;
        const runIndex = args.blocks.findIndex((item) => item.id === run.id);
        return runIndex >= 0 ? (list[runIndex] ?? null) : null;
    };
    const blocks = args.state.editSet.blocks.map((block) => {
        const index = args.blocks.findIndex(
            (item) => item.id === block.id || item.userBlockId === block.id,
        );
        const center = centerFor(block.id, index);
        if (center === null || center === undefined || !Number.isFinite(center)) return block;
        if (block.value === center) return block;
        changed = true;
        return { ...block, value: center };
    });
    if (!changed) return args.state;
    const edits = { ...args.state.editSet, revision: args.state.editSet.revision + 1, blocks };
    return withEdits(args.state, edits);
}

export function setParamBlockTransition(args: {
    readonly state: ParamBlockState;
    readonly left: ParamBlock;
    readonly right: ParamBlock;
    readonly transitionSec: number;
    /** 与派生视图同源的 clip 快照（过渡宽度按源域速率折算成帧）。 */
    readonly clips: readonly ParamBlockSourceClip[];
}): ParamBlockEditResult {
    if (args.left.projected === undefined) return { ok: false, reason: "not_found" };
    const framePeriodMs = args.state.framePeriodMs;
    try {
        const edits = setProjectedBlockTransition(
            args.clips,
            args.state.editSet,
            args.left.projected,
            args.transitionSec,
            framePeriodMs,
        );
        return { ok: true, blockId: args.left.id, state: withEdits(args.state, edits) };
    } catch {
        return { ok: false, reason: "not_found" };
    }
}

/** 视图块与当前参数曲线采样后的 UI 块。 */
export function blocksFromViewForPanel(rootTrackId: string, view: ParamBlockView): ParamBlock[] {
    return view.blocks.map((block) =>
        paramBlockFromProjected(rootTrackId, block, view.framePeriodMs),
    );
}

/**
 * 拖动时的**纵向值偏移**（唯一来源，预览与提交共用）。
 *
 * 音高 / 音分 / 度数 / 张力等是线性刻度，直接照指针位移。
 *
 * 对块拖动来说，`valueDelta` 本身就是指针在**值域纵轴**上的位移；把它再按
 * “锚点值”缩放会让块只移动一部分距离（锚点越小时越明显）。因此所有参数
 * 统一做线性偏移，保证块中心与指针同步；值域钳制仍由调用方的写入入口负责。
 */
export function clampParamBlockDragValue(args: {
    readonly param: string;
    readonly original: number;
    readonly valueDelta: number;
}): number {
    if (!Number.isFinite(args.original) || !Number.isFinite(args.valueDelta)) {
        return args.original;
    }
    const next = args.original + args.valueDelta;
    if (args.param === "pitch" && args.original === 0) return 0;
    if (isDynParam(args.param)) {
        // Negative dyn values are the "follow original" sentinel; a drag must
        // not materialize them. Positive targets clamp to the descriptor range
        // instead of collapsing below zero into the sentinel.
        if (args.original < 0) return args.original;
        return Math.min(DYN_VALUE_MAX, Math.max(DYN_VALUE_MIN, next));
    }
    return clampParamWriteValue(args.param, next);
}

function paramBlockDragValue(args: {
    readonly param: string;
    readonly original: number;
    readonly anchorValue: number;
    readonly valueDelta: number;
}): number {
    return clampParamBlockDragValue({
        param: args.param,
        original: args.original,
        valueDelta: args.valueDelta,
    });
}

/**
 * 块间圆滑：以相接点为对称中心，向左、右各作用 `transitionSec`。
 *
 * 过渡不是把原曲线替换成一条独立贝塞尔；它与相接点两侧的**原曲线**
 * 混合。左边界处权重 0（保留左块原走势），到相接点附近权重升高并平滑
 * 拉向目标值，再向右回落到右块原走势。两侧都改，但都不被覆盖。
 */
export function buildParamBlockTransitionSegments(args: {
    /** 相接点帧（左块右边界 = 右块左边界）。 */
    readonly boundaryFrame: number;
    /** 过渡宽度（帧），相接点左右各延展这么多帧。 */
    readonly widthFrames: number;
    readonly targetValue: number;
    /** 左块最左帧；过渡不能写出这个范围。 */
    readonly leftStartFrame: number;
    /** 右块最右帧；过渡不能写出这个范围。 */
    readonly rightEndFrame: number;
    /** 相接点两侧的原曲线；缺失时退回左 / 右块代表值。 */
    readonly sourceAt?: (frame: number) => number;
    readonly leftValue: number;
    readonly rightValue: number;
}): Array<{ startFrame: number; values: number[] }> {
    const width = Math.max(0, Math.floor(args.widthFrames));
    if (width <= 0) return [];
    const start = Math.max(args.leftStartFrame, args.boundaryFrame - width);
    const end = Math.min(args.rightEndFrame, args.boundaryFrame + width - 1);
    if (end < start) return [];
    const values: number[] = [];
    for (let frame = start; frame <= end; frame += 1) {
        /*
         * 以相接点为峰值的三角形权重：两端为 0、相接点为 1。
         * 目标值不是“覆盖”，只决定这一峰把原曲线拉多近。
         */
        const distance = Math.abs(frame - args.boundaryFrame);
        const t = Math.max(0, Math.min(1, 1 - distance / Math.max(1, width)));
        const smooth = t * t * (3 - 2 * t);
        const source = args.sourceAt?.(frame);
        const base = Number.isFinite(source)
            ? (source as number)
            : frame < args.boundaryFrame
              ? args.leftValue
              : args.rightValue;
        values.push(base + (args.targetValue - base) * smooth);
    }
    return values.length > 0 ? [{ startFrame: start, values }] : [];
}

/**
 * 手柄拖动时计算 live 预览：读取当前窗口的右块原曲线，按平滑权重拉向目标值。
 *
 * 与提交共用 `buildParamBlockTransitionSegments`，因此预览和最终写回后端
 * 是同一套数学；只写相接点右侧，左侧块永远不受影响。
 */
export function buildParamBlockTransitionPreview(args: {
    readonly boundaryFrame: number;
    readonly widthFrames: number;
    readonly targetValue: number;
    readonly leftStartFrame: number;
    readonly rightEndFrame: number;
    readonly leftValue: number;
    readonly rightValue: number;
    readonly paramView: {
        readonly startFrame: number;
        readonly stride: number;
        readonly edit: readonly number[];
    };
}): Array<{ startFrame: number; values: number[] }> {
    const pv = args.paramView;
    const stride = Math.max(1, Math.floor(pv.stride));
    const sourceAt = (frame: number): number => {
        const index = Math.round((frame - pv.startFrame) / stride);
        const value = pv.edit[index];
        return value !== undefined && Number.isFinite(value) ? value : args.targetValue;
    };
    return buildParamBlockTransitionSegments({ ...args, sourceAt });
}

/** 把全分辨率写入片段立即映射回当前参数窗口，供提交后的本地回显使用。 */
export function applyFullResSegmentsToParamView<
    T extends {
        readonly startFrame: number;
        readonly stride: number;
        readonly edit: readonly number[];
    },
>(args: {
    readonly paramView: T;
    readonly segments: readonly {
        readonly startFrame: number;
        readonly values: readonly number[];
    }[];
}): T & { readonly edit: number[] } {
    const next = args.paramView.edit.slice();
    const stride = Math.max(1, Math.round(args.paramView.stride) || 1);
    for (const segment of args.segments) {
        const startFrame = Math.floor(segment.startFrame);
        const firstIndex = Math.max(
            0,
            Math.ceil((startFrame - args.paramView.startFrame) / stride),
        );
        const lastIndex = Math.min(
            next.length - 1,
            Math.floor(
                (startFrame + segment.values.length - 1 - args.paramView.startFrame) / stride,
            ),
        );
        for (let targetIndex = firstIndex; targetIndex <= lastIndex; targetIndex += 1) {
            const frame = args.paramView.startFrame + targetIndex * stride;
            const value = segment.values[frame - startFrame];
            if (value === undefined) continue;
            next[targetIndex] = value;
        }
    }
    return { ...args.paramView, edit: next };
}

/** Compare committed values only at the frames actually sampled by the backend. */
export function paramViewContainsCommittedSegments(args: {
    readonly paramView: {
        readonly startFrame: number;
        readonly stride: number;
        readonly edit: readonly number[];
    };
    readonly segments: readonly {
        readonly startFrame: number;
        readonly values: readonly number[];
    }[];
    readonly tolerance?: number;
}): boolean {
    const stride = Math.max(1, Math.round(args.paramView.stride) || 1);
    const tolerance = Math.max(0, args.tolerance ?? 1e-5);
    for (const segment of args.segments) {
        const startFrame = Math.floor(segment.startFrame);
        const endFrame = startFrame + segment.values.length - 1;
        const firstIndex = Math.max(
            0,
            Math.ceil((startFrame - args.paramView.startFrame) / stride),
        );
        const lastIndex = Math.min(
            args.paramView.edit.length - 1,
            Math.floor((endFrame - args.paramView.startFrame) / stride),
        );
        for (let index = firstIndex; index <= lastIndex; index += 1) {
            const frame = args.paramView.startFrame + index * stride;
            const value = segment.values[frame - startFrame];
            const current = args.paramView.edit[index];
            if (
                value === undefined ||
                current === undefined ||
                !Number.isFinite(current) ||
                !Number.isFinite(value) ||
                Math.abs(current - value) > tolerance
            ) {
                return false;
            }
        }
    }
    return true;
}

/** 快速拖动时 pointerup 可能没有对应的最后一条 pointermove；松手坐标必须优先。 */
export function resolveParamBlockDragRelease(args: {
    readonly didDrag: boolean;
    readonly startClientY: number;
    readonly lastMoveClientY: number;
    readonly releaseClientY: number;
    readonly startClientX?: number;
    readonly lastMoveClientX?: number;
    readonly releaseClientX?: number;
}): { readonly didDrag: boolean; readonly clientY: number } {
    const clientY = Number.isFinite(args.releaseClientY)
        ? args.releaseClientY
        : args.lastMoveClientY;
    const hasX =
        args.startClientX !== undefined &&
        Number.isFinite(args.startClientX) &&
        Number.isFinite(args.releaseClientX ?? args.lastMoveClientX);
    const clientX = hasX ? (args.releaseClientX ?? args.lastMoveClientX)! : args.startClientX;
    const moved = hasX
        ? Math.hypot(clientX! - args.startClientX!, clientY - args.startClientY) > 3
        : Math.abs(clientY - args.startClientY) > 3;
    return { didDrag: args.didDrag || (Number.isFinite(clientY) && moved), clientY };
}

/**
 * 拖动期间块的显示中心只由**手势起点 + 指针位移**决定。
 *
 * 在此之前，块的 y 位置由实时曲线主簇反算。即使最终曲线正确，低缩放的
 * stride、局部静音帧和异步窗口切换也会让每一帧算出的主簇略有不同，表现为
 * 块在上下两个像素位置之间闪烁；松手后的全分辨率曲线还会再算一次，造成
 * 最后一次回跳。拖动和交接期间直接投影这一个确定值，曲线仍独立实时更新。
 */
export function projectParamBlockDragDisplayValue(args: {
    readonly startValue: number | null;
    readonly valueDelta: number;
    readonly param: string;
    readonly snapToSemitone: boolean;
}): number | null {
    if (
        args.startValue === null ||
        !Number.isFinite(args.startValue) ||
        !Number.isFinite(args.valueDelta)
    ) {
        return null;
    }
    const value = clampParamWriteValue(args.param, args.startValue + args.valueDelta);
    return args.param === "pitch" && args.snapToSemitone ? Math.round(value) : value;
}

/**
 * 创建块时把对应曲线整体平移到目标值，保留原有起伏与滑音轮廓。
 * 音高 0 表示无声帧，保持不动。
 */
export function shiftParamBlockCurveValuesToTarget(args: {
    readonly values: readonly number[];
    readonly target: number;
    readonly param: string;
    readonly curveOwned?: boolean;
    readonly flatten?: boolean;
}): number[] {
    if (!Number.isFinite(args.target)) return args.values.slice();
    if (args.flatten) {
        return args.values.map((value) => {
            if (!Number.isFinite(value)) return value;
            if (args.param === "pitch") return value > 0 ? args.target : value;
            return args.target;
        });
    }
    const editable = args.values.filter(
        (value) => Number.isFinite(value) && (args.param !== "pitch" || value > 0),
    );
    if (editable.length === 0)
        return args.curveOwned ? args.values.map(() => args.target) : args.values.slice();
    const center =
        modeCenterOfValues(editable, {
            param: args.param,
            isPitch: args.param === "pitch",
        }) ?? editable[0];
    const delta = args.target - (center as number);
    return args.values.map((value) =>
        Number.isFinite(value) && (args.param !== "pitch" || value > 0) ? value + delta : value,
    );
}

/**
 * 把选中的块投影成逐帧写入片段。
 *
 * - 时间范围取块的真实区间（绘制用的缝隙不参与编辑）；
 * - 每个块独立取 `sourceAt`，所以多选时不会把别的块的曲线当基准；
 * - 重叠帧后写覆盖先写（稳定排序，可复现）；
 * - `valueDelta` 由 {@link paramBlockDragValue} 逐帧应用，pitch 的 0 哨兵保持 0。
 */
export function buildParamBlockDragSegments(args: {
    readonly blocks: readonly { readonly members: readonly ParamBlockMember[] }[];
    readonly framePeriodMs: number;
    readonly param: string;
    readonly anchorValue: number;
    readonly valueDelta: number;
    readonly sourceAt: (frame: number) => number;
    readonly fillEmptyPitch?: boolean;
}): Array<{ startFrame: number; values: number[] }> {
    const fp = Math.max(1e-6, args.framePeriodMs);
    const spans = args.blocks
        .flatMap((block) => block.members)
        .map((member) => {
            /*
             * 块时间区间必须转成**同一采样网格**的帧下标；预览的 stride 可能
             * 大于 1，写入函数会按 `(frame - startFrame) / stride` 反解。
             * 若这里按 1 帧对齐而预览按 stride 网格对齐，边缘会差一帧，
             * 拖动时块高度就会落在两个位置之间闪。
             */
            const span = paramBlockFrameSpan(member.startSec, member.endSec, fp);
            return { startFrame: span.startFrame, endFrame: span.endFrameExclusive - 1 };
        })
        .sort((a, b) => a.startFrame - b.startFrame);
    if (spans.length === 0) return [];

    const merged: Array<{ startFrame: number; endFrame: number }> = [];
    for (const span of spans) {
        const previous = merged[merged.length - 1];
        if (previous !== undefined && span.startFrame <= previous.endFrame) {
            previous.endFrame = Math.max(previous.endFrame, span.endFrame);
        } else {
            merged.push({ ...span });
        }
    }

    return merged
        .map((span) => {
            /*
             * 先把源值采一遍并缓存，再判断整段是否全静音。旧实现为了
             * `emptyPitch` 先跑一次 Array.from(sourceAt)，随后映射循环再跑
             * 一次；长块 / 高采样率下这是 O(2n) 加一份临时数组。
             */
            const length = span.endFrame - span.startFrame + 1;
            const sourceValues = new Array<number>(length);
            let hasAudiblePitch = false;
            for (let i = 0; i < length; i += 1) {
                const source = args.sourceAt(span.startFrame + i);
                sourceValues[i] = source;
                if (source > 0) hasAudiblePitch = true;
            }
            const emptyPitch =
                args.fillEmptyPitch === true && args.param === "pitch" && !hasAudiblePitch;
            const values: number[] = new Array<number>(length);
            for (let i = 0; i < length; i += 1) {
                const source = sourceValues[i];
                if (emptyPitch) {
                    values[i] = paramBlockDragValue({
                        param: args.param,
                        original: args.anchorValue,
                        anchorValue: args.anchorValue,
                        valueDelta: args.valueDelta,
                    });
                    continue;
                }
                if (args.param === "pitch" && source <= 0) {
                    values[i] = source;
                    continue;
                }
                values[i] = paramBlockDragValue({
                    param: args.param,
                    original: source,
                    anchorValue: args.anchorValue,
                    valueDelta: args.valueDelta,
                });
            }
            return { startFrame: span.startFrame, values };
        })
        .filter((segment) => segment.values.length > 0);
}

/**
 * 拖动块的**实时预览**：只返回受影响的逐帧值。
 *
 * 与提交使用同一 `paramBlockDragValue` 与同一区间合并逻辑；数据源固定为
 * **手势开始时的已提交曲线**（`pv.orig`），所以预览不会把上一帧结果再累加一次。
 * 降采样的预览与全分辨率提交只在采样密度上不同，公式不分叉。
 */
export function buildParamBlockPreviewSegments(args: {
    readonly blocks: readonly { readonly members: readonly ParamBlockMember[] }[];
    readonly framePeriodMs: number;
    readonly param: string;
    readonly anchorValue: number;
    readonly valueDelta: number;
    readonly fillEmptyPitch?: boolean;
    readonly pv: {
        readonly startFrame: number;
        readonly stride: number;
        readonly edit: readonly number[];
        readonly orig: readonly number[];
    };
}): Array<{ startFrame: number; values: number[] }> {
    const step = Math.max(1, Math.floor(args.pv.stride));
    return buildParamBlockDragSegments({
        blocks: args.blocks,
        framePeriodMs: args.framePeriodMs,
        param: args.param,
        anchorValue: args.anchorValue,
        valueDelta: args.valueDelta,
        fillEmptyPitch: args.fillEmptyPitch,
        sourceAt: (frame) => {
            const index = Math.round((frame - args.pv.startFrame) / step);
            /*
             * 预览必须一直以**手势起点**的已提交曲线为基准。若读 live edit，
             * 上一帧的预览会成为下一帧的原值，拖动会被累积放大；松手提交
             * 则从真实曲线重算，落点因此与控制路径不一致。
             */
            const base = args.pv.orig;
            return index >= 0 && index < base.length ? Number(base[index]) || 0 : 0;
        },
    });
}

/**
 * 块命中测试：必须同时落在时间区间与**可见矩形**内。
 *
 * 【为什么不能只按时间列判断】块有真实高度；只按 x 选中会让用户点到块外、
 * 甚至点到曲线空白处仍被当成选块，多选时尤其容易误操作。
 */
export function hitTestParamBlocks<
    T extends {
        readonly id: string;
        readonly startSec: number;
        readonly endSec: number;
        readonly topPx: number;
        readonly heightPx: number;
    },
>(args: {
    readonly blocks: readonly T[];
    readonly sec: number;
    readonly localY: number;
    /** 命中容差（CSS px）：细块比 1px 边框更容易点中。 */
    readonly slopPx?: number;
}): T | null {
    const sec = finite(args.sec, Number.NaN);
    const y = finite(args.localY, Number.NaN);
    if (!Number.isFinite(sec) || !Number.isFinite(y)) return null;
    const slop = Math.max(0, finite(args.slopPx ?? 2, 2));
    // 反向遍历：后绘制的块在上，命中优先给用户实际看到的那个。
    for (let index = args.blocks.length - 1; index >= 0; index -= 1) {
        const block = args.blocks[index];
        if (block === undefined) continue;
        const startSec = Math.min(block.startSec, block.endSec);
        const endSec = Math.max(block.startSec, block.endSec);
        const top = block.topPx;
        const bottom = block.topPx + block.heightPx;
        if (sec < startSec || sec > endSec) continue;
        if (y < top - slop || y > bottom + slop) continue;
        return block;
    }
    return null;
}

/** 同时间位置、不同纵向高度的重叠块：优先命中 y 距离最近的，而不是数组最后一个。 */
export function hitTestParamBlocksAtPoint<
    T extends {
        readonly id: string;
        readonly startSec: number;
        readonly endSec: number;
        readonly topPx: number;
        readonly heightPx: number;
    },
>(args: {
    readonly blocks: readonly T[];
    readonly sec: number;
    readonly localY: number;
    readonly slopPx?: number;
}): T | null {
    const sec = finite(args.sec, Number.NaN);
    const y = finite(args.localY, Number.NaN);
    if (!Number.isFinite(sec) || !Number.isFinite(y)) return null;
    const slop = Math.max(0, finite(args.slopPx ?? 4, 4));
    let best: { block: T; distance: number } | null = null;
    for (const block of args.blocks) {
        if (sec < block.startSec || sec > block.endSec) continue;
        const center = block.topPx + block.heightPx / 2;
        const distance = Math.abs(y - center);
        if (distance > block.heightPx / 2 + slop) continue;
        if (best === null || distance < best.distance) best = { block, distance };
    }
    return best?.block ?? null;
}

/** 新块压到的旧块 id：创建预览与提交共用同一个“整块让位”判定。 */
function blocksCoveredByParamBlockCreate<
    T extends { readonly id: string; readonly startSec: number; readonly endSec: number },
>(args: {
    readonly blocks: readonly T[];
    readonly startSec: number;
    readonly endSec: number;
}): readonly string[] {
    const ids = new Set<string>();
    for (const block of args.blocks) {
        if (block.id === "__create__") continue;
        if (block.endSec <= args.startSec + 1e-9 || block.startSec >= args.endSec - 1e-9) {
            continue;
        }
        ids.add(block.id);
    }
    return [...ids];
}

/**
 * 新建块时旧块让位的**预览几何**：与提交语义一致，压住的旧块整块消失。
 *
 * 预览使用当前可见块（而不是持久化记录），让用户在松手前就看见最终布局；
 * 不能拆成左右残段，否则松手后残段会突然消失，造成“闪一下”的错觉。
 */
function previewParamBlockCreateLayout<
    T extends {
        readonly id: string;
        readonly startSec: number;
        readonly endSec: number;
        readonly averageValue?: number | null;
        readonly selected: boolean;
        readonly displayCenterValue?: number | null;
        /** pitch 块中心是否吸附半音行心；实时拖动时显式 false。 */
        readonly displayCenterSnapToSemitone?: boolean;
    },
>(args: {
    readonly blocks: readonly T[];
    readonly framePeriodMs?: number;
    readonly startSec: number;
    readonly endSec: number;
    readonly createId: string;
    readonly value: number | null;
}): Array<
    | T
    | {
          readonly id: string;
          readonly startSec: number;
          readonly endSec: number;
          readonly averageValue?: number | null;
          readonly selected: boolean;
          readonly displayCenterValue?: number | null;
      }
> {
    if (args.framePeriodMs) {
        const fp = args.framePeriodMs;
        const before = args.blocks.map((block) => ({
            id: block.id,
            startFrame: Math.round((block.startSec * 1000) / fp),
            endFrameExclusive: Math.round((block.endSec * 1000) / fp),
        }));
        const target = {
            id: args.createId,
            startFrame: Math.round((Math.min(args.startSec, args.endSec) * 1000) / fp),
            endFrameExclusive: Math.round((Math.max(args.startSec, args.endSec) * 1000) / fp),
        };
        const placed: Array<{ id: string; startFrame: number; endFrameExclusive: number }> = [];
        for (const block of before) {
            if (block.id === target.id) continue;
            if (
                block.endFrameExclusive <= target.startFrame ||
                block.startFrame >= target.endFrameExclusive
            ) {
                placed.push(block);
                continue;
            }
            if (block.startFrame < target.startFrame) {
                placed.push({ ...block, endFrameExclusive: target.startFrame });
            }
            if (block.endFrameExclusive > target.endFrameExclusive) {
                placed.push({
                    ...block,
                    id: block.startFrame < target.startFrame ? `${block.id}:right` : block.id,
                    startFrame: target.endFrameExclusive,
                });
            }
        }
        placed.push(target);
        return placed
            .sort((a, b) => a.startFrame - b.startFrame)
            .map((block) => {
                const original =
                    args.blocks.find((old) => old.id === block.id) ??
                    args.blocks.find(
                        (old) =>
                            old.startSec <= (block.startFrame * fp) / 1000 + 1e-9 &&
                            old.endSec >= (block.endFrameExclusive * fp) / 1000 - 1e-9,
                    );
                return {
                    ...original,
                    id: block.id,
                    startSec: (block.startFrame * fp) / 1000,
                    endSec: (block.endFrameExclusive * fp) / 1000,
                    selected: block.id === target.id,
                    averageValue: block.id === target.id ? args.value : original?.averageValue,
                    displayCenterValue:
                        block.id === target.id ? args.value : original?.displayCenterValue,
                };
            });
    }
    const next: Array<
        | T
        | {
              id: string;
              startSec: number;
              endSec: number;
              averageValue?: number | null;
              selected: boolean;
              displayCenterValue?: number | null;
          }
    > = [];
    const coveredIds = new Set(
        blocksCoveredByParamBlockCreate({
            blocks: args.blocks,
            startSec: args.startSec,
            endSec: args.endSec,
        }),
    );
    /*
     * 旧块被新块压住时整块消失，不拆成左右残段：预览阶段必须与松手后的
     * 提交语义一致（提交同样整块删除），否则松手瞬间会看见残段“跳回来”。
     */
    for (const block of args.blocks) {
        if (block.id === args.createId) continue;
        if (coveredIds.has(block.id)) continue;
        next.push(block);
    }
    next.push({
        id: args.createId,
        startSec: Math.min(args.startSec, args.endSec),
        endSec: Math.max(args.startSec, args.endSec),
        averageValue: args.value,
        selected: true,
    });
    return next;
}

/** 创建预览每帧都从按下时的块集重建，拖回后自动恢复曾覆盖的块。 */
export function projectParamBlockCreatePreviewFromBase<
    T extends {
        readonly id: string;
        readonly startSec: number;
        readonly endSec: number;
        readonly averageValue?: number | null;
        readonly selected: boolean;
        readonly displayCenterValue?: number | null;
        readonly displayCenterSnapToSemitone?: boolean;
    },
>(args: {
    readonly baseBlocks: readonly T[];
    readonly framePeriodMs?: number;
    readonly startSec: number;
    readonly endSec: number;
    readonly createId: string;
    readonly value: number | null;
}): ReturnType<typeof previewParamBlockCreateLayout<T>> {
    return previewParamBlockCreateLayout({ ...args, blocks: args.baseBlocks });
}

/**
 * 新建块的持久化值。保持与指针所在半音中心一致；宿主绘制时再加半键视觉偏移，
 * 这样“拖到哪个键”松手后仍然停在那个键上，而不是高/低半音。
 */
export function resolveCreatedParamBlockValue(args: {
    readonly pointerValue: number;
    readonly param: string;
    readonly pitchSnapEnabled: boolean;
    readonly pitchSnapUnit: "semitone" | "scale";
    readonly pitchSnapToleranceCents?: number;
    readonly projectScale?: ScaleLike | null;
    readonly toggleModifierActive: boolean;
}): number {
    const value = Number.isFinite(args.pointerValue) ? args.pointerValue : 0;
    if (args.param !== "pitch") return value;
    const effective = args.toggleModifierActive ? !args.pitchSnapEnabled : args.pitchSnapEnabled;
    if (!effective) return value;
    return getDrawPreviewValue({
        editParam: "pitch",
        rawValue: value,
        effectiveSnap: true,
        pitchSnapUnit: args.pitchSnapUnit,
        projectScale: args.pitchSnapUnit === "scale" ? (args.projectScale ?? undefined) : undefined,
        pitchSnapToleranceCents: args.pitchSnapToleranceCents,
    });
}

/** 创建块可用参数：当前根轨道组内、与区间相交的 Clip（重叠时取中心最近者）。 */
export function clipAtParamBlockCreateSpan<
    T extends {
        readonly startSec: number;
        readonly lengthSec: number;
    },
>(args: {
    readonly clips: readonly T[];
    readonly startSec: number;
    readonly endSec: number;
}): T | null {
    const center = (args.startSec + args.endSec) / 2;
    const centerSec = Number.isFinite(center) ? center : args.startSec;
    let best: { clip: T; contains: number; distance: number } | null = null;
    for (const clip of args.clips) {
        const end = clip.startSec + clip.lengthSec;
        if (args.endSec < clip.startSec - 1e-9 || args.startSec > end + 1e-9) continue;
        const contains = centerSec >= clip.startSec && centerSec <= end ? 0 : 1;
        const distance = Math.abs(centerSec - (clip.startSec + clip.lengthSec / 2));
        if (
            best === null ||
            contains < best.contains ||
            (contains === best.contains && distance < best.distance)
        ) {
            best = { clip, contains, distance };
        }
    }
    return best?.clip ?? null;
}

/**
 * 最近块命中：用于工具刚打开、宿主尚未出首帧时没有块几何的情况。
 *
 * 只按时间选取与指针最近、且横向距离在容差内的块；纵向不参与，因为此时
 * 还没有可信的像素位置。一旦宿主出帧，调用方应优先使用真实矩形命中。
 */
export function nearestParamBlockByTime<
    T extends {
        readonly id: string;
        readonly startSec: number;
        readonly endSec: number;
    },
>(blocks: readonly T[], sec: number, toleranceSec: number): T | null {
    const target = finite(sec, Number.NaN);
    if (!Number.isFinite(target) || blocks.length === 0) return null;
    const tolerance = Math.max(0, finite(toleranceSec, 0));
    let best: T | null = null;
    let bestDistance = Number.POSITIVE_INFINITY;
    for (const block of blocks) {
        const start = Math.min(block.startSec, block.endSec);
        const end = Math.max(block.startSec, block.endSec);
        const distance =
            target >= start && target <= end ? 0 : target < start ? start - target : target - end;
        if (distance > tolerance) continue;
        if (distance < bestDistance) {
            best = block;
            bestDistance = distance;
        }
    }
    return best;
}

/**
 * 块内参数值的**高密度簇中心**（默认块位置算法）。
 *
 * 不再是算术均值：先按 `binWidth` 分箱，取样本权重最高的主簇；相邻箱按半权
 * 计入，避免值恰好落在箱边界时被拆成两簇。簇内以实际样本取均值，因此多峰
 * 材料落在“大部分帧所在的位置”，长颤音的对称极值也不会抬高块。
 * 密度接近（差值在 1/8 内）时取更靠近中位数的簇，保持视觉稳定。
 *
 * @param binWidth 显式箱宽；缺省由 {@link inferBlockBinWidth} 按参数量级推断。
 */
/**
 * 块秒区间 → 统一采样帧跨度。所有预览 / 提交 / 反算都走这里，
 * 避免 ceil/round 混用导致边缘样本不一致。
 */
export function paramBlockFrameSpan(
    startSec: number,
    endSec: number,
    framePeriodMs: number,
): { startFrame: number; endFrameExclusive: number } {
    const fp = Math.max(1e-6, framePeriodMs);
    const startFrame = Math.max(0, Math.round((startSec * 1000) / fp));
    const endFrameExclusive = Math.max(startFrame + 1, Math.round((endSec * 1000) / fp));
    return { startFrame, endFrameExclusive };
}

export function averageParamValueInSpan(args: {
    readonly startSec: number;
    readonly endSec: number;
    readonly values: readonly number[];
    readonly startFrame: number;
    readonly stride: number;
    readonly framePeriodMs: number;
    /** 参数是否为音高：跳过 <= 0 的无声帧，避免把静音拉低块高。 */
    readonly isPitch: boolean;
    /** 参数名，用于选择箱宽 / 静音判定。 */
    readonly param?: string;
    /** 显式箱宽（参数值单位）；缺省按参数推断。 */
    readonly binWidth?: number;
}): number | null {
    const fp = Math.max(1e-6, args.framePeriodMs);
    const stride = Math.max(1, Math.floor(args.stride));
    if (!Array.isArray(args.values) || args.values.length === 0) return null;
    const { startFrame, endFrameExclusive: endFrame } = paramBlockFrameSpan(
        args.startSec,
        args.endSec,
        fp,
    );
    /*
     * The block start is not necessarily on the paramView sample grid. Start at
     * the first sample at or after the block start; iterating from blockStart
     * would read a sample before the block and miss the block's last sample.
     */
    const firstFrame =
        args.startFrame + Math.ceil((startFrame - args.startFrame) / stride) * stride;
    const samples: number[] = [];
    for (let frame = firstFrame; frame < endFrame; frame += stride) {
        const index = (frame - args.startFrame) / stride;
        if (index < 0 || index >= args.values.length) continue;
        const value = Number(args.values[index]);
        if (!Number.isFinite(value)) continue;
        if (args.isPitch && value <= 0) continue;
        samples.push(value);
    }
    if (samples.length === 0) {
        /*
         * 块短于一个采样步长时 firstFrame >= endFrame，严格循环一个样本都取不到。
         * 回退到块内最近的一个采样（而不是返回 null 让块没有高度）。
         */
        const index = Math.max(
            0,
            Math.min(args.values.length - 1, Math.round((startFrame - args.startFrame) / stride)),
        );
        const value = Number(args.values[index]);
        if (!Number.isFinite(value) || (args.isPitch && value <= 0)) return null;
        samples.push(value);
    }
    return modeCenterOfValues(samples, {
        param: args.param,
        isPitch: args.isPitch,
        binWidth: args.binWidth,
    });
}

/**
 * 一组参数值的密度众数中心（纯函数，宿主与面板共用）。
 *
 * 返回主簇的**样本均值**：箱只用于找出主簇，最终位置仍由真实样本决定，
 * 不会出现半音格的量化跳变。
 */
export function modeCenterOfValues(
    values: readonly number[],
    opts?: {
        readonly param?: string;
        readonly isPitch?: boolean;
        readonly binWidth?: number;
    },
): number | null {
    if (values.length === 0) return null;
    const sorted = values
        .filter((value) => Number.isFinite(value))
        .slice()
        .sort((a, b) => a - b);
    if (sorted.length === 0) return null;
    if (sorted.length === 1) return sorted[0] as number;
    const middle = Math.floor(sorted.length / 2);
    const median =
        sorted.length % 2 === 0
            ? ((sorted[middle - 1] as number) + (sorted[middle] as number)) / 2
            : (sorted[middle] as number);
    const q1 = sorted[Math.floor(sorted.length * 0.25)] as number;
    const q3 = sorted[Math.floor(sorted.length * 0.75)] as number;
    const spread = q3 - q1;
    const fallback = inferBlockBinWidth(opts?.param, opts?.isPitch === true, median);
    const binWidth = Math.max(
        1e-9,
        Number.isFinite(opts?.binWidth) && (opts?.binWidth as number) > 0
            ? (opts?.binWidth as number)
            : spread > fallback * 0.25
              ? Math.max(fallback, spread / 12)
              : fallback,
    );

    const counts = new Map<number, number[]>();
    for (const value of sorted) {
        // 相对中位数分箱：整组值平移 delta 后 median 与 bin 归属同步平移，
        // 主簇不会因绝对分箱边界改变而换一个峰。
        const bin = Math.floor((value - median) / binWidth);
        const bucket = counts.get(bin);
        if (bucket === undefined) counts.set(bin, [value]);
        else bucket.push(value);
    }
    const bins = [...counts.keys()].sort((a, b) => a - b);
    let best: { weight: number; center: number; distance: number } | null = null;
    for (const bin of bins) {
        const own = counts.get(bin) ?? [];
        const prev = counts.get(bin - 1) ?? [];
        const next = counts.get(bin + 1) ?? [];
        // 主箱必须非空；相邻箱只加权，不与主箱争夺“峰”的身份。
        if (own.length === 0) continue;
        const merged = [...prev, ...own, ...next];
        const weight = own.length + prev.length * 0.5 + next.length * 0.5;
        const center = merged.reduce((sum, value) => sum + value, 0) / merged.length;
        const candidate = {
            weight,
            center,
            distance: Math.abs(center - median),
        };
        /*
         * 只接受**明显更密**的簇；密度接近时保留第一个峰，并优先更靠近中位数的
         * 那个。不能让“后遍历到的峰”凭 1 个样本的差就替换，否则稳定音上零星
         * 的滑音会把整块拖走。
         */
        if (
            best === null ||
            candidate.weight > best.weight * 1.125 ||
            (candidate.weight >= best.weight * 0.875 && candidate.distance < best.distance - 1e-9)
        ) {
            best = candidate;
        }
    }
    return best === null ? (sorted[0] as number) : best.center;
}

/**
 * 从宿主镜像的逐帧采样中求某时间段的代表值。
 *
 * 采样窗口与块不相交时返回 null（宿主回退到 spec 的快照值），而不是伪造 0。
 */
export function modeCenterOfValuesInSamples(
    startSec: number,
    endSec: number,
    samples: {
        readonly values: readonly number[];
        readonly startFrame: number;
        readonly stride: number;
        readonly framePeriodMs: number;
        readonly isPitch: boolean;
        readonly param: string;
        readonly binWidth?: number;
    },
): number | null {
    if (samples.values.length === 0) return null;
    const fp = Math.max(1e-6, samples.framePeriodMs);
    const stride = Math.max(1, Math.floor(samples.stride));
    const { startFrame, endFrameExclusive: endFrame } = paramBlockFrameSpan(startSec, endSec, fp);
    const firstFrame =
        samples.startFrame + Math.ceil((startFrame - samples.startFrame) / stride) * stride;
    const values: number[] = [];
    for (let frame = firstFrame; frame < endFrame; frame += stride) {
        const index = (frame - samples.startFrame) / stride;
        if (index < 0 || index >= samples.values.length) continue;
        const value = Number(samples.values[index]);
        if (!Number.isFinite(value)) continue;
        if (samples.isPitch && value <= 0) continue;
        values.push(value);
    }
    return modeCenterOfValues(values, {
        param: samples.param,
        isPitch: samples.isPitch,
        binWidth: samples.binWidth,
    });
}

/** 当前参数的一行像素高；块高 = 它的 1.2 倍。 */
export function blockRowHeightPx(
    valueToY: ((value: number, heightPx: number) => number) | undefined,
    viewportHeightPx: number,
    param: string,
): number {
    if (param !== "pitch" || valueToY === undefined) return 16 * PARAM_BLOCK_ROW_HEIGHT_MULTIPLIER;
    const row = Math.abs(valueToY(60, viewportHeightPx) - valueToY(61, viewportHeightPx));
    return (Number.isFinite(row) && row > 1e-6 ? row : 16) * PARAM_BLOCK_ROW_HEIGHT_MULTIPLIER;
}

/**
 * 块中心对应的**曲线坐标**。
 *
 * 曲线用 `valueToY(midi + 0.5)` 把 MIDI N 画在 N 键中心；块中心必须走同一
 * 口径。非 pitch 参数没有半键概念，直接使用参数值。
 */
function paramBlockCenterValue(
    param: string,
    value: number,
    opts?: { readonly snapToSemitone?: boolean },
): number {
    if (param !== "pitch") return value;
    const snap = opts?.snapToSemitone !== false;
    return (snap ? Math.round(value) : value) + 0.5;
}

/**
 * 块中心 → 视口 y 的**唯一**投影。
 *
 * `valueToY` 必须与曲线共用同一个实时视口闭包。不能在调用方另外展开
 * `(value - min) / span`：那会漏掉 `+0.5`，并且非整数行高时会与曲线错位。
 */
export function projectParamBlockCenterY(args: {
    readonly param: string;
    readonly value: number;
    readonly heightPx: number;
    readonly valueToY: (value: number, heightPx: number) => number;
    /** pitch 默认吸附到行心；自由拖动显式中心时传 false。 */
    readonly snapToSemitone?: boolean;
}): number {
    return args.valueToY(
        paramBlockCenterValue(args.param, args.value, {
            snapToSemitone: args.snapToSemitone,
        }),
        Math.max(1, args.heightPx),
    );
}

/** 块的短标签：音高用音名，其余参数显示一位小数。 */
export function formatParamBlockLabel(
    value: number | null,
    ordinal: number,
    param: string,
): string {
    if (value === null || !Number.isFinite(value)) return `#${ordinal}`;
    if (param === "pitch") {
        const rounded = Math.round(value);
        return `#${ordinal} ${midiLabel(rounded)}`;
    }
    return `#${ordinal} ${value.toFixed(1)}`;
}

/** 纯函数音名（避免 paramBlocks 在 node 单测里引入渲染模块）。 */
function midiLabel(midi: number): string {
    const names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
    return `${names[((midi % 12) + 12) % 12]}${Math.floor(midi / 12) - 1}`;
}

/** 按参数量级推断块位置直方图的箱宽。 */
function inferBlockBinWidth(param: string | undefined, isPitch: boolean, value: number): number {
    if (isPitch || param === "pitch") return 1;
    if (
        param === "tension" ||
        param?.endsWith("_cents") === true ||
        param?.endsWith("_degrees") === true
    ) {
        return 50;
    }
    if (param === "volume" || param === "dyn" || param === "dyn_edit") return 0.04;
    return Math.max(1e-6, Math.abs(value) * 0.02);
}
