/**
 * 参数块点击选择手势（纯函数）。
 *
 * 与 Clip 的选择语义同族：Ctrl/Cmd 切换成员，Shift 从锚点范围选择，
 * 两键互斥时退回普通单选。这里只计算"下一次选区 + 锚点"，
 * 不启动拖动、不写 React 状态。
 */
import {
    addFrameRanges,
    normalizeSelection,
    subtractFrameRange,
    type ParamSelection,
} from "./paramSelection";
import {
    logicalBlocksOf,
    paramBlockIdsInSelection,
    paramBlockLogicalFrameRanges,
    type ParamBlockFrameRange,
} from "./paramBlockSelection";
import { logicalBlockId } from "./paramBlockTypes";

interface FrameBlock {
    readonly id: string;
    readonly userBlockId?: string;
    readonly startFrame: number;
    readonly endFrameExclusive: number;
}

export type BlockClickMode = "replace" | "toggle" | "range";

export interface ParamBlockSelectionAnchor {
    /** 逻辑块 id。 */
    readonly blockId: string;
    /** 点击位置的整数帧坐标。 */
    readonly frame: number;
}

export interface BlockClickOutcome {
    readonly selection: ParamSelection | null;
    readonly anchor: ParamBlockSelectionAnchor | null;
    readonly targetSelected: boolean;
}

/**
 * 锚点到目标点击帧之间的逻辑块。闭区间：相接的块会被纳入，与 Clip
 * 的区间判定同口径。
 *
 * 结果按整块展开；多 run 块只要一个 run 相交就纳入全部 run。锚点块若
 * 仍存在则强制并入 —— 用户锚定的块不能因为后来被裁剪/移动而丢失。
 */
export function resolveBlockRangeSelection(args: {
    readonly blocks: readonly FrameBlock[];
    readonly anchorFrame: number;
    readonly targetFrame: number;
    readonly anchorBlockId?: string;
}): ParamSelection | null {
    const lo = Math.min(args.anchorFrame, args.targetFrame);
    const hi = Math.max(args.anchorFrame, args.targetFrame);
    const logical = logicalBlocksOf(args.blocks);
    const chosen = logical.filter((block) =>
        block.runs.some((run) => run.startFrame <= hi && run.startFrame + run.frameCount >= lo),
    );
    if (
        args.anchorBlockId !== undefined &&
        logical.some((block) => block.id === args.anchorBlockId) &&
        !chosen.some((block) => block.id === args.anchorBlockId)
    ) {
        const anchor = logical.find((block) => block.id === args.anchorBlockId);
        if (anchor !== undefined) chosen.push(anchor);
    }
    if (chosen.length === 0) return null;
    chosen.sort((a, b) => a.startFrame - b.startFrame || a.id.localeCompare(b.id));
    return normalizeSelection(chosen.flatMap((block) => block.runs));
}

/**
 * 解析一次块点击。targetBlockId 可以是逻辑 id 或任一投影 run id。
 * 目标不存在时原样返回，targetSelected=false。
 */
export function resolveBlockClickSelection(args: {
    readonly blocks: readonly FrameBlock[];
    readonly selection: ParamSelection | null;
    readonly anchor: ParamBlockSelectionAnchor | null;
    readonly targetBlockId: string;
    readonly targetFrame: number;
    readonly mode: BlockClickMode;
}): BlockClickOutcome {
    const logical = logicalBlocksOf(args.blocks).find(
        (block) =>
            block.id === args.targetBlockId ||
            (block.runs.length > 0 &&
                args.blocks.some(
                    (candidate) =>
                        candidate.id === args.targetBlockId &&
                        logicalBlockId(candidate) === block.id,
                )),
    );
    if (logical === undefined) {
        return { selection: args.selection, anchor: args.anchor, targetSelected: false };
    }
    const targetAnchor: ParamBlockSelectionAnchor = {
        blockId: logical.id,
        frame: args.targetFrame,
    };
    if (args.mode === "replace") {
        return {
            selection: normalizeSelection(logical.runs),
            anchor: targetAnchor,
            targetSelected: true,
        };
    }
    if (args.mode === "toggle") {
        const selected = paramBlockIdsInSelection(args.blocks, args.selection).includes(logical.id);
        let selection: ParamSelection | null = args.selection;
        for (const run of logical.runs) {
            if (selected) {
                selection = subtractFrameRange(
                    selection,
                    run.startFrame,
                    run.startFrame + run.frameCount,
                );
            } else {
                selection = addFrameRanges(selection, [run]);
            }
        }
        return {
            selection,
            anchor: targetAnchor,
            targetSelected: !selected,
        };
    }
    const anchor = args.anchor;
    const effectiveAnchor =
        anchor !== null && logicalBlocksOf(args.blocks).some((block) => block.id === anchor.blockId)
            ? anchor
            : targetAnchor;
    let selection: ParamSelection | null;
    if (args.anchor === null && effectiveAnchor.blockId === logical.id) {
        selection = normalizeSelection(logical.runs);
    } else {
        selection = resolveBlockRangeSelection({
            blocks: args.blocks,
            anchorFrame: effectiveAnchor.frame,
            targetFrame: args.targetFrame,
            anchorBlockId: effectiveAnchor.blockId,
        });
    }
    return {
        selection,
        anchor: targetAnchor,
        targetSelected: paramBlockIdsInSelection(args.blocks, selection).includes(logical.id),
    };
}

/**
 * 下一次 Shift 点击的锚点回退顺序：
 * 1. 原锚点仍存在；
 * 2. 选区按时间排序的第一个逻辑块；
 * 3. 播放头所在逻辑块；
 * 4. null（Shift 退化为普通 replace）。
 */
export function resolveBlockAnchor(args: {
    readonly blocks: readonly FrameBlock[];
    readonly anchor: ParamBlockSelectionAnchor | null;
    readonly selection: ParamSelection | null;
    readonly playheadFrame?: number;
}): ParamBlockSelectionAnchor | null {
    const logical = logicalBlocksOf(args.blocks);
    const anchor = args.anchor;
    if (anchor !== null && logical.some((block) => block.id === anchor.blockId)) {
        return anchor;
    }
    const selected = paramBlockIdsInSelection(args.blocks, args.selection);
    const first = logical.find((block) => selected.includes(block.id));
    if (first !== undefined) {
        return { blockId: first.id, frame: first.startFrame };
    }
    if (args.playheadFrame !== undefined && Number.isFinite(args.playheadFrame)) {
        const atPlayhead = logical.find((block) =>
            block.runs.some(
                (run) =>
                    run.startFrame <= args.playheadFrame! &&
                    run.startFrame + run.frameCount > args.playheadFrame!,
            ),
        );
        if (atPlayhead !== undefined) {
            return { blockId: atPlayhead.id, frame: args.playheadFrame };
        }
    }
    return null;
}

/** 方便调用方把逻辑块 run 转成参数选区。 */
export function selectionForLogicalBlock(
    blocks: readonly FrameBlock[],
    blockId: string,
): ParamSelection | null {
    return normalizeSelection(paramBlockLogicalFrameRanges(blocks, blockId));
}

export type { ParamBlockFrameRange };
