/** 参数块选区工具：只处理派生视图帧，不读任何旧块记录。 */

import {
    normalizeSelection,
    selectionCoversFrameRange,
    type FrameRange,
    type ParamSelection,
} from "./paramSelection";
import { logicalBlockId } from "./paramBlockTypes";

export interface ParamBlockFrameRange {
    readonly startFrame: number;
    readonly frameCount: number;
}

type RangeInput = ParamBlockFrameRange | readonly ParamBlockFrameRange[] | null | undefined;

interface FrameBlock {
    readonly id: string;
    readonly userBlockId?: string;
    readonly startFrame: number;
    readonly endFrameExclusive: number;
}

/** 一个逻辑块的全部可见 run；id 永远是逻辑 id（userBlockId ?? id）。 */
export interface LogicalBlockFrames {
    readonly id: string;
    readonly runs: readonly ParamBlockFrameRange[];
    readonly startFrame: number;
    readonly endFrameExclusive: number;
}

function rangesOf(input: RangeInput): ParamBlockFrameRange[] {
    if (input === null || input === undefined) return [];
    return Array.isArray(input) ? [...input] : [input as ParamBlockFrameRange];
}

function runRange(block: FrameBlock): ParamBlockFrameRange {
    return {
        startFrame: block.startFrame,
        frameCount: Math.max(0, block.endFrameExclusive - block.startFrame),
    };
}

function mergeRanges(ranges: readonly ParamBlockFrameRange[]): ParamBlockFrameRange[] {
    const sorted = ranges
        .filter(
            (range) =>
                Number.isFinite(range.startFrame) &&
                Number.isFinite(range.frameCount) &&
                range.frameCount > 0,
        )
        .map((range) => ({ ...range }))
        .sort((a, b) => a.startFrame - b.startFrame);
    const out: ParamBlockFrameRange[] = [];
    for (const range of sorted) {
        const last = out[out.length - 1];
        if (last && range.startFrame <= last.startFrame + last.frameCount) {
            const end = Math.max(
                last.startFrame + last.frameCount,
                range.startFrame + range.frameCount,
            );
            (last as { frameCount: number }).frameCount = end - last.startFrame;
        } else {
            out.push(range);
        }
    }
    return out;
}

/**
 * 按逻辑 id 分组并合并 run。所有块命令都以这里的结果为单位；
 * 只有绘制 / 命中 / 过渡手柄才看单个 run。
 */
export function logicalBlocksOf(blocks: readonly FrameBlock[]): LogicalBlockFrames[] {
    const grouped = new Map<string, ParamBlockFrameRange[]>();
    for (const block of blocks) {
        const range = runRange(block);
        if (range.frameCount <= 0) continue;
        const id = logicalBlockId(block);
        const list = grouped.get(id) ?? [];
        list.push(range);
        grouped.set(id, list);
    }
    return [...grouped.entries()]
        .map(([id, runs]): LogicalBlockFrames | null => {
            const merged = mergeRanges(runs);
            if (merged.length === 0) return null;
            return {
                id,
                runs: merged,
                startFrame: Math.min(...merged.map((run) => run.startFrame)),
                endFrameExclusive: Math.max(
                    ...merged.map((run) => run.startFrame + run.frameCount),
                ),
            };
        })
        .filter((block): block is LogicalBlockFrames => block !== null)
        .sort((a, b) => a.startFrame - b.startFrame || a.id.localeCompare(b.id));
}

/** 同一逻辑块的全部可见片段（多 run 块必须整体选中/操作）。 */
export function paramBlockLogicalFrameRanges(
    blocks: readonly FrameBlock[],
    blockId: string,
): ParamBlockFrameRange[] {
    const identity = blocks.find((block) => block.id === blockId);
    const id = identity === undefined ? blockId : logicalBlockId(identity);
    return mergeRanges(blocks.filter((block) => logicalBlockId(block) === id).map(runRange));
}

/** 视图块 id 列表 → 逻辑块的全部 run。 */
export function logicalFrameRangesFromView(
    blocks: readonly {
        readonly id: string;
        readonly userBlockId?: string;
        readonly startFrame: number;
        readonly endFrameExclusive: number;
    }[],
    blockId: string,
): ParamBlockFrameRange[] {
    return paramBlockLogicalFrameRanges(blocks, blockId);
}

/** 选区碰到的每个逻辑块都扩展为它的全部可见片段。 */
export function expandParamBlockSelection(
    blocks: readonly FrameBlock[],
    input: RangeInput,
): ParamBlockFrameRange[] {
    const selection = rangesOf(input);
    const identities = new Set<string>();
    for (const block of blocks) {
        if (
            selection.some((range) => {
                const end = range.startFrame + Math.max(0, range.frameCount);
                return block.startFrame < end && block.endFrameExclusive > range.startFrame;
            })
        ) {
            identities.add(logicalBlockId(block));
        }
    }
    return mergeRanges(
        blocks.filter((block) => identities.has(logicalBlockId(block))).map(runRange),
    );
}

/**
 * 块高亮 / 命令作用域：只返回**全部 run 都被选区完整覆盖**的逻辑块。
 *
 * 相交判定会让"选区压住半块"也高亮，随后 Delete 却按整块执行；完整覆盖
 * 保证高亮的块与命令实际作用的块严格一致。
 */
export function paramBlockIdsInSelection(
    blocks: readonly FrameBlock[],
    input: RangeInput,
): string[] {
    const ranges = normalizeSelection(rangesOf(input) as FrameRange[]);
    if (ranges === null) return [];
    return logicalBlocksOf(blocks)
        .filter((block) =>
            block.runs.every((run) =>
                selectionCoversFrameRange(ranges, run.startFrame, run.startFrame + run.frameCount),
            ),
        )
        .map((block) => block.id);
}

/**
 * 块工具 Ctrl+A：只选择所有可见逻辑块实际覆盖的帧区间。
 * 块间空隙与轨道头尾不进入选区；无块时返回 null（不是整条曲线）。
 */
export function selectAllParamBlockRanges(blocks: readonly FrameBlock[]): ParamSelection | null {
    const ranges = logicalBlocksOf(blocks).flatMap((block) => block.runs);
    if (ranges.length === 0) return null;
    return normalizeSelection(mergeRanges(ranges));
}

export function paramBlockRestoreRanges(
    blocks: readonly FrameBlock[],
    selectedIds: readonly string[],
    selection: RangeInput,
): ParamBlockFrameRange[] {
    if (selectedIds.length > 0) {
        const identities = new Set<string>();
        for (const block of blocks) {
            if (selectedIds.some((id) => blockMatches(block, id))) {
                identities.add(logicalBlockId(block));
            }
        }
        return mergeRanges(
            blocks.filter((block) => identities.has(logicalBlockId(block))).map(runRange),
        );
    }
    return mergeRanges(rangesOf(selection).filter((range) => range.frameCount > 0));
}

function blockMatches(block: FrameBlock, id: string): boolean {
    return block.id === id || block.userBlockId === id || logicalBlockId(block) === id;
}
