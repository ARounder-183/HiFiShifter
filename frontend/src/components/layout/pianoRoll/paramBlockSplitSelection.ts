import {
    logicalFrameRangesFromView,
    logicalBlocksOf,
    paramBlockIdsInSelection,
    type ParamBlockFrameBlock,
} from "./paramBlockSelection";
import { normalizeSelection, type ParamSelection } from "./paramSelection";
import type { ParamBlockSelectionAnchor } from "./paramBlockSelectionGesture";

export interface ParamBlockSplitSelectionInput {
    readonly beforeBlocks: readonly ParamBlockFrameBlock[];
    readonly beforeSelection: ParamSelection | null;
    readonly splitFrame: number;
    readonly targetLogicalId: string;
    readonly leftBlockId?: string;
    readonly rightBlockId?: string;
    readonly viewAfter: {
        readonly blocks: readonly ParamBlockFrameBlock[];
    };
    readonly rightSpan?: { readonly startFrame: number; readonly endFrameExclusive: number };
}

export interface ParamBlockSplitSelectionResult {
    readonly selection: ParamSelection | null;
    readonly anchor: ParamBlockSelectionAnchor | null;
    readonly selectedLogicalIds: readonly string[];
}

/**
 * Split selection policy:
 * - select the right result;
 * - deselect the left result;
 * - preserve previously selected logical blocks strictly to the right of the
 *   split, while dropping previously selected blocks to its left.
 *
 * The target is always excluded first and then re-added through rightBlockId,
 * so it cannot accidentally keep both halves selected.
 */
export function resolveParamBlockSplitSelection(
    input: ParamBlockSplitSelectionInput,
): ParamBlockSplitSelectionResult {
    const selectedBefore = new Set(
        paramBlockIdsInSelection(input.beforeBlocks, input.beforeSelection),
    );
    const selectedRight = logicalBlocksOf(input.beforeBlocks).filter(
        (block) =>
            selectedBefore.has(block.id) &&
            block.id !== input.targetLogicalId &&
            block.startFrame >= input.splitFrame,
    );

    const nextIds = new Set<string>(selectedRight.map((block) => block.id));
    if (input.rightBlockId !== undefined) nextIds.add(input.rightBlockId);

    const ranges = [...nextIds].flatMap((id) =>
        logicalFrameRangesFromView(input.viewAfter.blocks, id),
    );
    const selection = ranges.length > 0 ? normalizeSelection(ranges) : null;
    const anchor =
        input.rightBlockId === undefined
            ? null
            : {
                  blockId: input.rightBlockId,
                  frame: input.rightSpan?.startFrame ?? input.splitFrame,
              };

    return { selection, anchor, selectedLogicalIds: [...nextIds] };
}
