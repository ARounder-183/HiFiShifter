import type { ParamLineBlock } from "./paramLineBlocks";
import type { ParamSelection } from "./paramSelection";

/** Time selection is independent of a block's vertical position or GL visibility. */
export function paramLineBlockIdsInSelection(
    blocks: readonly ParamLineBlock[],
    selection: ParamSelection | null,
): string[] {
    return blocks
        .filter((block) =>
            selection?.some(
                (range) =>
                    range.frameCount > 0 &&
                    block.startFrame < range.startFrame + range.frameCount &&
                    block.endFrameExclusive > range.startFrame,
            ),
        )
        .map((block) => block.id);
}

export function paramLineRestoreRanges(
    blocks: readonly ParamLineBlock[],
    ids: readonly string[],
    selection: ParamSelection | null,
): ParamSelection {
    const ranges = selection?.filter((range) => range.frameCount > 0);
    if (ranges?.length) return ranges;
    const selected = blocks.filter((block) => ids.includes(block.id));
    if (!selected.length) return [];
    const startFrame = Math.min(...selected.map((block) => block.startFrame));
    return [
        {
            startFrame,
            frameCount: Math.max(...selected.map((block) => block.endFrameExclusive)) - startFrame,
        },
    ];
}
