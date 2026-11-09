/** 参数块选区工具：只处理派生视图帧，不读任何旧块记录。 */

export interface ParamBlockFrameRange {
    readonly startFrame: number;
    readonly frameCount: number;
}

type RangeInput = ParamBlockFrameRange | readonly ParamBlockFrameRange[] | null | undefined;

function rangesOf(input: RangeInput): ParamBlockFrameRange[] {
    if (input === null || input === undefined) return [];
    return Array.isArray(input) ? [...input] : [input as ParamBlockFrameRange];
}

export function paramBlockIdsInSelection(
    blocks: readonly {
        readonly id: string;
        readonly startFrame: number;
        readonly endFrameExclusive: number;
    }[],
    input: RangeInput,
): string[] {
    const ranges = rangesOf(input).filter(
        (range) => Number.isFinite(range.startFrame) && Number.isFinite(range.frameCount),
    );
    if (ranges.length === 0) return [];
    return blocks
        .filter((block) =>
            ranges.some((range) => {
                const start = range.startFrame;
                const end = range.startFrame + Math.max(0, range.frameCount);
                return block.startFrame < end && block.endFrameExclusive > start;
            }),
        )
        .map((block) => block.id);
}

export function paramBlockRestoreRanges(
    blocks: readonly {
        readonly id: string;
        readonly startFrame: number;
        readonly endFrameExclusive: number;
    }[],
    selectedIds: readonly string[],
    selection: RangeInput,
): ParamBlockFrameRange[] {
    if (selectedIds.length > 0) {
        return blocks
            .filter((block) => selectedIds.includes(block.id))
            .map((block) => ({
                startFrame: block.startFrame,
                frameCount: Math.max(1, block.endFrameExclusive - block.startFrame),
            }));
    }
    return rangesOf(selection).filter((range) => range.frameCount > 0);
}
