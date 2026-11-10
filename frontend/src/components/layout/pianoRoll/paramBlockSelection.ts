/** 参数块选区工具：只处理派生视图帧，不读任何旧块记录。 */

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

function rangesOf(input: RangeInput): ParamBlockFrameRange[] {
    if (input === null || input === undefined) return [];
    return Array.isArray(input) ? [...input] : [input as ParamBlockFrameRange];
}

function logicalId(block: FrameBlock): string {
    return block.userBlockId ?? block.id;
}

function mergeRanges(ranges: readonly ParamBlockFrameRange[]): ParamBlockFrameRange[] {
    const sorted = ranges
        .filter((range) => Number.isFinite(range.startFrame) && Number.isFinite(range.frameCount))
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

/** 同一逻辑块的全部可见片段（多 run 块必须整体选中/操作）。 */
export function paramBlockLogicalFrameRanges(
    blocks: readonly FrameBlock[],
    blockId: string,
): ParamBlockFrameRange[] {
    const identity = blocks.find((block) => block.id === blockId);
    const id = identity === undefined ? blockId : logicalId(identity);
    return mergeRanges(
        blocks
            .filter((block) => logicalId(block) === id)
            .map((block) => ({
                startFrame: block.startFrame,
                frameCount: Math.max(1, block.endFrameExclusive - block.startFrame),
            })),
    );
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
            identities.add(logicalId(block));
        }
    }
    return mergeRanges(
        blocks
            .filter((block) => identities.has(logicalId(block)))
            .map((block) => ({
                startFrame: block.startFrame,
                frameCount: Math.max(1, block.endFrameExclusive - block.startFrame),
            })),
    );
}

export function paramBlockIdsInSelection(
    blocks: readonly FrameBlock[],
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
    blocks: readonly FrameBlock[],
    selectedIds: readonly string[],
    selection: RangeInput,
): ParamBlockFrameRange[] {
    if (selectedIds.length > 0) {
        const identities = new Set(
            blocks
                .filter((block) => selectedIds.includes(block.id))
                .map((block) => logicalId(block)),
        );
        return mergeRanges(
            blocks
                .filter((block) => identities.has(logicalId(block)))
                .map((block) => ({
                    startFrame: block.startFrame,
                    frameCount: Math.max(1, block.endFrameExclusive - block.startFrame),
                })),
        );
    }
    return mergeRanges(rangesOf(selection).filter((range) => range.frameCount > 0));
}
