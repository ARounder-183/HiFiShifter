/** A block is an interval on a root parameter line. It has no Clip parent. */
export interface ParamLineBlock {
    readonly id: string;
    readonly startFrame: number;
    readonly endFrameExclusive: number;
    readonly transitionFrames?: number;
}

export interface ParamLineBlockSet {
    readonly schemaVersion: 1;
    readonly revision: number;
    readonly framePeriodMs: number;
    readonly blocks: readonly ParamLineBlock[];
    readonly gameBaseline: {
        readonly revision: number;
        readonly segments: readonly ParamLineBlock[];
    };
}

export interface ParamLineTimeMapping {
    readonly oldStartFrame: number;
    readonly oldEndFrameExclusive: number;
    readonly newStartFrame: number;
    readonly newEndFrameExclusive: number;
}

const EPS = 1e-9;

function derivedId(...parts: (string | number)[]): string {
    const value = parts.join("|");
    let left = 0x811c9dc5,
        right = 0x9e3779b9;
    for (let i = 0; i < value.length; i++) {
        left = Math.imul(left ^ value.charCodeAt(i), 0x01000193);
        right = Math.imul(right ^ value.charCodeAt(i), 0x85ebca6b);
    }
    return `block:${(left >>> 0).toString(16)}${(right >>> 0).toString(16)}`;
}

export function normalizeParamLineBlocks(blocks: readonly ParamLineBlock[]): ParamLineBlock[] {
    const ids = new Set<string>();
    if (blocks.length > 20000) throw new Error("Parameter block count exceeded");
    const normalized = blocks
        .map((block) => {
            if (
                !block.id ||
                block.id.length > 512 ||
                ids.has(block.id) ||
                !Number.isSafeInteger(block.startFrame) ||
                !Number.isSafeInteger(block.endFrameExclusive) ||
                block.startFrame < 0 ||
                block.endFrameExclusive <= block.startFrame ||
                block.endFrameExclusive > 1_000_000 ||
                (block.transitionFrames !== undefined &&
                    (!Number.isSafeInteger(block.transitionFrames) || block.transitionFrames < 0))
            )
                throw new Error("Invalid parameter block interval");
            ids.add(block.id);
            return {
                id: block.id,
                startFrame: block.startFrame,
                endFrameExclusive: block.endFrameExclusive,
                ...(block.transitionFrames === undefined
                    ? {}
                    : { transitionFrames: block.transitionFrames }),
            };
        })
        .sort(
            (a, b) =>
                a.startFrame - b.startFrame ||
                a.endFrameExclusive - b.endFrameExclusive ||
                a.id.localeCompare(b.id),
        );
    for (let index = 1; index < normalized.length; index++) {
        if (normalized[index].startFrame < normalized[index - 1].endFrameExclusive)
            throw new Error("Overlapping parameter block intervals");
    }
    return normalized;
}

/**
 * Compute both preview and commit from the same immutable gesture basis.
 * Neighbours keep every uncovered part; a middle overlap leaves two intervals.
 */
export function placeParamLineBlock(
    basis: readonly ParamLineBlock[],
    target: ParamLineBlock,
): ParamLineBlock[] {
    if (
        !Number.isFinite(target.startFrame) ||
        !Number.isFinite(target.endFrameExclusive) ||
        target.startFrame < 0 ||
        target.endFrameExclusive <= target.startFrame + EPS
    )
        return [...basis];
    const result: ParamLineBlock[] = [];
    for (const block of basis) {
        if (block.id === target.id) continue;
        if (
            block.endFrameExclusive <= target.startFrame + EPS ||
            block.startFrame >= target.endFrameExclusive - EPS
        ) {
            result.push(block);
            continue;
        }
        const left = block.startFrame < target.startFrame - EPS;
        const right = block.endFrameExclusive > target.endFrameExclusive + EPS;
        if (left)
            result.push({ ...block, endFrameExclusive: target.startFrame, transitionFrames: 0 });
        if (right)
            result.push({
                ...block,
                id: left ? derivedId(block.id, "right", target.id) : block.id,
                startFrame: target.endFrameExclusive,
            });
    }
    result.push(target);
    return normalizeParamLineBlocks(result);
}

export function resizeParamLineBlock(
    basis: readonly ParamLineBlock[],
    id: string,
    startFrame: number,
    endFrameExclusive: number,
): ParamLineBlock[] {
    const block = basis.find((block) => block.id === id);
    return block === undefined
        ? [...basis]
        : placeParamLineBlock(basis, { ...block, startFrame, endFrameExclusive });
}

export function splitParamLineBlock(
    blocks: readonly ParamLineBlock[],
    id: string,
    sec: number,
    rightId: string,
): ParamLineBlock[] {
    return normalizeParamLineBlocks(
        blocks.flatMap((block) => {
            if (
                block.id !== id ||
                sec <= block.startFrame + EPS ||
                sec >= block.endFrameExclusive - EPS
            )
                return [block];
            return [
                { ...block, endFrameExclusive: sec, transitionFrames: 0 },
                { ...block, id: rightId, startFrame: sec },
            ];
        }),
    );
}

export function mergeParamLineBlocks(
    blocks: readonly ParamLineBlock[],
    ids: ReadonlySet<string>,
    id: string,
): ParamLineBlock[] {
    const selected = blocks.filter((block) => ids.has(block.id));
    if (selected.length < 2) return [...blocks];
    return placeParamLineBlock(
        blocks.filter((block) => !ids.has(block.id)),
        {
            ...selected[0],
            id,
            startFrame: Math.min(...selected.map((block) => block.startFrame)),
            endFrameExclusive: Math.max(...selected.map((block) => block.endFrameExclusive)),
        },
    );
}

/**
 * Apply the same project-time mapping used for linked parameter samples.
 * Only the part inside the changed Clip's old interval moves/scales/crops.
 */
export function mapParamLineBlocks(
    basis: readonly ParamLineBlock[],
    mapping: ParamLineTimeMapping,
): ParamLineBlock[] {
    const oldWidth = mapping.oldEndFrameExclusive - mapping.oldStartFrame;
    if (!(oldWidth > EPS)) return [...basis];
    const newWidth = Math.max(0, mapping.newEndFrameExclusive - mapping.newStartFrame);
    const outside: ParamLineBlock[] = [];
    const mapped: ParamLineBlock[] = [];
    for (const block of basis) {
        const start = Math.max(block.startFrame, mapping.oldStartFrame);
        const end = Math.min(block.endFrameExclusive, mapping.oldEndFrameExclusive);
        if (end <= start + EPS) {
            outside.push(block);
            continue;
        }
        const hasLeft = block.startFrame < start - EPS;
        const hasRight = block.endFrameExclusive > end + EPS;
        if (hasLeft) outside.push({ ...block, endFrameExclusive: start, transitionFrames: 0 });
        if (hasRight)
            outside.push({
                ...block,
                id: hasLeft ? derivedId(block.id, "outside:right") : block.id,
                startFrame: end,
            });
        if (newWidth > EPS)
            mapped.push({
                ...block,
                id: hasLeft || hasRight ? derivedId(block.id, "mapped") : block.id,
                startFrame: Math.round(
                    mapping.newStartFrame + ((start - mapping.oldStartFrame) * newWidth) / oldWidth,
                ),
                endFrameExclusive: Math.round(
                    mapping.newStartFrame + ((end - mapping.oldStartFrame) * newWidth) / oldWidth,
                ),
            });
    }
    return mapped
        .filter((block) => block.endFrameExclusive > block.startFrame)
        .reduce(
            (blocks, block) => placeParamLineBlock(blocks, block),
            normalizeParamLineBlocks(outside),
        );
}

/** Restore segmentation only; curve samples cannot be passed into this operation. */
export function restoreGameParamLineBlocks(
    blocks: readonly ParamLineBlock[],
    baseline: readonly ParamLineBlock[],
    startFrame: number,
    endFrameExclusive: number,
): ParamLineBlock[] {
    const placeholder = { id: "__restore_range__", startFrame, endFrameExclusive };
    let result = placeParamLineBlock(blocks, placeholder).filter(
        (block) => block.id !== placeholder.id,
    );
    for (const block of baseline) {
        const start = Math.max(startFrame, block.startFrame);
        const end = Math.min(endFrameExclusive, block.endFrameExclusive);
        if (end <= start) continue;
        result = placeParamLineBlock(result, {
            ...block,
            id: derivedId(block.id, "restore", start, end),
            startFrame: start,
            endFrameExclusive: end,
        });
    }
    return result;
}
