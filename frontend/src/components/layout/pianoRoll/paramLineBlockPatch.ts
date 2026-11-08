import {
    normalizeParamLineBlocks,
    placeParamLineBlock,
    type ParamLineBlock,
} from "./paramLineBlocks";

const equal = (left: ParamLineBlock, right: ParamLineBlock) =>
    left.startFrame === right.startFrame &&
    left.endFrameExclusive === right.endFrameExclusive &&
    (left.transitionFrames ?? 0) === (right.transitionFrames ?? 0);

export function rebaseParamLineBlocks(
    current: readonly ParamLineBlock[],
    before: readonly ParamLineBlock[],
    after: readonly ParamLineBlock[],
): ParamLineBlock[] {
    const beforeById = new Map(before.map((block) => [block.id, block]));
    const afterById = new Map(after.map((block) => [block.id, block]));
    const removed = new Set(
        before.filter((block) => !afterById.has(block.id)).map((block) => block.id),
    );
    const currentById = new Map(current.map((block) => [block.id, block]));
    for (const old of before) {
        const target = afterById.get(old.id);
        if (target && equal(old, target)) continue;
        const actual = currentById.get(old.id);
        if (actual && !equal(actual, old) && (!target || !equal(actual, target)))
            throw new Error("Conflict: parameter block changed during gesture");
    }
    let result = current.filter((block) => !removed.has(block.id));
    for (const block of after) {
        const old = beforeById.get(block.id);
        if (!old || !equal(old, block)) result = placeParamLineBlock(result, block);
    }
    return normalizeParamLineBlocks(result);
}
