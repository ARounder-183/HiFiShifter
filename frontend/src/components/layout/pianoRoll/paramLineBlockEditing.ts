import type { ParamBlock, ParamBlockState } from "./paramBlocks";
import {
    mergeParamLineBlocks,
    normalizeParamLineBlocks,
    placeParamLineBlock,
    resizeParamLineBlock,
    splitParamLineBlock,
    type ParamLineBlock,
} from "./paramLineBlocks";

export const newParamLineBlockId = () => `block:${crypto.randomUUID()}`;
export const lineFrame = (state: ParamBlockState, sec: number) =>
    Math.max(0, Math.round((sec * 1000) / (state.lineLayout?.framePeriodMs ?? 5)));

export function withParamLineBlocks(
    state: ParamBlockState,
    blocks: readonly ParamLineBlock[],
): ParamBlockState {
    if (!state.lineLayout) return state;
    return {
        ...state,
        lineEditBase: state.lineLayout,
        lineLayout: { ...state.lineLayout, blocks: normalizeParamLineBlocks(blocks) },
    };
}

export function lineBlocksForPanel(root: string, state: ParamBlockState): ParamBlock[] {
    const layout = state.lineLayout;
    if (!layout) return [];
    const seconds = layout.framePeriodMs / 1000;
    return layout.blocks.map((block) => ({
        id: block.id,
        rootTrackId: root,
        kind: "auto",
        members: [],
        startSec: block.startFrame * seconds,
        endSec: block.endFrameExclusive * seconds,
        averageValue: null,
        transitionSec: (block.transitionFrames ?? 0) * seconds,
        lineBlock: block,
        lineFramePeriodMs: layout.framePeriodMs,
    }));
}

export function createLineBlock(
    state: ParamBlockState,
    start: number,
    end: number,
    id = newParamLineBlockId(),
) {
    const startFrame = lineFrame(state, Math.min(start, end));
    const endFrameExclusive = lineFrame(state, Math.max(start, end));
    if (!state.lineLayout || endFrameExclusive <= startFrame)
        return { ok: false, reason: "too_short" } as const;
    return {
        ok: true,
        blockId: id,
        state: withParamLineBlocks(
            state,
            placeParamLineBlock(state.lineLayout.blocks, { id, startFrame, endFrameExclusive }),
        ),
    } as const;
}

export function splitLineBlock(state: ParamBlockState, block: ParamBlock, sec: number) {
    const frame = lineFrame(state, sec);
    if (
        !state.lineLayout ||
        !block.lineBlock ||
        frame <= block.lineBlock.startFrame ||
        frame >= block.lineBlock.endFrameExclusive
    )
        return { ok: false, reason: "too_short" } as const;
    return {
        ok: true,
        blockId: block.id,
        state: withParamLineBlocks(
            state,
            splitParamLineBlock(state.lineLayout.blocks, block.id, frame, newParamLineBlockId()),
        ),
    } as const;
}

export function resizeLineBlock(
    state: ParamBlockState,
    block: ParamBlock,
    edge: "left" | "right",
    targetSec: number,
    targetSpan?: { startSec: number; endSec: number } | null,
) {
    const current = block.lineBlock;
    if (!state.lineLayout || !current) return { ok: false, reason: "not_found" } as const;
    const start = targetSpan
        ? lineFrame(state, targetSpan.startSec)
        : edge === "left"
          ? lineFrame(state, targetSec)
          : current.startFrame;
    const end = targetSpan
        ? lineFrame(state, targetSpan.endSec)
        : edge === "right"
          ? lineFrame(state, targetSec)
          : current.endFrameExclusive;
    const blocks =
        end > start
            ? resizeParamLineBlock(state.lineLayout.blocks, block.id, start, end)
            : state.lineLayout.blocks.filter((item) => item.id !== block.id);
    return { ok: true, blockId: block.id, state: withParamLineBlocks(state, blocks) } as const;
}

export function mergeLineBlocks(state: ParamBlockState, ids: readonly string[]) {
    if (!state.lineLayout || ids.length < 2) return { ok: false, reason: "empty" } as const;
    const wanted = new Set(ids);
    const selected = state.lineLayout.blocks.filter((block) => wanted.has(block.id));
    if (selected.length < 2) return { ok: false, reason: "empty" } as const;
    for (let i = 1; i < selected.length; i++) {
        if (selected[i - 1].endFrameExclusive !== selected[i].startFrame)
            return { ok: false, reason: "not_adjacent" } as const;
    }
    return {
        ok: true,
        state: withParamLineBlocks(
            state,
            mergeParamLineBlocks(state.lineLayout.blocks, new Set(ids), newParamLineBlockId()),
        ),
    } as const;
}

export function deleteLineBlocks(state: ParamBlockState, ids: readonly string[]) {
    const wanted = new Set(ids);
    if (!state.lineLayout || !state.lineLayout.blocks.some((block) => wanted.has(block.id)))
        return { ok: false, reason: "not_found" } as const;
    return {
        ok: true,
        blockId: ids[0],
        state: withParamLineBlocks(
            state,
            state.lineLayout.blocks.filter((block) => !wanted.has(block.id)),
        ),
    } as const;
}
