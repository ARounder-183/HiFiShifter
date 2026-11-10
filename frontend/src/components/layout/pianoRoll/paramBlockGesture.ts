/**
 * 参数块手势计划（Rust `param_blocks::gesture` 的预览端口）。
 *
 * 【唯一入口】分割 / 伸缩的预览、提交、菜单可用性都调用这里；底层实现只有
 * `paramBlockEditing` 的一份。这样“菜单可点却切不了”“拖动看见 A、松手变成 B”
 * 不会再因为两套算法而复发。
 */

import {
    planResizeProjectedBlock,
    planSplitProjectedBlock,
    ParamBlockPlanError,
    type FrameSpan,
    type ParamBlockPlanErrorCode,
    type ResizeProjectedBlockPlan,
    type SplitProjectedBlockPlan,
} from "./paramBlockEditing";
import { composeParamBlockView } from "./paramBlockComposition";
import type { ParamBlockEditSet, ParamBlockView, ProjectedParamBlock } from "./paramBlockTypes";
import type { ParamBlockSourceClip } from "./paramBlockProjection";

export type ParamBlockGesture =
    | { readonly kind: "split"; readonly blockId: string; readonly frame: number }
    | {
          readonly kind: "resize";
          readonly blockId: string;
          readonly edge: "left" | "right";
          readonly targetFrame: number;
      };

export interface ParamBlockGesturePlan {
    readonly gesture: ParamBlockGesture;
    readonly edits: ParamBlockEditSet;
    readonly viewAfter: ParamBlockView;
    readonly targetSpan: FrameSpan | null;
    readonly affected: ReadonlyArray<{ readonly id: string; readonly span: FrameSpan }>;
    readonly removedIds: readonly string[];
    readonly warnings: readonly string[];
    readonly leftBlockId?: string;
    readonly rightBlockId?: string;
    readonly leftSpan?: FrameSpan;
    readonly rightSpan?: FrameSpan;
}

export interface ParamBlockGestureFailure {
    readonly code: ParamBlockPlanErrorCode;
    readonly detail?: string;
}

export type ParamBlockGestureResult =
    | { readonly ok: true; readonly plan: ParamBlockGesturePlan }
    | { readonly ok: false; readonly error: ParamBlockGestureFailure };

function failure(error: unknown): ParamBlockGestureFailure {
    if (error instanceof ParamBlockPlanError) {
        return {
            code: error.code,
            ...(error.detail === undefined ? {} : { detail: error.detail }),
        };
    }
    return {
        code: "internal",
        detail: error instanceof Error ? error.message : String(error),
    };
}

function affectedSpans(
    before: ParamBlockView,
    after: ParamBlockView,
): Array<{ id: string; span: FrameSpan }> {
    const beforeById = new Map(before.blocks.map((block) => [block.id, block]));
    const out: Array<{ id: string; span: FrameSpan }> = [];
    for (const block of after.blocks) {
        const previous = beforeById.get(block.id);
        if (
            previous === undefined ||
            previous.startFrame !== block.startFrame ||
            previous.endFrameExclusive !== block.endFrameExclusive
        ) {
            out.push({
                id: block.id,
                span: { startFrame: block.startFrame, endFrameExclusive: block.endFrameExclusive },
            });
        }
    }
    return out;
}

export function planParamBlockGesture(args: {
    readonly clips: readonly ParamBlockSourceClip[];
    readonly edits: ParamBlockEditSet;
    readonly framePeriodMs: number;
    readonly gesture: ParamBlockGesture;
}): ParamBlockGestureResult {
    const fp = Math.max(0.1, args.framePeriodMs);
    const before = composeParamBlockView("", args.clips, args.edits, fp);
    try {
        if (args.gesture.kind === "split") {
            const block = before.blocks.find((item) => item.id === args.gesture.blockId);
            if (block === undefined) {
                return { ok: false, error: { code: "block_not_found" } };
            }
            const plan: SplitProjectedBlockPlan = planSplitProjectedBlock(
                args.clips,
                args.edits,
                block,
                args.gesture.frame,
                fp,
            );
            return {
                ok: true,
                plan: {
                    gesture: args.gesture,
                    edits: plan.edits,
                    viewAfter: plan.viewAfter,
                    targetSpan: null,
                    affected: affectedSpans(before, plan.viewAfter),
                    removedIds: [],
                    warnings: [],
                    leftBlockId: plan.leftBlockId,
                    rightBlockId: plan.rightBlockId,
                    leftSpan: plan.leftSpan,
                    rightSpan: plan.rightSpan,
                },
            };
        }
        const block = before.blocks.find((item) => item.id === args.gesture.blockId);
        if (block === undefined) {
            return { ok: false, error: { code: "block_not_found" } };
        }
        const plan: ResizeProjectedBlockPlan = planResizeProjectedBlock(
            args.clips,
            args.edits,
            block,
            args.gesture.edge,
            args.gesture.targetFrame,
            fp,
        );
        return {
            ok: true,
            plan: {
                gesture: args.gesture,
                edits: plan.edits,
                viewAfter: plan.viewAfter,
                targetSpan: plan.targetSpan,
                affected: affectedSpans(before, plan.viewAfter),
                removedIds: plan.removedIds,
                warnings: plan.warnings,
            },
        };
    } catch (error) {
        return { ok: false, error: failure(error) };
    }
}

/** 菜单 / 快捷键共用的可行性检查：能生成计划才算“可切”。 */
export function canSplitParamBlockAtFrame(args: {
    readonly clips: readonly ParamBlockSourceClip[];
    readonly edits: ParamBlockEditSet;
    readonly framePeriodMs: number;
    readonly block: ProjectedParamBlock;
    readonly frame: number;
}): boolean {
    const result = planParamBlockGesture({
        clips: args.clips,
        edits: args.edits,
        framePeriodMs: args.framePeriodMs,
        gesture: { kind: "split", blockId: args.block.id, frame: args.frame },
    });
    return result.ok;
}
