import type { ParamBlock } from "./paramBlocks";
import { blockMatchesLogicalId, logicalBlockId } from "./paramBlockTypes";

/**
 * Pure geometry for vertical parameter-block drags.
 *
 * A logical block owns one persisted center even when it projects to several
 * visible runs. The drag must therefore resolve one start center per logical
 * block and fan it out to every run, instead of only resolving the run that
 * happens to own the pointer.
 */
export interface ParamBlockLogicalGroup {
    readonly id: string;
    readonly runs: readonly ParamBlock[];
}

export interface ParamBlockDragCenterInput {
    readonly selectedBlocks: readonly ParamBlock[];
    readonly anchorBlockId: string;
    readonly anchorValue?: number | null;
    readonly centerValues?: ReadonlyMap<string, number>;
    readonly param: string;
    readonly fallbackValue: number;
}

export interface ParamBlockDragCenters {
    readonly byRun: ReadonlyMap<string, number>;
    readonly byLogical: ReadonlyMap<string, number>;
}

function finite(value: number | null | undefined): number | null {
    return value !== null && value !== undefined && Number.isFinite(value) ? value : null;
}

/** Group projected runs by logical identity, deterministically ordered by time. */
export function groupBlocksByLogicalId(blocks: readonly ParamBlock[]): ParamBlockLogicalGroup[] {
    const grouped = new Map<string, ParamBlock[]>();
    for (const block of blocks) {
        const id = logicalBlockId(block);
        const runs = grouped.get(id);
        if (runs === undefined) grouped.set(id, [block]);
        else runs.push(block);
    }
    return [...grouped.entries()]
        .map(([id, runs]) => ({
            id,
            runs: runs
                .slice()
                .sort(
                    (left, right) =>
                        left.startSec - right.startSec || left.id.localeCompare(right.id),
                ),
        }))
        .sort(
            (left, right) =>
                left.runs[0]!.startSec - right.runs[0]!.startSec || left.id.localeCompare(right.id),
        );
}

/**
 * Resolve every selected logical block's start center from the same rendered
 * geometry as the pointer. The pointer value is deliberately not accepted:
 * using it would make non-anchor blocks jump to the pointer on pointerdown.
 */
export function resolveParamBlockDragCenters(
    input: ParamBlockDragCenterInput,
): ParamBlockDragCenters {
    const byRun = new Map<string, number>();
    const byLogical = new Map<string, number>();
    const groups = groupBlocksByLogicalId(input.selectedBlocks);
    const anchorLogicalId = logicalBlockId(
        input.selectedBlocks.find((block) => blockMatchesLogicalId(block, input.anchorBlockId)) ?? {
            id: input.anchorBlockId,
        },
    );

    for (const group of groups) {
        const isAnchorGroup = group.id === anchorLogicalId;
        const representative =
            group.runs.find((block) => blockMatchesLogicalId(block, input.anchorBlockId)) ??
            group.runs[0];
        if (representative === undefined) continue;

        const explicit = isAnchorGroup ? finite(input.anchorValue) : null;
        const renderedCenter =
            group.runs
                .map((block) => finite(input.centerValues?.get(block.id)))
                .find((value): value is number => value !== null) ?? null;
        const manualCenter = input.param === "pitch" ? finite(representative.manualValue) : null;
        const averageCenter = finite(representative.averageValue);
        const center =
            explicit ??
            renderedCenter ??
            manualCenter ??
            averageCenter ??
            finite(input.fallbackValue);
        if (center === null) continue;

        byLogical.set(group.id, center);
        for (const run of group.runs) byRun.set(run.id, center);
    }

    return { byRun, byLogical };
}
