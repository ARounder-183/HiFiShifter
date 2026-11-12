/**
 * 参数块渲染规格的唯一构建器（纯函数）。
 *
 * 所有预览（canonical / drag / resize / create / transition）都必须从完整的
 * 块列表经过这里生成 `PianoRollParamBlockSpec`。调用方只能覆盖几何字段，
 * 不能手写 blocks 数组；否则 transitionHandle、contiguous 标记与标签会在预览
 * 分支里悄悄丢失，表现为拖拽时连接线整层消失、松手后又跳回来。
 */
import type { PianoRollParamBlockSpec } from "./kernel/host/pianoRollKernelData";
import { formatParamBlockLabel, paramBlockTransitionHandles } from "./paramBlocks";

export interface ParamBlockRenderBlock {
    readonly id: string;
    readonly userBlockId?: string;
    readonly startSec: number;
    readonly endSec: number;
    readonly averageValue: number | null;
    readonly logicalCenterValue?: number | null;
    readonly frameCenterValue?: number | null;
    readonly displayCenterValue?: number | null;
    readonly displayCenterSnapToSemitone?: boolean;
    readonly displayHeightPx?: number | null;
    readonly topPx?: number;
    readonly heightPx?: number;
    readonly selected: boolean;
    readonly grouped: boolean;
    readonly label?: string;
    readonly transitionSec?: number;
    /** canonical = live；手势预览 = locked；display-only fallback = fallback. */
    readonly centerPolicy?: "live" | "locked" | "fallback";
    readonly sourceNote?: number | null;
    readonly fallbackValue?: number | null;
    readonly cutStart?: boolean;
    readonly cutEnd?: boolean;
    readonly sourceKey?: string;
    /** Stable full-source note-event identity; view-only. */
    readonly sourceEventId?: string;
}

export interface ParamBlockSpecColors {
    readonly borderRgba: readonly [number, number, number, number];
    readonly selectedRgba: readonly [number, number, number, number];
    readonly groupedRgba: readonly [number, number, number, number];
    readonly labelRgba?: readonly [number, number, number, number];
    readonly selectedLabelRgba?: readonly [number, number, number, number];
    readonly fillRgba?: readonly [number, number, number, number];
    readonly selectedFillRgba?: readonly [number, number, number, number];
    readonly groupedFillRgba?: readonly [number, number, number, number];
    readonly topHighlightRgba?: readonly [number, number, number, number];
    readonly transitionRgba?: readonly [number, number, number, number];
    readonly transitionPointRgba?: readonly [number, number, number, number];
    readonly transitionPointRadiusPx?: number;
    readonly topHighlightPx?: number;
    readonly cornerRadiusPx?: number;
}

/** 时间序稳定排序；同一时刻按 id 兜底，保证输出与输入数组顺序无关。 */
function sortParamBlockRenderBlocks<
    T extends Pick<ParamBlockRenderBlock, "id" | "startSec" | "endSec">,
>(blocks: readonly T[]): T[] {
    return blocks
        .slice()
        .sort((a, b) => a.startSec - b.startSec || a.endSec - b.endSec || a.id.localeCompare(b.id));
}

/**
 * 从完整块列表构建块规格。
 *
 * - contiguous 只由最终几何决定；
 * - transitionHandle 只挂在相接关系左侧的 run 上，宽度取左块 transitionSec；
 * - 同一逻辑块（userBlockId ?? id）的全部 run 共享一个标签序号。
 */
export function buildParamBlockRenderSpec(args: {
    readonly blocks: readonly ParamBlockRenderBlock[];
    readonly param: string;
    readonly valueMin: number;
    readonly valueMax: number;
    readonly valueToY: (value: number, heightPx: number) => number;
    readonly colors: ParamBlockSpecColors;
    readonly curveOwned?: boolean;
}): PianoRollParamBlockSpec {
    const sorted = sortParamBlockRenderBlocks(args.blocks);
    const handles = paramBlockTransitionHandles(sorted);
    // Only known, equal source identities can be contiguous. Two unknown
    // sources must never be treated as the same Clip/Take/cycle.
    const sameSource = (a: ParamBlockRenderBlock | undefined, b: ParamBlockRenderBlock) =>
        a?.sourceKey !== undefined && a.sourceKey === b.sourceKey;
    const handleByLeftId = new Map(handles.map((handle) => [handle.leftId, handle]));
    return {
        curveOwned: args.curveOwned ?? true,
        param: args.param,
        valueMin: args.valueMin,
        valueMax: args.valueMax,
        valueToY: args.valueToY,
        blocks: sorted.map((block, index) => {
            const handle = handleByLeftId.get(block.id);
            const previous = sorted[index - 1];
            const next = sorted[index + 1];
            return {
                ...block,
                topPx: block.topPx ?? 0,
                heightPx: block.heightPx ?? 0,
                contiguousStart:
                    sameSource(previous, block) &&
                    Math.abs(previous!.endSec - block.startSec) < 1e-6,
                contiguousEnd:
                    sameSource(next, block) && Math.abs(next!.startSec - block.endSec) < 1e-6,
                transitionHandle:
                    handle === undefined
                        ? null
                        : {
                              sec: handle.sec,
                              leftId: handle.leftId,
                              rightId: handle.rightId,
                              widthSec: handle.transitionSec,
                          },
                label: block.label ?? formatParamBlockLabel(block.averageValue, args.param),
                centerPolicy: block.centerPolicy ?? "live",
            };
        }),
        ...args.colors,
    };
}
