/**
 * 参数块层的**悬停**处理（纯几何判定 + 光标）。
 *
 * 【为什么单独成文件】它原本是面板模型里的一个 useCallback（96 行），只做一件事：
 * 把指针位置换成「命中过渡手柄 / 命中块边缘 / 命中块体 / 空白」四种状态，并据此设置
 * 光标与 hover ref。它与面板的任何 state / effect 顺序都无关。
 *
 * 【返回值的语义】true = 本次移动由块层认领。调用方（EditorArea）在 true 时**不再**
 * 调用通用的 onCanvasPointerMove —— 否则通用处理器会用「是否贴近参数线」重新算一遍
 * 光标与读数，把块层刚设好的状态覆盖掉。
 */
import type { MutableRefObject, PointerEvent as ReactPointerEvent, RefObject } from "react";
import { viewportPxToSec } from "../../renderKernel/timelineAxis.js";
import {
    PARAM_BLOCK_TRANSITION_HANDLE_OUTSET_X_PX,
    PARAM_BLOCK_TRANSITION_HANDLE_OUTSET_Y_PX,
    hitTestParamBlockEdges,
    hitTestParamBlockTransitionHandles,
    hitTestParamBlocks,
    paramBlockCursorForHover,
    paramBlockEdgeCandidates,
    type ParamBlock,
} from "../paramBlocks";
import type { PianoRollKernelHost } from "../kernel/host/pianoRollKernelHost";

/** 块悬停处理需要的外部状态（全部显式传入，便于逐字段核对）。 */
export interface ParamBlockHoverDeps {
    readonly paramBlockToolActive: boolean;
    readonly paramBlockHitSlopPx: number;
    readonly paramBlocksWithValues: readonly ParamBlock[];
    readonly hostRef: RefObject<PianoRollKernelHost | null>;
    readonly hoveredParamBlockIdRef: MutableRefObject<string | null>;
    /** 过渡手柄的 hover 键是**左块 id**（与 paramBlockTransitionHandles 的 leftId 对应）。 */
    readonly hoveredTransitionLeftIdRef: MutableRefObject<string | null>;
    /** 新建块手势进行中（光标是 crosshair 而不是 grabbing）。 */
    readonly createGestureActive: boolean;
    readonly setCanvasCursor: (cursor: string) => void;
    readonly invalidate: () => void;
}

/**
 * 把一次画布 pointermove 折成块层的悬停状态。
 *
 * @returns true 表示块层认领了这次移动（块模式下的**所有**移动都由块层认领）。
 */
export function handleParamBlockHover(
    deps: ParamBlockHoverDeps,
    event: ReactPointerEvent<HTMLCanvasElement>,
): boolean {
    if (!deps.paramBlockToolActive) return false;
    if (event.buttons !== 0) {
        // 手势期间保持拖动光标；由 pointerup 的默认光标接替。
        deps.setCanvasCursor(deps.createGestureActive ? "crosshair" : "grabbing");
        return true;
    }
    const rect = event.currentTarget.getBoundingClientRect();
    const localX = event.clientX - rect.left;
    const localY = event.clientY - rect.top;
    const axis = deps.hostRef.current?.getAxis();
    if (axis === undefined) return false;
    const sec = viewportPxToSec(axis, localX);
    const framedHits = deps.hostRef.current?.getParamBlockHitRects() ?? [];
    const edgeCandidateIds = new Set(
        paramBlockEdgeCandidates(deps.paramBlocksWithValues).map((block) => block.id),
    );
    const secToX = (value: number) => value * axis.pxPerSec - axis.scrollLeftPx;
    const transition = hitTestParamBlockTransitionHandles({
        // 手柄可见时即可抓住；首次 hover 不依赖块已选中。
        blocks: framedHits,
        clientX: localX,
        localY,
        secToX,
        outsetPx: PARAM_BLOCK_TRANSITION_HANDLE_OUTSET_X_PX,
        outsetYPx: PARAM_BLOCK_TRANSITION_HANDLE_OUTSET_Y_PX,
    });
    if (transition !== null) {
        deps.setCanvasCursor(paramBlockCursorForHover("transition"));
        deps.hoveredParamBlockIdRef.current = transition.leftId;
        deps.hoveredTransitionLeftIdRef.current = transition.leftId;
        return true;
    }
    const hit = hitTestParamBlocks({
        blocks: framedHits,
        sec,
        localY,
        slopPx: deps.paramBlockHitSlopPx,
    });
    if (hit === null) return false;
    const block = deps.paramBlocksWithValues.find((item) => item.id === hit.id);
    if (block === undefined) return false;
    const edge = hitTestParamBlockEdges({
        blocks: framedHits.filter((rect) => edgeCandidateIds.has(rect.id)),
        clientX: localX,
        localY,
        secToX,
        hitPx: 8,
    });
    /* 手柄在右边缘外侧上方，已在上面优先处理；这里只剩边缘与块体。 */
    deps.setCanvasCursor(
        paramBlockCursorForHover(
            edge !== null ? (edge.edge === "left" ? "left-edge" : "right-edge") : "body",
        ),
    );
    /*
     * 悬停状态只用于光标与局部强调；它不重建块数据（`drawParamBlocks` 在
     * 绘制期把 hovered 叠到 spec 上），只标脏重绘一帧。
     *
     * 【为什么必须真的写】过渡手柄的"可见即可抓"判据是 `selected || hovered`。
     * 旧实现把 hovered 写死成 null，于是**只有已选中的块**才抓得住手柄，
     * 而光标却按 hover 显示成 `ew-resize` —— 用户看到"光标说能拖、按下去
     * 却拖不动"。这条判据现在由真实命中提供。
     */
    const nextHoverBlockId = hit.id;
    const nextTransitionId = null;
    if (
        deps.hoveredParamBlockIdRef.current !== nextHoverBlockId ||
        deps.hoveredTransitionLeftIdRef.current !== nextTransitionId
    ) {
        /*
         * Hover is computed from the **host's frame geometry**, while
         * `invalidateParamBlockSpec()` rebuilds from React state. The two
         * can disagree for a frame (selection commit is async), and the
         * rebuilt spec may drop every transition handle. Keep the refs for
         * visual hover only; do not invalidate the structural spec here.
         */
        deps.hoveredParamBlockIdRef.current = nextHoverBlockId;
        deps.hoveredTransitionLeftIdRef.current = nextTransitionId;
        deps.invalidate();
    }
    return true;
}
