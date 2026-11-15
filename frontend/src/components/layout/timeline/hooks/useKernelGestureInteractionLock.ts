/**
 * 内核手势的**交互锁**（`beginInteraction` / `endInteraction` 的成对封装）。
 * 从 `useTimelineKernelSeekDrag` 拆出：锁的生命周期与拖拽预览逻辑彼此独立，
 * 单独成模块既守住主文件的行数预算，也让重叠手势的计数语义可被单独审查。
 */
import React from "react";

import { beginInteraction, endInteraction } from "../../../../features/session/sessionSlice";
import {
    beginClipGeometryPreview,
    endClipGeometryPreview,
} from "../../../../features/session/interactionSlice";
import { buildClipGeometryPreview, createClipGeometryPreviewToken } from "../clipGeometryPreview";
import type { TimelineSessionSlice } from "./useTimelineState";
import type { AppDispatch } from "../../../../app/store";

/**
 * 【为什么每个手势都必须持有】`sessionSlice` 在 `_interactionLockCount > 0`
 * 时丢弃后端回包的过期快照。手势期间若有一条 `*Remote.fulfilled` 抵达（例如
 * 上一次点击的 `selectClipRemote`、或后端的周期性状态推送），没有这把锁它的
 * 旧快照会**覆盖掉正在预览的乐观值**——表现为拖拽中途画面突然弹回原位。
 * 旧实现的每个连续手势都成对调用（`useClipDrag` / `useEditDrag` /
 * `useSlipDrag` / `useSnapOffsetDrag` 各一处），内核路径此前只有 snap offset 接了。
 *
 * 【为什么用计数而不是布尔去重】底层 `sessionSlice` 的 `_interactionLockCount`
 * 是计数器：手势 1 的异步提交（`.finally` 释放）尚未回来时，用户可能已经开始
 * 手势 2。布尔去重会让手势 1 的释放把锁一次归零 —— 手势 2 拖拽期间抵达的
 * `*Remote.fulfilled` 旧快照不再被丢弃，乐观预览被覆盖（画面弹回原位）。
 * 计数语义下每个手势都入账，锁只在**最后一个手势结束**时释放，与后端同口径。
 */
export function useKernelGestureInteractionLock(args: {
    dispatch: AppDispatch;
    sessionRef: React.MutableRefObject<TimelineSessionSlice>;
}) {
    const { dispatch, sessionRef } = args;
    const kernelGestureInteractionDepthRef = React.useRef(0);
    const kernelGestureGeometryTokenRef = React.useRef<string | null>(null);

    /**
     * @param options.capturesClipGeometry 本手势是否改变 Clip 几何。默认 true。
     *   纯播放头 seek 不改变任何 clip，发布全量快照只会让参数编辑器白重算一次。
     */
    const beginKernelGestureInteraction = React.useCallback(
        (options?: { capturesClipGeometry?: boolean }): void => {
            kernelGestureInteractionDepthRef.current += 1;
            if (kernelGestureInteractionDepthRef.current > 1) return;
            // 记录**按下时**的全量 clip 几何，供参数编辑器把响度基线/曲线搬到正确位置。
            // 时序要求：必须早于本次手势的任何乐观写入 —— 本函数在首个真实位移帧、
            // 且在任何预览派发之前调用，满足该要求。重叠手势共用这一份按下时快照，
            // 映射存活到最后一个手势结束。
            if (options?.capturesClipGeometry !== false) {
                const preview = buildClipGeometryPreview(sessionRef.current.clips);
                const token = createClipGeometryPreviewToken();
                kernelGestureGeometryTokenRef.current = token;
                dispatch(beginClipGeometryPreview({ token, ...preview }));
            }
            dispatch(beginInteraction());
        },
        [dispatch, sessionRef],
    );

    const endKernelGestureInteraction = React.useCallback((): void => {
        if (kernelGestureInteractionDepthRef.current === 0) return;
        kernelGestureInteractionDepthRef.current -= 1;
        if (kernelGestureInteractionDepthRef.current > 0) return;
        // 只宣告手势结束；映射本身由消费方保留到「提交之后取的快照」落地（否则
        // 松手瞬间会退回旧位置的基线，表现为闪一下）。
        const token = kernelGestureGeometryTokenRef.current;
        if (token !== null) dispatch(endClipGeometryPreview(token));
        kernelGestureGeometryTokenRef.current = null;
        dispatch(endInteraction());
    }, [dispatch]);

    return {
        depthRef: kernelGestureInteractionDepthRef,
        begin: beginKernelGestureInteraction,
        end: endKernelGestureInteraction,
    };
}
