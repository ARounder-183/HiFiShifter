/**
 * OverlapEditLayer — 交叉（重叠）区域内的“确定性可编辑层”。
 *
 * 背景：当两个 clip 在时间上重叠（如交叉淡化）时，它们各自的 DOM 边缘/淡入淡出
 * 手柄会互相遮盖——同一段屏幕空间同时属于两个 clip。单纯依赖 z-index/绘制顺序
 * 无法保证“两个 clip 的交叉处都可编辑”（要么后绘制覆盖先前的，要么反过来）。
 *
 * 本层用一个 z 高于所有 ClipItem 的独立层（z-[200]），在每一对重叠 clip 之间
 * **按位置**提供双方的编辑控件，交互模型与常规 clip 完全一致——抓的是
 * “绘制的包络线”和“淡化区域的边缘竖线”，而不是整片淡入淡出区域：
 *   - 重叠区左侧 → 后一个 clip：左边缘（截短/拉伸）+ 其淡入控件；
 *   - 重叠区右侧 → 前一个 clip：右边缘（截短/拉伸）+ 其淡出控件。
 *
 * 命中优先级（确定性，由 DOM 层叠顺序保证）：
 *   1. clip 自身的起始/终止边缘（trim/stretch）最优先——当淡入区域最右侧
 *      竖线恰好与前一个 clip 的终止边缘重合、或淡出区域最左侧竖线恰好与
 *      后一个 clip 的起始边缘重合时，一律触发“clip 边缘”交互。
 *   2. 淡化区域的边缘竖线（淡入最右侧 / 淡出最左侧）其次；
 *   3. 包络线命中块最次。
 */
import React from "react";
import type { ClipInfo } from "../../../features/session/sessionTypes";
import { CLIP_BODY_PADDING_Y, CLIP_HEADER_HEIGHT } from "./constants";
import { buildFadeHitTargets } from "./fadeHitTargets";

export type OverlapEditType =
    | "trim_left"
    | "trim_right"
    | "stretch_left"
    | "stretch_right"
    | "fade_in"
    | "fade_out";

function overlapLengthSec(
    a: { startSec: number; lengthSec: number },
    b: { startSec: number; lengthSec: number },
): number {
    const aEnd = a.startSec + a.lengthSec;
    const bEnd = b.startSec + b.lengthSec;
    return Math.min(aEnd, bEnd) - Math.max(a.startSec, b.startSec);
}

/** 有效 fade = 自动交叉淡化（>0 时覆盖）否则手动 fade（与 ClipItem / 渲染一致）。 */
function effectiveFadeInSec(clip: ClipInfo): number {
    return (clip.autoFadeInSec ?? 0) > 0 ? (clip.autoFadeInSec ?? 0) : (clip.fadeInSec ?? 0);
}
function effectiveFadeOutSec(clip: ClipInfo): number {
    return (clip.autoFadeOutSec ?? 0) > 0 ? (clip.autoFadeOutSec ?? 0) : (clip.fadeOutSec ?? 0);
}

type EditZone = {
    key: string;
    clipId: string;
    type: OverlapEditType;
    leftPx: number;
    widthPx: number;
    topPx: number;
    heightPx: number;
    cursor: string;
};

export const OverlapEditLayer = React.memo(function OverlapEditLayer({
    trackClips,
    pxPerSec,
    rowHeight,
    altPressed,
    selectedClipId,
    multiSelectedClipIds,
    multiSelectedSet,
    ensureSelected,
    selectClipRemote,
    recordLastClickPosition,
    startEditDrag,
}: {
    trackClips: ClipInfo[];
    pxPerSec: number;
    rowHeight: number;
    altPressed: boolean;
    selectedClipId: string | null;
    multiSelectedClipIds: string[];
    multiSelectedSet: Set<string>;
    ensureSelected: (clipId: string) => void;
    selectClipRemote: (clipId: string) => void;
    recordLastClickPosition?: (clientX: number) => void;
    startEditDrag: (
        e: React.PointerEvent,
        clipId: string,
        type: OverlapEditType,
    ) => void;
}) {
    const zones: EditZone[] = [];

    const bodyTopPx = CLIP_HEADER_HEIGHT;
    const bodyHeightPx = Math.max(1, rowHeight - CLIP_BODY_PADDING_Y - CLIP_HEADER_HEIGHT);
    const clipEdgeWidthPx = 10;

    for (let i = 0; i < trackClips.length; i += 1) {
        for (let j = i + 1; j < trackClips.length; j += 1) {
            const a = trackClips[i];
            const b = trackClips[j];
            if (overlapLengthSec(a, b) <= 1e-6) continue;

            // 时间上“前一个”= 起点更早；若起点相同则按 id 稳定排序。
            let earlier = a;
            let later = b;
            if (
                later.startSec < earlier.startSec ||
                (later.startSec === earlier.startSec && later.id < earlier.id)
            ) {
                [earlier, later] = [later, earlier];
            }

            const earlierStartPx = earlier.startSec * pxPerSec;
            const earlierEndPx = (earlier.startSec + earlier.lengthSec) * pxPerSec;
            const laterStartPx = later.startSec * pxPerSec;
            const laterEndPx = (later.startSec + later.lengthSec) * pxPerSec;
            const overlapStartPx = Math.max(earlierStartPx, laterStartPx);
            const overlapEndPx = Math.min(earlierEndPx, laterEndPx);
            if (overlapEndPx - overlapStartPx <= 0.5) continue;

            // ── 1) 后一个 clip 的淡入（只取重叠区内部分）──────────────────
            const laterFadeInSec = effectiveFadeInSec(later);
            if (laterFadeInSec > 0) {
                const targets = buildFadeHitTargets({
                    clipLeftPx: laterStartPx,
                    clipWidthPx: laterEndPx - laterStartPx,
                    bodyTop: bodyTopPx,
                    bodyHeight: bodyHeightPx,
                    fadeInPx: laterFadeInSec * pxPerSec,
                    fadeOutPx: 0,
                    fadeInCurve: later.fadeInCurve,
                    fadeOutCurve: "sine",
                    clipXFrom: overlapStartPx,
                    clipXTo: overlapEndPx,
                });
                for (let t = 0; t < targets.length; t += 1) {
                    const target = targets[t];
                    zones.push({
                        key: `${earlier.id}:${later.id}:later-fade:${t}`,
                        clipId: later.id,
                        type: "fade_in",
                        leftPx: target.left,
                        widthPx: target.width,
                        topPx: target.top,
                        heightPx: target.height,
                        cursor: "nwse-resize",
                    });
                }
            }

            // ── 2) 前一个 clip 的淡出（只取重叠区内部分）──────────────────
            const earlierFadeOutSec = effectiveFadeOutSec(earlier);
            if (earlierFadeOutSec > 0) {
                const targets = buildFadeHitTargets({
                    clipLeftPx: earlierStartPx,
                    clipWidthPx: earlierEndPx - earlierStartPx,
                    bodyTop: bodyTopPx,
                    bodyHeight: bodyHeightPx,
                    fadeInPx: 0,
                    fadeOutPx: earlierFadeOutSec * pxPerSec,
                    fadeInCurve: "sine",
                    fadeOutCurve: earlier.fadeOutCurve,
                    clipXFrom: overlapStartPx,
                    clipXTo: overlapEndPx,
                });
                for (let t = 0; t < targets.length; t += 1) {
                    const target = targets[t];
                    zones.push({
                        key: `${earlier.id}:${later.id}:earlier-fade:${t}`,
                        clipId: earlier.id,
                        type: "fade_out",
                        leftPx: target.left,
                        widthPx: target.width,
                        topPx: target.top,
                        heightPx: target.height,
                        cursor: "nesw-resize",
                    });
                }
            }

            // ── 3) clip 自身的起始/终止边缘（最优先渲染 = 最顶层）─────────
            // 后一个 clip 的左边缘（进入重叠区的一侧）。
            zones.push({
                key: `${earlier.id}:${later.id}:later-edge`,
                clipId: later.id,
                type: altPressed ? "stretch_left" : "trim_left",
                leftPx: laterStartPx - clipEdgeWidthPx / 2,
                widthPx: clipEdgeWidthPx,
                topPx: 0,
                heightPx: rowHeight,
                cursor: altPressed ? "col-resize" : "ew-resize",
            });
            // 前一个 clip 的右边缘（离开重叠区的一侧）。
            zones.push({
                key: `${earlier.id}:${later.id}:earlier-edge`,
                clipId: earlier.id,
                type: altPressed ? "stretch_right" : "trim_right",
                leftPx: earlierEndPx - clipEdgeWidthPx / 2,
                widthPx: clipEdgeWidthPx,
                topPx: 0,
                heightPx: rowHeight,
                cursor: altPressed ? "col-resize" : "ew-resize",
            });
        }
    }

    if (zones.length === 0) return null;

    const startDeferredEdit = (
        event: React.PointerEvent,
        clipId: string,
        type: OverlapEditType,
    ) => {
        if (event.button !== 0) return;
        event.preventDefault();
        event.stopPropagation();

        // 选择/点选语义：按下时先确保该 clip 进入（多）选集合。
        const isInMultiSelect =
            multiSelectedClipIds.length > 0 && multiSelectedSet.has(clipId);
        const clipIsSelected =
            multiSelectedClipIds.length > 0
                ? isInMultiSelect
                : selectedClipId === clipId;
        if (!clipIsSelected) {
            if (!isInMultiSelect || multiSelectedClipIds.length > 1) {
                ensureSelected(clipId);
            }
            selectClipRemote(clipId);
            recordLastClickPosition?.(event.clientX);
        }

        const startX = event.clientX;
        const startY = event.clientY;
        const pointerId = event.pointerId;
        const currentTarget = event.currentTarget as HTMLElement;
        let dragStarted = false;

        const onMove = (ev: PointerEvent) => {
            if (dragStarted || ev.pointerId !== pointerId) return;
            const dx = ev.clientX - startX;
            const dy = ev.clientY - startY;
            if (dx * dx + dy * dy < 81) return;
            dragStarted = true;
            startEditDrag(
                {
                    button: 0,
                    pointerId,
                    clientX: ev.clientX,
                    clientY: ev.clientY,
                    dragStartClientX: startX,
                    altKey: ev.altKey,
                    metaKey: ev.metaKey,
                    ctrlKey: ev.ctrlKey,
                    shiftKey: ev.shiftKey,
                    currentTarget,
                    nativeEvent: ev,
                    preventDefault: () => {},
                    stopPropagation: () => {},
                } as unknown as React.PointerEvent,
                clipId,
                type,
            );
        };
        const onEnd = (ev: PointerEvent) => {
            if (ev.pointerId !== pointerId) return;
            window.removeEventListener("pointermove", onMove, true);
            window.removeEventListener("pointerup", onEnd, true);
            window.removeEventListener("pointercancel", onEnd, true);
        };
        window.addEventListener("pointermove", onMove, true);
        window.addEventListener("pointerup", onEnd, true);
        window.addEventListener("pointercancel", onEnd, true);
    };

    return (
        <div
            data-hs-overlap-layer="1"
            className="absolute inset-0 z-[200] pointer-events-none"
        >
            {zones.map((zone) => (
                <div
                    key={zone.key}
                    className="absolute pointer-events-auto"
                    style={{
                        left: zone.leftPx,
                        width: zone.widthPx,
                        top: zone.topPx,
                        height: zone.heightPx,
                        cursor: zone.cursor,
                    }}
                    onPointerDown={(e) => startDeferredEdit(e, zone.clipId, zone.type)}
                />
            ))}
        </div>
    );
});
