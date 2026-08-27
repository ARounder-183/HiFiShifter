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
import {
    CLIP_BODY_PADDING_Y,
    CLIP_HEADER_HEIGHT,
    SNAP_OFFSET_HANDLE_SIZE_PX,
    SNAP_OFFSET_HIT_HEIGHT_PX,
    snapOffsetHandleXPx,
} from "./constants";
import { buildFadeHitTargets } from "./fadeHitTargets";
import { fadeGainSigned } from "./reaperFade";
import { modifierWatcher } from "./hooks/modifierWatcher";
import {
    buildCrossfadeGripInfoText,
    buildSingleFadeInfoText,
} from "./fadeTooltipText";
import type { FadeLengthFormatContext } from "./fadeTooltipText";
import type { EditDragChannelOpts } from "./hooks/useEditDrag";

export type OverlapEditType =
    | "trim_left"
    | "trim_right"
    | "stretch_left"
    | "stretch_right"
    | "fade_in"
    | "fade_out"
    | "crossfade_edges"
    | "snap_offset";

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

/**
 * 计算两条“真实淡入淡出包络曲线”（非直线近似）在重叠区内的交点。
 *
 * 画布上用 fadeCurveGain 绘制的是曲线（sine/exponential/scurve 等），
 * 直接用两端点连线求交点会在 Y 轴明显偏离视觉交叉点。这里用二分法
 * 精确求解两条单调曲线的交点，使交叉点手柄正好落在用户看到的交叉处。
 *
 * @returns 交点坐标；若两条曲线在重叠淡化区间内不相交则返回 null。
 */
function computeCrossfadeGripPoint(args: {
    /** 前一个 clip 的右边缘 X（px，时间轴坐标）。 */
    earlierEndPx: number;
    /** 前一个 clip 淡出包络的像素宽度。 */
    earlierFadePx: number;
    earlierShape: number;
    earlierDir: number;
    /** 后一个 clip 的左边缘 X（px，时间轴坐标）。 */
    laterStartPx: number;
    /** 后一个 clip 淡入包络的像素宽度。 */
    laterFadePx: number;
    laterShape: number;
    laterDir: number;
    bodyTop: number;
    bodyHeight: number;
}): { x: number; y: number } | null {
    const {
        earlierEndPx,
        earlierFadePx,
        earlierShape,
        earlierDir,
        laterStartPx,
        laterFadePx,
        laterShape,
        laterDir,
        bodyTop,
        bodyHeight,
    } = args;
    const earlierLeftPx = earlierEndPx - earlierFadePx;
    const laterRightPx = laterStartPx + laterFadePx;
    const lo = Math.max(earlierLeftPx, laterStartPx);
    const hi = Math.min(earlierEndPx, laterRightPx);
    if (hi - lo <= 0.01 || earlierFadePx <= 0 || laterFadePx <= 0) return null;

    // 两条曲线在重叠淡化区的 X 区间单调：A 淡出 y 随 x 增大而增大，
    // B 淡入 y 随 x 增大而减小，因此 yA-yB 严格单调 → 二分求零点。
    const yDiff = (x: number): number => {
        const tA = (x - earlierLeftPx) / earlierFadePx;
        const gainA = fadeGainSigned(earlierShape, earlierDir, "out", tA);
        const yA = bodyTop + bodyHeight * (1 - gainA);
        const tB = (x - laterStartPx) / laterFadePx;
        const gainB = fadeGainSigned(laterShape, laterDir, "in", tB);
        const yB = bodyTop + bodyHeight * (1 - gainB);
        return yA - yB;
    };

    let low = lo;
    let high = hi;
    const fLow = yDiff(low);
    const fHigh = yDiff(high);

    // 不相交：视觉交叉点在两个淡化区之外，不显示手柄。
    if (fLow * fHigh > 0) {
        return null;
    }

    for (let i = 0; i < 40; i += 1) {
        const mid = (low + high) / 2;
        const fMid = yDiff(mid);
        if (Math.abs(fMid) < 1e-3) {
            low = high = mid;
            break;
        }
        if (fLow * fMid < 0) {
            high = mid;
        } else {
            low = mid;
        }
    }

    const x = (low + high) / 2;
    const tA = (x - earlierLeftPx) / earlierFadePx;
    const gainA = fadeGainSigned(earlierShape, earlierDir, "out", tA);
    return {
        x,
        y: bodyTop + bodyHeight * (1 - gainA),
    };
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
    /** 交叉点拖拽时的另一个 clip id；仅 type === "crossfade_edges" 时有意义。 */
    partnerClipId?: string;
    /** 显式层叠优先级（交叉点手柄应高于所有淡入淡出/边缘控件）。 */
    zIndex?: number;
    /** 悬停信息浮标文本（data-tooltip）；缺省时不展示。 */
    tooltip?: string;
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
    startSnapOffsetDrag,
    fadeLengthFormatCtx,
    t,
}: {
    trackClips: ClipInfo[];
    pxPerSec: number;
    rowHeight: number;
    altPressed: boolean;
    /** 淡化长度 ToolTips 的相对时长时间上下文。 */
    fadeLengthFormatCtx: FadeLengthFormatContext;
    /** i18n 文案查询。 */
    t: (key: string) => string;
    selectedClipId: string | null;
    multiSelectedClipIds: string[];
    multiSelectedSet: Set<string>;
    ensureSelected: (clipId: string) => void;
    selectClipRemote: (clipId: string) => void;
    recordLastClickPosition?: (clientX: number) => void;
    startEditDrag: (
        e: React.PointerEvent,
        clipId: string,
        type: Exclude<OverlapEditType, "snap_offset">,
        channel?: EditDragChannelOpts,
    ) => void;
    /** SnapOffset 角部拖拽入口（重叠区左下角仍可调整吸附偏移）。 */
    startSnapOffsetDrag?: (e: React.PointerEvent, clipId: string) => void;
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
                    fadeInShape: Number.isFinite(later.fadeInShape) ? later.fadeInShape : 0,
                    fadeInDir: later.fadeInDir ?? 0,
                    fadeOutShape: 0,
                    fadeOutDir: 0,
                    clipXFrom: overlapStartPx,
                    clipXTo: overlapEndPx,
                });
                for (let targetIndex = 0; targetIndex < targets.length; targetIndex += 1) {
                    const target = targets[targetIndex];
                    zones.push({
                        key: `${earlier.id}:${later.id}:later-fade:${targetIndex}`,
                        clipId: later.id,
                        type: "fade_in",
                        leftPx: target.left,
                        widthPx: target.width,
                        topPx: target.top,
                        heightPx: target.height,
                        cursor: "nwse-resize",
                        tooltip: buildSingleFadeInfoText({
                            isOut: false,
                            shape: Number.isFinite(later.fadeInShape) ? later.fadeInShape : 0,
                            dir: later.fadeInDir ?? 0,
                            lengthSec: laterFadeInSec,
                            formatCtx: fadeLengthFormatCtx,
                            t,
                        }),
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
                    fadeInShape: 0,
                    fadeInDir: 0,
                    fadeOutShape: Number.isFinite(earlier.fadeOutShape) ? earlier.fadeOutShape : 0,
                    fadeOutDir: earlier.fadeOutDir ?? 0,
                    clipXFrom: overlapStartPx,
                    clipXTo: overlapEndPx,
                });
                for (let targetIndex = 0; targetIndex < targets.length; targetIndex += 1) {
                    const target = targets[targetIndex];
                    zones.push({
                        key: `${earlier.id}:${later.id}:earlier-fade:${targetIndex}`,
                        clipId: earlier.id,
                        type: "fade_out",
                        leftPx: target.left,
                        widthPx: target.width,
                        topPx: target.top,
                        heightPx: target.height,
                        cursor: "nesw-resize",
                        tooltip: buildSingleFadeInfoText({
                            isOut: true,
                            shape: Number.isFinite(earlier.fadeOutShape) ? earlier.fadeOutShape : 0,
                            dir: earlier.fadeOutDir ?? 0,
                            lengthSec: earlierFadeOutSec,
                            formatCtx: fadeLengthFormatCtx,
                            t,
                        }),
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

            // ── SnapOffset 命中区：**跟随后一个 clip 的 ◣ 三角位置**（偏移
            // 换算同渲染，左竖边严格对齐偏移值），zIndex 高于边缘/淡化控件；
            // 保证重叠区的吸附偏移仍可调整、与淡入淡出无关）。──────────
            zones.push({
                key: `${earlier.id}:${later.id}:later-snap-offset`,
                clipId: later.id,
                type: "snap_offset",
                leftPx:
                    laterStartPx +
                    snapOffsetHandleXPx(later.snapOffsetSec, pxPerSec) -
                    4,
                widthPx: SNAP_OFFSET_HANDLE_SIZE_PX + 5,
                topPx: rowHeight - SNAP_OFFSET_HIT_HEIGHT_PX,
                heightPx: SNAP_OFFSET_HIT_HEIGHT_PX,
                cursor: "ew-resize",
                zIndex: 320,
            });

            // ── 4) 交叉点手柄：两条淡入淡出包络线（按实际曲线）的交点 ─────────────
            // 拖动它 = 同时移动前 clip 的右缘（结束位置）与后 clip 的左缘（起始位置），
            // 相对偏移方式，保持两个 clip 的重叠长度不变（因此手动/自动淡化长度都不变）。
            if (laterFadeInSec > 0 && earlierFadeOutSec > 0) {
                const fA = Math.min(earlierFadeOutSec * pxPerSec, earlierEndPx - earlierStartPx);
                const fB = Math.min(laterFadeInSec * pxPerSec, laterEndPx - laterStartPx);
                const grip = computeCrossfadeGripPoint({
                    earlierEndPx,
                    earlierFadePx: fA,
                    earlierShape: Number.isFinite(earlier.fadeOutShape) ? earlier.fadeOutShape : 0,
                    earlierDir: earlier.fadeOutDir ?? 0,
                    laterStartPx,
                    laterFadePx: fB,
                    laterShape: Number.isFinite(later.fadeInShape) ? later.fadeInShape : 0,
                    laterDir: later.fadeInDir ?? 0,
                    bodyTop: bodyTopPx,
                    bodyHeight: bodyHeightPx,
                });
                if (grip) {
                    const gripSize = 16;
                    const gripTooltip = buildCrossfadeGripInfoText({
                        earlier: {
                            shape: Number.isFinite(earlier.fadeOutShape) ? earlier.fadeOutShape : 0,
                            dir: earlier.fadeOutDir ?? 0,
                            lengthSec: earlierFadeOutSec,
                        },
                        later: {
                            shape: Number.isFinite(later.fadeInShape) ? later.fadeInShape : 0,
                            dir: later.fadeInDir ?? 0,
                            lengthSec: laterFadeInSec,
                        },
                        formatCtx: fadeLengthFormatCtx,
                        t,
                    });
                    zones.push({
                        key: `${earlier.id}:${later.id}:crossfade-grip`,
                        clipId: earlier.id,
                        partnerClipId: later.id,
                        type: "crossfade_edges",
                        leftPx: grip.x - gripSize / 2,
                        widthPx: gripSize,
                        topPx: grip.y - gripSize / 2,
                        heightPx: gripSize,
                        // Alt（曲率修饰键）按住时光标切换，提示"拖动曲线"模式。
                        cursor: altPressed ? "move" : "ew-resize",
                        tooltip: gripTooltip,
                        zIndex: 300,
                    });
                }
            }
        }
    }

    if (zones.length === 0) return null;

    // SnapOffset 角区处理：立即进入偏移拖拽（无位移阈值），选择预备语义
    // 与其他区域一致。
    const startSnapOffsetEdit = (event: React.PointerEvent, clipId: string) => {
        if (event.button !== 0) return;
        event.preventDefault();
        event.stopPropagation();

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

        startSnapOffsetDrag?.(event, clipId);
    };

    const startDeferredEdit = (
        event: React.PointerEvent,
        clipId: string,
        type: Exclude<OverlapEditType, "snap_offset">,
        partnerClipId?: string,
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
        // 曲率拖拽环境：本层坐标即 lane 坐标，gain=1 基准线的客户 y =
        // 层容器 top + bodyTopPx；高度为行高推导的 body 高度。
        let fadePointerEnv:
            | { envTopClientY: number; bodyHeightPx: number }
            | undefined;
        {
            const layerRoot = currentTarget.closest("[data-hs-overlap-layer]");
            const hitTop = currentTarget.getBoundingClientRect().top;
            const hitTopLocal = Number((currentTarget as HTMLElement).dataset.hsZoneTop);
            const baseTopClientY =
                layerRoot != null
                    ? layerRoot.getBoundingClientRect().top
                    : hitTop - (Number.isFinite(hitTopLocal) ? hitTopLocal : 0);
            fadePointerEnv = {
                envTopClientY: baseTopClientY + CLIP_HEADER_HEIGHT,
                bodyHeightPx: Math.max(
                    1,
                    rowHeight - CLIP_BODY_PADDING_Y - CLIP_HEADER_HEIGHT,
                ),
            };
        }
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
                    currentTarget,
                    nativeEvent: ev,
                    preventDefault: () => {},
                    stopPropagation: () => {},
                } as unknown as React.PointerEvent,
                clipId,
                type,
                // 类型化通道：交叉配对与曲率环境不再走私到事件对象上。
                {
                    dragStartClientX: startX,
                    crossfadePartnerClipId: partnerClipId,
                    fadePointerEnv,
                },
            );
            // 手势开始：用起手事件初始化全局修饰键快照（此后每帧由
            // useEditDrag 从原生 pointermove 持续自愈）。
            modifierWatcher.refreshFromEvent(ev);
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
                        zIndex: zone.zIndex,
                    }}
                    data-hs-zone-top={String(zone.topPx)}
                    data-tooltip={zone.tooltip}
                    onPointerDown={(e) => {
                        if (zone.type === "snap_offset") {
                            startSnapOffsetEdit(e, zone.clipId);
                            return;
                        }
                        startDeferredEdit(e, zone.clipId, zone.type, zone.partnerClipId);
                    }}
                />
            ))}
        </div>
    );
});
