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
import { registerDragAbort } from "./gestureFocusGuard";
import type { ReactNode } from "react";

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
    buildCrossfadeGripInfoContent,
    buildCrossfadeGripInfoText,
    buildSingleFadeInfoContent,
    buildSingleFadeInfoText,
    publishFadeRichTooltip,
} from "./fadeTooltipText";
import type { FadeLengthFormatContext } from "./fadeTooltipText";
import type { Keybinding } from "../../../features/keybindings/types";
import { isNoneBinding } from "../../../features/keybindings/keybindingsSlice";
import { requestOpenFadeContextMenu, requestResetFadeCurvature } from "./fadeContextMenuBus";
import { noteFadeLinePointerDown } from "./hooks/fadeLineClickGesture";
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

type CrossfadeSides = { out: FadeContextSideLike; in: FadeContextSideLike };

type FadeContextSideLike = {
    clipId: string;
    isOut: boolean;
    shape: number;
    dir: number;
    lengthSec: number;
};

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
    /** 悬停信息浮标（纯文本回退 + 富内容经发布器注册）；缺省时不展示。 */
    tooltip?: string;
    /** 富内容节点（首行为内联曲线图标）。渲染时由回调 ref 发布。 */
    richTooltip?: ReactNode;
    /** 右键菜单载荷：该 zone 对应的包络侧（抓手为 null —— 由双侧快照承担）。 */
    contextSide?: FadeContextSideLike;
    /** 抓手专属：交叉点两侧包络快照。 */
    crossfadeSides?: CrossfadeSides;
    /** 是否为包络线本体命中（区别于区域边缘竖线）：双击重置仅对它生效。 */
    line?: boolean;
};

/**
 * pointerdown 现场判定循环修饰键是否按下（按下瞬间的事件本身是最可靠
 * 信号源；与 FadeHitLayer.cycleModifierHeld 同一套子集匹配规则）。
 */
function cycleModifierHeld(kb: Keybinding, event: PointerEvent): boolean {
    const requiredCtrl = kb.modifierOnly === true && kb.key === "control" ? true : Boolean(kb.ctrl);
    const requiredAlt = kb.modifierOnly === true && kb.key === "alt" ? true : Boolean(kb.alt);
    const requiredShift = kb.modifierOnly === true && kb.key === "shift" ? true : Boolean(kb.shift);
    return (
        (!requiredCtrl || event.ctrlKey || event.metaKey) &&
        (!requiredAlt || event.altKey) &&
        (!requiredShift || event.shiftKey)
    );
}

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
    seekFromClientX,
    fadeLengthFormatCtx,
    t,
    shapeCycleKb,
    onCrossfadeCycleClick,
    onFadeShapeSingleCycle,
}: {
    trackClips: ClipInfo[];
    pxPerSec: number;
    rowHeight: number;
    altPressed: boolean;
    /** 淡化长度 ToolTips 的相对时长时间上下文。 */
    fadeLengthFormatCtx: FadeLengthFormatContext;
    /** i18n 文案查询。 */
    t: (key: string) => string;
    /** 形状循环键绑定（抓手点击用）。 */
    shapeCycleKb: Keybinding | null;
    /** 抓手上的循环点击：同时切换交叉点两侧。 */
    onCrossfadeCycleClick?: (sides: Array<{ clipId: string; isOut: boolean }>) => void;
    /** 单条包络线上的循环点击（Ctrl+点击，非抓手）。 */
    onFadeShapeSingleCycle?: (side: { clipId: string; isOut: boolean }) => void;
    selectedClipId: string | null;
    multiSelectedClipIds: string[];
    multiSelectedSet: Set<string>;
    ensureSelected: (clipId: string) => void;
    selectClipRemote: (clipId: string) => void;
    recordLastClickPosition?: (clientX: number) => void;
    /** 播放头寻址（单击包络线 → 内侧边缘；交叉点 → 点击位置）。 */
    seekFromClientX: (clientX: number, commit: boolean) => void;
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
                        line: target.kind === "line",
                        tooltip: buildSingleFadeInfoText({
                            isOut: false,
                            shape: Number.isFinite(later.fadeInShape) ? later.fadeInShape : 0,
                            dir: later.fadeInDir ?? 0,
                            lengthSec: laterFadeInSec,
                            formatCtx: fadeLengthFormatCtx,
                            t,
                        }),
                        richTooltip: buildSingleFadeInfoContent({
                            isOut: false,
                            shape: Number.isFinite(later.fadeInShape) ? later.fadeInShape : 0,
                            dir: later.fadeInDir ?? 0,
                            lengthSec: laterFadeInSec,
                            formatCtx: fadeLengthFormatCtx,
                            t,
                        }),
                        contextSide: {
                            clipId: later.id,
                            isOut: false,
                            shape: Number.isFinite(later.fadeInShape) ? later.fadeInShape : 0,
                            dir: later.fadeInDir ?? 0,
                            lengthSec: laterFadeInSec,
                        },
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
                        line: target.kind === "line",
                        tooltip: buildSingleFadeInfoText({
                            isOut: true,
                            shape: Number.isFinite(earlier.fadeOutShape) ? earlier.fadeOutShape : 0,
                            dir: earlier.fadeOutDir ?? 0,
                            lengthSec: earlierFadeOutSec,
                            formatCtx: fadeLengthFormatCtx,
                            t,
                        }),
                        richTooltip: buildSingleFadeInfoContent({
                            isOut: true,
                            shape: Number.isFinite(earlier.fadeOutShape) ? earlier.fadeOutShape : 0,
                            dir: earlier.fadeOutDir ?? 0,
                            lengthSec: earlierFadeOutSec,
                            formatCtx: fadeLengthFormatCtx,
                            t,
                        }),
                        contextSide: {
                            clipId: earlier.id,
                            isOut: true,
                            shape: Number.isFinite(earlier.fadeOutShape) ? earlier.fadeOutShape : 0,
                            dir: earlier.fadeOutDir ?? 0,
                            lengthSec: earlierFadeOutSec,
                        },
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
                leftPx: laterStartPx + snapOffsetHandleXPx(later.snapOffsetSec, pxPerSec) - 4,
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
                    const gripRichTooltip = buildCrossfadeGripInfoContent({
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
                        richTooltip: gripRichTooltip,
                        crossfadeSides: {
                            out: {
                                clipId: earlier.id,
                                isOut: true,
                                shape: Number.isFinite(earlier.fadeOutShape)
                                    ? earlier.fadeOutShape
                                    : 0,
                                dir: earlier.fadeOutDir ?? 0,
                                lengthSec: earlierFadeOutSec,
                            },
                            in: {
                                clipId: later.id,
                                isOut: false,
                                shape: Number.isFinite(later.fadeInShape) ? later.fadeInShape : 0,
                                dir: later.fadeInDir ?? 0,
                                lengthSec: laterFadeInSec,
                            },
                        },
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

        const isInMultiSelect = multiSelectedClipIds.length > 0 && multiSelectedSet.has(clipId);
        const clipIsSelected =
            multiSelectedClipIds.length > 0 ? isInMultiSelect : selectedClipId === clipId;
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
        /** 手势延后判定：按下后未拖动即松开时触发（循环切换等点击语义）。 */
        deferredClick?: (ev: PointerEvent) => void,
    ) => {
        if (event.button !== 0) return;
        event.preventDefault();
        event.stopPropagation();

        // 选择/点选语义：按下时先确保该 clip 进入（多）选集合。
        const isInMultiSelect = multiSelectedClipIds.length > 0 && multiSelectedSet.has(clipId);
        const clipIsSelected =
            multiSelectedClipIds.length > 0 ? isInMultiSelect : selectedClipId === clipId;
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
        let fadePointerEnv: { envTopClientY: number; bodyHeightPx: number } | undefined;
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
                bodyHeightPx: Math.max(1, rowHeight - CLIP_BODY_PADDING_Y - CLIP_HEADER_HEIGHT),
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
        // 失焦取消：切屏期间 pointerup/pointercancel 不送达本窗口，blur
        // 时走与 onEnd 相同的收尾（真正的淡化/交叉拖拽由 useEditDrag
        // 自身的失焦守卫收尾并提交；此处只做监听清理）。失焦时手势并未
        // 完成——不合成"点击"语义（不触发循环切换/寻址）。
        let finished = false;
        const finish = () => {
            if (finished) return;
            finished = true;
            unregisterAbort();
            window.removeEventListener("pointermove", onMove, true);
            window.removeEventListener("pointerup", onEnd, true);
            window.removeEventListener("pointercancel", onEnd, true);
        };
        const onEnd = (ev: PointerEvent) => {
            if (ev.pointerId !== pointerId) return;
            window.removeEventListener("pointermove", onMove, true);
            window.removeEventListener("pointerup", onEnd, true);
            window.removeEventListener("pointercancel", onEnd, true);
            // 手势延后判定：未拖动即松开 = 点击语义（如 Ctrl 循环切换、单击寻址）。
            if (!dragStarted && deferredClick) {
                deferredClick(ev);
            }
        };
        const unregisterAbort = registerDragAbort(finish);
        window.addEventListener("pointermove", onMove, true);
        window.addEventListener("pointerup", onEnd, true);
        window.addEventListener("pointercancel", onEnd, true);
    };

    return (
        <div data-hs-overlap-layer="1" className="absolute inset-0 z-[200] pointer-events-none">
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
                    data-hs-crossfade-grip={zone.crossfadeSides ? "1" : undefined}
                    ref={(element) => {
                        publishFadeRichTooltip(element, zone.richTooltip ?? null);
                    }}
                    onPointerDown={(e) => {
                        // Ctrl（可配置）+左键按下：延后判定用户意图 ——
                        // 拖动超过阈值 = 抓手/淡化长度拖拽（保持原语义，
                        // 含 Ctrl 的反向缩放模式）；未拖动即松开 = 循环切换。
                        // 这样循环点击与交叉淡化手柄共用 Ctrl 也不再冲突。
                        const cycleHeld =
                            e.button === 0 &&
                            shapeCycleKb != null &&
                            !isNoneBinding(shapeCycleKb) &&
                            cycleModifierHeld(shapeCycleKb, e.nativeEvent);
                        if (cycleHeld && zone.type !== "snap_offset") {
                            startDeferredEdit(e, zone.clipId, zone.type, zone.partnerClipId, () => {
                                if (zone.crossfadeSides) {
                                    onCrossfadeCycleClick?.([
                                        {
                                            clipId: zone.crossfadeSides.out.clipId,
                                            isOut: true,
                                        },
                                        {
                                            clipId: zone.crossfadeSides.in.clipId,
                                            isOut: false,
                                        },
                                    ]);
                                } else if (zone.contextSide) {
                                    onFadeShapeSingleCycle?.(zone.contextSide);
                                }
                            });
                            return;
                        }
                        // 双击重置曲率：
                        // - 抓手（交叉点）= 同时重置两侧；
                        // - 包络线本体 = 只重置该侧；边缘竖线不参与。
                        // 循环键按住时循环优先，双击重置让位。
                        // 检测用时间窗 + zone 键（pointerdown detail 不可靠）。
                        const zoneClickKey = `${zone.clipId}:${zone.type}:${Math.round(zone.topPx)}:${zone.crossfadeSides ? "grip" : "env"}`;
                        const dblReset =
                            !cycleHeld &&
                            e.button === 0 &&
                            noteFadeLinePointerDown(zoneClickKey) === "double";
                        if (dblReset) {
                            if (zone.crossfadeSides) {
                                e.preventDefault();
                                e.stopPropagation();
                                requestResetFadeCurvature({
                                    sides: [
                                        {
                                            clipId: zone.crossfadeSides.out.clipId,
                                            isOut: true,
                                        },
                                        {
                                            clipId: zone.crossfadeSides.in.clipId,
                                            isOut: false,
                                        },
                                    ],
                                });
                                return;
                            }
                            if (zone.line === true && zone.contextSide) {
                                e.preventDefault();
                                e.stopPropagation();
                                requestResetFadeCurvature({
                                    sides: [
                                        {
                                            clipId: zone.contextSide.clipId,
                                            isOut: zone.contextSide.isOut,
                                        },
                                    ],
                                });
                                return;
                            }
                        }
                        if (zone.type === "snap_offset") {
                            startSnapOffsetEdit(e, zone.clipId);
                            return;
                        }
                        // 未拖动即松开 = 单击寻址：
                        // - 交叉点抓手 → 跳转到点击位置（保持原行为）；
                        // - 包络线本体 → 跳转到淡化区内侧边缘（淡入 → 右缘，
                        //   淡出 → 左缘）；
                        // - Clip 边缘（trim/stretch）→ 跳转到该边缘的准确位置。
                        // 双击重置分支已提前 return，第二击不会走到这里。
                        const zoneEl = e.currentTarget as HTMLElement;
                        startDeferredEdit(e, zone.clipId, zone.type, zone.partnerClipId, (ev) => {
                            if (zone.crossfadeSides) {
                                seekFromClientX(ev.clientX, true);
                                return;
                            }
                            const layerRoot = zoneEl.closest("[data-hs-overlap-layer]");
                            const clip = trackClips.find((entry) => entry.id === zone.clipId);
                            if (!clip || !layerRoot) return;
                            const baseLeft = layerRoot.getBoundingClientRect().left;
                            if (zone.type === "trim_left" || zone.type === "trim_right") {
                                // 边缘命中：跳到左缘（trim_left）或右缘（trim_right）。
                                const edgeSec =
                                    zone.type === "trim_left"
                                        ? clip.startSec
                                        : clip.startSec + clip.lengthSec;
                                seekFromClientX(baseLeft + edgeSec * pxPerSec, true);
                                return;
                            }
                            if (zone.line && zone.contextSide) {
                                const innerSec = zone.contextSide.isOut
                                    ? clip.startSec + clip.lengthSec - zone.contextSide.lengthSec
                                    : clip.startSec + zone.contextSide.lengthSec;
                                seekFromClientX(baseLeft + innerSec * pxPerSec, true);
                            }
                        });
                    }}
                    onContextMenu={(e) => {
                        e.preventDefault();
                        e.stopPropagation();
                        if (zone.crossfadeSides) {
                            // 抓手：双列 —— 先前者淡出，后后者淡入。
                            requestOpenFadeContextMenu({
                                clientX: e.clientX,
                                clientY: e.clientY,
                                primary: zone.crossfadeSides.out,
                                secondary: zone.crossfadeSides.in,
                            });
                            return;
                        }
                        if (!zone.contextSide) return;
                        requestOpenFadeContextMenu({
                            clientX: e.clientX,
                            clientY: e.clientY,
                            primary: zone.contextSide,
                            secondary: null,
                        });
                    }}
                />
            ))}
        </div>
    );
});
