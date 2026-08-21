/**
 * ClipItem 组件
 *
 * 时间轴上单个音频 Clip 的渲染组件，负责：
 * - 淡入/淡出可视化和交互手柄
 * - Clip 的选中、拖拽、右键菜单等交互逻辑
 * - 支持 trim/stretch 编辑手柄
 *
 * 波形渲染由 WaveformTrackCanvas（轨道级 Canvas）统一负责，
 * ClipItem 仅提供 DOM 交互层。
 */
import React from "react";

import { useI18n } from "../../../i18n/I18nProvider";
import { isPrimaryModifierDown } from "../../../utils/platform";
import type { ClipFormantMorph, ClipInfo } from "../../../features/session/sessionTypes";
import { CLIP_BODY_PADDING_Y, CLIP_HEADER_HEIGHT } from "./constants";
import { FadeHitLayer } from "./FadeHitLayer";
import { ClipEdgeHandles } from "./clip/ClipEdgeHandles";
import {
    ClipHeader,
    type ClipRenameClickCandidate,
    type ClipRenameController,
} from "./clip/ClipHeader";

const LEADING_OVERLAP_ALPHA = 0.5;

export const ClipItem = React.memo(function ClipItem({
    clip,
    rowHeight,
    pxPerSec,
    altPressed = false,
    selected,
    leadingOverlapSec = 0,
    isInMultiSelectedSet,
    multiSelectedCount,
    viewportStartSec,
    viewportEndSec,
    ensureSelected,
    selectClipRemote,
    openContextMenu,
    trackColor,
    seekFromClientX,
    startClipDrag,
    startEditDrag,
    toggleClipMuted,
    onCtrlToggleSelect,
    toggleMultiSelect: _toggleMultiSelect,
    onShiftRangeSelect,
    rangeSelectAnchorClipId,
    recordLastClickPosition,
    clearContextMenu,
    triggerRename,
    onRenameStart,
    onRenameCommit,
    onRenameDone,
    onRenameClickCandidate,
    onGainCommit,
    onFormantMorphCommit,
    activeGroupIds,
    disabledGroupIds,
    onToggleGroupDisabled,
    hovered = false,
}: {
    clip: ClipInfo;
    rowHeight: number;
    pxPerSec: number;
    altPressed?: boolean;
    selected: boolean;
    /** 该 clip 左侧前导重叠时长（秒），只用于重叠区可视化混合 */
    leadingOverlapSec?: number;
    isInMultiSelectedSet: boolean;
    multiSelectedCount: number;
    /** 可视区开始时间（秒） */
    viewportStartSec?: number;
    /** 可视区结束时间（秒） */
    viewportEndSec?: number;

    ensureSelected: (clipId: string) => void;
    selectClipRemote: (clipId: string) => void;
    openContextMenu: (clipId: string, clientX: number, clientY: number) => void;

    /** 轨道主题色，用于 Clip 背景色和选中边框 */
    trackColor?: string;
    seekFromClientX: (clientX: number, commit: boolean) => void;
    startClipDrag: (
        e: React.PointerEvent<HTMLDivElement>,
        clipId: string,
        clipstartSec: number,
        altPressedHint?: boolean,
    ) => void;
    startEditDrag: (
        e: React.PointerEvent,
        clipId: string,
        type:
            | "trim_left"
            | "trim_right"
            | "stretch_left"
            | "stretch_right"
            | "fade_in"
            | "fade_out"
            | "gain",
    ) => void;
    toggleClipMuted: (clipId: string, nextMuted: boolean) => void;
    /** Ctrl+左键选择切换（会更新主选中 clip） */
    onCtrlToggleSelect: (clipId: string) => void;
    /** Ctrl+左键多选切换 */
    toggleMultiSelect: (clipId: string) => void;
    /** Shift+点击范围选择（跨轨按包围矩形选中）；targetClientX 用于基于鼠标位置构建矩形 */
    onShiftRangeSelect: (
        clipId: string,
        anchorClipIdOverride?: string | null,
        targetClientX?: number,
    ) => void;
    /** Shift 范围选择锚点（点击前快照） */
    rangeSelectAnchorClipId: string | null;
    /** 记录最近的点击 clientX，用于 Shift 范围选择的锚点位置 */
    recordLastClickPosition?: (clientX: number) => void;

    clearContextMenu: () => void;

    /** 外部触发重命名（来自右键菜单） */
    triggerRename?: boolean;
    onRenameStart?: (clipId: string) => void;
    onRenameCommit?: (clipId: string, newName: string) => void;
    onRenameDone?: () => void;
    onRenameClickCandidate?: (candidate: ClipRenameClickCandidate | null) => void;
    onGainCommit?: (clipId: string, db: number) => void;
    onFormantMorphCommit?: (clipId: string, value: ClipFormantMorph, checkpoint: boolean) => void;
    activeGroupIds?: Set<string>;
    disabledGroupIds?: string[];
    onToggleGroupDisabled?: (groupId: string) => void;
    hovered?: boolean;
}) {
    const { t } = useI18n();
    const renameControllerRef = React.useRef<ClipRenameController | null>(null);

    // 不要对 left/width 取整：背景网格与时间标尺均按浮点像素位置绘制。
    // 若这里 Math.round，Clip 会相对网格最多向右偏 0.5px；在常用缩放
    // (100-200 px/s) 下就是约 2.5-5ms 的“网格偏右”观感，且随 pxPerSec
    // 与 BPM 改变而变化。保留浮点像素可让 Clip 与网格完全对齐。
    const left = Math.max(0, clip.startSec * pxPerSec);
    const width = Math.max(1, clip.lengthSec * pxPerSec);
    // body 区高度（与 WaveformTrackCanvas 一致）：轨道高 - 上下 padding - 头部高。
    const bodyHeight = Math.max(1, rowHeight - CLIP_BODY_PADDING_Y - CLIP_HEADER_HEIGHT);
    // 有效 fade = 自动交叉淡化（>0 时覆盖）否则手动 fade（对齐 REAPER 分离存储模型）。
    const effectiveFadeInSec = (clip.autoFadeInSec ?? 0) > 0 ? clip.autoFadeInSec! : (clip.fadeInSec ?? 0);
    const effectiveFadeOutSec = (clip.autoFadeOutSec ?? 0) > 0 ? clip.autoFadeOutSec! : (clip.fadeOutSec ?? 0);
    const leadingOverlapPx = Math.max(
        0,
        Math.min(width, Math.max(0, leadingOverlapSec) * pxPerSec),
    );
    const leadingOverlapMaskImage =
        leadingOverlapPx > 0
            ? `linear-gradient(to right, rgba(0,0,0,${LEADING_OVERLAP_ALPHA}) 0px, rgba(0,0,0,${LEADING_OVERLAP_ALPHA}) ${leadingOverlapPx}px, rgba(0,0,0,1) ${leadingOverlapPx}px, rgba(0,0,0,1) 100%)`
            : undefined;

    const isGroupHighlighted =
        activeGroupIds != null && clip.groupId != null && activeGroupIds.has(clip.groupId);

    const interactionHintBoxShadow =
        selected && isGroupHighlighted
            ? // blue inner ring (selected) + golden outer ring (grouped)
              "0 0 0 1px rgba(156, 196, 255, 0.68), 0 0 0 2px rgba(156, 196, 255, 0.16), 0 0 0 3px rgba(255, 200, 50, 0.60), 0 0 0 4px rgba(255, 200, 50, 0.18)"
            : selected
              ? "0 0 0 1px rgba(156, 196, 255, 0.68), 0 0 0 2px rgba(156, 196, 255, 0.16)"
              : isGroupHighlighted
                ? "0 0 0 1px rgba(255, 200, 50, 0.60), 0 0 0 2px rgba(255, 200, 50, 0.18)"
                : hovered && clip.groupId == null
                  ? "0 0 0 1px rgba(255, 255, 255, 0.24)"
                  : undefined;

    const startDeferredFadeEditDrag = React.useCallback(
        (e: React.PointerEvent<HTMLDivElement>, type: "fade_in" | "fade_out") => {
            e.preventDefault();
            e.stopPropagation();
            clearContextMenu();

            // Only check physical Alt key for click-selection bypass.
            // altPressed tracks the stretch modifier and must not interfere
            // with primary-modifier/Shift selection behavior.
            const altKeyDown = Boolean(e.altKey || e.nativeEvent.getModifierState?.("Alt"));
            const primaryModifierDown = isPrimaryModifierDown(e);
            const doShiftRangeSelect = e.shiftKey && !altKeyDown && !primaryModifierDown;
            const shiftRangeAnchorClipId = doShiftRangeSelect ? rangeSelectAnchorClipId : null;
            const doCtrlToggleOnly = primaryModifierDown && !e.shiftKey && !altKeyDown;
            const shouldPrimeSelection = !doCtrlToggleOnly && !doShiftRangeSelect;
            const primedSelection = shouldPrimeSelection && !selected;

            if (primedSelection) {
                ensureSelected(clip.id);
                selectClipRemote(clip.id);
                recordLastClickPosition?.(e.clientX);
            }

            const startX = e.clientX;
            const startY = e.clientY;
            const pointerId = e.pointerId;
            const targetEl = e.currentTarget as HTMLElement;
            let dragStarted = false;

            const onMove = (ev: PointerEvent) => {
                if (ev.pointerId !== pointerId || dragStarted) return;
                const dx = ev.clientX - startX;
                const dy = ev.clientY - startY;
                if (dx * dx + dy * dy < 9) return;
                dragStarted = true;
                startEditDrag(
                    {
                        button: 0,
                        pointerId,
                        clientX: startX,
                        dragStartClientX: startX,
                        currentTarget: targetEl,
                    } as unknown as React.PointerEvent,
                    clip.id,
                    type,
                );
            };

            const onEnd = (ev: PointerEvent) => {
                if (ev.pointerId !== pointerId) return;
                window.removeEventListener("pointermove", onMove, true);
                window.removeEventListener("pointerup", onEnd, true);
                window.removeEventListener("pointercancel", onEnd, true);
                if (!dragStarted) {
                    if (doCtrlToggleOnly) {
                        onCtrlToggleSelect(clip.id);
                        return;
                    }
                    if (doShiftRangeSelect) {
                        onShiftRangeSelect(clip.id, shiftRangeAnchorClipId, startX);
                        return;
                    }
                    if (shouldPrimeSelection && !primedSelection) {
                        if (multiSelectedCount !== 1 || !isInMultiSelectedSet) {
                            ensureSelected(clip.id);
                        }
                        selectClipRemote(clip.id);
                        recordLastClickPosition?.(e.clientX);
                    }
                    seekFromClientX(ev.clientX, true);
                }
            };

            window.addEventListener("pointermove", onMove, true);
            window.addEventListener("pointerup", onEnd, true);
            window.addEventListener("pointercancel", onEnd, true);
        },
        [
            clearContextMenu,
            clip.id,
            ensureSelected,
            isInMultiSelectedSet,
            multiSelectedCount,
            onCtrlToggleSelect,
            onShiftRangeSelect,
            rangeSelectAnchorClipId,
            recordLastClickPosition,
            seekFromClientX,
            selectClipRemote,
            selected,
            startEditDrag,
            altPressed,
        ],
    );

    // ========================================
    // DOM 视口剔除
    // ========================================
    if (viewportStartSec !== undefined && viewportEndSec !== undefined) {
        const clipEndSec = clip.startSec + clip.lengthSec;
        // 增加 1.5 秒的缓冲余量，防止快速滚动时边缘 DOM 突然卸载造成的闪烁
        const bufferSec = 1.5;
        if (
            clipEndSec < viewportStartSec - bufferSec ||
            clip.startSec > viewportEndSec + bufferSec
        ) {
            // 完全在屏幕/缓冲带之外，直接卸载此 Clip 的一切 DOM 节点
            return null;
        }
    }

    return (
        <div
            data-hs-clip-item="1"
            className="absolute overflow-visible group"
            style={{
                left,
                width,
                top: 0,
                height: rowHeight - CLIP_BODY_PADDING_Y,
                boxShadow: interactionHintBoxShadow,
                // 名称编辑时整块 DOM 需要压过 timeline Canvas（zIndex:1），
                // 否则 Canvas 中的原始名称会把输入框盖住。
                zIndex: triggerRename ? 60 : undefined,
            }}
            onPointerDownCapture={(e) => {
                // 正在编辑名称时，点击 Clip 内输入框以外的任意位置都先提交编辑。
                // 输入框自身的 pointerdown 会在命中 input 时跳过。
                const target = e.target as HTMLElement | null;
                const isInputTarget =
                    target?.closest?.("input,textarea,select,[contenteditable='true']") != null;
                const controller = renameControllerRef.current;
                if (!isInputTarget && controller?.isEditing()) {
                    controller.commit();
                }
            }}
            onContextMenu={(e) => {
                e.preventDefault();
                e.stopPropagation();
                const keepExistingMultiSelection = multiSelectedCount > 1;
                if (!keepExistingMultiSelection) {
                    ensureSelected(clip.id);
                    selectClipRemote(clip.id);
                }
                openContextMenu(clip.id, e.clientX, e.clientY);
            }}
            onPointerDown={(e) => {
                if (e.button !== 0) return;

                // altPressed tracks the stretch modifier (configurable), used only
                // for edge-handle behavior. For click-selection bypass (slip-edit),
                // we only check the physical Alt key to avoid breaking primary-modifier/Shift
                // selection when those keys are configured as stretch modifiers.
                const altKeyDown = Boolean(e.altKey || e.nativeEvent.getModifierState?.("Alt"));
                const primaryModifierDown = isPrimaryModifierDown(e);

                // Shift+点击范围选择在 pointerup 时处理（避免阻止拖动）
                const doShiftRangeSelect = e.shiftKey && !altKeyDown && !primaryModifierDown;
                const shiftRangeAnchorClipId = doShiftRangeSelect ? rangeSelectAnchorClipId : null;
                const doCtrlToggleOnly = primaryModifierDown && !e.shiftKey && !altKeyDown;
                const shouldPrimeSelection = !doCtrlToggleOnly && !doShiftRangeSelect;
                const primedSelection = shouldPrimeSelection && !selected;

                if (primedSelection) {
                    ensureSelected(clip.id);
                    selectClipRemote(clip.id);
                    recordLastClickPosition?.(e.clientX);
                }

                // Seek should happen on click, not on drag.
                // Track whether the pointer moved beyond a small deadzone.
                const allowSeek = !altKeyDown && !primaryModifierDown && !e.shiftKey;
                const startX = e.clientX;
                const startY = e.clientY;
                let moved = false;

                function onMove(ev: PointerEvent) {
                    if (ev.pointerId !== e.pointerId) return;
                    const dx = ev.clientX - startX;
                    const dy = ev.clientY - startY;
                    if (dx * dx + dy * dy >= 9) moved = true;
                }

                function onUp(ev: PointerEvent) {
                    if (ev.pointerId !== e.pointerId) return;
                    window.removeEventListener("pointermove", onMove, true);
                    window.removeEventListener("pointerup", onUp, true);
                    window.removeEventListener("pointercancel", onUp, true);
                    if (!moved) {
                        if (doShiftRangeSelect) {
                            onShiftRangeSelect(clip.id, shiftRangeAnchorClipId, startX);
                        } else if (shouldPrimeSelection && !primedSelection) {
                            if (multiSelectedCount !== 1 || !isInMultiSelectedSet) {
                                ensureSelected(clip.id);
                            }
                            selectClipRemote(clip.id);
                            recordLastClickPosition?.(e.clientX);
                        }
                        if (allowSeek) {
                            seekFromClientX(ev.clientX, true);
                        }
                    }
                }

                window.addEventListener("pointermove", onMove, true);
                window.addEventListener("pointerup", onUp, true);
                window.addEventListener("pointercancel", onUp, true);

                e.preventDefault();
                e.stopPropagation();
                clearContextMenu();

                startClipDrag(e, clip.id, clip.startSec, false);
            }}
        >
            <div
                className="absolute inset-0 overflow-visible"
                style={{
                    // 每个 clip 保持独立的层叠上下文（原设计）：重叠时的“同时可编辑”
                    // 由 TrackLane 的 OverlapEditLayer（z 高于一切 clip item）确定性提供，
                    // 这里不再依赖手柄 z 穿透其它 clip 的 body。
                    transform: "translateZ(0)",
                    backfaceVisibility: "hidden",
                    WebkitMaskImage: leadingOverlapMaskImage,
                    maskImage: leadingOverlapMaskImage,
                    WebkitMaskRepeat: leadingOverlapMaskImage ? "no-repeat" : undefined,
                    maskRepeat: leadingOverlapMaskImage ? "no-repeat" : undefined,
                    WebkitMaskSize: leadingOverlapMaskImage ? "100% 100%" : undefined,
                    maskSize: leadingOverlapMaskImage ? "100% 100%" : undefined,
                }}
            >
                <ClipEdgeHandles
                    clipId={clip.id}
                    altPressed={altPressed}
                    isInMultiSelectedSet={isInMultiSelectedSet}
                    multiSelectedCount={multiSelectedCount}
                    ensureSelected={ensureSelected}
                    selectClipRemote={selectClipRemote}
                    onCtrlToggleSelect={onCtrlToggleSelect}
                    onShiftRangeSelect={onShiftRangeSelect}
                    rangeSelectAnchorClipId={rangeSelectAnchorClipId}
                    recordLastClickPosition={recordLastClickPosition}
                    seekFromClientX={seekFromClientX}
                    startEditDrag={startEditDrag}
                />

                <ClipHeader
                    clip={clip}
                    clipWidthPx={width}
                    trackColor={trackColor}
                    transparentVisuals
                    isPitchAdjustment={clip.midiNoteCount != null}
                    startEditDrag={startEditDrag}
                    toggleClipMuted={toggleClipMuted}
                    triggerRename={triggerRename}
                    onRenameStart={onRenameStart}
                    onRenameCommit={onRenameCommit}
                    onRenameDone={onRenameDone}
                    onRenameClickCandidate={onRenameClickCandidate}
                    renameControllerRef={renameControllerRef}
                    onGainCommit={onGainCommit}
                    onFormantMorphCommit={onFormantMorphCommit}
                    onToggleGroupDisabled={onToggleGroupDisabled}
                    activeGroupIds={activeGroupIds}
                    disabledGroupIds={disabledGroupIds}
                />

                {/* Body block (does not fill the entire track row; leaves header lane above) */}
                <div
                    className="absolute left-0 right-0 bottom-0 overflow-visible"
                    style={{
                        top: CLIP_HEADER_HEIGHT,
                    }}
                >
                    {/* Body (waveform + edit handles) */}
                    <div className="absolute inset-0">
                        {/* Fade 角落创建/编辑手柄：始终存在（即使当前无淡化），
                            可从此拖拽“造出一个”淡化；淡化存在时也是有效抓取点
                            （对齐 REAPER 顶部角落三角）。完全透明、不做悬停高亮，
                            仅以 resize 光标提示。left/right 10px 避开
                            ClipEdgeHandles 的 10px 宽度。 */}
                        <div
                            className="absolute left-[10px] top-0 w-[16px] h-[16px] z-[55]"
                            style={{ cursor: "nwse-resize" }}
                            onPointerDown={(e) => {
                                startDeferredFadeEditDrag(e, "fade_in");
                            }}
                            data-tooltip={t("fade_in")}
                        />
                        <div
                            className="absolute right-[10px] top-0 w-[16px] h-[16px] z-[55]"
                            style={{ cursor: "nesw-resize" }}
                            onPointerDown={(e) => {
                                startDeferredFadeEditDrag(e, "fade_out");
                            }}
                            data-tooltip={t("fade_out")}
                        />

                        {/* Fade 拖拽控件：抓「绘制的包络线」和「淡化区域边缘竖线」，
                            而非整片淡入淡出区域（对齐 REAPER）。命中块很小，未命中处
                            会自然穿透到 clip body（拖拽移动 clip）。 */}
                        {(effectiveFadeInSec > 0 || effectiveFadeOutSec > 0) && (
                            <FadeHitLayer
                                clipLeftPx={0}
                                clipWidthPx={width}
                                bodyTop={0}
                                bodyHeight={bodyHeight}
                                fadeInPx={
                                    effectiveFadeInSec > 0
                                        ? Math.min(width, effectiveFadeInSec * pxPerSec)
                                        : 0
                                }
                                fadeOutPx={
                                    effectiveFadeOutSec > 0
                                        ? Math.min(width, effectiveFadeOutSec * pxPerSec)
                                        : 0
                                }
                                fadeInCurve={clip.fadeInCurve}
                                fadeOutCurve={clip.fadeOutCurve}
                                zIndex={40}
                                onFadeInPointerDown={(e) => startDeferredFadeEditDrag(e, "fade_in")}
                                onFadeOutPointerDown={(e) =>
                                    startDeferredFadeEditDrag(e, "fade_out")
                                }
                            />
                        )}

                        {/* 波形由 WaveformTrackCanvas（轨道级 Canvas）统一渲染，此处不再包含波形内容 */}
                    </div>
                </div>
            </div>
        </div>
    );
});
