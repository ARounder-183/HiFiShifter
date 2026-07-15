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
import type { ClipFormantMorph, ClipInfo } from "../../../features/session/sessionTypes";
import { CLIP_BODY_PADDING_Y, CLIP_HEADER_HEIGHT } from "./constants";
import { ClipEdgeHandles } from "./clip/ClipEdgeHandles";
import { ClipHeader } from "./clip/ClipHeader";

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
    onRenameCommit,
    onRenameDone,
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

    /** 外部触发重命名（来自右键菜单�?*/
    triggerRename?: boolean;
    onRenameCommit?: (clipId: string, newName: string) => void;
    onRenameDone?: () => void;
    onGainCommit?: (clipId: string, db: number) => void;
    onFormantMorphCommit?: (clipId: string, value: ClipFormantMorph, checkpoint: boolean) => void;
    activeGroupIds?: Set<string>;
    disabledGroupIds?: string[];
    onToggleGroupDisabled?: (groupId: string) => void;
    hovered?: boolean;
}) {
    const { t } = useI18n();

    const left = Math.max(0, Math.round(clip.startSec * pxPerSec));
    const width = Math.max(1, Math.round(clip.lengthSec * pxPerSec));
    const leadingOverlapPx = Math.max(
        0,
        Math.min(width, Math.round(Math.max(0, leadingOverlapSec) * pxPerSec)),
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
            // with Ctrl/Shift selection behavior.
            const altKeyDown = Boolean(e.altKey || e.nativeEvent.getModifierState?.("Alt"));
            const ctrlOrMeta = e.ctrlKey || e.metaKey;
            const doShiftRangeSelect = e.shiftKey && !altKeyDown && !ctrlOrMeta;
            const shiftRangeAnchorClipId = doShiftRangeSelect ? rangeSelectAnchorClipId : null;
            const doCtrlToggleOnly = ctrlOrMeta && !e.shiftKey && !altKeyDown;
            const shouldPrimeSelection = !doCtrlToggleOnly && !doShiftRangeSelect;

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
                    if (shouldPrimeSelection) {
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
                // we only check the physical Alt key to avoid breaking Ctrl/Shift
                // selection when those keys are configured as stretch modifiers.
                const altKeyDown = Boolean(e.altKey || e.nativeEvent.getModifierState?.("Alt"));
                const ctrlOrMeta = e.ctrlKey || e.metaKey;

                // Shift+点击范围选择在 pointerup 时处理（避免阻止拖动）
                const doShiftRangeSelect = e.shiftKey && !altKeyDown && !ctrlOrMeta;
                const shiftRangeAnchorClipId = doShiftRangeSelect ? rangeSelectAnchorClipId : null;
                const doCtrlToggleOnly = ctrlOrMeta && !e.shiftKey && !altKeyDown;
                const shouldPrimeSelection = !doCtrlToggleOnly && !doShiftRangeSelect;

                // Seek should happen on click, not on drag.
                // Track whether the pointer moved beyond a small deadzone.
                const allowSeek = !altKeyDown && !ctrlOrMeta && !e.shiftKey;
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
                        } else if (shouldPrimeSelection) {
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
            title={
                clip.midiNoteCount != null
                    ? `${t("clip_type_midi_prefix")} ${clip.name}`
                    : (clip.sourcePath ?? clip.name)
            }
        >
            <div
                className="absolute inset-0 overflow-visible"
                style={{
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
                    onRenameCommit={onRenameCommit}
                    onRenameDone={onRenameDone}
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
                        {/* Fade 角落 handle：始终存在，位于 body 左上�?右上角，用于�?0 开始拖拽出渐变 */}
                        {/* left-[10px]：避开左侧 edge handle 的 10px 宽度，确保两者不重叠 */}
                        <div
                            className="absolute left-[10px] top-0 w-[20px] h-[20px] z-[55]"
                            style={{ cursor: "nwse-resize" }}
                            onPointerDown={(e) => {
                                startDeferredFadeEditDrag(e, "fade_in");
                            }}
                            title={t("fade_in")}
                        />
                        {/* right-[10px]：避开右侧 edge handle 的 10px 宽度，确保两者不重叠 */}
                        <div
                            className="absolute right-[10px] top-0 w-[20px] h-[20px] z-[55]"
                            style={{ cursor: "nesw-resize" }}
                            onPointerDown={(e) => {
                                startDeferredFadeEditDrag(e, "fade_out");
                            }}
                            title={t("fade_out")}
                        />

                        {/* Fade handles: 操作区覆盖整�?fade 区域（fadeBeats > 0 时显示） */}
                        {(clip.fadeInSec ?? 0) > 0 && (
                            <div
                                className="absolute left-0 top-0 h-full z-[40] cursor-nwse-resize"
                                style={{
                                    width: Math.min(width, (clip.fadeInSec ?? 0) * pxPerSec),
                                }}
                                onPointerDown={(e) => {
                                    startDeferredFadeEditDrag(e, "fade_in");
                                }}
                                title={t("fade_in")}
                            ></div>
                        )}
                        {(clip.fadeOutSec ?? 0) > 0 && (
                            <div
                                className="absolute right-0 top-0 h-full z-[40] cursor-nesw-resize"
                                style={{
                                    width: Math.min(width, (clip.fadeOutSec ?? 0) * pxPerSec),
                                }}
                                onPointerDown={(e) => {
                                    startDeferredFadeEditDrag(e, "fade_out");
                                }}
                                title={t("fade_out")}
                            ></div>
                        )}

                        {/* 波形由 WaveformTrackCanvas（轨道级 Canvas）统一渲染，此处不再包含波形内容 */}
                    </div>
                </div>
            </div>
        </div>
    );
});
