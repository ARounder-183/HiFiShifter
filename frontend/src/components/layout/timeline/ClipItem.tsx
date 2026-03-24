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
import type { ClipInfo } from "../../../features/session/sessionTypes";
import { CLIP_BODY_PADDING_Y, CLIP_HEADER_HEIGHT } from "./constants";
import { fadeInAreaPath, fadeOutAreaPath } from "./paths";
import { ClipEdgeHandles } from "./clip/ClipEdgeHandles";
import { ClipHeader } from "./clip/ClipHeader";


type WaveformPreview = number[] | { l: number[]; r: number[] };

export const ClipItem = React.memo(function ClipItem({
    clip,
    rowHeight,
    pxPerSec,
    waveform: _waveform,
    altPressed = false,
    selected,
    isInMultiSelectedSet,
    multiSelectedCount,
    viewportStartSec: _viewportStartSec,
    viewportEndSec: _viewportEndSec,
    ensureSelected,
    selectClipRemote,
    openContextMenu,
    seekFromClientX,
    startClipDrag,
    startEditDrag,
    toggleClipMuted,
    toggleMultiSelect: _toggleMultiSelect,
    onShiftRangeSelect,
    clearContextMenu,
    triggerRename,
    onRenameCommit,
    onRenameDone,
    onGainCommit,
    trackColor,
}: {
    clip: ClipInfo;
    rowHeight: number;
    pxPerSec: number;
    waveform: WaveformPreview | undefined;
    altPressed?: boolean;
    selected: boolean;
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
    /** Ctrl+左键多选切换 */
    toggleMultiSelect: (clipId: string) => void;
    /** Shift+点击范围选择（跨轨按包围矩形选中） */
    onShiftRangeSelect: (clipId: string) => void;

    clearContextMenu: () => void;

    /** 外部触发重命名（来自右键菜单�?*/
    triggerRename?: boolean;
    onRenameCommit?: (clipId: string, newName: string) => void;
    onRenameDone?: () => void;
    onGainCommit?: (clipId: string, db: number) => void;
}) {
    const { t } = useI18n();

    const left = Math.max(0, clip.startSec * pxPerSec);
    const width = Math.max(1, clip.lengthSec * pxPerSec);
    const bodyHeight = Math.max(
        1,
        rowHeight - CLIP_BODY_PADDING_Y - CLIP_HEADER_HEIGHT,
    );

    const showRepeatMarker = false;
    const repeatMarkerX = 0;



    const startDeferredFadeEditDrag = React.useCallback(
        (
            e: React.PointerEvent<HTMLDivElement>,
            type: "fade_in" | "fade_out",
        ) => {
            e.preventDefault();
            e.stopPropagation();
            clearContextMenu();

            if (multiSelectedCount === 0 || !isInMultiSelectedSet) {
                ensureSelected(clip.id);
            }
            selectClipRemote(clip.id);

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
            seekFromClientX,
            selectClipRemote,
            startEditDrag,
        ],
    );

    return (
        <div
            data-hs-clip-item="1"
            className={`absolute cursor-pointer overflow-visible group ${clip.muted ? "opacity-60 grayscale" : "opacity-95"}`}
            style={{
                left,
                width,
                top: 0,
                height: rowHeight - CLIP_BODY_PADDING_Y,
                // 选中或 hover 时提升 z-index，确保 fade 手柄不被相邻 clip 遮挡
                zIndex: selected ? 2 : undefined,
            }}
            onMouseEnter={(e) => {
                (e.currentTarget as HTMLElement).style.zIndex = "2";
            }}
            onMouseLeave={(e) => {
                (e.currentTarget as HTMLElement).style.zIndex = selected
                    ? "2"
                    : "";
            }}
            onContextMenu={(e) => {
                e.preventDefault();
                e.stopPropagation();
                const keepExistingMultiSelection =
                    multiSelectedCount > 0 && isInMultiSelectedSet;
                if (!keepExistingMultiSelection) {
                    ensureSelected(clip.id);
                    selectClipRemote(clip.id);
                }
                openContextMenu(clip.id, e.clientX, e.clientY);
            }}
            onPointerDown={(e) => {
                if (e.button !== 0) return;

                const alt = Boolean(
                    altPressed ||
                    e.altKey ||
                    e.nativeEvent.getModifierState?.("Alt"),
                );

                // Shift+点击范围选择在 pointerup 时处理（避免阻止拖动）
                const doShiftRangeSelect = e.shiftKey && !alt && !e.ctrlKey && !e.metaKey;

                // Seek should happen on click, not on drag.
                // Track whether the pointer moved beyond a small deadzone.
                const allowSeek = !alt && !e.ctrlKey && !e.metaKey;
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
                    // Shift+点击且未移动时执行范围选择
                    if (doShiftRangeSelect && !moved) {
                        onShiftRangeSelect(clip.id);
                    } else if (!moved && allowSeek) {
                        seekFromClientX(ev.clientX, true);
                    }
                }

                window.addEventListener("pointermove", onMove, true);
                window.addEventListener("pointerup", onUp, true);
                window.addEventListener("pointercancel", onUp, true);

                e.preventDefault();
                e.stopPropagation();
                clearContextMenu();

                if (multiSelectedCount === 0 || !isInMultiSelectedSet) {
                    ensureSelected(clip.id);
                }
                selectClipRemote(clip.id);
                startClipDrag(e, clip.id, clip.startSec, alt);
            }}
            title={clip.sourcePath ?? clip.name}
        >
            <ClipEdgeHandles
                clipId={clip.id}
                altPressed={altPressed}
                multiSelectedCount={multiSelectedCount}
                isInMultiSelectedSet={isInMultiSelectedSet}
                ensureSelected={ensureSelected}
                selectClipRemote={selectClipRemote}
                seekFromClientX={seekFromClientX}
                startEditDrag={startEditDrag}
            />

            <ClipHeader
                clip={clip}
                clipWidthPx={width}
                ensureSelected={ensureSelected}
                selectClipRemote={selectClipRemote}
                startEditDrag={startEditDrag}
                toggleClipMuted={toggleClipMuted}
                isInMultiSelectedSet={isInMultiSelectedSet}
                multiSelectedCount={multiSelectedCount}
                triggerRename={triggerRename}
                onRenameCommit={onRenameCommit}
                onRenameDone={onRenameDone}
                onGainCommit={onGainCommit}
            />

            {/* Body block (does not fill the entire track row; leaves header lane above) */}
            <div
                className={`absolute left-0 right-0 bottom-0 rounded-sm shadow-sm overflow-visible border transition-colors ${selected
                        ? "border-white/90"
                        : "border-transparent group-hover:border-white/30"
                    }`}
                style={{
                    top: CLIP_HEADER_HEIGHT,
                    backgroundColor: trackColor
                        ? `color-mix(in oklab, ${trackColor} 30%, transparent)`
                        : "color-mix(in oklab, var(--qt-highlight) 35%, transparent)",
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
                                width: Math.min(
                                    width,
                                    (clip.fadeInSec ?? 0) * pxPerSec,
                                ),
                            }}
                            onPointerDown={(e) => {
                                startDeferredFadeEditDrag(e, "fade_in");
                            }}
                            title={t("fade_in")}
                        >
                            {/* 全区域条带：与可交互区域完全重合，右边缘竖线表示可拖拽边�?*/}
                            <div
                                className={
                                    "absolute inset-0 rounded-l-sm bg-white/8 border-r transition-opacity " +
                                    (selected
                                        ? "opacity-100 border-white/70"
                                        : "opacity-30 border-white/40 group-hover:opacity-100")
                                }
                            />
                        </div>
                    )}
                    {(clip.fadeOutSec ?? 0) > 0 && (
                        <div
                            className="absolute right-0 top-0 h-full z-[40] cursor-nesw-resize"
                            style={{
                                width: Math.min(
                                    width,
                                    (clip.fadeOutSec ?? 0) * pxPerSec,
                                ),
                            }}
                            onPointerDown={(e) => {
                                startDeferredFadeEditDrag(e, "fade_out");
                            }}
                            title={t("fade_out")}
                        >
                            {/* 全区域条带：与可交互区域完全重合，左边缘竖线表示可拖拽边�?*/}
                            <div
                                className={
                                    "absolute inset-0 rounded-r-sm bg-white/8 border-l transition-opacity " +
                                    (selected
                                        ? "opacity-100 border-white/70"
                                        : "opacity-30 border-white/40 group-hover:opacity-100")
                                }
                            />
                        </div>
                    )}

                    <div className="absolute inset-0 pointer-events-none z-30">
                        {showRepeatMarker ? (
                            <div
                                className="absolute top-0 bottom-0"
                                style={{
                                    left: Math.max(
                                        0,
                                        Math.min(width - 1, repeatMarkerX),
                                    ),
                                    width: 1,
                                    backgroundColor: "rgba(255,255,255,0.35)",
                                }}
                                title={t("repeat")}
                            />
                        ) : null}
                        {clip.fadeInSec > 0 ? (
                            <svg
                                className="absolute left-0 top-0 h-full"
                                width={Math.min(
                                    width,
                                    clip.fadeInSec * pxPerSec,
                                )}
                                height={bodyHeight}
                                viewBox={`0 0 ${Math.max(1, Math.min(width, clip.fadeInSec * pxPerSec))} ${Math.max(1, bodyHeight)}`}
                                preserveAspectRatio="none"
                            >
                                <path
                                    d={fadeInAreaPath(
                                        Math.max(
                                            1,
                                            Math.min(
                                                width,
                                                clip.fadeInSec * pxPerSec,
                                            ),
                                        ),
                                        Math.max(1, bodyHeight),
                                        24,
                                        clip.fadeInCurve ?? "sine",
                                    )}
                                    fill="rgba(255,255,255,0.14)"
                                    stroke="rgba(255,255,255,0.55)"
                                    strokeWidth="1"
                                    vectorEffect="non-scaling-stroke"
                                />
                            </svg>
                        ) : null}
                        {clip.fadeOutSec > 0 ? (
                            <svg
                                className="absolute right-0 top-0 h-full"
                                width={Math.min(
                                    width,
                                    clip.fadeOutSec * pxPerSec,
                                )}
                                height={bodyHeight}
                                viewBox={`0 0 ${Math.max(1, Math.min(width, clip.fadeOutSec * pxPerSec))} ${Math.max(1, bodyHeight)}`}
                                preserveAspectRatio="none"
                            >
                                <path
                                    d={fadeOutAreaPath(
                                        Math.max(
                                            1,
                                            Math.min(
                                                width,
                                                clip.fadeOutSec * pxPerSec,
                                            ),
                                        ),
                                        Math.max(1, bodyHeight),
                                        24,
                                        clip.fadeOutCurve ?? "sine",
                                    )}
                                    fill="rgba(255,255,255,0.14)"
                                    stroke="rgba(255,255,255,0.55)"
                                    strokeWidth="1"
                                    vectorEffect="non-scaling-stroke"
                                />
                            </svg>
                        ) : null}
                    </div>

                    {/* 波形由 WaveformTrackCanvas（轨道级 Canvas）统一渲染，此处不再包含波形内容 */}
                </div>
            </div>
        </div>
    );
});
