import React, { useRef, useState } from "react";
import type { ClipFormantMorph, ClipInfo } from "../../../../features/session/sessionTypes";
import { CLIP_HEADER_HEIGHT } from "../constants";
import { formatGainDbValue, gainToDb } from "../math";
import { AppTooltipBubble } from "../../../../components/AppTooltip";
import { useI18n } from "../../../../i18n/I18nProvider";
import { useAppTheme } from "../../../../theme/AppThemeProvider";
import { resolveTimelineClipHeaderVisibility } from "../runtime/timelineClipHeaderVisibility";
import { buildTimelineClipVisualStyle } from "../runtime/timelineCanvasStyle";
import { ClipFormantButton } from "./ClipFormantButton";

export const ClipHeader: React.FC<{
    clip: ClipInfo;
    clipWidthPx: number;
    trackColor?: string;
    transparentVisuals?: boolean;
    isPitchAdjustment?: boolean;
    startEditDrag: (e: React.PointerEvent, clipId: string, type: "gain") => void;
    toggleClipMuted: (clipId: string, nextMuted: boolean) => void;
    /** 触发内联重命名（由 ClipContextMenu 的"重命名"菜单项调用） */
    triggerRename?: boolean;
    onRenameCommit?: (clipId: string, newName: string) => void;
    onRenameDone?: () => void;
    /** 增益提交（dB 值；输入框提交会 clamp 到 -12~+12，双击旋钮/数值标签重置为 0 dB） */
    onGainCommit?: (clipId: string, db: number) => void;
    onFormantMorphCommit?: (clipId: string, value: ClipFormantMorph, checkpoint: boolean) => void;
    onToggleGroupDisabled?: (groupId: string) => void;
    activeGroupIds?: Set<string>;
    disabledGroupIds?: string[];
}> = ({
    clip,
    clipWidthPx,
    trackColor,
    transparentVisuals = false,
    isPitchAdjustment = false,
    startEditDrag,
    toggleClipMuted,
    triggerRename = false,
    onRenameCommit,
    onRenameDone,
    onGainCommit,
    onToggleGroupDisabled,
    activeGroupIds,
    disabledGroupIds,
}) => {
    const { t } = useI18n();
    const { mode, fontFamily } = useAppTheme();
    const isDark = mode === "dark";
    const gainDb = gainToDb(clip.gain);
    const clampedGainDb = Math.min(12, Math.max(-12, gainDb));
    const gainKnobDeg = (clampedGainDb / 12) * 135;
    const [gainDragBaseDb, setGainDragBaseDb] = useState<number | null>(null);
    const [gainHovered, setGainHovered] = useState(false);
    const [gainTooltipPos, setGainTooltipPos] = useState<{ x: number; y: number } | null>(null);
    const gainTooltip =
        gainDragBaseDb == null
            ? t("gain_value_tooltip").replace("{gain}", formatGainDbValue(clampedGainDb))
            : t("gain_value_tooltip_drag")
                  .replace("{gain}", formatGainDbValue(clampedGainDb))
                  .replace("{delta}", formatGainDbValue(clampedGainDb - gainDragBaseDb));
    const showGainTooltip = gainHovered || gainDragBaseDb != null;
    const clipTooltipText =
        clip.midiNoteCount != null
            ? `${t("clip_type_midi_prefix")} ${clip.name}`
            : (clip.sourcePath ?? clip.name);

    // 根据 clip 像素宽度决定显示哪些元素（从右往左依次隐藏）
    // >= 152px: 全显示 | 116-152: 隐藏名称 | 96-116: 隐藏播放速率 | 68-96: 隐藏增益值+F | 52-68: 隐藏F | 32-52: 只留增益旋钮 | < 32px: 全隐藏
    const {
        showAny,
        showChain,
        showMute,
        showFormant,
        showGainKnob,
        showPlaybackRate,
        showGainLabel: showGainVal,
        showName,
    } = resolveTimelineClipHeaderVisibility(clipWidthPx, isPitchAdjustment);
    const visualStyle = buildTimelineClipVisualStyle({
        widthPx: clipWidthPx,
        trackColor,
        selected: false,
        muted: Boolean(clip.muted),
        gain: clip.gain,
        playbackRate: clip.playbackRate,
        name: clip.name,
        fontFamily,
        isPitchAdjustment,
    });

    // ── 增益数值输入框 ──────────────────────────────────────────────────────
    const [gainEditing, setGainEditing] = useState(false);
    const [gainInputVal, setGainInputVal] = useState("");
    const gainInputRef = useRef<HTMLInputElement>(null);

    function commitGainEdit() {
        const parsed = parseFloat(gainInputVal);
        if (!isNaN(parsed)) {
            // clamp 到 -12 ~ +12 dB
            const clamped = Math.min(12, Math.max(-12, parsed));
            onGainCommit?.(clip.id, clamped);
        }
        setGainEditing(false);
    }

    function cancelGainEdit() {
        setGainEditing(false);
    }

    // ── 名称内联编辑 ────────────────────────────────────────────────────────
    const [nameEditing, setNameEditing] = useState(false);
    const [nameInputVal, setNameInputVal] = useState("");
    const nameInputRef = useRef<HTMLInputElement>(null);

    // 外部触发重命名（来自右键菜单）
    React.useEffect(() => {
        if (triggerRename && !nameEditing) {
            setNameInputVal(clip.name);
            setNameEditing(true);
            setTimeout(() => {
                nameInputRef.current?.select();
            }, 0);
        }
    }, [triggerRename]);

    function commitNameEdit() {
        const trimmed = nameInputVal.trim();
        const finalName = trimmed.length > 0 ? trimmed : clip.name;
        onRenameCommit?.(clip.id, finalName);
        setNameEditing(false);
        onRenameDone?.();
    }

    function cancelNameEdit() {
        setNameEditing(false);
        onRenameDone?.();
    }

    if (!showAny) return null;
    const hideVisuals = transparentVisuals && !nameEditing && !gainEditing;

    return (
        <div
            className="absolute left-1 right-1 flex items-center gap-1 z-50 select-none"
            style={{
                top: 1,
                height: CLIP_HEADER_HEIGHT,
            }}
        >
            {/* 增益拖拽把手 */}
            {showGainKnob && (
                <div
                    aria-label={gainTooltip}
                    style={{ cursor: "ns-resize", opacity: hideVisuals ? 0 : 1 }}
                    onPointerEnter={(e) => {
                        setGainHovered(true);
                        setGainTooltipPos({ x: e.clientX, y: e.clientY });
                    }}
                    onPointerMove={(e) => {
                        setGainTooltipPos({ x: e.clientX, y: e.clientY });
                    }}
                    onPointerLeave={() => {
                        setGainHovered(false);
                    }}
                    onPointerDown={(e) => {
                        e.preventDefault();
                        e.stopPropagation();
                        setGainHovered(true);
                        setGainTooltipPos({ x: e.clientX, y: e.clientY });

                        const pointerId = e.pointerId;
                        const targetEl = e.currentTarget as HTMLElement;
                        const startX = e.clientX;
                        const startY = e.clientY;
                        let dragStarted = false;

                        const onMove = (ev: PointerEvent) => {
                            if (ev.pointerId !== pointerId) return;
                            setGainTooltipPos({ x: ev.clientX, y: ev.clientY });
                            if (dragStarted) return;
                            const dx = ev.clientX - startX;
                            const dy = ev.clientY - startY;
                            if (dx * dx + dy * dy < 9) return;
                            dragStarted = true;
                            setGainDragBaseDb(clampedGainDb);
                            startEditDrag(
                                {
                                    button: 0,
                                    pointerId,
                                    currentTarget: targetEl,
                                } as unknown as React.PointerEvent,
                                clip.id,
                                "gain",
                            );
                        };

                        const onEnd = (ev: PointerEvent) => {
                            if (ev.pointerId !== pointerId) return;
                            const knobRect = targetEl.getBoundingClientRect();
                            const stillOverKnob =
                                ev.clientX >= knobRect.left &&
                                ev.clientX <= knobRect.right &&
                                ev.clientY >= knobRect.top &&
                                ev.clientY <= knobRect.bottom;
                            setGainHovered(stillOverKnob);
                            setGainDragBaseDb(null);
                            window.removeEventListener("pointermove", onMove, true);
                            window.removeEventListener("pointerup", onEnd, true);
                            window.removeEventListener("pointercancel", onEnd, true);
                        };

                        window.addEventListener("pointermove", onMove, true);
                        window.addEventListener("pointerup", onEnd, true);
                        window.addEventListener("pointercancel", onEnd, true);
                    }}
                    onDoubleClick={(e) => {
                        // 双击旋钮重置为 0 dB
                        e.preventDefault();
                        e.stopPropagation();
                        onGainCommit?.(clip.id, 0);
                    }}
                >
                    <div
                        className="relative rounded-full border"
                        style={{
                            width: visualStyle.gainKnobRadius * 2 + 4,
                            height: visualStyle.gainKnobRadius * 2 + 4,
                            borderColor: visualStyle.gainKnobStroke,
                            backgroundColor: visualStyle.gainKnobFill,
                        }}
                    >
                        <span
                            className="absolute left-1/2 top-1/2 w-[2px] h-[7px] -translate-x-1/2 -translate-y-full rounded-full"
                            style={{
                                backgroundColor: visualStyle.gainKnobIndicator,
                                transform: `translate(-50%, -100%) rotate(${gainKnobDeg}deg)`,
                                transformOrigin: "50% 100%",
                            }}
                        />
                        <span
                            className="absolute left-1/2 top-1/2 h-[4px] w-[4px] -translate-x-1/2 -translate-y-1/2 rounded-full"
                            style={{ backgroundColor: visualStyle.gainKnobCoreFill }}
                        />
                    </div>
                </div>
            )}

            {/* 编组链按钮 */}
            {clip.groupId != null &&
                showChain &&
                (() => {
                    const isGroupActive =
                        activeGroupIds != null && activeGroupIds.has(clip.groupId);
                    const isGroupDisabled =
                        disabledGroupIds != null && disabledGroupIds.includes(clip.groupId);

                    let bgColor: string;
                    let borderColor: string;
                    let iconColor: string;
                    let boxShadow: string | undefined;
                    let iconSvg: React.ReactNode;
                    let title: string;

                    if (isGroupDisabled) {
                        bgColor = "rgba(220, 70, 70, 0.45)";
                        borderColor = "rgba(200, 50, 50, 0.80)";
                        iconColor = "rgba(180, 40, 40, 1)";
                        boxShadow = "0 0 4px rgba(200, 50, 50, 0.50)";
                        title = t("enable_group");
                        iconSvg = (
                            <svg
                                width="10"
                                height="10"
                                viewBox="0 0 24 24"
                                fill="none"
                                stroke="currentColor"
                                strokeWidth="2.5"
                                strokeLinecap="round"
                                strokeLinejoin="round"
                            >
                                <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71" />
                                <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71" />
                                <line x1="2" y1="2" x2="22" y2="22" />
                            </svg>
                        );
                    } else if (isGroupActive) {
                        bgColor = "rgba(255, 200, 50, 0.55)";
                        borderColor = "rgba(255, 200, 50, 0.90)";
                        iconColor = "rgba(200, 140, 20, 1)";
                        boxShadow = "0 0 4px rgba(255, 200, 50, 0.50)";
                        title = t("disable_group");
                        iconSvg = (
                            <svg
                                width="10"
                                height="10"
                                viewBox="0 0 24 24"
                                fill="none"
                                stroke="currentColor"
                                strokeWidth="2.5"
                                strokeLinecap="round"
                                strokeLinejoin="round"
                            >
                                <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71" />
                                <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71" />
                            </svg>
                        );
                    } else {
                        bgColor = visualStyle.muteBadgeFill;
                        borderColor = visualStyle.muteBadgeStroke;
                        iconColor = "rgba(200, 200, 210, 1)";
                        boxShadow = undefined;
                        title = t("disable_group");
                        iconSvg = (
                            <svg
                                width="10"
                                height="10"
                                viewBox="0 0 24 24"
                                fill="none"
                                stroke="currentColor"
                                strokeWidth="2.5"
                                strokeLinecap="round"
                                strokeLinejoin="round"
                            >
                                <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71" />
                                <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71" />
                            </svg>
                        );
                    }

                    return (
                        <button
                            className="rounded flex items-center justify-center border transition-all text-[10px] font-bold"
                            onPointerDown={(e) => {
                                e.preventDefault();
                                e.stopPropagation();
                            }}
                            onClick={(e) => {
                                e.preventDefault();
                                e.stopPropagation();
                                onToggleGroupDisabled?.(clip.groupId!);
                            }}
                            data-tooltip={title}
                            style={{
                                opacity: hideVisuals ? 0 : 1,
                                width: visualStyle.muteBadgeWidth,
                                height: visualStyle.muteBadgeHeight,
                                backgroundColor: bgColor,
                                borderColor: borderColor,
                                color: iconColor,
                                boxShadow: boxShadow,
                            }}
                        >
                            {iconSvg}
                        </button>
                    );
                })()}

            {/* 静音按钮 */}
            {showMute && (
                <button
                    className="rounded flex items-center justify-center border transition-all text-[10px] font-bold"
                    onPointerDown={(e) => {
                        e.preventDefault();
                        e.stopPropagation();
                    }}
                    onClick={(e) => {
                        e.preventDefault();
                        e.stopPropagation();
                        toggleClipMuted(clip.id, !clip.muted);
                    }}
                    data-tooltip={clip.muted ? t("clip_unmute") : t("clip_mute")}
                    style={{
                        opacity: hideVisuals ? 0 : 1,
                        width: visualStyle.muteBadgeWidth,
                        height: visualStyle.muteBadgeHeight,
                        backgroundColor: visualStyle.muteBadgeFill,
                        borderColor: visualStyle.muteBadgeStroke,
                        color: visualStyle.muteBadgeTextFill,
                    }}
                >
                    M
                </button>
            )}

            <ClipFormantButton
                clip={clip}
                hidden={!showFormant}
                opacity={hideVisuals ? 0 : 1}
                width={visualStyle.muteBadgeWidth}
                height={visualStyle.muteBadgeHeight}
                baseBackgroundColor={visualStyle.muteBadgeFill}
                baseBorderColor={visualStyle.muteBadgeStroke}
                baseTextColor={visualStyle.muteBadgeTextFill}
            />

            {/* Clip 名称区域 */}
            {showName && (
                <div className="flex-1 min-w-0">
                    {nameEditing ? (
                        <input
                            ref={nameInputRef}
                            className="w-full text-xs font-medium rounded px-1 outline-none"
                            style={{
                                color: isDark ? "rgba(255,255,255,0.95)" : "rgba(0,0,0,0.88)",
                                backgroundColor: isDark
                                    ? "rgba(0,0,0,0.45)"
                                    : "rgba(255,255,255,0.70)",
                                border: `1px solid ${isDark ? "rgba(255,255,255,0.40)" : "rgba(0,0,0,0.35)"}`,
                            }}
                            value={nameInputVal}
                            onChange={(e) => setNameInputVal(e.target.value)}
                            onKeyDown={(e) => {
                                e.stopPropagation();
                                if (e.key === "Enter") commitNameEdit();
                                if (e.key === "Escape") cancelNameEdit();
                            }}
                            onBlur={commitNameEdit}
                            onPointerDown={(e) => e.stopPropagation()}
                            onDoubleClick={(e) => e.stopPropagation()}
                        />
                    ) : (
                        <div
                            className="text-xs font-medium drop-shadow-md truncate cursor-default"
                            data-tooltip={clipTooltipText}
                            style={{
                                color: visualStyle.textFill,
                                opacity: hideVisuals ? 0 : 1,
                            }}
                            onDoubleClick={(e) => {
                                e.preventDefault();
                                e.stopPropagation();
                                setNameInputVal(clip.name);
                                setNameEditing(true);
                                setTimeout(() => nameInputRef.current?.select(), 0);
                            }}
                        >
                            {clip.name}
                        </div>
                    )}
                </div>
            )}

            {/* 播放倍率 / 增益数值显示 */}
            {showGainVal && (
                <div className="ml-auto flex items-center gap-2 min-w-0">
                    {showPlaybackRate && (
                        <div
                            className="text-[10px] tracking-wide"
                            style={{
                                color: "rgba(208, 216, 223, 0.76)",
                                opacity: hideVisuals ? 0 : 1,
                            }}
                        >
                            {visualStyle.playbackRateLabel}
                        </div>
                    )}
                    {gainEditing ? (
                        <input
                            ref={gainInputRef}
                            className="w-14 text-xs rounded px-1 outline-none text-right"
                            style={{
                                color: isDark ? "rgba(255,255,255,0.94)" : "rgba(0,0,0,0.88)",
                                backgroundColor: isDark
                                    ? "rgba(0,0,0,0.45)"
                                    : "rgba(255,255,255,0.70)",
                                border: `1px solid ${isDark ? "rgba(255,255,255,0.40)" : "rgba(0,0,0,0.35)"}`,
                            }}
                            value={gainInputVal}
                            onChange={(e) => setGainInputVal(e.target.value)}
                            onKeyDown={(e) => {
                                e.stopPropagation();
                                if (e.key === "Enter") commitGainEdit();
                                if (e.key === "Escape") cancelGainEdit();
                            }}
                            onBlur={commitGainEdit}
                            onPointerDown={(e) => e.stopPropagation()}
                        />
                    ) : (
                        <div
                            className="text-xs drop-shadow-md cursor-ns-resize"
                            style={{
                                color: "rgba(233, 239, 244, 0.82)",
                                opacity: hideVisuals ? 0 : 1,
                            }}
                            onDoubleClick={(e) => {
                                // 双击数值重置为 0 dB
                                e.preventDefault();
                                e.stopPropagation();
                                onGainCommit?.(clip.id, 0);
                            }}
                        >
                            {clampedGainDb >= 0 ? "+" : ""}
                            {clampedGainDb.toFixed(1)}dB
                        </div>
                    )}
                </div>
            )}
            <AppTooltipBubble
                text={gainTooltip}
                position={showGainTooltip ? gainTooltipPos : null}
            />
        </div>
    );
};
