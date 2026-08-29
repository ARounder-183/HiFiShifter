import React, { useLayoutEffect, useRef, useState } from "react";
import type { ClipFormantMorph, ClipInfo } from "../../../../features/session/sessionTypes";
import { activeClipTakeName, clipDisplayName } from "../../../../features/session/sessionTypes";
import { CLIP_HEADER_HEIGHT } from "../constants";
import { formatGainDbValue, gainToDb } from "../math";
import { AppTooltipBubble } from "../../../../components/AppTooltip";
import { useI18n } from "../../../../i18n/I18nProvider";
import { useAppTheme } from "../../../../theme/AppThemeProvider";
import { resolveTimelineClipHeaderVisibility } from "../runtime/timelineClipHeaderVisibility";
import { buildTimelineClipVisualStyle } from "../runtime/timelineCanvasStyle";
import { ClipFormantButton } from "./ClipFormantButton";

export interface ClipRenameController {
    isEditing: () => boolean;
    commit: () => void;
    cancel: () => void;
}

/** 名称上的第一次点击候选，用于第二次点击落在播放头上时仍能进入重命名 */
export interface ClipRenameClickCandidate {
    clipId: string;
    pointerId: number;
    clientX: number;
    clientY: number;
    time: number;
}

export const ClipHeader: React.FC<{
    clip: ClipInfo;
    clipWidthPx: number;
    trackColor?: string;
    transparentVisuals?: boolean;
    isPitchAdjustment?: boolean;
    startEditDrag: (e: React.PointerEvent, clipId: string, type: "gain") => void;
    toggleClipMuted: (clipId: string, nextMuted: boolean) => void;
    /** 触发内联重命名（由 ClipContextMenu 的"重命名"菜单项或双击名称触发） */
    triggerRename?: boolean;
    /** 名称内联编辑开始时通知父级（用于提升图层并隐藏 Canvas 中的原始名称） */
    onRenameStart?: (clipId: string) => void;
    onRenameCommit?: (clipId: string, newName: string) => void;
    onRenameDone?: () => void;
    /** 名称区域第一次按下时上报双击候选；第二次点击已处理或编辑结束时传 null 清除 */
    onRenameClickCandidate?: (candidate: ClipRenameClickCandidate | null) => void;
    /** 供父级在 Clip 其他区域按下时主动提交当前编辑 */
    renameControllerRef?: React.MutableRefObject<ClipRenameController | null>;
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
    onRenameStart,
    onRenameCommit,
    onRenameDone,
    onRenameClickCandidate,
    renameControllerRef,
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
    const displayName = clipDisplayName(clip);
    const editTakeName = activeClipTakeName(clip);
    const clipTooltipText =
        clip.midiNoteCount != null
            ? `${t("clip_type_midi_prefix")} ${displayName}`
            : (clip.sourcePath ?? displayName);

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
        name: displayName,
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
    // 双击名称不依赖浏览器 dblclick：Clip 的 pointerdown 会 preventDefault，
    // 某些 WebView 中会因此收不到 dblclick。这里用 pointerdown 时间/位置自行判定双击。
    const nameDoubleClickStateRef = useRef<{
        pointerId: number;
        clientX: number;
        clientY: number;
        time: number;
    } | null>(null);
    const suppressNextNameMouseDownRef = useRef(false);

    // 外部触发重命名（来自右键菜单）
    React.useEffect(() => {
        if (triggerRename && !nameEditing) {
            setNameInputVal(editTakeName);
            setNameEditing(true);
            setTimeout(() => {
                const input = nameInputRef.current;
                if (input) {
                    input.focus();
                    input.select();
                }
            }, 0);
        }
    }, [triggerRename]);

    function commitNameEdit() {
        const trimmed = nameInputVal.trim();
        const finalName = trimmed.length > 0 ? trimmed : editTakeName;
        onRenameCommit?.(clip.id, finalName);
        setNameEditing(false);
        onRenameDone?.();
    }

    function beginNameEditing() {
        onRenameClickCandidate?.(null);
        onRenameStart?.(clip.id);
        setNameInputVal(editTakeName);
        setNameEditing(true);
        setTimeout(() => {
            // 第二次点击的 mousedown 要么被名称 div 拦截，要么命中刚挂载的 input；
            // 这里兜底清除拦截标记，避免影响下一次单击。
            suppressNextNameMouseDownRef.current = false;
            const input = nameInputRef.current;
            if (input) {
                input.focus();
                input.select();
            }
        }, 0);
    }

    function cancelNameEdit() {
        setNameEditing(false);
        onRenameDone?.();
    }

    // 把当前输入框的提交/取消能力暴露给父级。
    // Clip 内部点击输入框以外的区域时，父级会在 pointerdown 捕获阶段调用 commit()。
    useLayoutEffect(() => {
        if (!renameControllerRef) return;
        renameControllerRef.current = {
            isEditing: () => nameEditing,
            commit: () => commitNameEdit(),
            cancel: () => cancelNameEdit(),
        };
        return () => {
            renameControllerRef.current = null;
        };
    });

    // 名称编辑状态必须始终可渲染：窄 clip 平时不显示名称区域，
    // 但从右键菜单触发重命名时仍需显示输入框，否则点击"重命名"无任何反馈。
    if (!showAny && !nameEditing) return null;
    // 轨道使用 Canvas 绘制 clip 的静态视觉（增益旋钮/M/F 等），DOM 层只负责交互。
    // 编辑名称/增益时只应显示对应的输入框，其余 DOM 控件继续保持透明，
    // 否则它们会叠在 Canvas 控件上方，造成进入重命名后控件轻微“样式变化”。
    const hideVisuals = transparentVisuals;

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
                        // 仅左键触发音量旋钮手势/提示：中键不得启动 gain 拖拽。
                        if (e.button !== 0) return;
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
            {(showName || nameEditing) && (
                <div
                    className={showName ? "flex-1 min-w-0" : "absolute"}
                    style={
                        showName
                            ? undefined
                            : {
                                  // 窄 clip 平时没有名称占位，编辑时把输入框做成覆盖层，
                                  // 避免 minWidth 参与 flex 排版而挤压/移动增益旋钮、M、F。
                                  top: 1,
                                  left: Math.min(
                                      visualStyle.leadingControlsWidth,
                                      Math.max(0, clipWidthPx - 2),
                                  ),
                                  right: 0,
                              }
                    }
                >
                    {nameEditing ? (
                        <input
                            ref={nameInputRef}
                            className="w-full text-xs font-medium rounded px-1 outline-none"
                            style={{
                                color: isDark ? "rgba(255,255,255,0.95)" : "rgba(0,0,0,0.88)",
                                backgroundColor: isDark
                                    ? "rgba(0,0,0,0.95)"
                                    : "rgba(255,255,255,0.96)",
                                border: `1px solid ${isDark ? "rgba(255,255,255,0.40)" : "rgba(0,0,0,0.35)"}`,
                                // 窄 clip 平时不显示名称，触发重命名时给输入框一个最小可用宽度
                                minWidth: showName ? undefined : 120,
                                height: showName ? undefined : 16,
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
                            onPointerDown={(e) => {
                                const previous = nameDoubleClickStateRef.current;
                                const now = performance.now();
                                const isDoubleClick =
                                    previous != null &&
                                    previous.pointerId === e.pointerId &&
                                    Math.abs(previous.clientX - e.clientX) <= 6 &&
                                    Math.abs(previous.clientY - e.clientY) <= 6 &&
                                    now - previous.time <= 500;

                                if (!isDoubleClick) {
                                    nameDoubleClickStateRef.current = {
                                        pointerId: e.pointerId,
                                        clientX: e.clientX,
                                        clientY: e.clientY,
                                        time: now,
                                    };
                                    // 第一次点击仍会走普通 Clip 点击逻辑并可能移动播放头；
                                    // 把候选信息上报给时间轴，让第二次点击即使落在播放头上也能重命名。
                                    onRenameClickCandidate?.({
                                        clipId: clip.id,
                                        pointerId: e.pointerId,
                                        clientX: e.clientX,
                                        clientY: e.clientY,
                                        time: now,
                                    });
                                    return;
                                }

                                // 第二次按下：阻止 Clip 的点击/拖拽逻辑并进入重命名。
                                e.preventDefault();
                                e.stopPropagation();
                                nameDoubleClickStateRef.current = null;
                                suppressNextNameMouseDownRef.current = true;
                                beginNameEditing();
                            }}
                            onMouseDown={(e) => {
                                if (!suppressNextNameMouseDownRef.current) return;
                                suppressNextNameMouseDownRef.current = false;
                                e.preventDefault();
                                e.stopPropagation();
                            }}
                        >
                            {displayName}
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
