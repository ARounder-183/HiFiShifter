/**
 * ClipItem 组件
 *
 * 时间轴上单个音频 Clip 的渲染组件，负责：
 * - 淡入/淡出可视化和交互手柄
 * - Clip 的选中、拖拽、右键菜单等交互逻辑
 * - 支持 trim/stretch 编辑手柄
 *
 * 波形渲染由 TimelineSurface 内的 TimelineWaveformSurface（共享 WebGL2 波形面）
 * 统一负责，ClipItem 仅提供 DOM 交互层。
 */
import React from "react";

import { useAppSelector } from "../../../app/hooks";
import type { Keybinding } from "../../../features/keybindings/types";
import { DEFAULT_KEYBINDINGS } from "../../../features/keybindings/defaultKeybindings";
import { isModifierActive } from "../../../features/keybindings/keybindingsSlice";
import type { FadeLengthFormatContext } from "./fadeTooltipText";
import {
    buildSingleFadeInfoContent,
    buildSingleFadeInfoText,
    publishFadeRichTooltip,
} from "./fadeTooltipText";
import { resolveCurvatureEditBase } from "./reaperFade";
import { useI18n } from "../../../i18n/I18nProvider";
import { resolveClipSelectionModifiers } from "../../../features/keybindings/clipSelectionModifiers";
import type { ClipFormantMorph, ClipInfo } from "../../../features/session/sessionTypes";
import type { EditDragChannelOpts } from "./hooks/useEditDrag";
import {
    FADE_CORNER_CAP_HEIGHT_PX,
    FADE_CORNER_CAP_WIDTH_PX,
    FADE_CORNER_EDGE_WIDTH_PX,
    fadeCornerReservePx,
    CLIP_BODY_PADDING_Y,
    CLIP_HEADER_HEIGHT,
    SNAP_OFFSET_HANDLE_SIZE_PX,
    SNAP_OFFSET_HIT_HEIGHT_PX,
    snapOffsetHandleXPx,
} from "./constants";
import { FadeHitLayer } from "./FadeHitLayer";
import { modifierWatcher } from "./hooks/modifierWatcher";
import { hitInactiveTakeLane, resolveTakeLaneLayouts } from "./takeLanes";
import { ClipEdgeHandles } from "./clip/ClipEdgeHandles";
import {
    ClipHeader,
    type ClipRenameClickCandidate,
    type ClipRenameController,
} from "./clip/ClipHeader";

/** 重叠遮罩：被压住的 Clip 以该不透明度透出。0.5 会把两个彩色块混成脏色。 */
const LEADING_OVERLAP_ALPHA = 0.3;

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
    startSnapOffsetDrag,
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
    showAllTakes = true,
    onActivateTake,
    fadeShapeCycleKb = null,
    multiSelectToggleKb = DEFAULT_KEYBINDINGS["modifier.clipMultiSelectToggle"],
    rangeSelectKb = DEFAULT_KEYBINDINGS["modifier.clipRangeSelect"],
    pitchDragKb = DEFAULT_KEYBINDINGS["modifier.clipPitchDrag"],
    onClipPitchDragStart,
    onFadeShapeCycleClick,
    fadeLengthFormatCtx,
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
            | "gain"
            | "crossfade_edges",
        /** 延迟起手的类型化私有通道（曲率环境、相对拖拽锚点、交叉配对）。 */
        channel?: EditDragChannelOpts,
    ) => void;
    /** SnapOffset 三角手柄拖拽（左下角；拖动调整吸附偏移）。 */
    startSnapOffsetDrag?: (e: React.PointerEvent, clipId: string) => void;
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
    /** 与轨道波形渲染一致的“显示全部 Take”设置。 */
    showAllTakes?: boolean;
    /** 点击 inactive take 波形 lane 时触发（优先级低于常规编辑手势）。 */
    onActivateTake?: (clipId: string, takeId: string) => void;
    /** 形状循环键绑定（无绑定时不启用点击切换）。 */
    fadeShapeCycleKb?: Keybinding | null;
    /** modifier.clipMultiSelectToggle 绑定（按住并点击切换多选） */
    multiSelectToggleKb?: Keybinding;
    /** modifier.clipRangeSelect 绑定（按住并点击范围选择） */
    rangeSelectKb?: Keybinding;
    /** modifier.clipPitchDrag 绑定（按住并垂直拖拽波形调整音高） */
    pitchDragKb?: Keybinding;
    /** 音高拖拽手势入口（useClipPitchDrag 提供） */
    onClipPitchDragStart?: (e: React.PointerEvent<HTMLDivElement>, clipId: string) => void;
    /** 淡化长度 ToolTips 的相对时长时间上下文。 */
    fadeLengthFormatCtx: FadeLengthFormatContext;
    /** 修饰键下左键点击包络线 → 循环切换该侧曲线类型。 */
    onFadeShapeCycleClick?: (clipId: string, side: "in" | "out") => void;
}) {
    const { t } = useI18n();
    const isPlaying = useAppSelector((state) => state.session.runtime.isPlaying);
    const renameControllerRef = React.useRef<ClipRenameController | null>(null);

    // 不要对 left/width 取整：背景网格与时间标尺均按浮点像素位置绘制。
    // 若这里 Math.round，Clip 会相对网格最多向右偏 0.5px；在常用缩放
    // (100-200 px/s) 下就是约 2.5-5ms 的“网格偏右”观感，且随 pxPerSec
    // 与 BPM 改变而变化。保留浮点像素可让 Clip 与网格完全对齐。
    const left = Math.max(0, clip.startSec * pxPerSec);
    const width = Math.max(1, clip.lengthSec * pxPerSec);
    // body 区高度（与 WaveformTrackCanvas 一致）：轨道高 - 上下 padding - 头部高。
    const bodyHeight = Math.max(1, rowHeight - CLIP_BODY_PADDING_Y - CLIP_HEADER_HEIGHT);
    // 多 Take lane 布局与波形面/点击命中共用同一套 takeLanes 数学：分隔线、
    // 可点击区与 WebGL 渲染的 lane 边界逐像素一致。
    const takeLaneLayouts = resolveTakeLaneLayouts(clip, showAllTakes, bodyHeight);
    // 有效 fade = 自动交叉淡化（>0 时覆盖）否则手动 fade（对齐 REAPER 分离存储模型）。
    const effectiveFadeInSec =
        (clip.autoFadeInSec ?? 0) > 0 ? clip.autoFadeInSec! : (clip.fadeInSec ?? 0);
    const effectiveFadeOutSec =
        (clip.autoFadeOutSec ?? 0) > 0 ? clip.autoFadeOutSec! : (clip.fadeOutSec ?? 0);
    // 角部手柄悬停信息：使用与包络线一致的三行格式；长度显示当前有效值
    //（未创建淡化时为 0，明确告诉用户“从这里拖出一个淡变”）。
    const cornerTooltipIn = buildSingleFadeInfoText({
        isOut: false,
        shape: resolveCurvatureEditBase(Number(clip.fadeInShape) || 0).shape,
        dir: Number(clip.fadeInDir ?? 0),
        lengthSec: effectiveFadeInSec,
        formatCtx: fadeLengthFormatCtx,
        t: (key) => t(key as Parameters<typeof t>[0]),
    });
    const cornerTooltipOut = buildSingleFadeInfoText({
        isOut: true,
        shape: resolveCurvatureEditBase(Number(clip.fadeOutShape) || 0).shape,
        dir: Number(clip.fadeOutDir ?? 0),
        lengthSec: effectiveFadeOutSec,
        formatCtx: fadeLengthFormatCtx,
        t: (key) => t(key as Parameters<typeof t>[0]),
    });
    // 角部手柄的富内容浮标：类型行以内联曲线图标替代文字名称，悬停时由
    // AppTooltipProvider 的富内容表渲染（data-tooltip 文本保留作回退）。
    const cornerRichContentIn = buildSingleFadeInfoContent({
        isOut: false,
        shape: resolveCurvatureEditBase(Number(clip.fadeInShape) || 0).shape,
        dir: Number(clip.fadeInDir ?? 0),
        lengthSec: effectiveFadeInSec,
        formatCtx: fadeLengthFormatCtx,
        t: (key) => t(key as Parameters<typeof t>[0]),
    });
    const cornerRichContentOut = buildSingleFadeInfoContent({
        isOut: true,
        shape: resolveCurvatureEditBase(Number(clip.fadeOutShape) || 0).shape,
        dir: Number(clip.fadeOutDir ?? 0),
        lengthSec: effectiveFadeOutSec,
        formatCtx: fadeLengthFormatCtx,
        t: (key) => t(key as Parameters<typeof t>[0]),
    });
    const cornerInCapRef = React.useRef<HTMLDivElement | null>(null);
    const cornerInEdgeRef = React.useRef<HTMLDivElement | null>(null);
    const cornerOutCapRef = React.useRef<HTMLDivElement | null>(null);
    const cornerOutEdgeRef = React.useRef<HTMLDivElement | null>(null);
    React.useLayoutEffect(() => {
        publishFadeRichTooltip(cornerInCapRef.current, cornerRichContentIn);
        publishFadeRichTooltip(cornerInEdgeRef.current, cornerRichContentIn);
        publishFadeRichTooltip(cornerOutCapRef.current, cornerRichContentOut);
        publishFadeRichTooltip(cornerOutEdgeRef.current, cornerRichContentOut);
    }, [cornerRichContentIn, cornerRichContentOut]);

    const leadingOverlapPx = Math.max(
        0,
        Math.min(width, Math.max(0, leadingOverlapSec) * pxPerSec),
    );
    const leadingOverlapMaskImage =
        leadingOverlapPx > 0
            ? `linear-gradient(to right, rgba(0,0,0,${LEADING_OVERLAP_ALPHA}) 0px, rgba(0,0,0,${LEADING_OVERLAP_ALPHA}) ${leadingOverlapPx}px, rgba(0,0,0,1) ${leadingOverlapPx}px, rgba(0,0,0,1) 100%)`
            : undefined;

    const interactionHintBoxShadow =
        hovered && clip.groupId == null ? "0 0 0 1px rgba(0, 0, 0, 0.35)" : undefined;

    // Clip 总高：边缘所有权切分（淡化角控件 vs 裁短）依赖它，取单一来源。
    const clipHeightPx = Math.max(1, rowHeight - CLIP_BODY_PADDING_Y);
    /**
     * 淡化角控件在 body 区内（header 之下）保留的高度；其下沿以下归裁短。
     * 角控件不得覆盖 header —— header 上有旋钮/badge/名称等交互控件。
     */
    const fadeCornerReserve = fadeCornerReservePx(bodyHeight);

    const startDeferredFadeEditDrag = React.useCallback(
        (e: React.PointerEvent<HTMLDivElement>, type: "fade_in" | "fade_out") => {
            // 仅左键触发渐变编辑：中键（aux click/自动滚动）与右键（上下文菜单）
            // 不得抢占渐变握把手势。
            if (e.button !== 0) return;
            e.preventDefault();
            e.stopPropagation();
            clearContextMenu();

            // 注意：双击重置曲率的判定在 FadeHitLayer 的 onPointerDown 里完成
            // （包络线本体命中才参与）。这里不能再调 noteFadeLinePointerDown ——
            // 同一键被记录两次（间隔 0ms）会被误判为双击，导致每次按下都触发
            // 重置并提前返回，拖拽（曲率/长度）全部失效。

            // Only check physical Alt key for click-selection bypass.
            // altPressed tracks the stretch modifier and must not interfere
            // with primary-modifier/Shift selection behavior.
            const selectionMods = resolveClipSelectionModifiers({
                event: e,
                multiSelectToggleKb,
                rangeSelectKb,
            });
            const doShiftRangeSelect = selectionMods.rangeSelectActive;
            const shiftRangeAnchorClipId = doShiftRangeSelect ? rangeSelectAnchorClipId : null;
            const doCtrlToggleOnly = selectionMods.multiSelectToggleActive;
            const shouldPrimeSelection = selectionMods.shouldPrimeSelection;
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
            // 曲率拖拽环境：包络“gain=1 基准线”的客户坐标。**基准 y 在按下
            // 瞬间从 hit 元素几何推导一次**，之后手势全程固定 —— 这与指针的
            // 原始 clientY 无关（滚动/布局变化下两者解耦），曲率求解恒一致。
            let fadePointerEnv: { envTopClientY: number; bodyHeightPx: number } | undefined;
            {
                const hitYLocal = Number(targetEl.dataset.hsFadeY);
                if (Number.isFinite(hitYLocal)) {
                    const rect = targetEl.getBoundingClientRect();
                    fadePointerEnv = {
                        envTopClientY: rect.top - hitYLocal,
                        bodyHeightPx: bodyHeight,
                    };
                }
            }
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
                        currentTarget: targetEl,
                    } as unknown as React.PointerEvent,
                    clip.id,
                    type,
                    // 类型化通道：不再把私有字段走私到事件对象上。
                    { dragStartClientX: startX, fadePointerEnv },
                );
                // 手势已开始：用起手事件初始化全局修饰键快照（后续每帧由
                // useEditDrag 从原生 pointermove 持续自愈）。
                modifierWatcher.refreshFromEvent(ev);
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
                    // 所有淡化命中（包络线 / 边缘竖线 / 角部，均展示淡入淡出
                    // ToolTips）单击（未拖动）→ 播放头跳到淡化区内侧边缘（淡入
                    // → 右缘，淡出 → 左缘），与 REAPER 行为对齐。交叉点
                    // （OverlapEditLayer 抓手）仍按点击位置跳转。坐标用 **clip
                    // 根元素** rect 计算 —— hit 元素自身位于 clip 内部，取其
                    // rect 会多算一级偏移。
                    const fadeSec = type === "fade_in" ? effectiveFadeInSec : effectiveFadeOutSec;
                    const innerEdgeSec =
                        type === "fade_in"
                            ? clip.startSec + fadeSec
                            : clip.startSec + clip.lengthSec - fadeSec;
                    const clipRoot = targetEl.closest("[data-hs-clip-item]");
                    const clipRootRect = clipRoot?.getBoundingClientRect();
                    if (clipRootRect) {
                        seekFromClientX(
                            clipRootRect.left + (innerEdgeSec - clip.startSec) * pxPerSec,
                            true,
                        );
                    } else {
                        seekFromClientX(ev.clientX, true);
                    }
                }
            };

            window.addEventListener("pointermove", onMove, true);
            window.addEventListener("pointerup", onEnd, true);
            window.addEventListener("pointercancel", onEnd, true);
        },
        [
            clearContextMenu,
            clip.id,
            clip.startSec,
            clip.lengthSec,
            ensureSelected,
            isInMultiSelectedSet,
            multiSelectedCount,
            multiSelectToggleKb,
            rangeSelectKb,
            onCtrlToggleSelect,
            onShiftRangeSelect,
            rangeSelectAnchorClipId,
            recordLastClickPosition,
            seekFromClientX,
            selectClipRemote,
            selected,
            startEditDrag,
            altPressed,
            effectiveFadeInSec,
            effectiveFadeOutSec,
            pxPerSec,
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
            data-hs-clip-id={clip.id}
            className="absolute overflow-visible group"
            style={{
                left,
                width,
                top: 0,
                height: clipHeightPx,
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
                const selectionMods = resolveClipSelectionModifiers({
                    event: e,
                    multiSelectToggleKb,
                    rangeSelectKb,
                });
                const altKeyDown = selectionMods.altKeyDown;

                // Shift+点击范围选择在 pointerup 时处理（避免阻止拖动）
                const doShiftRangeSelect = selectionMods.rangeSelectActive;
                const shiftRangeAnchorClipId = doShiftRangeSelect ? rangeSelectAnchorClipId : null;
                const doCtrlToggleOnly = selectionMods.multiSelectToggleActive;
                const shouldPrimeSelection = selectionMods.shouldPrimeSelection;
                const primedSelection = shouldPrimeSelection && !selected;

                // 音高拖拽修饰键（默认 Alt+Shift）：按住并垂直拖拽波形 =
                // 调整 Clip 范围内的音高。拦截在普通移动/Slip 之前；
                // 选区预备语义与 Slip 拖拽一致（上方 primedSelection）。
                if (isModifierActive(pitchDragKb, e)) {
                    e.preventDefault();
                    e.stopPropagation();
                    clearContextMenu();
                    onClipPitchDragStart?.(e, clip.id);
                    return;
                }

                // Inactive take 命中只接管“无移动、无编辑修饰键的 click”。
                // 拖拽仍然进入下方正常 Clip move 流程；header/edge/fade/snap
                // 等子控件已在更早阶段 stopPropagation，因此不会被抢占。
                let clickedInactiveTakeId: string | null = null;
                if (e.button === 0 && !altKeyDown && !doShiftRangeSelect && !doCtrlToggleOnly) {
                    const bounds = e.currentTarget.getBoundingClientRect();
                    const localBodyY = e.clientY - bounds.top - CLIP_HEADER_HEIGHT;
                    clickedInactiveTakeId =
                        hitInactiveTakeLane(clip, showAllTakes, bodyHeight, localBodyY)?.takeId ??
                        null;
                }

                if (primedSelection) {
                    ensureSelected(clip.id);
                    selectClipRemote(clip.id);
                    recordLastClickPosition?.(e.clientX);
                }

                // Seek should happen on click, not on drag.
                // Track whether the pointer moved beyond a small deadzone.
                const allowSeek =
                    !altKeyDown &&
                    !selectionMods.multiSelectToggleRaw &&
                    !selectionMods.rangeSelectRaw &&
                    clickedInactiveTakeId == null;
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
                        if (clickedInactiveTakeId) {
                            // 暂停/停止状态下，点击 inactive take 除了切换，
                            // 还会把播放光标带到点击位置；播放中不打断当前位置。
                            if (!isPlaying) {
                                seekFromClientX(ev.clientX, true);
                            }
                            onActivateTake?.(clip.id, clickedInactiveTakeId);
                            return;
                        }
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
                    bodyHeightPx={bodyHeight}
                    altPressed={altPressed}
                    isInMultiSelectedSet={isInMultiSelectedSet}
                    multiSelectedCount={multiSelectedCount}
                    ensureSelected={ensureSelected}
                    selectClipRemote={selectClipRemote}
                    onCtrlToggleSelect={onCtrlToggleSelect}
                    onShiftRangeSelect={onShiftRangeSelect}
                    multiSelectToggleKb={multiSelectToggleKb}
                    rangeSelectKb={rangeSelectKb}
                    rangeSelectAnchorClipId={rangeSelectAnchorClipId}
                    recordLastClickPosition={recordLastClickPosition}
                    seekFromClientX={seekFromClientX}
                    startEditDrag={startEditDrag}
                />

                {/* Fade 角落创建/编辑手柄（L 形）：**body 区**的左上/右上 ——
                    顶部横帽与竖条都在 header 之下，不与 header 上的旋钮 /
                    badge / 名称争事件；竖条覆盖 body 上部直至 fadeCornerReserve，
                    此高度带以下才是 EdgeHandles 的裁短/延长区（几何切分）。
                    保留高度随 body 高度自适应，避免矮 Clip 上裁短手势无处下手。
                    悬停信息为富内容三行淡变 ToolTips（首行为内联曲线图标，
                    经 publishFadeRichTooltip 注册）；长度显示当前有效值
                    （未创建时为 0）。z 高于 SnapOffset 握把不影响：互不相交。 */}
                <div
                    ref={cornerInCapRef}
                    className="absolute left-0 z-[65]"
                    style={{
                        top: CLIP_HEADER_HEIGHT,
                        width: FADE_CORNER_CAP_WIDTH_PX,
                        height: FADE_CORNER_CAP_HEIGHT_PX,
                        cursor: "nwse-resize",
                    }}
                    onPointerDown={(e) => {
                        startDeferredFadeEditDrag(e, "fade_in");
                    }}
                    data-tooltip={cornerTooltipIn}
                />
                <div
                    ref={cornerInEdgeRef}
                    className="absolute left-0 z-[65]"
                    style={{
                        top: CLIP_HEADER_HEIGHT + FADE_CORNER_CAP_HEIGHT_PX,
                        width: FADE_CORNER_EDGE_WIDTH_PX,
                        height: Math.max(0, fadeCornerReserve - FADE_CORNER_CAP_HEIGHT_PX),
                        cursor: "ew-resize",
                    }}
                    onPointerDown={(e) => {
                        startDeferredFadeEditDrag(e, "fade_in");
                    }}
                    data-tooltip={cornerTooltipIn}
                />
                <div
                    ref={cornerOutCapRef}
                    className="absolute right-0 z-[65]"
                    style={{
                        top: CLIP_HEADER_HEIGHT,
                        width: FADE_CORNER_CAP_WIDTH_PX,
                        height: FADE_CORNER_CAP_HEIGHT_PX,
                        cursor: "nesw-resize",
                    }}
                    onPointerDown={(e) => {
                        startDeferredFadeEditDrag(e, "fade_out");
                    }}
                    data-tooltip={cornerTooltipOut}
                />
                <div
                    ref={cornerOutEdgeRef}
                    className="absolute right-0 z-[65]"
                    style={{
                        top: CLIP_HEADER_HEIGHT + FADE_CORNER_CAP_HEIGHT_PX,
                        width: FADE_CORNER_EDGE_WIDTH_PX,
                        height: Math.max(0, fadeCornerReserve - FADE_CORNER_CAP_HEIGHT_PX),
                        cursor: "ew-resize",
                    }}
                    onPointerDown={(e) => {
                        startDeferredFadeEditDrag(e, "fade_out");
                    }}
                    data-tooltip={cornerTooltipOut}
                />

                {/* SnapOffset 命中握把（透明）：**跟随 ◣ 三角位置**（三角
                    视觉由轨道级 Canvas 绘制），z 高于左缘 trim/stretch 条
                    （z-60，全高、始终可命中）——否则三角所在处会被边缘
                    拖拽抢走。此处只负责把按下事件路由给吸附偏移拖拽。 */}
                <div
                    className="absolute bottom-0 z-[70]"
                    style={{
                        // 收敛在 Clip 宽度内：offset≈length 时握把若按三角 x
                        // 原样放置会伸出右缘约 11px，以 z-70 吞掉相邻 Clip 左下
                        // 条带的 trim_left 手势（z-60）。
                        left: Math.min(
                            Math.max(-4, snapOffsetHandleXPx(clip.snapOffsetSec, pxPerSec) - 1),
                            Math.max(-4, width - SNAP_OFFSET_HANDLE_SIZE_PX),
                        ),
                        width: SNAP_OFFSET_HANDLE_SIZE_PX + 3,
                        height: SNAP_OFFSET_HIT_HEIGHT_PX,
                        cursor: "ew-resize",
                    }}
                    onPointerDown={(e) => {
                        if (e.button !== 0) return;
                        e.preventDefault();
                        e.stopPropagation();
                        startSnapOffsetDrag?.(e, clip.id);
                    }}
                    data-tooltip={t("clip_snap_offset")}
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
                    {/* Body (waveform + edit handles)。data-hs-clip-body 供
                        淡化曲率拖拽把指针 clientY 映射回包络增益。 */}
                    <div className="absolute inset-0" data-hs-clip-body="1">
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
                                fadeInShape={
                                    Number.isFinite(clip.fadeInShape) ? clip.fadeInShape : 0
                                }
                                fadeInDir={clip.fadeInDir ?? 0}
                                fadeOutShape={
                                    Number.isFinite(clip.fadeOutShape) ? clip.fadeOutShape : 0
                                }
                                fadeOutDir={clip.fadeOutDir ?? 0}
                                zIndex={40}
                                effectiveFadeInSec={effectiveFadeInSec}
                                effectiveFadeOutSec={effectiveFadeOutSec}
                                formatCtx={fadeLengthFormatCtx}
                                t={(key) => t(key as Parameters<typeof t>[0])}
                                shapeCycleKb={fadeShapeCycleKb}
                                clipId={clip.id}
                                onShapeCycleClick={(side) => onFadeShapeCycleClick?.(clip.id, side)}
                                onFadeInPointerDown={(e) => startDeferredFadeEditDrag(e, "fade_in")}
                                onFadeOutPointerDown={(e) =>
                                    startDeferredFadeEditDrag(e, "fade_out")
                                }
                            />
                        )}

                        {/* 波形由 TimelineWaveformSurface（共享 WebGL2 波形面）统一渲染，此处不再包含波形内容 */}
                    </div>
                </div>

                {/* 多 Take lane 分界线：与波形面/点击命中共用同一套 takeLanes 布局，
                    首 lane 顶边即 header 边界不画。放在 mask 容器之外 —— 前导重叠
                    的 DOM 渐隐只作用于交互层视觉，不淡化分界线（对齐旧 Canvas
                    多 Take 实现）；pointer-events-none 不参与任何手势。 */}
                {(takeLaneLayouts ?? []).slice(1).map((lane) => (
                    <div
                        key={`take-lane-separator-${lane.index}`}
                        className="pointer-events-none absolute left-0 right-0 h-px"
                        style={{
                            top: CLIP_HEADER_HEIGHT + lane.top,
                            // 亮色块上一律深色分线（白线在彩色块上看不见）。
                            backgroundColor: "rgba(0, 0, 0, 0.18)",
                        }}
                    />
                ))}
            </div>
        </div>
    );
});
