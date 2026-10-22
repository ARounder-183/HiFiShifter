import { resolveScaleNotes, transposePitchByScaleSteps } from "../../../../utils/musicalScales";
import type { ScaleLike } from "../../../../utils/musicalScales";
import { clamp } from "../../timeline";
import { buildScaleSegments } from "../../../../utils/tempoMap";
import { PITCH_MAX_MIDI, PITCH_MIN_MIDI } from "../constants";
import {
    frameRangeEndCut,
    frameRangeStartCut,
    selectionToFrameRanges,
    type FrameRange,
    type ParamSelection,
} from "../paramSelection";
import { clipboardPreviewSpans } from "../paramClipboardMapping";
import { DYN_DEFAULT_VIEW, DYN_VALUE_MAX, isDynParam, VOLUME_DEFAULT_VIEW } from "../paramRanges";
import { supportsParamAxisUnit } from "../paramAxisUnits";
import { framesToTime } from "../utils";
import { viewportEndSec, viewportStartSec } from "../../renderKernel/timelineAxis.js";
import {
    CHILD_PITCH_OFFSET_CENTS_RANGE,
    CHILD_PITCH_OFFSET_DEGREES_RANGE,
    isChildPitchOffsetCentsParam,
    isChildPitchOffsetDegreesParam,
    isChildFormantOffsetCentsParam,
} from "../childPitchOffsetParams";
import type { ParamName, ValueViewport } from "../types";
import { verticalScrollTopFromCenter } from "../verticalScrollMapping";
import {
    resolveClipboardPreviewColor,
    resolveDetectedCurveColors,
    resolveSecondaryCurveColor,
} from "../colors";
import { resolveSecondaryOverlayValues } from "../secondaryOverlaySelection";
import { getFixedDashPattern } from "../render";
import type { PianoRollCurveLayer } from "../kernel/host/pianoRollKernelData";
import type { TimelineAxis } from "../../renderKernel/timelineAxis";
import { secToViewportPx } from "../../renderKernel/timelineAxis";
import type {
    PianoRollGridSpec,
    PianoRollSelectionBandSpec,
} from "../kernel/host/pianoRollKernelData";
import { normalizeCssColor, resolvePianoRollColors } from "../colors";
import { parseRgbaColor } from "../../timeline/runtime/timelineClipGlRenderer";
import { PARAM_EDITOR_VERTICAL_SCROLL_RANGE_PX, toDashTuple } from "./panelHelpers";
import type { PianoRollPanelModel } from "./usePianoRollPanelModel";

export function pitchDeltaToDegreeSteps(
    basePitch: number,
    targetPitch: number,
    scale: ScaleLike,
): number {
    if (!Number.isFinite(basePitch) || !Number.isFinite(targetPitch)) {
        return 0;
    }
    if (Math.abs(targetPitch - basePitch) <= 1e-9) return 0;

    const minStep: number = CHILD_PITCH_OFFSET_DEGREES_RANGE.min;
    const maxStep: number = CHILD_PITCH_OFFSET_DEGREES_RANGE.max;
    const minPitch = transposePitchByScaleSteps(basePitch, minStep, scale);
    const maxPitch = transposePitchByScaleSteps(basePitch, maxStep, scale);
    const lowPitch = Math.min(minPitch, maxPitch);
    const highPitch = Math.max(minPitch, maxPitch);
    if (targetPitch <= lowPitch) {
        return minPitch <= maxPitch ? minStep : maxStep;
    }
    if (targetPitch >= highPitch) {
        return minPitch <= maxPitch ? maxStep : minStep;
    }

    let left = minStep;
    let right = maxStep;
    const ascending = minPitch <= maxPitch;
    for (let i = 0; i < 24; i += 1) {
        const mid = (left + right) / 2;
        const midPitch = transposePitchByScaleSteps(basePitch, mid, scale);
        if (midPitch < targetPitch === ascending) {
            left = mid;
        } else {
            right = mid;
        }
    }
    return (left + right) / 2;
}

export function buildGridSpec(
    deps: Pick<
        PianoRollPanelModel,
        | "clampViewport"
        | "editParam"
        | "editParamAxisUnit"
        | "effectiveProjectScale"
        | "fontFamily"
        | "getCurrentViewportForScrollbar"
        | "getParamValueBoundsForScrollbar"
        | "prAxis"
        | "processorParamsRef"
        | "s"
        | "themeMode"
        | "valueToY"
    >,
): PianoRollGridSpec | null {
    const {
        clampViewport,
        editParam,
        editParamAxisUnit,
        effectiveProjectScale,
        fontFamily,
        getCurrentViewportForScrollbar,
        getParamValueBoundsForScrollbar,
        prAxis,
        processorParamsRef,
        s,
        themeMode,
        valueToY,
    } = deps;

    // 显式标注为字面量联合：不加标注时 TS 会把嵌套三元推断成 `string`，
    // 导致返回值无法赋给 `PianoRollGridSpec["kind"]`。
    //
    // 【fallback 分支为什么必须有】内核模式下轴画布整张归 GL（`skipAxisCanvas`
    // 恒为 true，见下方 drawPianoRoll 调用点），这里返回 null 就意味着该参数
    // 的左侧刻度与网格**没有任何绘制者** —— 整列空白。曾表现为：切到音量 /
    // 声像 / 张力 / 气声等一般数值参数时左侧刻度完全消失。凡自动化曲线参数
    // 一律给出 fallback kind（GL 的 resolveAxisKind / 签名层早已支持），
    // 只有非曲线参数（理论上一条都不该走到这）才维持 null。
    const descForKind = processorParamsRef.current.find((d) => d.id === editParam);
    const isAutomationCurve = descForKind?.kind.type === "automation_curve";
    const kind: PianoRollGridSpec["kind"] | null =
        editParam === "pitch"
            ? "pitch"
            : isChildPitchOffsetCentsParam(editParam)
              ? "cents"
              : isChildPitchOffsetDegreesParam(editParam)
                ? "degrees"
                : isChildFormantOffsetCentsParam(editParam)
                  ? "formantCents"
                  : // 动态面板用倍率刻度（1.0× / 0.5× / 0.25×…），
                    // 与纵轴标签的 dB 换算表配合，见 axisMarkInstances。
                    isDynParam(editParam)
                    ? "level"
                    : isAutomationCurve
                      ? "fallback"
                      : null;
    if (kind === null) return null;

    const bounds = getParamValueBoundsForScrollbar(editParam);
    const view = clampViewport(editParam, getCurrentViewportForScrollbar(editParam));
    const colors = resolvePianoRollColors(themeMode === "dark");
    const finite = (c: readonly number[]) => c.every((v) => Number.isFinite(v));
    /**
     * 解析 CSS 颜色为数值 RGBA；解析失败（NaN）时退回**不透明黑**。
     *
     * 特殊说明：宁可颜色不对也不要上传 NaN——NaN 会让整个实例属性失效、
     * 整层几何消失（比"颜色错"严重得多，且更难归因）。
     */
    const toRgba = (css: string): [number, number, number, number] => {
        // 必须先归一化：`parseRgbaColor` 只认 rgb()/rgba()，hex 会被解析成
        // **不透明洋红**（故意设计，用于暴露漏解析）。本配色表的 whiteKey /
        // blackKey 正是 hex，不归一化会让整个键盘变洋红（曾实际发生）。
        const parsed = parseRgbaColor(normalizeCssColor(css));
        return finite(parsed) ? parsed : [0, 0, 0, 1];
    };

    const base = {
        kind,
        view: { center: view.center, span: view.span },
        absMin: bounds.min,
        absMax: bounds.max,
        valueToY: (value: number, heightPx: number) => valueToY(editParam, value, heightPx),
        strongRgba: toRgba(colors.pitchGridC),
        weakRgba: toRgba(colors.pitchGridOther),
        // 文字与刻度线（阶段 2 Task 5）：GL 侧据此渲染轴标签与刻度。
        paramName: editParam,
        // 纵轴展示单位（音量 / 动态的倍率 ↔ dB）。只对支持切换的参数下发，
        // 其余参数保持缺省 —— 签名里的 `axisUnit ?? ""` 因此不会给非切换参数
        // 附加一个无意义的常量。
        ...(supportsParamAxisUnit(editParam) ? { axisUnit: editParamAxisUnit } : {}),
        // 与传给 drawPianoRoll 的 fontFamily 同一个值（第 3268 行），
        // 保证两种渲染模式的字形完全一致。
        fontFamily,
        tensionLabelRgba: toRgba(colors.tensionLabel),
        tensionLineRgba: toRgba(colors.tensionLine),
    };

    // 键盘轴颜色只在音高参数下提供：非 pitch 时 GL 层据此判定"没有键盘"
    // 并清空几何（否则切到别的参数后键盘会残留在画布上）。
    if (kind !== "pitch") return base;

    // 音阶高亮：条件与旧 Canvas2D 分支逐字对齐——只在 `always` 且有工程音阶时
    // 高亮。两条路径：
    // - **分段**（有 Tempo Map 且换过音阶）：每段按自己的音阶、画自己那段 x 范围。
    //   段的时间范围在这里**投影成视口 x** 再交给 GL 层（它是纯视口坐标、不含
    //   时间轴，见 `buildPitchGridInstances` 的说明）。
    // - **均匀**（默认）：整宽一条线，只高亮工程音阶的音级。
    //
    // 范围取「可见区间 ± 5s」并按 0.02s 量化（与旧实现同一取法）：量化让滚动
    // 在 0.02s 内不改变段边界，减少几何重建；±5s 保证视口边缘的段完整。
    const highlightActive = s.scaleHighlightMode === "always";
    const segStartSec = Math.round(Math.max(0, viewportStartSec(prAxis) - 5) / 0.02) * 0.02;
    const segEndSec = Math.round((viewportEndSec(prAxis) + 5) / 0.02) * 0.02;
    const scaleSegments = highlightActive
        ? (buildScaleSegments(s.tempoMap, effectiveProjectScale, segStartSec, segEndSec) ?? [])
              .map((segment) => ({
                  startSec: segment.startSec,
                  endSec: segment.endSec,
                  // 段没有自己的音阶（`null`）时退回工程音阶——与旧实现
                  // `buildScaleSegments` 的 `current ?? projectScale` 同一口径。
                  notes: resolveScaleNotes(segment.scale ?? effectiveProjectScale),
              }))
              .filter((segment) => segment.notes.length > 0 && segment.endSec > segment.startSec)
        : [];
    // 分段存在时不再走高亮的均匀路径（两者互斥，分段优先）。
    const scaleNotes =
        highlightActive && scaleSegments.length === 0
            ? resolveScaleNotes(effectiveProjectScale)
            : [];

    return {
        ...base,
        // 钢琴背景（恒开）：黑键行背景带。GL 侧保证带子在所有网格线之前发射，
        // 音阶强调线再压在其上。
        blackKeyRowBandRgba: toRgba(colors.blackKeyRowBand),
        scaleNotes,
        scaleSegments,
        scaleHighlightRgba: toRgba(colors.scaleHighlight),
        whiteKeyRgba: toRgba(colors.whiteKey),
        blackKeyRgba: toRgba(colors.blackKey),
        blackKeyGradientRgba: toRgba(colors.blackKeyGradient),
        cSeparatorRgba: toRgba(colors.cSeparator),
        keySeparatorRgba: toRgba(colors.keySeparator),
        axisBorderRgba: toRgba(colors.axisBorder),
        cLabelRgba: toRgba(colors.cLabel),
        whiteKeyLabelRgba: toRgba(colors.whiteKeyLabel),
        blackKeyLabelRgba: toRgba(colors.blackKeyLabel),
    };
}

export function valueToY(
    deps: Pick<PianoRollPanelModel, "paramViewsRef" | "pitchViewRef" | "processorParamsRef">,
    param: ParamName,
    v: number,
    h: number,
): number {
    const { paramViewsRef, pitchViewRef, processorParamsRef } = deps;

    const H = Math.max(1, h);
    if (param === "pitch") {
        const absMin = PITCH_MIN_MIDI;
        const absMax = PITCH_MAX_MIDI;
        const view = pitchViewRef.current;
        const span = clamp(view.span, 1e-6, absMax - absMin);
        const min = clamp(view.center - span / 2, absMin, absMax - span);
        const t = (clamp(v, absMin, absMax) - min) / Math.max(1e-9, span);
        return (1 - t) * H;
    }

    if (isChildPitchOffsetCentsParam(param)) {
        const absMin = CHILD_PITCH_OFFSET_CENTS_RANGE.min;
        const absMax = CHILD_PITCH_OFFSET_CENTS_RANGE.max;
        const view = paramViewsRef.current[param] ?? {
            center: (absMin + absMax) / 2,
            span: absMax - absMin,
        };
        const span = clamp(view.span, 1e-6, absMax - absMin);
        const min = clamp(view.center - span / 2, absMin, absMax - span);
        const t = (clamp(v, absMin, absMax) - min) / Math.max(1e-9, span);
        return (1 - t) * H;
    }

    if (isChildPitchOffsetDegreesParam(param)) {
        const absMin = CHILD_PITCH_OFFSET_DEGREES_RANGE.min;
        const absMax = CHILD_PITCH_OFFSET_DEGREES_RANGE.max;
        const view = paramViewsRef.current[param] ?? {
            center: (absMin + absMax) / 2,
            span: absMax - absMin,
        };
        const span = clamp(view.span, 1e-6, absMax - absMin);
        const min = clamp(view.center - span / 2, absMin, absMax - span);
        const t = (clamp(v, absMin, absMax) - min) / Math.max(1e-9, span);
        return (1 - t) * H;
    }

    const desc = processorParamsRef.current.find((d) => d.id === param);
    const absMin = desc?.kind.type === "automation_curve" ? desc.kind.min_value : 0;
    const absMax = desc?.kind.type === "automation_curve" ? desc.kind.max_value : 1;
    const view = paramViewsRef.current[param] ?? {
        center: (absMin + absMax) / 2,
        span: absMax - absMin || 1,
    };
    const span = clamp(view.span, 1e-6, absMax - absMin || 1);
    const min = clamp(view.center - span / 2, absMin, absMax - span);
    const t = (clamp(v, absMin, absMax) - min) / Math.max(1e-9, span);
    return (1 - t) * H;
}

export function yToViewportT(y: number, h: number): number {
    const H = Math.max(1, h);
    return clamp(y / H, 0, 1);
}

export function yToValue(
    deps: Pick<PianoRollPanelModel, "paramViewsRef" | "pitchViewRef" | "processorParamsRef">,
    param: ParamName,
    y: number,
    h: number,
): number {
    const { paramViewsRef, pitchViewRef, processorParamsRef } = deps;

    const H = Math.max(1, h);
    const t = 1 - clamp(y / H, 0, 1);
    if (param === "pitch") {
        const absMin = PITCH_MIN_MIDI;
        const absMax = PITCH_MAX_MIDI;
        const view = pitchViewRef.current;
        const span = clamp(view.span, 1e-6, absMax - absMin);
        const min = clamp(view.center - span / 2, absMin, absMax - span);
        return clamp(min + t * span, absMin, absMax);
    }

    if (isChildPitchOffsetCentsParam(param)) {
        const absMin = CHILD_PITCH_OFFSET_CENTS_RANGE.min;
        const absMax = CHILD_PITCH_OFFSET_CENTS_RANGE.max;
        const view = paramViewsRef.current[param] ?? {
            center: (absMin + absMax) / 2,
            span: absMax - absMin,
        };
        const span = clamp(view.span, 1e-6, absMax - absMin);
        const min = clamp(view.center - span / 2, absMin, absMax - span);
        return clamp(min + t * span, absMin, absMax);
    }

    if (isChildPitchOffsetDegreesParam(param)) {
        const absMin = CHILD_PITCH_OFFSET_DEGREES_RANGE.min;
        const absMax = CHILD_PITCH_OFFSET_DEGREES_RANGE.max;
        const view = paramViewsRef.current[param] ?? {
            center: (absMin + absMax) / 2,
            span: absMax - absMin,
        };
        const span = clamp(view.span, 1e-6, absMax - absMin);
        const min = clamp(view.center - span / 2, absMin, absMax - span);
        return clamp(min + t * span, absMin, absMax);
    }

    const desc = processorParamsRef.current.find((d) => d.id === param);
    const absMin = desc?.kind.type === "automation_curve" ? desc.kind.min_value : 0;
    const absMax = desc?.kind.type === "automation_curve" ? desc.kind.max_value : 1;
    const view = paramViewsRef.current[param] ?? {
        center: (absMin + absMax) / 2,
        span: absMax - absMin || 1,
    };
    const span = clamp(view.span, 1e-6, absMax - absMin || 1);
    const min = clamp(view.center - span / 2, absMin, absMax - span);
    return clamp(min + t * span, absMin, absMax);
}

export function clampViewport(
    deps: Pick<PianoRollPanelModel, "processorParamsRef">,
    param: ParamName,
    v: ValueViewport,
): ValueViewport {
    const { processorParamsRef } = deps;

    if (param === "pitch") {
        const absMin = PITCH_MIN_MIDI;
        const absMax = PITCH_MAX_MIDI;
        const span = clamp(v.span, 6, absMax - absMin);
        const center = clamp(v.center, absMin + span / 2, absMax - span / 2);
        return { center, span };
    }
    if (isChildPitchOffsetCentsParam(param)) {
        const absMin = CHILD_PITCH_OFFSET_CENTS_RANGE.min;
        const absMax = CHILD_PITCH_OFFSET_CENTS_RANGE.max;
        const span = clamp(v.span, 100, absMax - absMin);
        const center = clamp(v.center, absMin + span / 2, absMax - span / 2);
        return { center, span };
    }
    if (isChildPitchOffsetDegreesParam(param)) {
        const absMin = CHILD_PITCH_OFFSET_DEGREES_RANGE.min;
        const absMax = CHILD_PITCH_OFFSET_DEGREES_RANGE.max;
        const span = clamp(v.span, 1, absMax - absMin);
        const center = clamp(v.center, absMin + span / 2, absMax - span / 2);
        return { center, span };
    }
    const desc = processorParamsRef.current.find((d) => d.id === param);
    const absMin = desc?.kind.type === "automation_curve" ? desc.kind.min_value : 0;
    const absMax = desc?.kind.type === "automation_curve" ? desc.kind.max_value : 1;
    const range = Math.max(1e-6, absMax - absMin);
    const span = clamp(v.span, range * 0.05, range);
    const center = clamp(v.center, absMin + span / 2, absMax - span / 2);
    return { center, span };
}

export function getParamValueBoundsForScrollbar(
    deps: Pick<PianoRollPanelModel, "processorParamsRef">,
    param: ParamName,
): { min: number; max: number } {
    const { processorParamsRef } = deps;

    if (param === "pitch") {
        return { min: PITCH_MIN_MIDI, max: PITCH_MAX_MIDI };
    }
    if (isChildPitchOffsetCentsParam(param)) {
        return {
            min: CHILD_PITCH_OFFSET_CENTS_RANGE.min,
            max: CHILD_PITCH_OFFSET_CENTS_RANGE.max,
        };
    }
    if (isChildPitchOffsetDegreesParam(param)) {
        return {
            min: CHILD_PITCH_OFFSET_DEGREES_RANGE.min,
            max: CHILD_PITCH_OFFSET_DEGREES_RANGE.max,
        };
    }

    const desc = processorParamsRef.current.find((d) => d.id === param);
    if (desc?.kind.type === "automation_curve") {
        return {
            min: desc.kind.min_value,
            max: desc.kind.max_value,
        };
    }
    // `dyn_edit` 是动态的历史别名（后端描述符只有 "dyn"）。若在这里退化成
    // {0,1}，动态面板的值域会从 0..4 塌成 0..1，网格与轴线全部按错误值域
    // 绘制（表现为"动态显示成了音量的标尺"的另一种形态）。它与 dyn 同值域。
    if (isDynParam(param)) {
        return { min: 0, max: DYN_VALUE_MAX };
    }
    return { min: 0, max: 1 };
}

export function getCurrentViewportForScrollbar(
    deps: Pick<
        PianoRollPanelModel,
        "getParamValueBoundsForScrollbar" | "paramViewsRef" | "pitchViewRef"
    >,
    param: ParamName,
): ValueViewport {
    const { getParamValueBoundsForScrollbar, paramViewsRef, pitchViewRef } = deps;

    if (param === "pitch") {
        return pitchViewRef.current;
    }

    const bounds = getParamValueBoundsForScrollbar(param);
    return (
        paramViewsRef.current[param] ??
        // 默认视口按参数语义分流（见 paramRanges 的两个常量说明）：
        // - dyn：0..1.25，0 dB 在 80% 高度 —— 面板主体留给 −∞..0 dB 的
        //   编辑区间（目标电平几乎不会超过 0 dB）；
        // - volume：0..2，1.0 在中线（>1 的提升是常态操作）。
        // 自定义过的视口（paramViewsRef 已有记录）原样尊重。
        (isDynParam(param)
            ? { center: DYN_DEFAULT_VIEW.center, span: DYN_DEFAULT_VIEW.span }
            : param === "volume"
              ? { center: VOLUME_DEFAULT_VIEW.center, span: VOLUME_DEFAULT_VIEW.span }
              : {
                    center: (bounds.min + bounds.max) / 2,
                    span: Math.max(1e-6, bounds.max - bounds.min),
                })
    );
}

export function syncVerticalScrollbarForViewport(
    deps: Pick<
        PianoRollPanelModel,
        | "clampViewport"
        | "getParamValueBoundsForScrollbar"
        | "hostRef"
        | "kernelDataRef"
        | "lastMirroredScrollTopRef"
        | "scrollerRef"
    >,
    param: ParamName,
    view: ValueViewport,
): void {
    const {
        clampViewport,
        getParamValueBoundsForScrollbar,
        hostRef,
        kernelDataRef,
        lastMirroredScrollTopRef,
        scrollerRef,
    } = deps;

    const clampedView = clampViewport(param, view);
    const bounds = getParamValueBoundsForScrollbar(param);

    const host = hostRef.current;
    if (host) {
        // 就地刷新值域镜像（见特殊说明 2）。
        kernelDataRef.current.valueDomain = {
            min: bounds.min,
            max: bounds.max,
            span: clampedView.span,
        };
        host.setValueCenter(clampedView.center);
        return;
    }

    const scroller = scrollerRef.current;
    if (!scroller) return;

    const nextTop = verticalScrollTopFromCenter({
        min: bounds.min,
        max: bounds.max,
        span: clampedView.span,
        center: clampedView.center,
        scrollRangePx: PARAM_EDITOR_VERTICAL_SCROLL_RANGE_PX,
    });

    if (Math.abs(scroller.scrollTop - nextTop) > 0.75) {
        scroller.scrollTop = nextTop;
        lastMirroredScrollTopRef.current = nextTop;
    }
}

export function buildCurveLayers(
    deps: Pick<
        PianoRollPanelModel,
        | "clipboardRef"
        | "detectedPitchCurves"
        | "editParam"
        | "liveEditOverrideRef"
        | "paramView"
        | "pitchEnabled"
        | "referencePitchOverlays"
        | "secSpanClipRect"
        | "secondaryParamViews"
        | "selectionClipRect"
        | "selectionRef"
        | "themeMode"
        | "valueToY"
        | "viewSizeRef"
        | "visibleSecondaryParamIds"
    >,
    axis: TimelineAxis,
): PianoRollCurveLayer[] {
    const {
        clipboardRef,
        detectedPitchCurves,
        editParam,
        liveEditOverrideRef,
        paramView,
        pitchEnabled,
        referencePitchOverlays,
        secSpanClipRect,
        secondaryParamViews,
        selectionClipRect,
        selectionRef,
        themeMode,
        valueToY,
        viewSizeRef,
        visibleSecondaryParamIds,
    } = deps;

    const layers: PianoRollCurveLayer[] = [];
    const isDark = themeMode === "dark";
    // 与 Canvas2D 路径同一份配色表（`render.ts` 也调 `resolvePianoRollColors`）。
    const colors = resolvePianoRollColors(isDark);
    const h = viewSizeRef.current.h;
    const project = (param: ParamName, value: number) => valueToY(param, value, h);

    // ── ① 参考线（pitch 模式）────────────────────────────────────
    if (editParam === "pitch") {
        for (const overlay of referencePitchOverlays) {
            const values = resolveSecondaryOverlayValues({
                orig: overlay.paramView.orig,
                edit: overlay.paramView.edit,
            });
            if (values.length < 2) continue;
            layers.push({
                values,
                param: "pitch",
                startFrame: overlay.paramView.startFrame,
                stride: overlay.paramView.stride,
                framePeriodMs: overlay.paramView.framePeriodMs,
                lineWidthPx: overlay.highlighted ? 3.2 : 2.6,
                rgba: parseRgbaColor(normalizeCssColor(overlay.strokeColor)),
                dash: null,
                projection: "curve",
                valueToY: (v) => project("pitch", v),
            });
        }
    }

    // ── ② 检测曲线（pitch 模式，按 clip 循环调色）─────────────────
    //
    // 投影模式必须是 `"detected"`（不是 `"curve"`）：
    // - 检测曲线自带**绝对起始秒** `curveStartSec`，塞进 `startFrame` 会把它丢掉，
    //   曲线整体平移到时间轴原点；
    // - 检测曲线的 `midi <= 0` 表示**无声帧**，必须跳过，否则相邻有声点之间会
    //   拉出一条贯穿底部的垂直尖刺（GL 迁移后实际出现过）。
    // 两条语义都与 `drawCurveTimed` 不同，见 `projectDetectedCurvePoints` 说明。
    if (editParam === "pitch") {
        const palette = resolveDetectedCurveColors(isDark);
        detectedPitchCurves.forEach((curve, ci) => {
            if (!curve.midiCurve || curve.midiCurve.length < 2) return;
            layers.push({
                values: curve.midiCurve,
                param: "pitch",
                startFrame: 0,
                stride: 1,
                framePeriodMs: curve.framePeriodMs,
                lineWidthPx: 2,
                rgba: parseRgbaColor(normalizeCssColor(palette[ci % palette.length])),
                dash: null,
                projection: "detected",
                curveStartSec: curve.curveStartSec,
                valueToY: (v) => project("pitch", v),
            });
        });
    }

    // ── ③ 副参数曲线 ────────────────────────────────────────────
    if (pitchEnabled && visibleSecondaryParamIds.length > 0) {
        visibleSecondaryParamIds.forEach((paramId, index) => {
            const pv = secondaryParamViews[paramId];
            if (!pv || Math.max(pv.orig.length, pv.edit.length) < 2) return;
            const values = resolveSecondaryOverlayValues({ orig: pv.orig, edit: pv.edit });
            layers.push({
                values,
                param: paramId,
                startFrame: pv.startFrame,
                stride: pv.stride,
                framePeriodMs: pv.framePeriodMs,
                lineWidthPx: 2,
                rgba: parseRgbaColor(
                    normalizeCssColor(resolveSecondaryCurveColor(isDark, paramId, index)),
                ),
                dash: null,
                projection: "curve",
                valueToY: (v) => valueToY(paramId as ParamName, v, h),
            });
        });
    }

    // ── ④⑤⑥ 主参数曲线（原始 / 编辑 / 选区高亮）─────────────────
    const pv = pitchEnabled ? paramView : null;
    if (pv) {
        const editValues =
            liveEditOverrideRef.current && liveEditOverrideRef.current.key === pv.key
                ? liveEditOverrideRef.current.edit
                : pv.edit;

        if (pv.orig.length >= 2) {
            layers.push({
                values: pv.orig,
                param: editParam,
                startFrame: pv.startFrame,
                stride: pv.stride,
                framePeriodMs: pv.framePeriodMs,
                lineWidthPx: 1.8,
                rgba: parseRgbaColor(normalizeCssColor(colors.origCurve)),
                dash: toDashTuple(getFixedDashPattern(6, 6)),
                projection: "curve",
                valueToY: (v) => project(editParam, v),
            });
        }

        if (editValues.length >= 2) {
            layers.push({
                values: editValues,
                param: editParam,
                startFrame: pv.startFrame,
                stride: pv.stride,
                framePeriodMs: pv.framePeriodMs,
                lineWidthPx: 2.6,
                rgba: parseRgbaColor(normalizeCssColor(colors.editCurve)),
                dash: null,
                projection: "curve",
                valueToY: (v) => project(editParam, v),
            });
        }

        // 选区高亮：裁剪到选区矩形（与 Canvas2D 的 ctx.clip 对应）。
        // 多段选区：每段推一层（同一曲线、各自的 clipRect）——曲线被画多次、
        // 每次只露出该段窗口，视觉等价于 Canvas2D 的「多矩形并集 clip」。
        const selection = selectionRef.current;
        if (selection && selection.length > 0 && editValues.length >= 2) {
            for (const range of selection) {
                const clip = selectionClipRect(axis, range);
                if (!clip || clip.w <= 0) continue;
                layers.push({
                    values: editValues,
                    param: editParam,
                    startFrame: pv.startFrame,
                    stride: pv.stride,
                    framePeriodMs: pv.framePeriodMs,
                    lineWidthPx: 3.6,
                    rgba: parseRgbaColor(normalizeCssColor(colors.selectionCurve)),
                    dash: null,
                    projection: "curve",
                    clipRect: clip,
                    valueToY: (v) => project(editParam, v),
                });
            }
        }

        // ── ⑦ 剪贴板预览（不同的投影语义：从落点起点按原始帧距排布）──
        // 与 Canvas2D 路径（render.ts）共用 paramClipboardMapping 的唯一映射
        // 规则：选区（帧）→ 帧区间 → 逐段落点 span，每段推一层。预览画的
        // 就是粘贴会落下的数据，断层两侧的截断与 Canvas2D 路径完全一致。
        // 时间换算用目标帧周期（粘贴按帧号落盘，用剪贴板帧周期会让预览与
        // 结果错位，见 clipboardPreviewSpans 说明）。
        const preview = clipboardRef.current;
        if (preview && selection && selection.length > 0 && preview.param === editParam) {
            const frameRanges = selectionToFrameRanges(selection);
            const spans = clipboardPreviewSpans({
                targetRanges: frameRanges,
                clipboard: preview,
                targetFramePeriodMs: pv.framePeriodMs,
            });
            for (const span of spans) {
                if (span.values.length === 0) continue;
                const spanEndSec = span.startSec + (span.values.length * span.framePeriodMs) / 1000;
                const clip = secSpanClipRect(axis, span.startSec, spanEndSec);
                if (!clip || clip.w <= 0) continue;
                layers.push({
                    values: span.values,
                    param: editParam,
                    startFrame: 0,
                    stride: 1,
                    framePeriodMs: span.framePeriodMs,
                    lineWidthPx: 2,
                    rgba: parseRgbaColor(normalizeCssColor(resolveClipboardPreviewColor(isDark))),
                    dash: toDashTuple(getFixedDashPattern(4, 4)),
                    projection: "clipboard",
                    clipStartSec: span.startSec,
                    clipEndSec: spanEndSec,
                    clipRect: clip,
                    valueToY: (v) => project(editParam, v),
                });
            }
        }
    }

    return layers;
}

export function buildSelectionBandSpec(
    deps: Pick<PianoRollPanelModel, "paramView" | "themeMode">,
    selection: ParamSelection | null,
): PianoRollSelectionBandSpec | null {
    const { paramView, themeMode } = deps;

    if (!selection || selection.length === 0) return null;
    const framePeriodMs = paramView?.framePeriodMs ?? 5;
    const colors = resolvePianoRollColors(themeMode === "dark");
    return {
        spansSec: selection.map((range) => ({
            startSec: framesToTime(frameRangeStartCut(range), framePeriodMs),
            endSec: framesToTime(frameRangeEndCut(range), framePeriodMs),
        })),
        fillRgba: parseRgbaColor(normalizeCssColor(colors.selectionBand)),
        borderRgba: parseRgbaColor(normalizeCssColor(colors.selectionBorder)),
    };
}

export function secSpanClipRect(
    deps: Pick<PianoRollPanelModel, "viewSizeRef">,
    axis: TimelineAxis,
    startSec: number,
    endSec: number,
): { x: number; y: number; w: number; h: number } | null {
    const { viewSizeRef } = deps;

    const x0 = secToViewportPx(axis, startSec);
    const x1 = secToViewportPx(axis, endSec);
    const w = x1 - x0;
    if (!(w > 0)) return null;
    return { x: x0, y: 0, w, h: viewSizeRef.current.h };
}

export function selectionClipRect(
    deps: Pick<PianoRollPanelModel, "paramView" | "secSpanClipRect">,
    axis: TimelineAxis,
    range: FrameRange,
): { x: number; y: number; w: number; h: number } | null {
    const { paramView, secSpanClipRect } = deps;

    const framePeriodMs = paramView?.framePeriodMs ?? 5;
    return secSpanClipRect(
        axis,
        framesToTime(frameRangeStartCut(range), framePeriodMs),
        framesToTime(frameRangeEndCut(range), framePeriodMs),
    );
}
