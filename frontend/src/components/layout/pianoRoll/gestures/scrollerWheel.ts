/**
 * 滚轮手势（缩放 / 滚动 / 参数值 / 颤音调整）。
 *
 * 【为什么单独成文件】它原本是 `usePianoRollInteractions` 里的一个 `useCallback`
 * （293 行）。这些回调只做一件事，却和另外几十个回调挤在同一个 hook 里。
 *
 * 【依赖怎么拿】上下文由调用方打包传入（`Pick<PointerDeps, …>`，字段清单由 AST
 * 扫描得出），函数体原样搬过来，开头解构回局部变量。
 */
import { isModifierActive, isNoneBinding } from "../../../../features/keybindings/keybindingsSlice";
import { nativeScrollbarZoneAt } from "../../../../utils/nativeScrollbar";
import { accumulatePinchSteps, pinchDeltaFromWheel } from "../../../../utils/pinchGesture";
import { MAX_PX_PER_SEC, MIN_PX_PER_SEC, clamp } from "../../timeline";
import { resolveHorizontalWheelZoom } from "../../timeline/runtime/timelineScrollRange";
import { resolveTimelineMinPxPerSec } from "../../timeline/runtime/timelineZoomBounds";
import { getParamEditorWheelAction, getVibratoDragWheelTarget } from "../wheelGesture";
import type { Keybinding } from "../../../../features/keybindings/types";
import type { ScrollbarZone } from "../wheelGesture";

import type { PointerDeps } from "./pointerDeps";

/** 本回调需要的上下文。 */
type Deps = Pick<
    PointerDeps,
    | "applyVibratoDragAdjustment"
    | "canvasRef"
    | "clampViewport"
    | "currentParamRange"
    | "dynamicProjectSec"
    | "editParam"
    | "getPlayheadSec"
    | "horizontalZoomChainRef"
    | "horizontalZoomKb"
    | "invalidate"
    | "onHorizontalZoom"
    | "paramViewsRef"
    | "pitchViewRef"
    | "playheadZoomEnabled"
    | "prVerticalZoomKb"
    | "pxPerSecRef"
    | "scrollHorizontalKb"
    | "scrollLeftRef"
    | "scrollVerticalKb"
    | "scrollbarZoomKb"
    | "scrollerRef"
    | "setParamViewport"
    | "setPitchView"
    | "syncScrollLeft"
    | "syncTimelineEnabled"
    | "timelineOffsetRef"
    | "vibratoAmplitudeAdjustKb"
    | "vibratoFrequencyAdjustKb"
    | "vibratoPinchRef"
    | "vibratoStateRef"
    | "yToValue"
    | "yToViewportT"
>;

/** 滚轮手势（缩放 / 滚动 / 参数值 / 颤音调整）。 */
export function handleScrollerWheel(
    deps: Deps,
    e: globalThis.WheelEvent,
    forcedScrollbarZone?: ScrollbarZone,
): void {
    const {
        applyVibratoDragAdjustment,
        canvasRef,
        clampViewport,
        currentParamRange,
        dynamicProjectSec,
        editParam,
        getPlayheadSec,
        horizontalZoomChainRef,
        horizontalZoomKb,
        invalidate,
        onHorizontalZoom,
        paramViewsRef,
        pitchViewRef,
        playheadZoomEnabled,
        prVerticalZoomKb,
        pxPerSecRef,
        scrollHorizontalKb,
        scrollLeftRef,
        scrollVerticalKb,
        scrollbarZoomKb,
        scrollerRef,
        setParamViewport,
        setPitchView,
        syncScrollLeft,
        syncTimelineEnabled,
        timelineOffsetRef,
        vibratoAmplitudeAdjustKb,
        vibratoFrequencyAdjustKb,
        vibratoPinchRef,
        vibratoStateRef,
        yToValue,
        yToViewportT,
    } = deps;

    const el = scrollerRef.current;
    if (!el) return;

    const vib = vibratoStateRef.current;
    if (vib) {
        e.preventDefault();
        /*
         * 捏合（`ctrl`/`meta` + wheel）在拖拽期间 = **连续**调深度。
         *
         * 【为什么放在最前】它必须先于下面那套"按轴向与修饰键判定深度/速率"
         * 的离散逻辑被拦下：捏合是一串高频小 delta，走离散路径会被
         * `Math.max(1, …)` 放大成"深度飞走"。累积成整步后与滚轮同速，
         * 且因为是连续的，手感比一格一格的滚轮平滑得多。
         */
        const pinchDelta = pinchDeltaFromWheel(e);
        if (pinchDelta != null) {
            const steps = accumulatePinchSteps(
                vibratoPinchRef.current,
                pinchDelta,
                performance.now(),
            );
            if (steps !== 0) {
                applyVibratoDragAdjustment({
                    target: "depth",
                    direction: steps > 0 ? 1 : -1,
                    steps: Math.abs(steps),
                    shiftHeld: e.shiftKey,
                    fineEvent: e,
                    clientX: e.clientX,
                    clientY: e.clientY,
                });
            }
            return;
        }
        const ampRequested =
            isNoneBinding(vibratoAmplitudeAdjustKb) ||
            isModifierActive(vibratoAmplitudeAdjustKb, e);
        const freqRequested =
            isNoneBinding(vibratoFrequencyAdjustKb) ||
            isModifierActive(vibratoFrequencyAdjustKb, e);
        const wheelTarget = getVibratoDragWheelTarget({
            deltaX: e.deltaX,
            deltaY: e.deltaY,
            deltaMode: e.deltaMode,
            depthRequested: ampRequested,
            rateRequested: freqRequested,
        });

        // During vibrato drag, wheel always adjusts vibrato (amplitude by default,
        // frequency via modifier or horizontal scroll). If no modifier is held and
        // neither binding is "None", fall back to amplitude adjustment so the wheel
        // never gets blocked or interpreted as zoom/scroll.
        const resolvedTarget = wheelTarget !== "none" ? wheelTarget : "depth";

        e.preventDefault();
        const controlDelta =
            resolvedTarget === "rate" && Math.abs(e.deltaX) > Math.abs(e.deltaY)
                ? e.deltaX
                : e.deltaY;
        // 【与捏合分支同源】把 wheel 增量按 deltaMode 折算成像素（行 16 / 页
        // 400，与 `pinchDeltaFromWheel` 同一约定），再交给整步累积器。
        // 逐事件 `Math.max(1, …)` 是给"一格一个事件"的机械滚轮设计的；高分辨率
        // / 触控板吐出的是一串小 delta，每个都放大成整步会让深度 / 速率以事件
        // 频率飞走。累积到满一格（100 单位）才走一步，与滚轮同速。
        const wheelMode = e.deltaMode ?? 0;
        const wheelModeScale = wheelMode === 1 ? 16 : wheelMode === 2 ? 400 : 1;
        const steps = accumulatePinchSteps(
            vibratoPinchRef.current,
            controlDelta * wheelModeScale,
            performance.now(),
        );
        if (steps === 0) return;
        const direction =
            resolvedTarget === "depth" ? (controlDelta < 0 ? 1 : -1) : controlDelta < 0 ? -1 : 1;
        applyVibratoDragAdjustment({
            target: resolvedTarget,
            direction,
            steps: Math.abs(steps),
            shiftHeld: e.shiftKey,
            fineEvent: e,
            // 滚轮事件自带指针位置：用它当锚点，气泡不会因为"指针没动过"
            // 而停在旧坐标上。
            clientX: e.clientX,
            clientY: e.clientY,
        });
        return;
    }

    const noModifierPressed = !e.ctrlKey && !e.metaKey && !e.altKey && !e.shiftKey;
    const isWheelBindingRequested = (kb: Keybinding) => {
        if (isNoneBinding(kb)) return noModifierPressed;
        return isModifierActive(kb, e);
    };
    // ── 悬停原生滚动条：滚轮语义只归属该滚动条的轴 ──────────────
    // 无修饰键 = 该轴滚动（竖直条 → 视口纵向平移；水平条 → 横向滚动）；
    // 按住 modifier.scrollbarZoom（默认 Alt）= 该轴缩放。优先于全局绑定。
    const scrollbarZone = forcedScrollbarZone ?? nativeScrollbarZoneAt(el, e.clientX, e.clientY);
    const scrollbarZoomRequested =
        scrollbarZone != null &&
        !isNoneBinding(scrollbarZoomKb) &&
        isModifierActive(scrollbarZoomKb, e);
    const horizontalScrollModifierActive = isWheelBindingRequested(scrollHorizontalKb);
    const wheelAction = getParamEditorWheelAction({
        deltaX: e.deltaX,
        deltaY: e.deltaY,
        horizontalScrollRequested: horizontalScrollModifierActive,
        verticalPanRequested: isWheelBindingRequested(scrollVerticalKb),
        verticalZoomRequested: isWheelBindingRequested(prVerticalZoomKb),
        horizontalZoomRequested: isWheelBindingRequested(horizontalZoomKb),
        scrollbarZone,
        scrollbarZoomRequested,
    });

    const applyVerticalPanDelta = (deltaY: number) => {
        const h = Math.max(1, el.clientHeight);
        const delta = (-deltaY / h) * 0.5;
        if (editParam === "pitch") {
            const cur = pitchViewRef.current;
            const next = clampViewport("pitch", {
                span: cur.span,
                center: cur.center + delta * cur.span,
            });
            setPitchView(next);
        } else {
            const fallbackRange = currentParamRange ?? { min: 0, max: 1 };
            const cur = paramViewsRef.current[editParam] ?? {
                center: (fallbackRange.min + fallbackRange.max) / 2,
                span: Math.max(1e-6, fallbackRange.max - fallbackRange.min),
            };
            const next = clampViewport(editParam, {
                span: cur.span,
                center: cur.center + delta * cur.span,
            });
            setParamViewport(editParam, next);
        }
        invalidate();
    };

    const horizontalDelta = Math.abs(e.deltaX) > 0.5 ? e.deltaX : e.deltaY;

    if (wheelAction === "free-scroll") {
        e.preventDefault();
        el.scrollLeft += e.deltaX;
        syncScrollLeft(el);
        applyVerticalPanDelta(e.deltaY);
        return;
    }

    // Scroll modifier: convert wheel to horizontal scroll
    if (wheelAction === "horizontal-scroll") {
        e.preventDefault();
        el.scrollLeft += horizontalDelta;
        syncScrollLeft(el);
        return;
    }

    // Scroll modifier: convert wheel to vertical scroll
    if (wheelAction === "vertical-pan") {
        e.preventDefault();
        applyVerticalPanDelta(e.deltaY);
        return;
    }

    // Anchor zoom to the actual drawable viewport (canvas), not the scroller.
    // The scroller may include rulers/padding, which makes zoom feel off-center.
    const canvas = canvasRef.current;
    const bounds = (canvas ?? el).getBoundingClientRect();

    const pointerXRaw = e.clientX - bounds.left;
    const pointerYRaw = e.clientY - bounds.top;

    // We rely on preventDefault to stop native scrolling while zooming.
    e.preventDefault();

    if (wheelAction === "vertical-zoom") {
        if (
            pointerXRaw < 0 ||
            pointerYRaw < 0 ||
            pointerXRaw > bounds.width ||
            pointerYRaw > bounds.height
        ) {
            return;
        }
        const h = Math.max(1, bounds.height);
        const y = clamp(pointerYRaw, 0, h);
        const t = yToViewportT(y, h);
        const valueAtPointer = yToValue(editParam, y, h);

        const factor = e.deltaY < 0 ? 0.9 : 1.1;
        if (editParam === "pitch") {
            const cur = pitchViewRef.current;
            const nextSpan = cur.span * factor;
            const next = clampViewport("pitch", {
                span: nextSpan,
                center: valueAtPointer - (0.5 - t) * nextSpan,
            });
            setPitchView(next);
        } else {
            const fallbackRange = currentParamRange ?? { min: 0, max: 1 };
            const cur = paramViewsRef.current[editParam] ?? {
                center: (fallbackRange.min + fallbackRange.max) / 2,
                span: Math.max(1e-6, fallbackRange.max - fallbackRange.min),
            };
            const nextSpan = cur.span * factor;
            const next = clampViewport(editParam, {
                span: nextSpan,
                center: valueAtPointer - (0.5 - t) * nextSpan,
            });
            setParamViewport(editParam, next);
        }
        invalidate();
        return;
    }

    // Wheel: horizontal zoom (time axis)
    if (wheelAction !== "horizontal-zoom") {
        return;
    }
    const dir = e.deltaY < 0 ? 1 : -1;
    const factor = dir > 0 ? 1.1 : 0.9;
    const totalSec = Math.max(0, dynamicProjectSec);
    const minPxPerSec = resolveTimelineMinPxPerSec({
        baseMinPxPerSec: MIN_PX_PER_SEC,
        projectSec: totalSec,
        viewportWidthPx: el.clientWidth,
    });
    // 与轨道视图共用同一套水平缩放逻辑（秒为单位）：
    // 鼠标锚点 / 播放光标锚点 / 平滑右延滚动范围全部一致。
    const pendingZoom = horizontalZoomChainRef.current;
    const zoomResult = resolveHorizontalWheelZoom({
        factor,
        basePxPerSec: pendingZoom?.nextPxPerSec ?? pxPerSecRef.current,
        baseScrollLeft: pendingZoom?.nextScrollLeft ?? scrollLeftRef.current,
        totalSec,
        viewportWidth: el.clientWidth,
        playheadZoomEnabled: Boolean(playheadZoomEnabled),
        playheadSec: getPlayheadSec?.() ?? null,
        anchorScreenX: pointerXRaw,
        minPxPerSec,
        maxPxPerSec: MAX_PX_PER_SEC,
        minScrollLeft: syncTimelineEnabled ? -timelineOffsetRef.current : 0,
        anchorOffsetPx: syncTimelineEnabled ? timelineOffsetRef.current : 0,
    });
    if (!zoomResult) return;

    horizontalZoomChainRef.current = {
        nextPxPerSec: zoomResult.nextPxPerSec,
        nextScrollLeft: zoomResult.nextScrollLeft,
    };
    onHorizontalZoom(zoomResult.nextPxPerSec, zoomResult.nextScrollLeft);
}
