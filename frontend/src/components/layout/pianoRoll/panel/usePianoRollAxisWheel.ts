/**
 * 参数编辑器**左轴列**（琴键列 / 数值刻度）的滚轮路由。
 *
 * 轴列在滚动容器**之外**，但滚轮语义必须与主画布完全一致：同一个
 * `getParamEditorWheelAction` 判定（keybinding 命中 → 动作），同一套 y 投影
 * （分母是**绘图区**高度 `viewSize.h`，不是轴列高度），同一组上下限
 * （`clampViewport`）。轴列因此**不定义第二种语义**：它只把事件翻译成
 * "水平滚动 / 竖直平移 / 竖直缩放 / 水平缩放（仅拦截）" 四选一。
 *
 * 【为什么从面板模型里搬出来】这段 172 行的 handler 原本夹在自动跟随、试听、
 * 读数三条 effect 之间，但它的输入全部是只读的（视口 ref + 两个 setter + 一个
 * 标脏入口），输出为零——是一段**纯路由**。搬走后，模型里只剩"这四条 effect 的
 * 调用点"，而轴列的滚轮语义第一次可以单独读完。
 */
import { useEffect } from "react";
import { isModifierActive, isNoneBinding } from "../../../../features/keybindings/keybindingsSlice";
import type { Keybinding } from "../../../../features/keybindings/types";
import type { ProcessorParamDescriptor } from "../../../../types/api";
import { clamp } from "../../timeline";
import { PITCH_MAX_MIDI, PITCH_MIN_MIDI } from "../constants";
import type { ParamName, ValueViewport } from "../types";
import { getParamEditorWheelAction } from "../wheelGesture";

type PianoRollAxisWheelInputs = {
    /** 左轴列的宿主元素（滚轮监听挂在这里）。 */
    axisWrapRef: { current: HTMLDivElement | null };
    /** 绘图区尺寸（y 投影的分母，**不是**轴列高度）。 */
    viewSizeRef: { current: { w: number; h: number } };
    editParam: ParamName;
    /** 音高视口（琴键 → MIDI 的投影来源）。 */
    pitchViewRef: { current: ValueViewport };
    /** 各参数视口（数值刻度的投影来源）。 */
    paramViewsRef: { current: Record<string, ValueViewport> };
    /** 参数描述符（决定非音高参数的上下限）。 */
    processorParamsRef: { current: ProcessorParamDescriptor[] };
    setPitchView: (next: ValueViewport) => void;
    setParamViewport: (param: string, next: ValueViewport) => void;
    /** 标脏入口（改完视口后请求一帧）。 */
    invalidate: () => void;
    /** 把视口钳进参数的实际上下限（与滚动条、绘制同一份口径）。 */
    clampViewport: (param: ParamName, v: ValueViewport) => ValueViewport;
    /** 原生滚动容器（水平滚动直接写它，再交给内核采纳）。 */
    scrollerRef: { current: HTMLDivElement | null };
    /** 采纳一次原生滚动位置（内核真值入口）。 */
    syncScrollLeftRef: { current: (scroller: HTMLDivElement) => void };
    /** 四个滚轮 keybinding（与主画布同一份）。 */
    scrollHorizontalKb: Keybinding;
    pianoKeysVerticalScrollKb: Keybinding;
    pianoKeysVerticalZoomKb: Keybinding;
    horizontalZoomKb: Keybinding;
};

/** 左轴列的滚轮路由（副作用顺序与拆分前一致：只有一个 effect）。 */
export function usePianoRollAxisWheel({
    axisWrapRef,
    viewSizeRef,
    editParam,
    pitchViewRef,
    paramViewsRef,
    processorParamsRef,
    setPitchView,
    setParamViewport,
    invalidate,
    clampViewport,
    scrollerRef,
    syncScrollLeftRef,
    scrollHorizontalKb,
    pianoKeysVerticalScrollKb,
    pianoKeysVerticalZoomKb,
    horizontalZoomKb,
}: PianoRollAxisWheelInputs) {
    // Piano keys (axis) area: keep touchpad wheel behavior aligned with the main editor.
    useEffect(() => {
        const el = axisWrapRef.current;
        if (!el) return;

        const handler = (e: WheelEvent) => {
            // During vibrato/line drag, wheel always adjusts vibrato parameters.
            // Defer to the scroller's wheel handler which has full vibrato drag logic.
            if (document.body.hasAttribute("data-piano-roll-vibrato-drag-active")) {
                e.preventDefault();
                return;
            }

            const noModifierPressed = !e.ctrlKey && !e.metaKey && !e.altKey && !e.shiftKey;
            const isWheelBindingRequested = (kb: Keybinding) => {
                if (isNoneBinding(kb)) return noModifierPressed;
                return isModifierActive(kb, e);
            };
            const horizontalScrollRequested = isWheelBindingRequested(scrollHorizontalKb);
            const pianoVerticalScrollRequested = isWheelBindingRequested(pianoKeysVerticalScrollKb);
            const pianoVerticalZoomRequested = isWheelBindingRequested(pianoKeysVerticalZoomKb);
            const horizontalZoomRequested = isWheelBindingRequested(horizontalZoomKb);

            const bounds = el.getBoundingClientRect();
            // 竖直分母必须用**绘图区**高度（`viewSize.h`），不是纵轴列的高度：
            // 列比滚动视口高出一条自绘水平滚动条的行，且轴位图/琴键是按
            // `viewSize.h` 投影的（见内核的 AXIS_TICK_LABEL_DESCENT_PX 说明）。
            // 用列高会让滚轮/悬停算出的值比渲染出来的位置偏低。
            const h = Math.max(1, viewSizeRef.current.h);
            const pointerY = clamp(e.clientY - bounds.top, 0, h);
            // t: 0=top, 1=bottom — same semantics as usePianoRollInteractions
            const t = pointerY / h;

            const wheelAction = getParamEditorWheelAction({
                deltaX: e.deltaX,
                deltaY: e.deltaY,
                horizontalScrollRequested,
                verticalPanRequested: pianoVerticalScrollRequested,
                verticalZoomRequested: pianoVerticalZoomRequested,
                horizontalZoomRequested,
            });

            const applyVerticalPanDelta = (deltaY: number) => {
                const delta = (-deltaY / h) * 0.5;
                if (editParam === "pitch") {
                    const cur = pitchViewRef.current;
                    const next = clampViewport("pitch", {
                        span: cur.span,
                        center: cur.center + delta * cur.span,
                    });
                    setPitchView(next);
                } else {
                    const cur = paramViewsRef.current[editParam] ?? {
                        center: 0.5,
                        span: 1,
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
                const scroller = scrollerRef.current;
                if (scroller) {
                    scroller.scrollLeft += e.deltaX;
                    syncScrollLeftRef.current(scroller);
                }
                applyVerticalPanDelta(e.deltaY);
                return;
            }

            if (wheelAction === "horizontal-scroll") {
                e.preventDefault();
                const scroller = scrollerRef.current;
                if (!scroller) return;
                scroller.scrollLeft += horizontalDelta;
                syncScrollLeftRef.current(scroller);
                return;
            }

            if (wheelAction === "vertical-pan") {
                e.preventDefault();
                applyVerticalPanDelta(e.deltaY);
                return;
            }

            if (wheelAction === "horizontal-zoom") {
                // 主画布/轨道视图负责水平缩放；钢琴键区只拦截，避免原生滚动抢走事件。
                e.preventDefault();
                return;
            }

            if (wheelAction !== "vertical-zoom") {
                return;
            }

            e.preventDefault();

            const valueAtPointer =
                editParam === "pitch"
                    ? (() => {
                          const view = pitchViewRef.current;
                          const absMin = PITCH_MIN_MIDI;
                          const absMax = PITCH_MAX_MIDI;
                          const span = clamp(view.span, 1e-6, absMax - absMin);
                          const min = clamp(view.center - span / 2, absMin, absMax - span);
                          return clamp(min + (1 - t) * span, absMin, absMax);
                      })()
                    : (() => {
                          const desc = processorParamsRef.current?.find(
                              (d: ProcessorParamDescriptor) => d.id === editParam,
                          );
                          const absMin =
                              desc?.kind.type === "automation_curve" ? desc.kind.min_value : 0;
                          const absMax =
                              desc?.kind.type === "automation_curve" ? desc.kind.max_value : 1;
                          const view = paramViewsRef.current[editParam] ?? {
                              center: (absMin + absMax) / 2,
                              span: absMax - absMin || 1,
                          };
                          const span = clamp(view.span, 1e-6, absMax - absMin || 1);
                          const min = clamp(view.center - span / 2, absMin, absMax - span);
                          return clamp(min + (1 - t) * span, absMin, absMax);
                      })();

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
                const cur = paramViewsRef.current[editParam] ?? {
                    center: 0.5,
                    span: 1,
                };
                const nextSpan = cur.span * factor;
                const next = clampViewport(editParam, {
                    span: nextSpan,
                    center: valueAtPointer - (0.5 - t) * nextSpan,
                });
                setParamViewport(editParam, next);
            }
            invalidate();
        };

        el.addEventListener("wheel", handler, {
            passive: false,
        } as globalThis.AddEventListenerOptions);
        return () => {
            el.removeEventListener("wheel", handler);
        };
    }, [
        editParam,
        setPitchView,
        setParamViewport,
        invalidate,
        scrollHorizontalKb,
        pianoKeysVerticalScrollKb,
        pianoKeysVerticalZoomKb,
        horizontalZoomKb,
    ]);
}
