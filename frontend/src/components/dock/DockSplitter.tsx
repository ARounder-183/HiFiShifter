/*
 * 分隔条：把指针位移换算成"两侧目标尺寸"，交给 `DockSplit` 落样式。
 *
 * 【为什么样式不由这里写】拖拽期间的直写样式必须与 React 最终写入的样式**完全
 * 一致**，否则 React 的差异更新（只写变化过的属性）会与直写打架：曾经在松手时
 * `style.flex = ""` 清空行内样式，被清掉的 `flex-basis` 落回 `auto`，而 React 只
 * 重写 `flexGrow`，最终成了 `flex: <ratio> 1 auto` —— 不是按比例分配，分界线
 * 因此"拖了却没正确生效"。样式规则（含 `paneStyle`）属于 `DockSplit`，所以由它
 * 负责写，这里只报告目标尺寸。
 *
 * 【为什么拖动期间只改 DOM】与仓库既有的时间轴/参数编辑器分隔条同一策略：每帧
 * 走 React 会让整个停靠树重渲染。尺寸在松手时才进 Redux。
 */

import { useCallback, useRef, useState } from "react";

import { resolveSplitDragTarget } from "../../features/dock/dockTree";
import { DOCK_SPLITTER_PX, type DockSplitNode } from "../../features/dock/dockTypes";
import { shouldSuppressHoverSideEffects } from "../../utils/penInput";
import { registerDragAbort } from "../../utils/gestureFocusGuard";

export interface DockSplitterProps {
    dir: DockSplitNode["dir"];
    /** 分隔条所在的 flex 容器（尺寸基准）。 */
    containerRef: React.RefObject<HTMLDivElement | null>;
    /** 当前节点固定的是哪一侧（拖拽期间保持不变，见下方说明）。 */
    fixed: DockSplitNode["fixed"];
    /** 两侧最小尺寸（像素）。 */
    minA: number;
    minB: number;
    /** 拖拽中：报告两侧的目标尺寸（只改 DOM，不进 Redux）。 */
    onLiveSize: (next: { ratio: number; fixed: DockSplitNode["fixed"] }) => void;
    /** 松手：提交最终尺寸。 */
    onCommit: (next: { ratio: number; fixed: DockSplitNode["fixed"] }) => void;
    /** 双击复位：回到两侧均分。 */
    onReset: () => void;
}

/** 一次拖拽的目标尺寸（也是最终提交的载荷）。 */
type SplitTarget = { ratio: number; fixed: DockSplitNode["fixed"] };

/** 键盘步进（CSS px）：方向键一格；按住 Shift 走精细档（与数字字段的手感同源）。 */
const KEYBOARD_STEP_PX = 16;
const KEYBOARD_FINE_STEP_PX = 4;

export function DockSplitter({
    dir,
    containerRef,
    fixed,
    minA,
    minB,
    onLiveSize,
    onCommit,
    onReset,
}: DockSplitterProps) {
    const [dragging, setDragging] = useState(false);
    const latestRef = useRef<SplitTarget | null>(null);
    const separatorRef = useRef<HTMLDivElement | null>(null);
    /** aria-valuenow（两侧占比 %）：聚焦/键控时从 DOM 实测刷新。 */
    const [valueNowPct, setValueNowPct] = useState<number | null>(null);

    const onPointerDown = useCallback(
        (event: React.PointerEvent<HTMLDivElement>) => {
            // 数位笔 / 触摸不触发分隔条：4px 窄条 + 按下即生效，画线时极易误扫。
            if (shouldSuppressHoverSideEffects(event.nativeEvent)) return;
            if (event.button !== 0) return;
            const container = containerRef.current;
            if (!container) return;

            event.preventDefault();
            setDragging(true);

            const rect = container.getBoundingClientRect();
            const horizontal = dir === "row";
            const available = Math.max(
                0,
                (horizontal ? rect.width : rect.height) - DOCK_SPLITTER_PX,
            );

            /**
             * 指针位置 → 两侧目标尺寸。
             *
             * 【固定侧必须保持固定】原先无论哪一侧固定，拖拽后都写成"侧 A 固定"，
             * 于是右侧停靠栏从"固定 360px"悄悄变成"自由伸缩"，窗口一缩放行为就变。
             * 现在保持原有的固定侧，只改它的像素值。
             */
            const compute = (clientX: number, clientY: number): SplitTarget =>
                resolveSplitDragTarget({
                    fixed,
                    minA,
                    minB,
                    available,
                    pointerOffset: horizontal ? clientX - rect.left : clientY - rect.top,
                });

            const apply = (clientX: number, clientY: number) => {
                const next = compute(clientX, clientY);
                latestRef.current = next;
                onLiveSize(next);
            };

            apply(event.clientX, event.clientY);

            const onMove = (moveEvent: PointerEvent) => apply(moveEvent.clientX, moveEvent.clientY);
            // 失焦（Alt+Tab / 最小化）时 pointerup 不会送回本窗口：只挂
            // pointerup/pointercancel 会让 `data-dragging` 与 `pointermove` 监听一起
            // 留着，切回来后**不按键移动鼠标也会继续改尺寸**。
            let unregisterAbort = () => {};
            const onUp = () => {
                window.removeEventListener("pointermove", onMove);
                window.removeEventListener("pointerup", onUp);
                window.removeEventListener("pointercancel", onUp);
                unregisterAbort();
                setDragging(false);
                // 不清空行内样式：`onLiveSize` 写入的就是 React 最终要写的值，
                // 提交后 React 的差异更新会发现"无需改动"，DOM 保持正确。
                if (latestRef.current) onCommit(latestRef.current);
            };
            unregisterAbort = registerDragAbort(onUp);
            window.addEventListener("pointermove", onMove);
            window.addEventListener("pointerup", onUp);
            window.addEventListener("pointercancel", onUp);
        },
        [containerRef, dir, fixed, minA, minB, onCommit, onLiveSize],
    );

    /*
     * 键盘操作（WAI-ARIA "focusable separator" 模式）。
     *
     * 【为什么实测而不是读 props】当前分界位置不在本组件的 props 里（拖拽期
     * 间只有 `fixed` 一侧有权威像素值，`fixed === null` 时只有 Redux 里的
     * ratio），而分隔条在容器里的**几何位置**本身就是两侧当前尺寸的事实：
     * `分隔条左缘（row）/ 上缘（column）− 容器对应缘` 恰好等于侧 A 的当前
     * 尺寸。聚焦与每次键控都重新实测，鼠标拖拽后的漂移也自然被带上。
     *
     * 方向键是离散动作，立即提交（与滚轮/数字字段的方向键同一语义），不走
     * 拖拽期的 onLiveSize 直写路径。
     */
    const measureSplit = useCallback((): { available: number; aSize: number } | null => {
        const container = containerRef.current;
        const separator = separatorRef.current;
        if (!container || !separator) return null;
        const contRect = container.getBoundingClientRect();
        const sepRect = separator.getBoundingClientRect();
        const horizontal = dir === "row";
        const available = Math.max(
            0,
            (horizontal ? contRect.width : contRect.height) - DOCK_SPLITTER_PX,
        );
        if (available <= 0) return null;
        const aSize = horizontal ? sepRect.left - contRect.left : sepRect.top - contRect.top;
        return { available, aSize };
    }, [containerRef, dir]);

    const commitKeyboardStep = useCallback(
        (nextASize: number) => {
            const m = measureSplit();
            if (!m) return;
            const next = resolveSplitDragTarget({
                fixed,
                minA,
                minB,
                available: m.available,
                pointerOffset: nextASize,
            });
            setValueNowPct(Math.round(next.ratio * 100));
            onCommit(next);
        },
        [fixed, minA, minB, measureSplit, onCommit],
    );

    const onKeyDown = useCallback(
        (event: React.KeyboardEvent<HTMLDivElement>) => {
            const horizontal = dir === "row";
            const forwardKey = horizontal ? "ArrowRight" : "ArrowDown";
            const backwardKey = horizontal ? "ArrowLeft" : "ArrowUp";
            const m = measureSplit();
            if (!m) return;
            if (event.key === forwardKey || event.key === backwardKey) {
                event.preventDefault();
                const step =
                    (event.shiftKey ? KEYBOARD_FINE_STEP_PX : KEYBOARD_STEP_PX) *
                    (event.key === forwardKey ? 1 : -1);
                commitKeyboardStep(m.aSize + step);
                return;
            }
            if (event.key === "Home") {
                event.preventDefault();
                commitKeyboardStep(minA);
                return;
            }
            if (event.key === "End") {
                event.preventDefault();
                commitKeyboardStep(Math.max(minA, m.available - minB));
            }
        },
        [commitKeyboardStep, dir, measureSplit, minA, minB],
    );

    return (
        <div
            // `hs-touch-none`：分隔条按下即拖，触摸上必须收回手势所有权，
            // 否则浏览器会把它解释成滚动并派发 pointercancel（拖到一半就断）。
            className="hs-dock-splitter hs-touch-none"
            data-dir={dir}
            data-dragging={dragging ? "true" : "false"}
            role="separator"
            aria-orientation={dir === "row" ? "vertical" : "horizontal"}
            // 可聚焦分隔条：读屏用户与键盘用户唯一可用的尺寸调整通道。
            tabIndex={0}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={valueNowPct ?? undefined}
            aria-valuetext={valueNowPct == null ? undefined : `${valueNowPct}%`}
            ref={separatorRef}
            onFocus={() => {
                const m = measureSplit();
                if (m) setValueNowPct(Math.round((m.aSize / m.available) * 100));
            }}
            onKeyDown={onKeyDown}
            onPointerDown={onPointerDown}
            onDoubleClick={onReset}
        />
    );
}
