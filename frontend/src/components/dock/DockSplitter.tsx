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
            const onUp = () => {
                window.removeEventListener("pointermove", onMove);
                window.removeEventListener("pointerup", onUp);
                window.removeEventListener("pointercancel", onUp);
                setDragging(false);
                // 不清空行内样式：`onLiveSize` 写入的就是 React 最终要写的值，
                // 提交后 React 的差异更新会发现"无需改动"，DOM 保持正确。
                if (latestRef.current) onCommit(latestRef.current);
            };
            window.addEventListener("pointermove", onMove);
            window.addEventListener("pointerup", onUp);
            window.addEventListener("pointercancel", onUp);
        },
        [containerRef, dir, fixed, minA, minB, onCommit, onLiveSize],
    );

    return (
        <div
            className="hs-dock-splitter"
            data-dir={dir}
            data-dragging={dragging ? "true" : "false"}
            role="separator"
            aria-orientation={dir === "row" ? "vertical" : "horizontal"}
            onPointerDown={onPointerDown}
            onDoubleClick={onReset}
        />
    );
}
