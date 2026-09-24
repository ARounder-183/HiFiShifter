/*
 * 分隔条：拖动改变两侧尺寸。
 *
 * 【拖动期间只写 DOM】与仓库既有的时间轴/参数编辑器分隔条同一策略
 * （见重构前的 `App.tsx` 手搓 splitter）：pointermove 里直接改 pane 的
 * flex 样式，绕过 React 重渲染；松手才 dispatch 落库。区别是这次不再依赖
 * `container.children[i]` 这种脆弱的下标约定 —— 两侧 pane 的 ref 由
 * `DockSplit` 显式持有并传进来。
 *
 * 【拖完是改比例还是改像素】取决于该分割节点原本的形态：本来就是固定像素
 * （如右侧停靠栏）的，拖完仍固定像素；否则改比例，让窗口缩放时按比例分配。
 * 这样"我把它拖到 360px"在两种形态下都不会被一次窗口缩放悄悄改掉。
 */

import { useCallback, useRef, useState } from "react";

import { DOCK_SPLITTER_PX, type DockSplitNode } from "../../features/dock/dockTypes";
import { shouldSuppressHoverSideEffects } from "../../utils/penInput";

export interface DockSplitterProps {
    dir: DockSplitNode["dir"];
    /** 该分割节点的当前形态，决定拖完写比例还是写像素。 */
    ratio: number;
    fixed: DockSplitNode["fixed"];
    /** 两侧 pane 的 ref（顺序与 `splitRect` 的 a/b 一致）。 */
    paneARef: React.RefObject<HTMLDivElement | null>;
    paneBRef: React.RefObject<HTMLDivElement | null>;
    /** 两侧最小尺寸（像素）。 */
    minA: number;
    minB: number;
    onCommit: (next: { ratio: number; fixed: DockSplitNode["fixed"] }) => void;
    /** 双击复位：回到两侧均分。 */
    onReset: () => void;
}

export function DockSplitter({
    dir,
    ratio,
    fixed,
    paneARef,
    paneBRef,
    minA,
    minB,
    onCommit,
    onReset,
}: DockSplitterProps) {
    const [dragging, setDragging] = useState(false);
    const latestRef = useRef({ ratio, fixed });

    const onPointerDown = useCallback(
        (event: React.PointerEvent<HTMLDivElement>) => {
            // 数位笔 / 触摸不触发分隔条：4px 窄条 + 按下即生效，画线时极易误扫。
            if (shouldSuppressHoverSideEffects(event.nativeEvent)) return;
            if (event.button !== 0) return;
            const paneA = paneARef.current;
            const paneB = paneBRef.current;
            const parent = paneA?.parentElement;
            if (!paneA || !paneB || !parent) return;

            event.preventDefault();
            setDragging(true);

            const rect = parent.getBoundingClientRect();
            const available = (dir === "row" ? rect.width : rect.height) - DOCK_SPLITTER_PX;
            const total = available;

            const apply = (clientX: number, clientY: number) => {
                const offset = dir === "row" ? clientX - rect.left : clientY - rect.top;
                const maxA = Math.max(minA, available - minB);
                const aSize = Math.min(maxA, Math.max(minA, offset));
                paneA.style.flex = `0 0 ${aSize}px`;
                paneB.style.flex = "1 1 0";
                latestRef.current = {
                    ratio: total > 0 ? aSize / total : 0.5,
                    fixed: fixed ? { side: "a", px: Math.round(aSize) } : null,
                };
            };

            apply(event.clientX, event.clientY);

            const onMove = (moveEvent: PointerEvent) => apply(moveEvent.clientX, moveEvent.clientY);
            const onUp = () => {
                window.removeEventListener("pointermove", onMove);
                window.removeEventListener("pointerup", onUp);
                window.removeEventListener("pointercancel", onUp);
                setDragging(false);
                // 交给 React 重新接管样式，并落库。
                paneA.style.flex = "";
                paneB.style.flex = "";
                onCommit(latestRef.current);
            };
            window.addEventListener("pointermove", onMove);
            window.addEventListener("pointerup", onUp);
            window.addEventListener("pointercancel", onUp);
        },
        [dir, fixed, minA, minB, onCommit, paneARef, paneBRef],
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
