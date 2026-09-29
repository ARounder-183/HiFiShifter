/*
 * 布局树的递归渲染。
 *
 * 【为什么尺寸用 flex 而不是算好的像素】`splitRect` 已经能算出精确矩形（落点
 * 判定用它），但渲染若也按像素写死，窗口每次缩放都要重算整棵树并触发全量
 * 重渲染。改用 flex：比例侧 `flex-grow: ratio / flex-basis: 0`，固定侧
 * `flex: 0 0 <px>` —— 与 `splitRect` 的分配规则等价（剩余空间按 grow 分配），
 * 却由浏览器在合成阶段完成，窗口缩放零 React 成本。
 *
 * 两处必须保持一致：`splitRect` 用于落点判定与浮动夹紧，flex 用于渲染。
 * 它们的等价关系是"available * ratio"与"flex-basis:0 按 grow 分 available"。
 */

import { useCallback, useRef } from "react";

import { useAppDispatch, useAppSelector } from "../../app/hooks";
import { setSplitRatioOf } from "../../features/dock/dockSlice";
import { panelMinSize } from "../../features/dock/dockPanel";
import { MIN_PANE_PX, applyLivePaneStyles, paneStyle } from "../../features/dock/dockTree";
import { isPanelForm } from "../../features/dock/dockTree";
import type { DockLayout, DockNode, DockSplitNode } from "../../features/dock/dockTypes";
import { getPanel } from "../../features/dock/panelRegistry";
import { DockSplitter } from "./DockSplitter";
import { DockZone } from "./DockZone";

export function DockNodeView({ node }: { node: DockNode }) {
    if (node.t === "split") return <DockSplit node={node} />;
    return <DockZone node={node} />;
}

function DockSplit({ node }: { node: DockSplitNode }) {
    const dispatch = useAppDispatch();
    const layout = useAppSelector((state) => state.dock.layout);
    const paneARef = useRef<HTMLDivElement | null>(null);
    const paneBRef = useRef<HTMLDivElement | null>(null);
    const containerRef = useRef<HTMLDivElement | null>(null);

    const horizontal = node.dir === "row";
    const minA = subtreeMinSize(layout, node.a, horizontal);
    const minB = subtreeMinSize(layout, node.b, horizontal);

    /**
     * 拖拽期间把两侧写成"目标尺寸"。
     *
     * 【关键约束：直写的样式必须与 `paneStyle` 的最终结果**完全一致**】
     * React 的样式差异更新只写**变化过的**属性。若拖拽期间写了 `flex` 简写、
     * 松手时再清空，被清掉的 `flex-basis` 会落回 `auto`，而 React 只会重写
     * `flexGrow` —— 最终得到 `flex: <ratio> 1 auto`，**不是**按比例分配，分界线
     * 因此"拖了但没正确生效"。所以这里不写简写、也不在松手时清空：直接写入
     * `paneStyle` 针对**目标节点**算出的样式对象，React 随后的差异更新会发现
     * 值与它要写的一致，从而不再改动，DOM 保持正确。
     */
    const applyLiveSize = useCallback(
        (target: { ratio: number; fixed: DockSplitNode["fixed"] }) => {
            applyLivePaneStyles(paneARef.current, paneBRef.current, {
                ...node,
                ratio: target.ratio,
                fixed: target.fixed,
            });
        },
        [node],
    );

    const onCommit = useCallback(
        (next: { ratio: number; fixed: DockSplitNode["fixed"] }) => {
            dispatch(setSplitRatioOf({ splitId: node.id, ratio: next.ratio, fixed: next.fixed }));
        },
        [dispatch, node.id],
    );
    const onReset = useCallback(() => {
        dispatch(setSplitRatioOf({ splitId: node.id, ratio: 0.5, fixed: null }));
    }, [dispatch, node.id]);

    return (
        <div ref={containerRef} className="hs-dock-split" data-dir={node.dir}>
            <div ref={paneARef} className="hs-dock-pane" style={paneStyle(node, "a")}>
                <DockNodeView node={node.a} />
            </div>
            <DockSplitter
                dir={node.dir}
                containerRef={containerRef}
                fixed={node.fixed}
                minA={minA}
                minB={minB}
                onLiveSize={applyLiveSize}
                onCommit={onCommit}
                onReset={onReset}
            />
            <div ref={paneBRef} className="hs-dock-pane" style={paneStyle(node, "b")}>
                <DockNodeView node={node.b} />
            </div>
        </div>
    );
}

/**
 * 子树在该方向上的最小尺寸。
 *
 * 取子树内所有面板定义的最小值中的最大值 —— 一组面板挤在一起时，最小的那个
 * 决定了下限。没有声明最小尺寸的面板按 `MIN_PANE_PX` 算。面板成员的值来自
 * **它自己那棵树**的递归（`panelMinSize`）：面板的最小值不是注册表里的静态
 * 声明，否则嵌套面板会被外层分割挤成不可用的窄条。
 */
function subtreeMinSize(layout: DockLayout, node: DockNode, horizontal: boolean): number {
    if (node.t === "split") {
        return Math.max(
            subtreeMinSize(layout, node.a, horizontal),
            subtreeMinSize(layout, node.b, horizontal),
        );
    }
    let min = 0;
    for (const formId of node.tabs) {
        const form = layout.forms[formId];
        if (form && isPanelForm(form)) {
            min = Math.max(min, panelMinSize(layout, formId, horizontal) || MIN_PANE_PX);
            continue;
        }
        const definition = form ? getPanel(form.panelId) : undefined;
        const value = horizontal ? definition?.minWidth : definition?.minHeight;
        min = Math.max(min, value ?? MIN_PANE_PX);
    }
    return min || MIN_PANE_PX;
}
