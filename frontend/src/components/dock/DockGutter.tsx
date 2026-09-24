/*
 * 沟槽分隔条：面板**内部**固定宽/高分区之间的可拖拽边界。
 *
 * 与 `DockSplitter` 的区别：分隔条管的是"两个面板之间"（布局树的结构），
 * 沟槽管的是"一个面板内部的固定分区"——时间轴左侧轨道头、参数编辑器左侧
 * 琴键轴。后者不参与停靠树的划分，但对用户而言同样是"我调过的尺寸必须记住"，
 * 因此它的尺寸也存进布局（`DockGutterSizes`）。
 *
 * 【拖动期间只写 DOM】与仓库既有的分隔条策略一致：pointermove 里直接改目标
 * 元素的样式，松手才提交到 Redux + 落盘。区别是这里把"改哪个元素"交给调用方
 * 提供的 `onLive`，因此同一个组件能服务于任意内部结构。
 */

import { useCallback, useRef, useState } from "react";

import { shouldSuppressHoverSideEffects } from "../../utils/penInput";

export interface DockGutterProps {
    /** `col` = 竖直条（调宽度），`row` = 水平条（调高度）。 */
    dir: "col" | "row";
    value: number;
    min: number;
    max: number;
    /**
     * 拖动中的实时应用（只改 DOM，不要 setState）。
     *
     * 调用方拿到的是**已钳制**的像素值。
     */
    onLive: (px: number) => void;
    /** 松手提交（进 Redux + 落盘）。 */
    onCommit: (px: number) => void;
    /** 双击复位；缺省时不响应双击。 */
    onReset?: () => void;
    /**
     * 拖动方向系数：`1` = 向右/下拖增大，`-1` = 向左/上拖增大。
     *
     * 面板贴在右侧或下方时需要 -1（拖拽方向与尺寸变化相反）。
     */
    sign?: 1 | -1;
    /** 无障碍标签（i18n 后的文本）。 */
    label?: string;
}

export function DockGutter({
    dir,
    value,
    min,
    max,
    onLive,
    onCommit,
    onReset,
    sign = 1,
    label,
}: DockGutterProps) {
    const [dragging, setDragging] = useState(false);
    const latestRef = useRef(value);

    const onPointerDown = useCallback(
        (event: React.PointerEvent<HTMLDivElement>) => {
            // 数位笔 / 触摸不触发：5px 窄条 + 按下即生效，画线时极易误扫。
            if (shouldSuppressHoverSideEffects(event.nativeEvent)) return;
            if (event.button !== 0) return;
            event.preventDefault();
            event.stopPropagation();
            setDragging(true);

            const startX = event.clientX;
            const startY = event.clientY;
            const start = value;
            latestRef.current = start;

            const apply = (clientX: number, clientY: number) => {
                const delta = dir === "col" ? clientX - startX : clientY - startY;
                const next = Math.min(max, Math.max(min, Math.round(start + sign * delta)));
                latestRef.current = next;
                onLive(next);
            };

            const onMove = (moveEvent: PointerEvent) => apply(moveEvent.clientX, moveEvent.clientY);
            const onUp = () => {
                window.removeEventListener("pointermove", onMove);
                window.removeEventListener("pointerup", onUp);
                window.removeEventListener("pointercancel", onUp);
                setDragging(false);
                onCommit(latestRef.current);
            };
            window.addEventListener("pointermove", onMove);
            window.addEventListener("pointerup", onUp);
            window.addEventListener("pointercancel", onUp);
        },
        [dir, max, min, onCommit, onLive, sign, value],
    );

    return (
        <div
            className="hs-dock-gutter"
            data-dir={dir}
            data-dragging={dragging ? "true" : "false"}
            role="separator"
            aria-orientation={dir === "col" ? "vertical" : "horizontal"}
            aria-label={label}
            aria-valuenow={value}
            aria-valuemin={min}
            aria-valuemax={max}
            onPointerDown={onPointerDown}
            onDoubleClick={onReset}
        />
    );
}
