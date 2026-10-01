/**
 * 手绘单周期编辑器。
 *
 * 【为什么需要它】`table` 波形此前只能由「从选区提取」产生 —— 用户想手捏一个
 * 周期没有入口。这里补上：横轴 = 一个周期（64 格），纵轴 = −1..1，带中线与四分
 * 线网格；按住拖动即以画笔落点写入，松手完成一次笔画。
 *
 * 【与渲染的关系】写进草稿的是 `cycle: { kind: "table", table }` —— 预览、试听、
 * 拖拽、应用全管线无差别支持（这是统一周期模型当初的承诺）。首尾相接由渲染端
 * 保证（`sampleCycle` 把表当循环），编辑器只把左右缘各画一格高亮作视觉提示。
 */

import { useEffect, useRef } from "react";
import { Flex } from "@radix-ui/themes";

import { readDevicePixelRatio } from "../../../utils/devicePixelLine";
import { AppButton, useRepeatPress } from "../../../ui";
import {
    cycleEditorPoint,
    cycleEditorY,
    paintCycleBin,
    paintCycleSegment,
    smoothCycleTable,
} from "./vibratoCycleEdit";

export interface VibratoCycleEditorProps {
    table: number[];
    disabled?: boolean;
    onChange: (table: number[]) => void;
    /** 复位到进入手绘前的形状；省略则不显示该按钮。 */
    onReset?: () => void;
    /**
     * 复位到**当前选中的参数式形状**（按钮文案里带上形状名）。
     *
     * 与 `onReset` 的区别：那个回到"你进来之前的样子"（可能是一张提取/手绘出来的表），
     * 这个把表重新采样成下拉框里当前那个形状 —— 手绘画歪了想从头来，或者想把提取出来
     * 的波形换成规整的正弦，都靠它。
     */
    onResetToShape?: () => void;
    smoothLabel: string;
    resetLabel: string;
    resetShapeLabel?: string;
    ariaLabel: string;
}

/** 画布高度（CSS 像素）。 */
const EDITOR_HEIGHT = 120;

function tokenColor(name: string, fallback: string): string {
    if (typeof window === "undefined") return fallback;
    const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return value || fallback;
}

export function VibratoCycleEditor({
    table,
    disabled = false,
    onChange,
    onReset,
    onResetToShape,
    smoothLabel,
    resetLabel,
    resetShapeLabel,
    ariaLabel,
}: VibratoCycleEditorProps) {
    const canvasRef = useRef<HTMLCanvasElement | null>(null);
    const containerRef = useRef<HTMLDivElement | null>(null);
    const drawingRef = useRef(false);
    const lastPointRef = useRef<{ bin: number; value: number } | null>(null);

    /*
     * 最新的表与回调：长按重复时每一拍都要基于**上一拍的结果**继续平滑。
     * 若闭包捕获的是按下那一刻的 `table`，连按十次也只会把同一份表平滑十遍 ——
     * 看起来像"长按没反应"。写入放在 effect 里（不是渲染期），与 `useFrameCommitter`
     * 的处理一致。
     */
    const latestRef = useRef({ table, onChange });
    useEffect(() => {
        latestRef.current = { table, onChange };
    });

    /** 短按平滑一次；按住则连续平滑（与键盘自动重复同一套手感）。 */
    const smoothPress = useRepeatPress({
        disabled,
        onTrigger: () => {
            const { table: current, onChange: apply } = latestRef.current;
            apply(smoothCycleTable(current));
        },
    });

    useEffect(() => {
        const canvas = canvasRef.current;
        const container = containerRef.current;
        if (!canvas || !container) return;

        const draw = () => {
            const width = container.clientWidth;
            if (width <= 0) return;
            const height = EDITOR_HEIGHT;
            const dpr = readDevicePixelRatio();
            canvas.width = Math.round(width * dpr);
            canvas.height = Math.round(height * dpr);
            canvas.style.width = `${width}px`;
            canvas.style.height = `${height}px`;

            const ctx = canvas.getContext("2d");
            if (!ctx) return;
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
            ctx.clearRect(0, 0, width, height);

            const accent = tokenColor("--qt-accent", "#6aa9ff");
            const muted = tokenColor("--qt-text-muted", "#8a8a8a");
            const divider = tokenColor("--qt-divider", "#3a3a3a");

            // 值 → y 的换算与命中测试共用 `cycleEditorY`（内缩量同源，见其注释）：
            // 各写一套就会让"看到的峰顶"与"能画到 1.0 的那一行"错开。
            const valueToY = (value: number) => cycleEditorY(value, height);

            // 网格：中线 + 四分线（横），四分位置（纵）。
            ctx.strokeStyle = divider;
            ctx.lineWidth = 1;
            for (const frac of [0.25, 0.5, 0.75]) {
                const y = height * frac;
                ctx.globalAlpha = frac === 0.5 ? 0.9 : 0.4;
                ctx.beginPath();
                ctx.moveTo(0, y + 0.5);
                ctx.lineTo(width, y + 0.5);
                ctx.stroke();
            }
            ctx.globalAlpha = 0.4;
            for (const frac of [0.25, 0.5, 0.75]) {
                const x = width * frac;
                ctx.beginPath();
                ctx.moveTo(x + 0.5, 0);
                ctx.lineTo(x + 0.5, height);
                ctx.stroke();
            }
            ctx.globalAlpha = 1;

            // 首尾各一格高亮：提示"这里与对面相接"。
            if (table.length > 0) {
                const cellWidth = width / table.length;
                ctx.globalAlpha = 0.12;
                ctx.fillStyle = accent;
                ctx.fillRect(0, 0, Math.max(1, cellWidth), height);
                ctx.fillRect(width - Math.max(1, cellWidth), 0, Math.max(1, cellWidth), height);
                ctx.globalAlpha = 1;
            }

            // 波形：画 n+1 个点（末点 = 首点），让周期在视觉上闭合。
            const n = table.length;
            if (n >= 2) {
                ctx.strokeStyle = disabled ? muted : accent;
                ctx.lineWidth = 1.5;
                ctx.lineJoin = "round";
                ctx.beginPath();
                for (let i = 0; i <= n; i += 1) {
                    const value = table[i % n] ?? 0;
                    const x = (i / n) * width;
                    const y = valueToY(value);
                    if (i === 0) ctx.moveTo(x, y);
                    else ctx.lineTo(x, y);
                }
                ctx.stroke();
            }
        };

        draw();
        const observer = new ResizeObserver(draw);
        observer.observe(container);
        return () => observer.disconnect();
    }, [table, disabled]);

    const pointAt = (event: { clientX: number; clientY: number }) => {
        const container = containerRef.current;
        if (!container) return null;
        const rect = container.getBoundingClientRect();
        return cycleEditorPoint(
            event.clientX - rect.left,
            event.clientY - rect.top,
            rect.width,
            EDITOR_HEIGHT,
            table.length,
        );
    };

    const handlePointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
        if (disabled || event.button !== 0) return;
        const point = pointAt(event);
        if (!point) return;
        drawingRef.current = true;
        lastPointRef.current = point;
        event.currentTarget.setPointerCapture(event.pointerId);
        onChange(paintCycleBin(table, point.bin, point.value));
    };

    const handlePointerMove = (event: React.PointerEvent<HTMLDivElement>) => {
        if (disabled || !drawingRef.current) return;
        const last = lastPointRef.current;
        if (!last) return;
        // 用合并事件补齐快笔：一次 pointermove 里可能有多个采样点。
        const native = event.nativeEvent;
        const points =
            typeof native.getCoalescedEvents === "function"
                ? native.getCoalescedEvents()
                : [native];
        let next = table;
        let cursor = last;
        for (const sample of points.length > 0 ? points : [native]) {
            const point = pointAt(sample);
            if (!point) continue;
            next = paintCycleSegment(next, cursor, point);
            cursor = point;
        }
        lastPointRef.current = cursor;
        onChange(next);
    };

    const endStroke = (event: React.PointerEvent<HTMLDivElement>) => {
        if (!drawingRef.current) return;
        drawingRef.current = false;
        lastPointRef.current = null;
        if (event.currentTarget.hasPointerCapture(event.pointerId)) {
            event.currentTarget.releasePointerCapture(event.pointerId);
        }
    };

    return (
        <div data-testid="vibrato-cycle-editor">
            <div
                ref={containerRef}
                className="w-full cursor-crosshair"
                style={{ height: EDITOR_HEIGHT, touchAction: disabled ? undefined : "none" }}
                onPointerDown={handlePointerDown}
                onPointerMove={handlePointerMove}
                onPointerUp={endStroke}
                onPointerCancel={endStroke}
            >
                <canvas
                    ref={canvasRef}
                    role="img"
                    aria-label={ariaLabel}
                    className="block w-full rounded border border-qt-border"
                    style={{ height: EDITOR_HEIGHT }}
                />
            </div>
            <Flex align="center" gap="2" mt="2" wrap="wrap">
                <AppButton
                    size="sm"
                    emphasis="soft"
                    disabled={disabled}
                    // 短按平滑一次，按住连续平滑（`useRepeatPress`）。
                    {...smoothPress}
                >
                    {smoothLabel}
                </AppButton>
                {onReset ? (
                    <AppButton size="sm" emphasis="soft" disabled={disabled} onClick={onReset}>
                        {resetLabel}
                    </AppButton>
                ) : null}
                {onResetToShape && resetShapeLabel ? (
                    <AppButton
                        size="sm"
                        emphasis="soft"
                        disabled={disabled}
                        onClick={onResetToShape}
                    >
                        {resetShapeLabel}
                    </AppButton>
                ) : null}
            </Flex>
        </div>
    );
}
