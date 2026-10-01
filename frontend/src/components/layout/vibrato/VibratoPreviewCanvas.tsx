/**
 * 颤音预设的波形预览画布。
 *
 * 【为什么必须有】没有它，用户在预设编辑器里是在**盲调**：深度、速率、渐入、
 * 不规则度这几个参数两两耦合，只看数字无法预判听感。这里把
 * `buildVibratoCurve` 的真实输出画出来，参数一改立刻重画。
 *
 * 纵轴按预设**自身**的幅度定标（见 `previewScaleCents`）：按参数值域定标的话，
 * 30 cents 的颤音会是一条直线。真实幅度由旁边的读数表达。
 *
 * 【可编辑】传入 `handles` 与手势回调后，画布从"只读渲染"升级为"所见即所编"：
 * 左右手柄拖动改渐入 / 渐出，主体拖动改相位与深度。手势只上报**区域与像素位移**，
 * 具体的字段换算由 `vibratoPreviewGestures.ts` 的纯函数完成 —— 画布不持有草稿。
 */

import { useEffect, useRef } from "react";
import { readDevicePixelRatio } from "../../../utils/devicePixelLine";
import { previewScaleCents, type VibratoPreviewSamples } from "./vibratoDialogLogic";
import {
    cursorForZone,
    hitTestPreviewZone,
    type PreviewHandleLayout,
    type PreviewZone,
} from "./vibratoPreviewGestures";

/** 画布几何：绘制与手势换算共用同一套标尺。 */
export interface VibratoPreviewGeometry {
    /** CSS 像素宽度。 */
    width: number;
    /** CSS 像素高度。 */
    height: number;
    /** 纵向每像素对应的 cents（与绘制同一套换算）。 */
    centsPerPx: number;
}

/** 手势开始时上报的几何 + 按下瞬间的修饰键状态。 */
export interface VibratoPreviewGestureInfo extends VibratoPreviewGeometry {
    /**
     * 按下瞬间的修饰键状态。
     *
     * 【为什么起点就要给】「精细调整」按增量缩放位移（见 `utils/fineAxisDrag.ts`），
     * 起手时修饰键是否已按下决定了首帧走哪个比例 —— 起手就按住时不该按"刚按下"
     * 的过渡比例处理。
     */
    modifiers: VibratoPreviewModifiers;
}

/** 手势开始时的修饰键状态（用于「精细调整」这类按修饰键缩放的手势）。 */
export interface VibratoPreviewModifiers {
    ctrlKey: boolean;
    shiftKey: boolean;
    altKey: boolean;
    metaKey: boolean;
}

/** 从指针事件读出修饰键状态。 */
function readModifiers(event: {
    ctrlKey: boolean;
    shiftKey: boolean;
    altKey: boolean;
    metaKey: boolean;
}): VibratoPreviewModifiers {
    return {
        ctrlKey: event.ctrlKey,
        shiftKey: event.shiftKey,
        altKey: event.altKey,
        metaKey: event.metaKey,
    };
}

export interface VibratoPreviewCanvasProps {
    samples: VibratoPreviewSamples;
    /** CSS 像素高度。 */
    height?: number;
    ariaLabel?: string;
    /**
     * 纵轴半幅（cents）。由调用方**一次性拟合**并保持稳定时，波形高度就等于深度，
     * 用户能直接判断幅度大小；省略时按峰值自适应（只读预览够用，但深度一变整幅
     * 就被重新缩放，看不出大小）。
     */
    halfCents?: number;
    /** 手柄的归一化横向位置（0..1）。提供时画出手柄并接受手势。 */
    handles?: PreviewHandleLayout;
    /** 手势开始。 */
    onGestureStart?: (zone: PreviewZone, info: VibratoPreviewGestureInfo) => void;
    /** 手势移动：自起点累计的像素位移，以及当前的修饰键状态。 */
    onGestureMove?: (deltaX: number, deltaY: number, modifiers: VibratoPreviewModifiers) => void;
    /** 手势结束。 */
    onGestureEnd?: () => void;
}

/** 读取语义色令牌；缺失（测试 / 非浏览器环境）时回退到中性色。 */
function tokenColor(name: string, fallback: string): string {
    if (typeof window === "undefined") return fallback;
    const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return value || fallback;
}

/** 手柄方块边长（CSS 像素）。 */
const HANDLE_SIZE = 7;

/**
 * 把 `[0, count)` 按"该点是否有值"切成若干连续段。
 *
 * 【为什么需要】音高曲线里的「未检测」帧以 `NaN` 给出（见 `vibratoDialogLogic` 的
 * `buildAppliedPreview`）。不切段的话，Canvas 会把断口两侧直接连起来 / 填起来 ——
 * "这里没有数据"就被画成了"这里有一条线"，正是要避免的误读。
 */
function finiteRuns(values: readonly number[], count: number): Array<[number, number]> {
    const runs: Array<[number, number]> = [];
    let start = -1;
    for (let i = 0; i < count; i += 1) {
        const ok = Number.isFinite(values[Math.min(i, values.length - 1)]);
        if (ok) {
            if (start < 0) start = i;
        } else if (start >= 0) {
            runs.push([start, i - 1]);
            start = -1;
        }
    }
    if (start >= 0) runs.push([start, count - 1]);
    return runs;
}

/** 轴标签的紧凑写法：整数不带小数点，非整数最多一位。 */
function formatAxisCents(value: number): string {
    return String(Math.round(value * 10) / 10);
}

export function VibratoPreviewCanvas({
    samples,
    height = 120,
    ariaLabel,
    halfCents: explicitHalfCents,
    handles,
    onGestureStart,
    onGestureMove,
    onGestureEnd,
}: VibratoPreviewCanvasProps) {
    const canvasRef = useRef<HTMLCanvasElement | null>(null);
    const containerRef = useRef<HTMLDivElement | null>(null);
    /** 最近一次绘制的几何：手势换算要用与绘制**同一套**标尺。 */
    const geometryRef = useRef<VibratoPreviewGeometry>({ width: 0, height, centsPerPx: 1 });
    /** 手势起点（未按下时为 null）。 */
    const gestureRef = useRef<{ zone: PreviewZone; x: number; y: number } | null>(null);
    const interactive = Boolean(handles && (onGestureStart || onGestureMove));

    useEffect(() => {
        const canvas = canvasRef.current;
        const container = containerRef.current;
        if (!canvas || !container) return;

        const draw = () => {
            const width = container.clientWidth;
            if (width <= 0) return;
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
            const panel = tokenColor("--qt-panel", "#1e1e1e");

            const midY = height / 2;
            const verticalReach = height / 2 - 6;
            // 显式纵轴（半幅 cents）优先：它由调用方**一次性拟合**，编辑期间保持不变，
            // 于是波形高度直接等于深度，用户能直观判断大小。省略时回落到按峰值自适应
            // （缩略图 / 无手势的只读预览用得上）。
            const halfCents =
                explicitHalfCents != null && explicitHalfCents > 0
                    ? explicitHalfCents
                    : previewScaleCents(samples.peakCents);
            const toY = (cents: number) => midY - (cents / halfCents) * verticalReach;
            geometryRef.current = {
                width,
                height,
                centsPerPx: halfCents / Math.max(1, verticalReach),
            };

            const count = Math.max(2, samples.wave.length);
            const toX = (index: number) => (index / (count - 1)) * width;

            /**
             * 折线：遇到无值的采样点**抬笔**，下一个有效点重新起笔。
             *
             * 直接 `lineTo(NaN)` 在 Canvas2D 里等价于"跳过这一点"，断口两侧仍会被连成
             * 一条直线 —— 那正好把"这里没有数据"画成了"这里有一条线"。显式抬笔才能
             * 让断口真的断开。
             */
            const drawPolyline = (values: readonly number[], xAt: (index: number) => number) => {
                ctx.beginPath();
                let pen = false;
                for (let i = 0; i < values.length; i += 1) {
                    const value = values[i];
                    if (!Number.isFinite(value)) {
                        pen = false;
                        continue;
                    }
                    const x = xAt(i);
                    const y = toY(value);
                    if (pen) ctx.lineTo(x, y);
                    else {
                        ctx.moveTo(x, y);
                        pen = true;
                    }
                }
                ctx.stroke();
            };

            // 刻度网格 + 读数：没有它，"波形占画布多少"仍然只是相对量；有了它，
            // 用户能直接把波峰高度读成 cents。上下边缘各标一次，中间画到 1/4 的细线。
            ctx.strokeStyle = divider;
            ctx.lineWidth = 1;
            ctx.font = "10px sans-serif";
            ctx.textBaseline = "middle";
            for (const fraction of [1, 0.5]) {
                for (const sign of [1, -1]) {
                    const cents = sign * halfCents * fraction;
                    const y = toY(cents);
                    ctx.globalAlpha = fraction === 1 ? 0.55 : 0.28;
                    ctx.beginPath();
                    ctx.moveTo(0, y + 0.5);
                    ctx.lineTo(width, y + 0.5);
                    ctx.stroke();
                    if (fraction === 1) {
                        ctx.globalAlpha = 0.75;
                        ctx.fillStyle = muted;
                        // 标签贴在线的内侧，避免被裁掉；单位由下方读数承担。
                        const label = `${sign > 0 ? "+" : "−"}${formatAxisCents(halfCents)}`;
                        ctx.fillText(label, 3, y + (sign > 0 ? 7 : -7));
                    }
                }
            }
            ctx.globalAlpha = 1;
            ctx.fillStyle = muted;

            // 零基线：波形围绕它摆动，视觉上是"音高中心"。
            ctx.strokeStyle = divider;
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(0, midY + 0.5);
            ctx.lineTo(width, midY + 0.5);
            ctx.stroke();

            // 包络带：±envelope 的填充区间，渐入 / 渐强 / 渐出入画就靠它。
            // 按连续段分别填充 —— 未检测帧处留空，而不是横着连成一条带。
            if (samples.envelope.length > 0) {
                const envelopeAt = (i: number) =>
                    samples.envelope[Math.min(i, samples.envelope.length - 1)];
                ctx.globalAlpha = 0.16;
                ctx.fillStyle = muted;
                for (const [from, to] of finiteRuns(samples.envelope, count)) {
                    ctx.beginPath();
                    for (let i = from; i <= to; i += 1) ctx.lineTo(toX(i), toY(envelopeAt(i)));
                    for (let i = to; i >= from; i -= 1) ctx.lineTo(toX(i), toY(-envelopeAt(i)));
                    ctx.closePath();
                    ctx.fill();
                }
                ctx.globalAlpha = 1;
            }

            // 套用前的原曲线：弱化的虚线，"颤音叠在哪条运动之上"一眼可见。
            // 只在"套用到选区"预览里出现（管理器预览没有原曲线）。
            const original = samples.original;
            if (original && original.length >= 2) {
                ctx.save();
                ctx.strokeStyle = muted;
                ctx.lineWidth = 1;
                ctx.setLineDash([4, 3]);
                drawPolyline(original, (i) => (i / (original.length - 1)) * width);
                ctx.restore();
            }

            // 波形本体。
            ctx.strokeStyle = accent;
            ctx.lineWidth = 1.5;
            ctx.lineJoin = "round";
            drawPolyline(samples.wave, toX);

            // 渐入 / 渐出手柄：小方块落在包络斜坡的起止处，语言与时间轴 clip 的
            // fade 手柄一致 —— 用户已经学会在那里拖。
            if (handles) {
                const drawHandle = (frac: number) => {
                    const index = Math.round(frac * (count - 1));
                    const envValue = samples.envelope.length
                        ? samples.envelope[Math.min(index, samples.envelope.length - 1)]
                        : 0;
                    const x = toX(index);
                    const y = toY(envValue);
                    const half = HANDLE_SIZE / 2;
                    ctx.fillStyle = panel;
                    ctx.strokeStyle = accent;
                    ctx.lineWidth = 1.5;
                    ctx.beginPath();
                    ctx.rect(x - half, y - half, HANDLE_SIZE, HANDLE_SIZE);
                    ctx.fill();
                    ctx.stroke();
                };
                drawHandle(handles.attackFrac);
                drawHandle(handles.releaseFrac);
            }
        };

        draw();

        // 宽度随停靠面板 / 对话框变化，需要跟着重画（画布不会自己缩放）。
        const observer = new ResizeObserver(draw);
        observer.observe(container);
        return () => observer.disconnect();
    }, [samples, height, handles, explicitHalfCents]);

    const localPoint = (event: React.PointerEvent<HTMLDivElement>) => {
        const rect = event.currentTarget.getBoundingClientRect();
        return { x: event.clientX - rect.left, y: event.clientY - rect.top };
    };

    const handlePointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
        if (!interactive || !handles || event.button !== 0) return;
        const { x, y } = localPoint(event);
        const zone = hitTestPreviewZone(x, geometryRef.current.width, handles);
        gestureRef.current = { zone, x, y };
        event.currentTarget.setPointerCapture(event.pointerId);
        onGestureStart?.(zone, { ...geometryRef.current, modifiers: readModifiers(event) });
    };

    const handlePointerMove = (event: React.PointerEvent<HTMLDivElement>) => {
        if (!interactive || !handles) return;
        const { x, y } = localPoint(event);
        const gesture = gestureRef.current;
        if (!gesture) {
            // 未按下时只更新光标，给出"这里能拖"的提示。
            event.currentTarget.style.cursor = cursorForZone(
                hitTestPreviewZone(x, geometryRef.current.width, handles),
            );
            return;
        }
        onGestureMove?.(x - gesture.x, y - gesture.y, readModifiers(event));
    };

    const endGesture = (event: React.PointerEvent<HTMLDivElement>) => {
        if (!gestureRef.current) return;
        gestureRef.current = null;
        if (event.currentTarget.hasPointerCapture(event.pointerId)) {
            event.currentTarget.releasePointerCapture(event.pointerId);
        }
        onGestureEnd?.();
    };

    return (
        <div
            ref={containerRef}
            className="w-full"
            style={{ height, touchAction: interactive ? "none" : undefined }}
            // 纵轴半幅（cents）：暴露出来便于测试断言"标尺是拟合档位而不是跟着当前值走"。
            data-axis-cents={explicitHalfCents ?? undefined}
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={endGesture}
            onPointerCancel={endGesture}
            onPointerLeave={(event) => {
                if (!gestureRef.current) event.currentTarget.style.cursor = "";
            }}
            data-testid={interactive ? "vibrato-preview-interactive" : undefined}
        >
            <canvas
                ref={canvasRef}
                role="img"
                aria-label={ariaLabel}
                className="block w-full"
                style={{ height }}
            />
        </div>
    );
}
