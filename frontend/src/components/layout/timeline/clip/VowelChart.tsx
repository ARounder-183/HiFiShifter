/* eslint-disable react-refresh/only-export-components -- 文件同时导出组件与 Hook/常量（刷新边界按文件粒度接受） */
import React from "react";
import { registerDragAbort } from "../gestureFocusGuard";
import { VOWEL_GUIDE_LINES, VOWEL_POINTS } from "./vowelChartLayout";
import { formantChartPointerDownShouldPreventDefault } from "./clipFormantInteractionGuards";

const F1_MIN = 250;
const F1_MAX = 1000;
const F2_MIN = 540;
const F2_MAX = 2600;
const WIDTH = 420;
const HEIGHT = 320;
const PAD_LEFT = 26;
const PAD_RIGHT = 26;
const PAD_TOP = 28;
const PAD_BOTTOM = 20;
const PLOT_WIDTH = WIDTH - PAD_LEFT - PAD_RIGHT;
const PLOT_HEIGHT = HEIGHT - PAD_TOP - PAD_BOTTOM;

function clamp(value: number, min: number, max: number): number {
    return Math.min(max, Math.max(min, value));
}

export function formantsToChartPoint(
    f1: number,
    f2: number,
    f2Min = F2_MIN,
    f2Max = F2_MAX,
    f1Min = F1_MIN,
    f1Max = F1_MAX,
    width = PLOT_WIDTH,
    height = PLOT_HEIGHT,
) {
    const clampedF1 = clamp(f1, f1Min, f1Max);
    const clampedF2 = clamp(f2, f2Min, f2Max);
    return {
        x: ((f2Max - clampedF2) / (f2Max - f2Min)) * width,
        y: ((clampedF1 - f1Min) / (f1Max - f1Min)) * height,
    };
}

export function chartPointToFormants(
    x: number,
    y: number,
    width = PLOT_WIDTH,
    height = PLOT_HEIGHT,
    f2Min = F2_MIN,
    f2Max = F2_MAX,
    f1Min = F1_MIN,
    f1Max = F1_MAX,
) {
    return {
        f2: clamp(f2Max - (x / width) * (f2Max - f2Min), f2Min, f2Max),
        f1: clamp(f1Min + (y / height) * (f1Max - f1Min), f1Min, f1Max),
    };
}

function formantsToPoint(f1: number, f2: number) {
    const point = formantsToChartPoint(f1, f2);
    const x = PAD_LEFT + point.x;
    const y = PAD_TOP + point.y;
    return { x, y };
}

function pointToFormants(clientX: number, clientY: number, rect: DOMRect) {
    const x = clamp(clientX - rect.left, PAD_LEFT, WIDTH - PAD_RIGHT);
    const y = clamp(clientY - rect.top, PAD_TOP, HEIGHT - PAD_BOTTOM);
    const formants = chartPointToFormants(x - PAD_LEFT, y - PAD_TOP, PLOT_WIDTH, PLOT_HEIGHT);
    return {
        targetF2Hz: Math.round(formants.f2),
        targetF1Hz: Math.round(formants.f1),
    };
}

export { formantChartPointerDownShouldPreventDefault };

export const VowelChart: React.FC<{
    targetF1Hz: number;
    targetF2Hz: number;
    disabled?: boolean;
    onChange: (next: { targetF1Hz: number; targetF2Hz: number }) => void;
}> = ({ targetF1Hz, targetF2Hz, disabled = false, onChange }) => {
    const svgRef = React.useRef<SVGSVGElement | null>(null);
    const draggingRef = React.useRef(false);

    const updateFromPointer = React.useCallback(
        (clientX: number, clientY: number) => {
            const rect = svgRef.current?.getBoundingClientRect();
            if (!rect) return;
            onChange(pointToFormants(clientX, clientY, rect));
        },
        [onChange],
    );

    React.useEffect(() => {
        const onMove = (event: PointerEvent) => {
            if (!draggingRef.current || disabled) return;
            updateFromPointer(event.clientX, event.clientY);
        };
        const onEnd = () => {
            draggingRef.current = false;
            unregisterAbort();
        };
        // 失焦取消：切屏期间 pointerup/pointercancel 不送达本窗口，blur 时
        // 复位拖拽布尔（否则切回后任何鼠标移动都会持续改写共振峰）。
        const unregisterAbort = registerDragAbort(onEnd);
        window.addEventListener("pointermove", onMove, true);
        window.addEventListener("pointerup", onEnd, true);
        window.addEventListener("pointercancel", onEnd, true);
        return () => {
            unregisterAbort();
            window.removeEventListener("pointermove", onMove, true);
            window.removeEventListener("pointerup", onEnd, true);
            window.removeEventListener("pointercancel", onEnd, true);
        };
    }, [disabled, updateFromPointer]);

    const point = formantsToPoint(targetF1Hz, targetF2Hz);

    return (
        <svg
            ref={svgRef}
            width={WIDTH}
            height={HEIGHT}
            viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
            className="rounded border"
            style={{
                background: "var(--qt-graph-bg)",
                borderColor: "var(--qt-border)",
                cursor: disabled ? "not-allowed" : "crosshair",
                opacity: disabled ? 0.55 : 1,
                userSelect: "none",
                WebkitUserSelect: "none",
                touchAction: "none",
            }}
            onPointerDown={(event) => {
                if (!formantChartPointerDownShouldPreventDefault({ disabled })) return;
                event.preventDefault();
                event.stopPropagation();
                draggingRef.current = true;
                updateFromPointer(event.clientX, event.clientY);
            }}
        >
            <polygon
                points={`${PAD_LEFT + PLOT_WIDTH * 0.62},${HEIGHT - PAD_BOTTOM} ${WIDTH - PAD_RIGHT},${HEIGHT - PAD_BOTTOM} ${WIDTH - PAD_RIGHT},${PAD_TOP + PLOT_HEIGHT * 0.58}`}
                fill="var(--qt-subtle-3)"
            />
            {Array.from({ length: 10 }).map((_, idx) => {
                const y = PAD_TOP + (PLOT_HEIGHT / 9) * idx;
                return (
                    <line
                        key={`h-${idx}`}
                        x1={PAD_LEFT}
                        y1={y}
                        x2={WIDTH - PAD_RIGHT}
                        y2={y}
                        stroke={
                            idx === 0 || idx === 9
                                ? "var(--qt-graph-grid-strong)"
                                : "var(--qt-graph-grid-weak)"
                        }
                        strokeWidth="1"
                    />
                );
            })}
            {Array.from({ length: 16 }).map((_, idx) => {
                const x = PAD_LEFT + (PLOT_WIDTH / 15) * idx;
                return (
                    <line
                        key={`v-${idx}`}
                        x1={x}
                        y1={PAD_TOP}
                        x2={x}
                        y2={HEIGHT - PAD_BOTTOM}
                        stroke={
                            idx === 0 || idx === 15
                                ? "var(--qt-graph-grid-strong)"
                                : "var(--qt-graph-grid-weak)"
                        }
                        strokeWidth="1"
                    />
                );
            })}
            {VOWEL_GUIDE_LINES.map((labels, index) => {
                const path = labels
                    .map((label, pointIndex) => {
                        const vowel = VOWEL_POINTS.find((entry) => entry.label === label);
                        if (!vowel) return null;
                        const pos = formantsToPoint(vowel.f1, vowel.f2);
                        return `${pointIndex === 0 ? "M" : "L"} ${pos.x} ${pos.y}`;
                    })
                    .filter((segment): segment is string => Boolean(segment))
                    .join(" ");
                if (!path) return null;
                return (
                    <path
                        key={`guide-${index}`}
                        d={path}
                        fill="none"
                        stroke="var(--qt-border)"
                        strokeOpacity="0.55"
                        strokeWidth="1.25"
                    />
                );
            })}
            {VOWEL_POINTS.map((vowel) => {
                const pos = formantsToPoint(vowel.f1, vowel.f2);
                return (
                    <text
                        key={vowel.label}
                        x={pos.x}
                        y={pos.y}
                        fontSize="15"
                        fontWeight="600"
                        textAnchor="middle"
                        dominantBaseline="middle"
                        fill="var(--qt-text)"
                    >
                        {vowel.label}
                    </text>
                );
            })}
            <line
                x1={PAD_LEFT}
                y1={PAD_TOP}
                x2={WIDTH - PAD_RIGHT}
                y2={PAD_TOP}
                stroke="var(--qt-text)"
                strokeWidth="2"
            />
            <line
                x1={WIDTH - PAD_RIGHT}
                y1={PAD_TOP}
                x2={WIDTH - PAD_RIGHT}
                y2={HEIGHT - PAD_BOTTOM}
                stroke="var(--qt-text)"
                strokeWidth="2"
            />
            <polygon
                points={`${PAD_LEFT},${PAD_TOP} ${PAD_LEFT + 10},${PAD_TOP - 5} ${PAD_LEFT + 10},${PAD_TOP + 5}`}
                fill="var(--qt-text)"
            />
            <polygon
                points={`${WIDTH - PAD_RIGHT},${HEIGHT - PAD_BOTTOM} ${WIDTH - PAD_RIGHT - 5},${HEIGHT - PAD_BOTTOM - 10} ${WIDTH - PAD_RIGHT + 5},${HEIGHT - PAD_BOTTOM - 10}`}
                fill="var(--qt-text)"
            />
            <text x={PAD_LEFT + 6} y={18} fontSize="11" fill="var(--qt-text)">
                {F2_MAX}
            </text>
            <text x={WIDTH / 2 - 20} y={18} fontSize="11" fill="var(--qt-text)">
                F2 (Hz)
            </text>
            <text
                x={WIDTH - PAD_RIGHT - 12}
                y={18}
                fontSize="11"
                textAnchor="end"
                fill="var(--qt-text)"
            >
                {F2_MIN}
            </text>
            <text x={WIDTH - PAD_RIGHT + 12} y={PAD_TOP + 4} fontSize="11" fill="var(--qt-text)">
                {F1_MIN}
            </text>
            <text
                x={WIDTH - 6}
                y={HEIGHT / 2 - 4}
                fontSize="11"
                textAnchor="end"
                fill="var(--qt-text)"
            >
                F1
            </text>
            <text
                x={WIDTH - 6}
                y={HEIGHT / 2 + 10}
                fontSize="11"
                textAnchor="end"
                fill="var(--qt-text)"
            >
                (Hz)
            </text>
            <text
                x={WIDTH - PAD_RIGHT + 12}
                y={HEIGHT - PAD_BOTTOM}
                fontSize="11"
                fill="var(--qt-text)"
            >
                {F1_MAX}
            </text>
            <circle
                cx={point.x}
                cy={point.y}
                r="7"
                fill="var(--qt-highlight)"
                stroke="var(--qt-window)"
                strokeWidth="2"
            />
        </svg>
    );
};
