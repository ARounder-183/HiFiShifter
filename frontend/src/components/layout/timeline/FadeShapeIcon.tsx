/**
 * FadeShapeIcon — REAPER 七预设淡变形状的微型 SVG 图标。
 *
 * 图标曲线直接用 reaperFade.fadeGainSigned 采样生成（与时间轴画布同一公式
 * 核心），保证"图标所见 = 画布所见 = 音频所听"。采样曲率取该形状的**默认
 * 曲率**（defaultFadeDirFor，与切换形状时的重置值同源）—— 否则快收/陡峭族
 * 在 dir=0 处恰是它们的视觉线性点，图标会退化成直线。
 *
 * 绘制采用足够的采样密度并对陡峭端自适应细分，避免"突然折过去"的观感。
 */
import { useMemo } from "react";

import { defaultFadeDirFor, FADE_PRESETS, fadeGainSigned } from "./reaperFade";

const PRESET_BY_SHAPE = new Map(FADE_PRESETS.map((preset) => [preset.shape, preset.id]));

export type FadeShapeIconProps = {
    /** REAPER 形状 id（整数 0..6；小数变体取整后匹配最近预设）。 */
    shape: number;
    size?: number;
    /** 淡出行时水平镜像（淡出方向）。 */
    mirrored?: boolean;
};

/**
 * 把形状按其默认曲率采样为 viewBox=size² 的 polyline points。
 *
 * 陡峭曲线（e 很大/很小）在中低采样密度下会出现可见折角 —— 这里按形状
 * 的陡峭程度分配采样密度：均匀基线 + 两端高密度加密（对数分布），确保
 * 指数曲线在像素尺度上是平滑的。
 */
function samplePolylinePoints(shape: number, dir: number, size: number): string {
    const pad = 2;
    const inner = size - pad * 2;
    // 基础 24 点；陡峭族加倍。再加两端各 8 个对数加密点。
    const isSteep = Math.trunc(shape) === 3 || Math.trunc(shape) === 4 || Math.trunc(shape) === 6;
    const baseSteps = isSteep ? 40 : 24;

    const ts: number[] = [];
    for (let i = 0; i < baseSteps; i += 1) {
        ts.push(i / (baseSteps - 1));
    }
    // 两端对数加密：靠近 t=0 与 t=1 处补点（陡峭曲线的"爆发段"）。
    for (let k = 1; k <= 6; k += 1) {
        ts.push(0.02 / Math.pow(2, k));
        ts.push(1 - 0.02 / Math.pow(2, k));
    }
    ts.sort((a, b) => a - b);

    const pts: string[] = [];
    for (const t of ts) {
        const gain = fadeGainSigned(shape, dir, "in", t);
        // y 轴向下：增益 0 → 底部，1 → 顶部。
        const x = pad + t * inner;
        const y = pad + (1 - gain) * inner;
        pts.push(`${x.toFixed(2)},${y.toFixed(2)}`);
    }
    return pts.join(" ");
}

export function FadeShapeIcon({ shape, size = 18, mirrored }: FadeShapeIconProps) {
    const normalizedShape = Number.isFinite(shape) ? Math.trunc(shape) : 0;
    const presetId = PRESET_BY_SHAPE.get(normalizedShape) ?? "linear";
    // REAPER 语义：每个形状有自己的默认曲率（切换时重置到该值）；
    // 图标展示的就是这个"出厂形态"。
    const iconDir = defaultFadeDirFor(normalizedShape, false);
    const points = useMemo(
        () => samplePolylinePoints(normalizedShape, iconDir, size),
        [normalizedShape, iconDir, size],
    );
    const pad = 2;

    return (
        <svg
            width={size}
            height={size}
            viewBox={`0 0 ${size} ${size}`}
            aria-hidden="true"
            focusable="false"
            style={mirrored ? { transform: "scaleX(-1)" } : undefined}
        >
            {presetId === "linear" ? (
                <line
                    x1={pad}
                    y1={size - pad}
                    x2={size - pad}
                    y2={pad}
                    stroke="currentColor"
                    strokeWidth="1.5"
                    strokeLinecap="round"
                    fill="none"
                />
            ) : (
                <polyline
                    points={points}
                    stroke="currentColor"
                    strokeWidth="1.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    fill="none"
                />
            )}
        </svg>
    );
}
