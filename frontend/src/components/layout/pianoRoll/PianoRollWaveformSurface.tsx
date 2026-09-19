/**
 * 参数编辑器（Piano Roll）波形面适配层。
 *
 * 【主要内容】把 clip 峰值条目组装成单行 `WaveformSceneRow[]`，并把参数编辑器
 * 的水平缩放 / 滚动位置包装成 `TimelineAxis` 交给共享的 `WaveformSurface`。
 *
 * 【作用】参数编辑器与时间线共用同一块波形面，本组件只负责「单波形带」这一
 * 布局差异与坐标投影的构造。
 *
 * 【与其他模块的关系】
 * - 上游：`PianoRollPanel` 传入 clip 峰值、宽度与 scrollLeft / pxPerSec。
 * - 横向：视口实时同步走 `pianoRollViewportBus`；时间↔像素换算一律由
 *   `timelineAxis.ts` 提供。
 * - 下游：`waveform/WaveformSurface`。
 */

import React from "react";

import { WaveformSurface } from "../../../waveform/WaveformSurface";
import type { WaveformSceneClip, WaveformSceneRow } from "../../../waveform/sceneBuilder";
import type { WaveformColors } from "../../../theme/waveformColors";
import type { WaveformAmplitudeMap } from "../../../waveform/geometry";
import { linearAmplitudeMap } from "../../../waveform/geometry";
import { createTimelineAxis } from "../renderKernel/timelineAxis.js";
import type { ClipPeaksEntry } from "./useClipsPeaksForPianoRoll";
import { pianoRollViewportBus } from "./pianoRollViewportBus";

/**
 * 波形绘制所需的**响度自动化**数据（volume 曲线 + 动态目标/基线）。
 *
 * 【统一可听结果】最终混音的增益链是
 * `源峰值 × clip增益×淡化 × volume(t) × dyn增益(t)`。波形要画的正是这条链的
 * 结果 —— 与「当前编辑的是哪个参数」无关（见 makeLoudnessAmplitudeMap）。
 *
 * `target`/`volume` 都允许来自**编辑中的 live 覆盖**（写在 ref 上、不触发 React
 * 渲染），因此用取值函数而非数组：重绘那一刻读最新值，映射对象引用保持稳定。
 */
export interface LoudnessAutomationSource {
    /** 快照第 0 帧的绝对帧号（整工程快照恒 0）。 */
    startFrame: number;
    /** 快照帧步长。 */
    stride: number;
    /** 帧周期（毫秒）。 */
    framePeriodMs: number;
    /** 逐帧音量包络（无数据帧 = 1.0）。 */
    volume: readonly number[];
    /** 逐帧动态目标电平（哨兵已解析）。 */
    dynTarget: readonly number[];
    /** 逐帧原声电平基线；空数组 = 分析未就绪（动态增益恒 1）。 */
    dynBaseline: readonly number[];
}

/** 编辑中的曲线覆盖（与发起编辑的 paramView 窗口对齐）。 */
export interface LoudnessLiveCurve {
    startFrame: number;
    stride: number;
    values: readonly number[];
}

/**
 * 静音保护下限：原声低于此值时**放大**请求被拒绝（增益 1），衰减/静音请求
 * 照常生效（与后端 `DYN_MIN_REF` 一致）。
 */
const DYN_MIN_REF = 0.05;

/** 增益上限（与后端 `DYN_MAX_GAIN` 一致）。 */
const DYN_MAX_GAIN = 4;

/**
 * 在 `(startFrame, stride)` 对齐的曲线数组上按**绝对帧**线性插值取样。
 *
 * 越界返回 null（由调用方决定回退语义 —— volume 回退 1.0，动态回退"沿用原声"）。
 */
function sampleCurveLinear(
    values: readonly number[],
    startFrame: number,
    stride: number,
    frameF: number,
): number | null {
    if (values.length === 0 || !(stride > 0)) return null;
    const indexF = (frameF - startFrame) / stride;
    if (!Number.isFinite(indexF)) return null;
    if (indexF < 0 || indexF >= values.length) return null;
    const i0 = Math.floor(indexF);
    const i1 = Math.min(i0 + 1, values.length - 1);
    const frac = indexF - i0;
    const a = values[i0] ?? 0;
    const b = values[i1] ?? a;
    return a + (b - a) * frac;
}

/**
 * 参数编辑器波形的**统一幅度映射**：把源波形画成应用响度自动化之后的可听结果。
 *
 * 【语义】`映射(value, gain, t) = value × gain × volume(t) × dynGain(t)`，其中
 * - `gain` 是几何层传入的 clip 增益 × 淡化；
 * - `volume(t)` 是音量包络（曲线外回退 1.0 —— 越界持有末值对音量是合理的，
 *   与后端 sample_automation_curve 一致）；
 * - `dynGain(t) = 目标/基线`（带静音保护与上限；曲线外 / 无基线 → 1，不持有
 *   末值污染后续帧 —— 与后端 dyn 采样器同一语义）。
 *
 * 【为什么对所有参数生效】用户在**任何**参数面板里都必须实时看到音频波形
 * （画一笔音量/动态曲线波形立刻跟着动）——波形表达"将听到的声音"，曲线叠加层
 * 才表达"正在编辑的参数"。这也保证了音量↔动态互转前后波形逐像素不变（等效性的
 * 直观佐证）。
 *
 * 【live 优先】编辑中的参数读 live 覆盖（ref，指针频率更新、不进 React），
 * 窗口外回退整工程快照；`revision()` 是几何缓存的失效计数（ref 累加，
 * 不经 React 状态）。
 *
 * 【降级】动态无基线（分析未就绪）→ dynGain 恒 1，volume 仍然生效 ——
 * 绝不返回全零映射。
 */
export function makeLoudnessAmplitudeMap(
    source: LoudnessAutomationSource,
    live: {
        volume: () => LoudnessLiveCurve | null;
        dyn: () => LoudnessLiveCurve | null;
    },
    revision: () => number,
): WaveformAmplitudeMap {
    if (!(source.framePeriodMs > 0)) return linearAmplitudeMap;

    const sampleLiveOrSnapshot = (
        liveCurve: LoudnessLiveCurve | null,
        snapshotValues: readonly number[],
        frameF: number,
    ): number | null => {
        if (liveCurve && liveCurve.values.length > 0) {
            const sampled = sampleCurveLinear(
                liveCurve.values,
                liveCurve.startFrame,
                liveCurve.stride,
                frameF,
            );
            if (sampled !== null) return sampled;
        }
        return sampleCurveLinear(
            snapshotValues,
            source.startFrame,
            source.stride,
            frameF,
        );
    };

    const map = ((value: number, gain: number, timeSec: number | null) => {
        const v = value * gain;
        if (!Number.isFinite(v)) return 0;
        if (timeSec === null || !Number.isFinite(timeSec)) {
            // 拿不到时间（理论上不会发生）→ 退化为不施加响度自动化。
            return v;
        }
        const frameF = (timeSec * 1000) / source.framePeriodMs;
        if (!Number.isFinite(frameF)) return v;

        // ① 音量包络：曲线外回退 1.0（与后端越界持有末值语义一致）。
        const vol = sampleLiveOrSnapshot(live.volume(), source.volume, frameF) ?? 1.0;

        // ② 动态增益：无基线（分析未就绪）→ 1；目标为哨兵（沿用原声）→ 1；
        //    基线低于静音门限 → 只拒绝放大（衰减/静音照常，与后端
        //    compute_dyn_gain 同口径 —— 把曲线拉到 0 后噪声底帧也是静音）；
        //    其余 = 目标/基线，钳制到上限。
        let dynGain = 1;
        if (source.dynBaseline.length > 0) {
            const target = sampleLiveOrSnapshot(live.dyn(), source.dynTarget, frameF);
            const base = sampleCurveLinear(
                source.dynBaseline,
                source.startFrame,
                source.stride,
                frameF,
            );
            if (target !== null && target >= 0 && base !== null) {
                if (base <= 0) {
                    // 真静音（分析电平 0）：画了静音 = 静音。
                    dynGain = target <= 0 ? 0 : 1;
                } else if (base < DYN_MIN_REF && target > base) {
                    // 噪声底帧：只拒绝"放大"请求。
                    dynGain = 1;
                } else {
                    dynGain = Math.min(Math.max(target / base, 0), DYN_MAX_GAIN);
                }
            }
        }

        return v * vol * dynGain;
    }) as WaveformAmplitudeMap & { revision: () => number };
    // 修订号读者：几何缓存据此判定"引用未变但内部数据变了"需要重建几何。
    // 用函数而非数字：拖动时每帧变化的计数不能经过 React 状态，否则整块面板
    // 会以指针频率重渲染。
    map.revision = revision;
    return map;
}

function toSceneClip(entry: ClipPeaksEntry): WaveformSceneClip | null {
    if (!entry.sourcePath || entry.muted) return null;
    return {
        id: entry.clipId,
        sourcePath: entry.sourcePath,
        startSec: entry.startSec,
        lengthSec: entry.lengthSec,
        sourceStartSec: entry.sourceStartSec,
        sourceEndSec: entry.sourceEndSec,
        durationSec: entry.sourceDurationSec,
        sourceSampleRate: entry.sourceSampleRate,
        playbackRate: entry.playbackRate,
        reversed: entry.reversed,
        loopEnabled: entry.loopEnabled,
        gain: entry.gain,
        muted: false,
        fadeInSec: entry.fadeInSec,
        fadeOutSec: entry.fadeOutSec,
        autoFadeInSec: entry.autoFadeInSec,
        autoFadeOutSec: entry.autoFadeOutSec,
        fadeInShape: entry.fadeInShape,
        fadeInDir: entry.fadeInDir,
        fadeOutShape: entry.fadeOutShape,
        fadeOutDir: entry.fadeOutDir,
    };
}

export const PianoRollWaveformSurface = React.memo(function PianoRollWaveformSurface(props: {
    clips: readonly ClipPeaksEntry[];
    widthPx: number;
    heightPx: number;
    scrollLeftPx: number;
    pxPerSec: number;
    colors: WaveformColors;
    /**
     * 幅度映射。缺省为线性直投（与时间线一致）；参数编辑器传
     * `makeLoudnessAmplitudeMap(...)` 把波形画成应用响度自动化（volume × dyn）
     * 之后的可听结果，见该函数的说明。
     */
    amplitudeMap?: WaveformAmplitudeMap;
}) {
    const rows = React.useMemo<WaveformSceneRow[]>(
        () => [
            {
                topPx: 0,
                waveformTopPx: 0,
                waveformHeightPx: props.heightPx,
                clips: props.clips
                    .map(toSceneClip)
                    .filter((clip): clip is WaveformSceneClip => clip != null),
            },
        ],
        [props.clips, props.heightPx],
    );
    // 统一坐标投影：缓存以避免每次渲染新建对象导致波形面 memo 失效。
    // 总线驱动时 WaveformSurface 会用总线快照覆盖 scrollLeftPx / pxPerSec。
    const axis = React.useMemo(
        () =>
            createTimelineAxis({
                pxPerSec: props.pxPerSec,
                scrollLeftPx: props.scrollLeftPx,
                viewportWidthPx: props.widthPx,
                dpr: window.devicePixelRatio || 1,
            }),
        [props.pxPerSec, props.scrollLeftPx, props.widthPx],
    );

    return (
        <WaveformSurface
            rows={rows}
            heightPx={props.heightPx}
            axis={axis}
            color={props.colors.stroke}
            style={{ opacity: 0.86 }}
            viewportSource={pianoRollViewportBus}
            amplitudeMap={props.amplitudeMap}
        />
    );
});
