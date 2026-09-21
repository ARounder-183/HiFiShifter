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
import type { WaveformAmplitudeFactors, WaveformAmplitudeMap } from "../../../waveform/geometry";
import { linearAmplitudeMap } from "../../../waveform/geometry";
import { createTimelineAxis } from "../renderKernel/timelineAxis.js";
import type { ClipPeaksEntry } from "./useClipsPeaksForPianoRoll";
import { computeDynGain } from "./paramRanges";
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
 * 静音保护下限与增益上限：**不再在本文件内联**。
 *
 * 增益语义（含门限与上限）的唯一真源是 `paramRanges::computeDynGain`，
 * 它与后端 `common_params::compute_dyn_gain` 逐分支同构。本文件曾各自
 * 内联一份常量与公式，导致前后端口径可能分叉（用户看到"波形能提升、
 * 实际播放不提升"）。热路径只需调用该函数（零分配）。
 */

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
 *
 * 【时域因子视图】本映射是**纯乘性**的（与峰值 value 无关），因此除逐值调用外
 * 还挂载 `factorAt(t)`（见 `WaveformAmplitudeFactors`）：几何层按列内切片调用
 * 时只需每切求值**一次**因子、min/max 复用，把逐切片重复的曲线取样与 live
 * 覆盖解析收敛成一次。两条路径共用同一个 `factorAt` —— 逐值调用就是它乘上
 * `value × gain`，因此不可能分叉。
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
        return sampleCurveLinear(snapshotValues, source.startFrame, source.stride, frameF);
    };

    /**
     * 时刻 → 乘性因子（volume × dynGain）。两条调用路径的唯一实现：
     * 逐值映射（`value × gain × factorAt(t)`）与几何层的按切片因子视图。
     *
     * 退化情形一律返回 1（不施加响度自动化）：拿不到时间 / 时间非有限 ——
     * 与旧实现的早退分支逐值等价。返回 `null` = 数据异常导致因子非有限，
     * 调用方按原语义产出非有限值（几何层据此跳过该切片）。
     */
    const factorAt = (timeSec: number | null): number | null => {
        if (timeSec === null || !Number.isFinite(timeSec)) return 1;
        const frameF = (timeSec * 1000) / source.framePeriodMs;
        if (!Number.isFinite(frameF)) return 1;

        // ① 音量包络：曲线外回退 1.0（与后端越界持有末值语义一致）。
        const vol = sampleLiveOrSnapshot(live.volume(), source.volume, frameF) ?? 1.0;

        // ② 动态增益：交给与后端逐分支同构的 `computeDynGain`（唯一实现）。
        //    基线为空 = 分析未就绪 → 不施加任何增益（绝不凭空造增益）。
        let dynGain = 1;
        if (source.dynBaseline.length > 0) {
            const target = sampleLiveOrSnapshot(live.dyn(), source.dynTarget, frameF);
            const base = sampleCurveLinear(
                source.dynBaseline,
                source.startFrame,
                source.stride,
                frameF,
            );
            if (target !== null && base !== null) {
                dynGain = computeDynGain(target, base);
            }
        }

        const factor = vol * dynGain;
        return Number.isFinite(factor) ? factor : null;
    };

    /**
     * 时间窗内可达电平的**硬上界**：`max over t∈窗 ( vol(t) × 目标电平(t) )`。
     *
     * 【为什么它是硬上界】逐样本地，`|x(s)| ≤ 原声基线(t_s)`（基线是该点原声
     * 电平），故真实输出
     *   `输出(s) = |x(s)| × vol × 目标/基线 ≤ vol(t_s) × 目标(t_s)`。
     * 因此窗口内任何时刻的真实输出都不超过本值 —— 显示超过它的部分一定是
     * 幻峰（来自"桶窗比分母窗宽"导致的 `桶峰 × 大增益` 越界）。
     *
     * 【为什么未编辑区不会被压】那里 `目标 = 原声`，本值 = `vol × max(原声)`；
     * 而显示用的桶峰 ≤ 窗内原声基线最大 ≤ 该值 ⇒ 钳制从不生效（已实测验证）。
     *
     * 【降级】无动态基线（分析未就绪）时动态不生效（增益恒 1），上界退化为
     * `max(vol)`；无音量数据时 vol 恒 1。任一曲线样本非有限即返回 null
     * （不钳制，保持既有行为）。
     */
    const levelCeilingOverWindow = (
        windowStartSec: number,
        windowEndSec: number,
    ): number | null => {
        if (
            !Number.isFinite(windowStartSec) ||
            !Number.isFinite(windowEndSec) ||
            !(source.framePeriodMs > 0)
        ) {
            return null;
        }
        const lo = Math.min(windowStartSec, windowEndSec);
        const hi = Math.max(windowStartSec, windowEndSec);
        const f0 = (lo * 1000) / source.framePeriodMs;
        const f1 = (hi * 1000) / source.framePeriodMs;
        if (!Number.isFinite(f0) || !Number.isFinite(f1)) return null;
        // 动态未启用（无基线）：因子只有音量（≤2）与 bucketPeak（≤1），
        // 显示值有界，不存在幻峰 → 不钳制（保持既有的"溢出削顶"观感）。
        if (source.dynBaseline.length === 0) return null;
        // 逐帧枚举窗口（帧周期 5ms，窗口最宽为 L2 桶 ≈ 85ms → 至多数十帧）。
        const first = Math.max(0, Math.floor(f0));
        const last = Math.ceil(f1);
        let ceiling = 0;
        for (let f = first; f <= last; f += 1) {
            const vol = sampleLiveOrSnapshot(live.volume(), source.volume, f) ?? 1.0;
            // 目标电平（哨兵已在后端出口解析成原声基线）；`≤0` = 画静音。
            const target = sampleLiveOrSnapshot(live.dyn(), source.dynTarget, f);
            if (target === null) continue;
            const level = Math.max(target, 0) * Math.max(vol, 0);
            if (!Number.isFinite(level)) return null;
            if (level > ceiling) ceiling = level;
        }
        return ceiling;
    };

    const map = ((value: number, gain: number, timeSec: number | null) => {
        const v = value * gain;
        if (!Number.isFinite(v)) return 0;
        const factor = factorAt(timeSec);
        // factor 为 null = 曲线数据异常（非有限）：与原实现「vol × dynGain 直接
        // 相乘」同语义地产出非有限值，由几何层跳过该切片。
        return factor === null ? Number.NaN : v * factor;
    }) as WaveformAmplitudeMap & { revision: () => number } & WaveformAmplitudeFactors;
    // 修订号读者：几何缓存据此判定"引用未变但内部数据变了"需要重建几何。
    // 用函数而非数字：拖动时每帧变化的计数不能经过 React 状态，否则整块面板
    // 会以指针频率重渲染。
    map.revision = revision;
    // 时域因子视图：几何层据此把每切片的 min/max 两次逐值调用换成一次求值。
    map.factorAt = factorAt;
    // 窗口电平上界：几何层据此把"桶峰 × 因子"钳在物理可达范围内，
    // 消除近零原声处的幻峰（见 levelCeilingOverWindow 与接口文档）。
    map.levelCeilingOverWindow = levelCeilingOverWindow;
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
        channelMode: entry.channelMode,
        sourceChannels: entry.sourceChannels,
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
