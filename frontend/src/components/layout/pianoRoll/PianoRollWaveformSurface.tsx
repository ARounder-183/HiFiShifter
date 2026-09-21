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
import { dynContentFade, dynLevelTargetingGain } from "./paramRanges";
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
 * 查询窗口查表的**帧数上限**。
 *
 * 【量纲】查表以**整数帧**为网格（帧周期 = 曲线自身采样周期），故缓冲大小由
 * 窗口覆盖的帧数决定。超过上限时整块退回逐值求值 —— 行为完全不变，只是慢，
 * 代价换来"不为了极端缩放的整工程窗口一次性分配几十 MB 常驻缓冲"。
 *
 * 1 << 17 = 131072 帧 ≈ 10.9 分钟（帧周期 5ms）：覆盖常见工程在最粗缩放下的
 * 全览窗口。更长的工程在整工程全览时会走逐值回退（此时列数已被 16 切片封顶，
 * 逐值成本仍有界）。
 */
const MAX_LUT_FRAMES = 1 << 17;

/**
 * 无内容判据（`dynContentFade`）所用原声的**平滑窗口**（毫秒）。
 *
 * 取 100ms —— 略宽于最粗的 mipmap 桶（L2 ≈ 85ms），使"有没有内容"的判据在整桶
 * 内一致，从而与水平缩放无关。详见 `contentBaselineAt` 的说明。
 */
const CONTENT_WINDOW_MS = 100;

/**
 * 查表条目的种类（`kind` 数组）。
 *
 * 用于精确复刻 `levelCeilingOverWindow` 逐帧枚举时的三分支语义：
 * 目标电平取不到 = 该帧**跳过**（不参与上界，但也**不**使整体失效）；
 * 乘积非有限 = 整体返回 `null`（不钳制）。
 */
const LUT_LEVEL_SKIP = 0;
const LUT_LEVEL_VALUE = 1;
const LUT_LEVEL_NONFINITE = 2;

/**
 * 查表时「该帧的取值来自哪一路数据」。
 *
 * 【为什么必须逐帧记录来源】`sampleLiveOrSnapshot` 的语义是**逐查询点**选路
 * （live 覆盖得到就用 live，否则用整工程快照）。查表以整数帧为网格，插值时
 * 两端点若来自不同路（live 窗口边界恰好落在这一格），就无法用单路数据线性
 * 插值表达 —— 此时按 `LUT_SRC_*` 判定为"不可插值"，交回逐值路径，从而保证
 * 查表路径与逐值路径**逐值等价**。
 */
const LUT_SRC_NONE = 0;
const LUT_SRC_SNAPSHOT = 1;
const LUT_SRC_LIVE = 2;

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
 * 某帧**物理可达的输出电平上界**：`vol × 目标 × 无内容可达系数(原声)`。
 *
 * 【为什么它不等于 `vol × 目标`】增益律是
 * `增益 = 目标 / max(原声, 下限) × 无内容淡出(原声)`，而逐样本 `|x| ≤ 原声`，
 * 于是该帧真正可达的输出电平是
 *
 *     |x| × 增益 ≤ 原声 × vol × 目标 / max(原声, 下限) × 淡出(原声)
 *                = vol × 目标 × 无内容可达系数(原声)
 *
 * 其中系数恰是**未画帧的增益** `computeDynGain(原声, 原声)`：下限之上为 1
 * （于是上界退化回 `vol × 目标`，与既有实现逐值一致），下限之下远小于 1
 * —— 无内容处再怎么把目标画高也到不了那么响。
 *
 * 【为什么必须带上它（幻峰"满高平台"的根因）】上界钳制的用意是把"桶峰 × 大增益"
 * 这种口径失配造成的越界压回去。但**桶峰来自宽桶（粗缩放的 L2 ≈ 85ms），而分母
 * 原声是窄窗（20ms）估计**，两者口径不同 ⇒ `桶峰/原声` 可以远大于 1；若上界仍写
 * `vol × 目标`，原声接近 0 处的显示值就会被"顶到 `vol × 目标`"——用户看到的不是
 * 尖刺而是**满高平台**，且命中哪些列随水平缩放（桶的划分）变化，正是"缩到一定
 * 程度后近零处出现随机伪影"。
 *
 * @param target 该帧目标电平（哨兵已在后端出口解析）。
 * @param vol 该帧音量。
 * @param base 该帧原声基线；NaN = 取不到（`computeDynGain` 对非有限原声返回 1，
 *   与 `factorAt` 的兜底一致）。
 */
function reachableLevel(target: number, vol: number, base: number, fade: number): number {
    // 未画帧的增益（电平对齐部分）× 与显示相同的淡出系数 —— 两者相乘即"该帧
    // 在不放大无内容的前提下所能达到的最高电平"。
    const contentFactor = dynLevelTargetingGain(base, base) * fade;
    return Math.max(target, 0) * Math.max(vol, 0) * contentFactor;
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

    const framePeriodMs = source.framePeriodMs;
    // 秒 → 帧的换算系数。热路径上秒→帧要转换 3.4 万次以上，用乘法替掉除法
    //（`f = sec * 1000 / fp` → `f = sec * secToFrame`），省下的除以数万计。
    const secToFrame = 1000 / framePeriodMs;
    // 「动态是否可用」在建表前定型。`source` 是每次取数新建的不可变快照
    //（字段声明为 `readonly number[]`），故可在闭包创建时捕获 —— 逐查询点读
    // `source.dynBaseline.length` 会在 3.4 万次调用里反复走属性链。
    const hasBaseline = source.dynBaseline.length > 0;

    /**
     * 无内容判据用的**平滑原声**（滑窗均值，O(1) 查询）。
     *
     * 【为什么必须平滑】波形按 mipmap 桶绘制：粗缩放命中的 L2 桶宽 ≈ 85ms，而原声
     * 基线是 5ms 帧栅格、20ms 窗的逐帧估计 —— 在"接近 0"的段落（词间停顿、抖动
     * 噪声底）它**逐帧抖动若干 dB**。淡出项随原声陡变，若用桶内某一帧的点采样，
     * "桶峰"与"淡出"就来自不同时刻，同一段素材在不同缩放下（桶的划分不同）会画出
     * 截然不同的高度 —— 用户看到的"随机的伪影"。
     *
     * 取 100ms（略宽于最粗的 L2 桶）的**滑窗均值**作判据：判据在整桶内一致 ⇒ 显示
     * 与缩放无关；对抖动噪声底它给出均值（低于桶峰）⇒ 更贴近"无内容"的实情；
     * 对真实轻声（−34…−55 dBFS，下限之上）仍在下限之上 ⇒ 淡出系数仍为 1。
     *
     * 用前缀和实现：一次 O(n) 构建，之后每次查询常数时间（查表路径按帧构建 LUT，
     * 共 n 次查询，摊销后与逐帧求值同阶）。
     */
    const contentWindowKnots = (() => {
        const stride = source.stride > 0 ? source.stride : 1;
        const perKnotMs = stride * framePeriodMs;
        return Math.max(0, Math.round(CONTENT_WINDOW_MS / 2 / Math.max(1e-6, perKnotMs)));
    })();
    let contentPrefix: Float64Array | null = null;
    const contentBaselineAt = (frameF: number): number => {
        const values = source.dynBaseline;
        if (values.length === 0) return 0;
        if (contentPrefix === null) {
            const out = new Float64Array(values.length + 1);
            for (let i = 0; i < values.length; i += 1) {
                const v = values[i];
                out[i + 1] = (out[i] as number) + (Number.isFinite(v) && v > 0 ? v : 0);
            }
            contentPrefix = out;
        }
        const prefix = contentPrefix;
        const stride = source.stride > 0 ? source.stride : 1;
        const center = Math.round((frameF - source.startFrame) / stride);
        const lo = Math.max(0, center - contentWindowKnots);
        const hi = Math.min(values.length - 1, center + contentWindowKnots);
        if (hi < lo) return 0;
        const sum = (prefix[hi + 1] as number) - (prefix[lo] as number);
        const count = hi - lo + 1;
        return count > 0 ? sum / count : 0;
    };
    /** 该帧的**无内容淡出系数**（判据用平滑原声，见上）。 */
    const contentFadeAt = (frameF: number): number => dynContentFade(contentBaselineAt(frameF));

    /**
     * 时刻 → 乘性因子（volume × dynGain）的**逐值实现**。
     *
     * 这是查表之外的原始路径，也是查表越界 / 不可插值时的回退。两条路径都在，
     * 因此"查表未命中"永远只是慢一些，不会改变结果。
     *
     * 退化情形一律返回 1（不施加响度自动化）：拿不到时间 / 时间非有限 ——
     * 与旧实现的早退分支逐值等价。返回 `null` = 数据异常导致因子非有限，
     * 调用方按原语义产出非有限值（几何层据此跳过该切片）。
     *
     * @param frameF 帧位置（可为小数）。
     */
    const factorAtDirect = (frameF: number): number | null => {
        // ① 音量包络：曲线外回退 1.0（与后端越界持有末值语义一致）。
        const vol = sampleLiveOrSnapshot(live.volume(), source.volume, frameF) ?? 1.0;

        // ② 动态增益 = 电平对齐（点采样原声）× 无内容淡出（**平滑**原声，见
        //    `contentBaselineAt`）。基线为空 = 分析未就绪 → 不施加增益。
        let dynGain = 1;
        if (hasBaseline) {
            const target = sampleLiveOrSnapshot(live.dyn(), source.dynTarget, frameF);
            const base = sampleCurveLinear(
                source.dynBaseline,
                source.startFrame,
                source.stride,
                frameF,
            );
            if (target !== null && base !== null) {
                dynGain = dynLevelTargetingGain(target, base) * contentFadeAt(frameF);
            }
        }

        const factor = vol * dynGain;
        return Number.isFinite(factor) ? factor : null;
    };

    // ── 查询窗口查表（beginWindow）────────────────────────────────────
    //
    // 【为什么要查表】一次几何重建按「像素列 × 列内切片」查询因子，典型窗口
    // （1 行 / 2624px / 150 px/s / L0 峰值密度）约 **3.4 万次**查询，而查询点
    // 全部落在同一个可见窗口内。逐值路径每次查询都要：调用 provider 解析 live
    // 覆盖（`live.volume()` / `live.dyn()` + 参数归属判定）、逐通道
    // `sampleCurveLinear`（含除法与越界判空）、再算 `computeDynGain`；上界钳制
    // 还要对窗口内每个整数帧重复同样的两路取样。
    //
    // 实测（同窗口，拖动音量）：单次重建 ~10.9ms，其中因子求值 ~4.1ms、上界
    // 钳制 ~4.1ms、无因子的几何本身仅 0.57ms。查表把「与查询点无关的准备工作」
    // 提到**每次重建一次**：逐整数帧预采样 vol / 目标电平 / 原声基线，于是
    // - `factorAt(t)` = 2 次线性插值 + `computeDynGain`（无 provider 调用、
    //   无除法、无归属判定）；
    // - `levelCeilingOverWindow` = 在预乘好的 level 表上取区间最大值。
    //
    // 【为什么逐值等价】`sampleCurveLinear` 定义的曲线在相邻帧之间是线性的，
    // 且任何一路数据的帧步长必 ≥ 1 帧（`stride` 是整数帧数），故"整数帧上的
    // 精确值 + 帧内线性插值"与"在查询点直接插值"逐值相等；用 `Float64Array`
    // 存储不引入 f32 误差。上界钳制更是**按定义**就在整数帧上取乘积的最大值
    // （见 `levelCeilingOverWindow` 的 `f += 1` 枚举），查表与逐帧枚举逐值相等。
    // 唯一无法用单路数据表达的是"一格的端点跨了 live / 快照两路"（见
    // `LUT_SRC_*`），此时该次查询回退逐值路径。
    let lutStartFrame = 0;
    let lutCount = 0;
    let lutReady = false;
    let lutVol = new Float64Array(0);
    let lutTarget = new Float64Array(0);
    let lutBase = new Float64Array(0);
    let lutLevel = new Float64Array(0);
    let lutLevelKind = new Uint8Array(0);
    /** 每帧每通道的来源（2 位一个通道：vol | target | base）。 */
    let lutTag = new Uint8Array(0);
    /** 每帧的**无内容淡出系数**（判据用平滑原声；与 `factorAt` 的合成一致）。 */
    let lutContentFade = new Float64Array(0);
    /**
     * 每**格**（相邻两帧之间）是否可安全线性插值：三个通道的来源都一致才为 1。
     *
     * 把「三次 2 位字段比较」提前成一次数组读 + 一次判零 —— 查询侧是热路径，
     * 建表侧只做一遍。`(a ^ b) & 0b111111 === 0` 等价于三个 2 位字段逐对相等。
     */
    let lutCellOk = new Uint8Array(0);
    /** 建表缓存键：快照引用 + 修订号 + 已覆盖的帧区间。 */
    let lutKeySnapshot: LoudnessAutomationSource | null = null;
    let lutKeyRevision = -1;
    let lutCoverStart = 0;
    let lutCoverEnd = -1;

    /**
     * 某通道在某**帧**上的取值来源。
     *
     * 与 `sampleLiveOrSnapshot` 的分支条件同构（live 覆盖得到就用 live）：只做
     * 区间判定、不重复取样，故建表时每帧只取样一次。
     */
    const sourceAtFrame = (liveCurve: LoudnessLiveCurve | null, frameF: number): number => {
        if (liveCurve !== null && liveCurve.values.length > 0 && liveCurve.stride > 0) {
            const indexF = (frameF - liveCurve.startFrame) / liveCurve.stride;
            if (indexF >= 0 && indexF < liveCurve.values.length) return LUT_SRC_LIVE;
        }
        return LUT_SRC_SNAPSHOT;
    };

    /**
     * 建立查询窗口的查表（O(窗口帧数)）。
     *
     * @param fLo 起始帧（含）。
     * @param fHi 结束帧（含）。
     */
    const buildLut = (fLo: number, fHi: number): void => {
        const count = fHi - fLo + 1;
        if (!(count > 0) || count > MAX_LUT_FRAMES) {
            // 窗口过大：整块退回逐值求值（行为不变，只是慢）。
            lutReady = false;
            return;
        }
        if (lutVol.length < count) {
            lutVol = new Float64Array(count);
            lutTarget = new Float64Array(count);
            lutBase = new Float64Array(count);
            lutLevel = new Float64Array(count);
            lutLevelKind = new Uint8Array(count);
            lutTag = new Uint8Array(count);
            lutCellOk = new Uint8Array(count > 0 ? count - 1 : 0);
            lutContentFade = new Float64Array(count);
        }
        // live 覆盖只解析一次：此前它在**每个查询点**都被重新解析（一次重建数万次）。
        const liveVolume = live.volume();
        const liveDyn = live.dyn();
        const { startFrame, stride, volume, dynTarget, dynBaseline } = source;
        for (let f = fLo; f <= fHi; f += 1) {
            const i = f - fLo;
            const vol = sampleLiveOrSnapshot(liveVolume, volume, f);
            const volSrc = vol === null ? LUT_SRC_NONE : sourceAtFrame(liveVolume, f);
            lutVol[i] = vol ?? 1.0;

            let target: number | null = null;
            let targetSrc = LUT_SRC_NONE;
            let base: number | null = null;
            let baseSrc = LUT_SRC_NONE;
            if (hasBaseline) {
                target = sampleLiveOrSnapshot(liveDyn, dynTarget, f);
                targetSrc = target === null ? LUT_SRC_NONE : sourceAtFrame(liveDyn, f);
                base = sampleCurveLinear(dynBaseline, startFrame, stride, f);
                baseSrc = base === null ? LUT_SRC_NONE : LUT_SRC_SNAPSHOT;
            }
            lutTarget[i] = target ?? 0;
            lutBase[i] = base ?? 0;
            lutTag[i] = volSrc | (targetSrc << 2) | (baseSrc << 4);
            lutContentFade[i] = contentFadeAt(f);

            // level 表：复刻 levelCeilingOverWindow 逐帧枚举的三分支语义
            //（目标取不到 = 该帧跳过；乘积非有限 = 整体失效）。
            if (target === null) {
                lutLevelKind[i] = LUT_LEVEL_SKIP;
                lutLevel[i] = 0;
            } else {
                const level = reachableLevel(
                    target,
                    vol ?? 1.0,
                    base ?? Number.NaN,
                    contentFadeAt(f),
                );
                if (Number.isFinite(level)) {
                    lutLevelKind[i] = LUT_LEVEL_VALUE;
                    lutLevel[i] = level;
                } else {
                    lutLevelKind[i] = LUT_LEVEL_NONFINITE;
                    lutLevel[i] = 0;
                }
            }
        }
        lutStartFrame = fLo;
        lutCount = count;
        lutCoverStart = fLo;
        lutCoverEnd = fHi;
        for (let i = 0; i + 1 < count; i += 1) {
            lutCellOk[i] = ((lutTag[i] as number) ^ (lutTag[i + 1] as number)) & 0x3f ? 0 : 1;
        }
        lutReady = true;
    };

    /**
     * 时刻 → 乘性因子（volume × dynGain）。两条调用路径的唯一实现：
     * 逐值映射（`value × gain × factorAt(t)`）与几何层的按切片因子视图。
     *
     * 热路径（查表命中）是直筒式计算：1 次索引换算 + 3 次同源线性插值 +
     * `computeDynGain`，没有 provider 调用、没有字符串 / 参数归属判定、没有
     * 分支分派。查表未覆盖（窗口边缘、`MAX_LUT_FRAMES` 退化）或该格两端点跨了
     * live / 整工程快照两路时回退 `factorAtDirect` —— 两条路径逐值等价，因此
     * 回退只会慢，不会改变结果。
     */
    const factorAt = (timeSec: number | null): number | null => {
        if (timeSec === null) return 1;
        const frameF = timeSec * secToFrame;
        if (!Number.isFinite(frameF)) return 1;
        const lutF = frameF - lutStartFrame;
        const i0 = Math.floor(lutF);
        // `i0 + 1 < lutCount` 同时保证 `lutCellOk[i0]` 在界内（其长度为 count-1）。
        if (i0 >= 0 && i0 + 1 < lutCount && (lutCellOk[i0] as number) === 1) {
            const i1 = i0 + 1;
            const frac = lutF - i0;
            const vA = lutVol[i0] as number;
            const vol = vA + ((lutVol[i1] as number) - vA) * frac;
            let dynGain = 1;
            if (hasBaseline) {
                const tag0 = lutTag[i0] as number;
                // 高位 2 位 = 原声基线来源、中位 = 目标电平来源；两者都非 NONE
                // 才与原语义一样求值（非有限输入由 `dynLevelTargetingGain` 退化为 1）。
                if ((tag0 >> 2) & 3) {
                    if ((tag0 >> 4) & 3) {
                        const tA = lutTarget[i0] as number;
                        const bA = lutBase[i0] as number;
                        const fA = lutContentFade[i0] as number;
                        dynGain =
                            dynLevelTargetingGain(
                                tA + ((lutTarget[i1] as number) - tA) * frac,
                                bA + ((lutBase[i1] as number) - bA) * frac,
                            ) *
                            (fA + ((lutContentFade[i1] as number) - fA) * frac);
                    }
                }
            }
            const factor = vol * dynGain;
            return Number.isFinite(factor) ? factor : null;
        }
        return factorAtDirect(frameF);
    };

    /**
     * 声明本次几何重建会查询的时间范围（`WaveformAmplitudeFactors::beginWindow`），
     * 据此建立/复用查询窗口查表。
     */
    const beginWindow = (windowStartSec: number, windowEndSec: number): void => {
        if (!Number.isFinite(windowStartSec) || !Number.isFinite(windowEndSec)) return;
        const lo = Math.min(windowStartSec, windowEndSec);
        const hi = Math.max(windowStartSec, windowEndSec);
        // 前后各留 2 帧余量：查询点是**切片桶边**的时刻，可能落在窗口端点外
        // 侧一帧以内；留余量让边缘列也走查表（否则每列都要回退一次逐值）。
        const fLo = Math.max(0, Math.floor(lo * secToFrame) - 2);
        const fHi = Math.ceil(hi * secToFrame) + 2;
        const rev = revision();
        if (
            lutReady &&
            lutKeySnapshot === source &&
            lutKeyRevision === rev &&
            lutCoverStart <= fLo &&
            lutCoverEnd >= fHi
        ) {
            return;
        }
        lutKeySnapshot = source;
        lutKeyRevision = rev;
        buildLut(fLo, fHi);
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
     *
     * 【实现】窗口内的整数帧是**有限且连续**的，故直接查 level 表取区间最大值
     * （逐帧枚举的语义原样保留在表里，见 buildLut）；查表未覆盖时回退逐帧枚举。
     */
    const levelCeilingOverWindow = (
        windowStartSec: number,
        windowEndSec: number,
    ): number | null => {
        // `framePeriodMs > 0` 无需在此复查：本工厂入口已对非正帧周期早退。
        if (!Number.isFinite(windowStartSec) || !Number.isFinite(windowEndSec)) {
            return null;
        }
        const lo = Math.min(windowStartSec, windowEndSec);
        const hi = Math.max(windowStartSec, windowEndSec);
        const f0 = lo * secToFrame;
        const f1 = hi * secToFrame;
        if (!Number.isFinite(f0) || !Number.isFinite(f1)) return null;
        // 动态未启用（无基线）：因子只有音量（≤2）与 bucketPeak（≤1），
        // 显示值有界，不存在幻峰 → 不钳制（保持既有的"溢出削顶"观感）。
        if (!hasBaseline) return null;
        // 逐帧枚举窗口（帧周期 5ms，窗口最宽为 L2 桶 ≈ 85ms → 至多数十帧）。
        const first = Math.max(0, Math.floor(f0));
        const last = Math.ceil(f1);
        if (lutReady) {
            const iLo = first - lutStartFrame;
            const iHi = last - lutStartFrame;
            if (iLo >= 0 && iHi < lutCount) {
                let ceiling = 0;
                for (let i = iLo; i <= iHi; i += 1) {
                    const kind = lutLevelKind[i];
                    if (kind === LUT_LEVEL_SKIP) continue;
                    if (kind === LUT_LEVEL_NONFINITE) return null;
                    const level = lutLevel[i] as number;
                    if (level > ceiling) ceiling = level;
                }
                return ceiling;
            }
        }
        let ceiling = 0;
        for (let f = first; f <= last; f += 1) {
            const vol = sampleLiveOrSnapshot(live.volume(), source.volume, f) ?? 1.0;
            // 目标电平（哨兵已在后端出口解析成原声基线）；`≤0` = 画静音。
            const target = sampleLiveOrSnapshot(live.dyn(), source.dynTarget, f);
            if (target === null) continue;
            const base = sampleCurveLinear(source.dynBaseline, source.startFrame, source.stride, f);
            const level = reachableLevel(target, vol, base ?? Number.NaN, contentFadeAt(f));
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
    // 查询窗口声明：几何层在每次重建开始时调用一次，据此建立/复用查表
    //（见查表段的说明）。未声明时一切照旧走逐值路径 —— 只是慢一些。
    map.beginWindow = beginWindow;
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
