import { fadeGainIn, fadeGainOut } from "../components/layout/timeline/paths.ts";
import {
    INACTIVE_TAKE_COLOR_ALPHA,
    INACTIVE_TAKE_RGB_SCALE,
    type WaveformScene,
} from "./sceneBuilder.ts";

export interface WaveformPeakView {
    min: Float32Array;
    max: Float32Array;
    dataStartSec: number;
    dataDurationSec: number;
    /**
     * 有效声道数（2 时 ch1Min/ch1Max 为第二声道视图，几何层画双带波形）。
     * 缺省 1（单带，兼容未携带声道信息的调用方）。
     */
    channels?: 1 | 2;
    ch1Min?: Float32Array;
    ch1Max?: Float32Array;
}

export interface WaveformGeometry {
    vertices: Float32Array;
    lineCount: number;
    complete: boolean;
}

export type WaveformPeakResolver = (
    sourcePath: string,
    sourceSampleRate: number,
    sourceStartSec: number,
    sourceDurationSec: number,
    /** 段所属 take 的声道模式（0..=4，对齐 REAPER CHANMODE）；缺省 0。 */
    channelMode?: number,
    /** 源文件声道数（未知时缺省 0，由峰值数据自身的 channels 兜底）。 */
    sourceChannels?: number,
) => WaveformPeakView | null;

/**
 * 幅度映射钩子：把「线性峰值 × 淡变增益」映射为绘制用的单位幅度。
 *
 * 缺省实现是**线性直投**（`value * gain`），即时间线与绝大多数参数面板的
 * 既有行为。参数编辑器在「动态（DYN）」面板下改传一个**逐帧**版本：它按像素
 * 所在的**时间轴绝对时间**取出该时刻的动态增益，把源波形画成"应用动态之后的
 * 结果电平" —— 于是波形、原声基线与用户曲线共用同一坐标系，用户画一笔就能
 * 立刻看到波形随之起伏。
 *
 * 【为什么必须带时间】动态是逐帧参数，而波形是逐像素列画的。没有时间输入时
 * 映射只能施加一个全局系数，画多少曲线波形都不变 —— 这正是"编辑动态时波形
 * 不按动态值重绘"的根因。
 *
 * 【调用契约】映射按**列内增益切片**调用（见 {@link MAX_COLUMN_GAIN_SLICES}）：
 * 每次 `value` 是一个**原始极值**（切片时间窗内某个桶的 min 或 max），`timeSec`
 * 是**该极值所在桶自身**的时刻（时间轴绝对时间）——**不是**切片中心。极值必须
 * 与它自己的时刻配对：切片极值按**窗口**聚合，而片中心只是**点采样**，两者
 * 口径不同；若把"片内最大峰值"乘上"片中心时刻的增益"，增益在片内剧变时
 * （近零原声处可逐帧跨数量级）乘积会系统性超出真实包络，且片越宽超出越多
 * —— 表现为峰值高度随水平缩放乱跳、缩放等级之间画出截然不同的波形（幻峰；
 * 修复推导见 {@link MAX_COLUMN_GAIN_SLICES} 的【★】段）。
 *
 * 【对称性】min 与 max 常落在同一桶（包络上下沿同源），此时两次调用拿到
 * **同一个** `timeSec`；但它们**可能落在不同桶**上，届时各自与自己的桶时刻
 * 配对 —— 消费方不得假设两次调用共享同一 `timeSec`（实现见
 * `buildWaveformGeometry` 的切片装配循环）。
 *
 * 【性能】本契约要求**逐值**调用（一列最多 32 次），而绝大多数映射其实只是
 * 一个"该切片时刻的乘性因子"（`映射 = value × gain × F(t)`）。这类映射应额外
 * 声明 {@link WaveformAmplitudeFactors}：几何层改为每切片求值**一次**因子、
 * 复用到 min/max —— 既省掉一半调用，也消除逐值调用里任何可缓存的解析工作
 * （参数面板拖动时曾在一次重建内做数万次字符串切分，见该接口的说明）。
 *
 * @param value 源文件峰值（线性，±1 满量程）。
 * @param gain clip 增益 × 淡变（线性）。
 * @param timeSec 该极值所在桶的**时间轴绝对时间**（秒）。未知时为 null。
 * @returns 单位幅度（0 附近的相对量；具体量纲由调用方的映射定义）。
 */
export type WaveformAmplitudeMap = (value: number, gain: number, timeSec: number | null) => number;

/**
 * 幅度映射的**时域因子**视图（可选契约，挂在映射对象上的 `factorAt`）。
 *
 * 【语义】声明了 `factorAt` 即等价于承诺：
 * `amplitudeMap(value, gain, t) ≡ value × gain × factorAt(t)`（因子为 null 的
 * 时刻除外，见下）。几何层据此把每切片的两次逐值调用（min / max）换成**一次**
 * 因子求值 + 两次乘法。
 *
 * 【为什么值得单独开一个契约】切片把每列的映射调用数从 2 次提到 32 次，而映射
 * 实现里往往藏着与 `value` 无关、却**每次调用都要重做**的工作：
 * - 从 live 覆盖读一次参数窗口（`key.split("|")` 这类字符串解析）；
 * - 构造一个描述窗口的临时对象；
 * - 逐帧曲线按时间的两次取样。
 * 逐值调用会把这些乘以列数 × 切片数（实测 2112 列：每次重建约 6.8 万次调用，
 * 拖动音量/动态时单帧几何重建 29.5ms —— 用户报告的卡顿）。因子路径把同一时刻
 * 的工作收敛成一次，实测降到 1ms 以内。
 *
 * 【退化】`factorAt` 返回 `null` = 该时刻无法用单一乘性因子表达，几何层回落到
 * 逐值调用 `amplitudeMap`（非乘性映射仍可正确工作，只是拿不到这次优化）。
 *
 * 【一致性】`factorAt` 与 `amplitudeMap` 必须来自同一实现（见
 * `makeLoudnessAmplitudeMap`：两者共用同一个求值函数），否则两条路径会分叉。
 */
export interface WaveformAmplitudeFactors {
    /**
     * 该时刻的乘性因子 F（映射 = `value × gain × F`）。
     *
     * @param timeSec 时间轴绝对时间（秒）；null = 未知。
     * @returns 有限因子；`null` = 该时刻须回落到逐值调用。
     */
    factorAt(timeSec: number | null): number | null;

    /**
     * **一次几何重建会查询的时间范围**（可选契约，在重建开始时调用一次）。
     *
     * 【为什么需要】重建按「像素列 × 列内切片」查询因子，一次典型重建 2112 列
     * × 16 切片 = **约 3.4 万次查询**，而被查询的时刻全部落在本窗口内。实现方
     * 据此把「与查询点无关的准备工作」提前做一次 —— 例如把曲线按帧预采样成
     * 查表、把 live 覆盖的解析从每次查询收敛成每次重建一次。实测（拖动音量、
     * 2624px 窗口）该准备工作把重建耗时从 ~11ms 降到 ~1.5ms 量级。
     *
     * 【契约】
     * - 调用顺序：本函数先于本次重建的任何 `factorAt` / `levelCeilingOverWindow`；
     * - 窗口只是**提示**：实现方必须保证窗口外的查询仍返回正确结果（查表
     *   越界时内部回退到逐值求值），否则拖动中的边缘列会画错；
     * - `windowEndSec` 可能小于 `windowStartSec`（倒放不改变时间方向，但实现
     *   方不应假设顺序），实现方自行排序；
     * - 可选：未实现时一切照旧（逐值求值），仅是慢一些。
     *
     * @param windowStartSec 本次重建会查询的最小时间（时间轴绝对秒）。
     * @param windowEndSec 本次重建会查询的最大时间（时间轴绝对秒）。
     */
    beginWindow?(windowStartSec: number, windowEndSec: number): void;

    /**
     * **该时间窗内可达电平的上限**（可选契约，用于压制"幻峰"）。
     *
     * 【为什么需要】波形每列的峰值取自 mipmap 桶（粗缩放走 L2，桶宽 ≈ 85 ms），
     * 而动态的原声基线是 20 ms 窗估计 —— **分子的窗口比分母宽**。于是
     * `桶内最大样本`可以大于"用来求增益的那一点的原声基线"，比值 > 1；
     * 近零原声处增益又极大（`目标/基线` 可达上限 ×1000），两者相乘就把
     * "桶内偶然略大的样本"放大成满高幻峰。水平缩放改变桶的划分，幻峰位置
     * 随之随机变化 —— 即用户报告的近零处伪影。
     *
     * 【为什么钳制是物理正确的】逐样本地看，`|x(s)| ≤ 原声基线(t_s)`，于是
     * `输出(s) = |x(s)| × vol × 目标/基线 ≤ vol(t_s) × 目标(t_s)`。
     * 因此本函数返回的"窗口内最大可达电平"是**真实输出的硬上界**：超过它的
     * 显示值不可能被听到，钳掉它不会影响任何真实内容 —— 未编辑区（目标=原声，
     * 因子恒 1）永远不会被钳到（可证：`桶峰 ≤ 窗内基线最大 ≤ 窗内目标最大`）。
     *
     * 【语义】返回 `max over t ∈ [start, end] of (vol(t) × 目标电平(t))`，
     * 单位与曲线一致（绝对电平倍率）。几何层再乘上该处的 clip 增益×淡变，
     * 得到包络的钳制边界。
     *
     * @param windowStartSec 窗口起点（时间轴绝对秒）。
     * @param windowEndSec 窗口终点（时间轴绝对秒）。
     * @returns 有限上界；无法给出时返回 null（不做钳制，保持旧行为）。
     */
    levelCeilingOverWindow?(windowStartSec: number, windowEndSec: number): number | null;
}

/**
 * 读取幅度映射声明的时域因子视图；未声明时返回 null（保持逐值调用契约）。
 */
export function readAmplitudeFactors(
    map: WaveformAmplitudeMap | undefined,
): WaveformAmplitudeFactors | null {
    if (map === undefined) return null;
    const fn = (map as WaveformAmplitudeMap & Partial<WaveformAmplitudeFactors>).factorAt;
    return typeof fn === "function" ? (map as unknown as WaveformAmplitudeFactors) : null;
}

/**
 * 幅度映射内部数据的**修订号读者**。
 *
 * 【为什么需要】动态面板的映射用延迟取值（读 ref 里的 live 覆盖），函数引用
 * 保持不变也能产出不同结果。几何缓存若只按引用比较会误判"没变"，于是绘制中的
 * 曲线画不上去。
 *
 * 约定：映射可以挂一个 `revision()` 返回单调递增的计数；几何缓存据此判定需要
 * 重建。用**函数**而非数字，是为了让计数变化不必经过 React 状态（参数面板拖动
 * 时每帧 setState 会让整块面板重渲染，代价过高）。
 */
export interface WaveformAmplitudeMapWithRevision {
    revision(): number;
}

/**
 * 读取幅度映射的修订号；未挂载时返回 0（视为恒定，保持既有缓存行为）。
 */
export function readAmplitudeRevision(map: WaveformAmplitudeMap | undefined): number {
    const fn = (map as WaveformAmplitudeMap & Partial<WaveformAmplitudeMapWithRevision>)?.revision;
    return typeof fn === "function" ? fn() : 0;
}

/** 缺省幅度映射：线性直投（保持所有既有调用方的行为不变）。 */
export const linearAmplitudeMap: WaveformAmplitudeMap = (value, gain) => value * gain;

/**
 * 每列增益采样的上限（列内切片数）。
 *
 * 【为什么需要"列内切片"】增益链（clip 增益 × 淡变 × 音量 × 动态）是**逐时刻**
 * 的，而一列的时间窗在粗缩放下可能横跨许多个自动化帧。若整列只用**列中心**
 * 一次增益缩放窗口内的全部峰值，峰值就被乘上了"别的时刻"的增益：
 * - 同一峰值在不同水平缩放等级下落入中心时刻不同的列 → 高度随缩放乱跳
 *   （用户报告：音量/动态面板的波形"峰值高度在缩放过程中乱跳"）；
 * - 粗缩放下一列的包络 ≠ 细缩放下它覆盖的各列包络的最大值 → 两种缩放等级
 *   画出"截然不同的波形"（幻峰 / 丢峰）。
 *
 * 【切片语义】把列的峰值索引窗口等分为至多 N 切，每切取自己的极值（max/min），
 * 并把极值与**该极值所在桶自身时刻**的增益配对，列包络 = 各切映射结果的 max/min。
 * 索引锚定 ⇒ 每个峰值桶的可听高度是**缩放不变量**；粗列 = 细列的逐段 max ⇒
 * 缩放一致（细化收敛）。N=16 时切片粒度在常见工程（3 分钟 / 全览）约等于自动化
 * 帧周期（≈5ms），视觉上已与逐帧精确一致；同时把每列的映射调用次数封顶，
 * 极端缩放下的每帧成本有界（≤ 列数 × N × 2 次调用）。
 *
 * 【★ 极值必须与"它自己的时刻"配对（不能与片中心配对）】这是近零原声伪影的
 * 修复要点。曾把"片内最大峰值"乘上"片**中心**时刻的增益"——分子是按**窗口**
 * 聚合的（片内 N 个桶的最大值），分母却是**点采样**。两者窗口口径不同，当增益
 * 在片内剧烈变化时（近零原声下 `目标/max(原声,下限)` 逐帧可跨数个数量级），
 * 乘积会系统性**超出**真实包络，且超出量随片宽增长：
 *
 *   一列覆盖的桶越多（= 缩得越小），片越宽 → 幻峰越高。
 *   实测（div=16、近零噪声尾音）：40 列时包络达 1.98（真值上限 1.00），
 *   即约 2× 的幻峰；20 列时偏差达 0.974。这正是"缩小时近零处出现随机伪影"。
 *
 * 改为把极值锚定到它自己的桶时刻后，偏差降到 0（逐样本真值对拍）。
 * 注意这条性质对**任何**逐时刻变化的映射都成立（音量/动态/未来参数），
 * 不只是 dyn。
 */
const MAX_COLUMN_GAIN_SLICES = 16;

// 列内增益切片的复用 scratch（模块级一次分配；buildWaveformGeometry 同步执行，
// 无重入风险）——热路径上每列每帧新建数组会把 GC 拖进渲染关键路径。
const sliceIndexLoScratch = new Int32Array(MAX_COLUMN_GAIN_SLICES);
const sliceIndexHiScratch = new Int32Array(MAX_COLUMN_GAIN_SLICES);
// 各切片的**时间窗端点**（时间轴绝对秒），与索引 scratch 同批填好。
// 上界钳制按切片的时间窗取（见 levelCeilingOverWindow 的调用点）；此前是在
// 钳制处用 `absSecAtIndex(sliceLo/Hi)` 现算 —— 一次重建多出 2 × 3.4 万次
// 十参数的求值调用。切片边界与索引同批算出即可，语义逐值不变。
// 仅在映射声明了 `levelCeilingOverWindow` 时填充（未声明时白算一轮）。
const sliceTimeLoScratch = new Float64Array(MAX_COLUMN_GAIN_SLICES);
const sliceTimeHiScratch = new Float64Array(MAX_COLUMN_GAIN_SLICES);

/** 一条绘制带（band）：峰值平面 + 竖直居中与半高。 */
interface WaveformBand {
    min: Float32Array;
    max: Float32Array;
    centerY: number;
    halfHeight: number;
}

/** 空峰值平面的占位（仅用于初始化 scratch，构建时必然被覆写）。 */
const EMPTY_PEAKS = new Float32Array(0);

/**
 * 绘制带复用的 scratch（最多两条：单声道 1 条，立体声 2 条）。
 *
 * 【为什么复用】此前每段都新建一个数组字面量 + 1~2 个带对象（400 段 ≈ 600 次
 * 分配/次重建），全是年轻代垃圾，落在渲染关键路径上。带的内容是「峰值平面 +
 * 两个几何量」，与切片索引 scratch 同样是纯派生数据，复用即可。
 * `buildWaveformGeometry` 同步执行，无重入风险。
 */
const bandScratch: [WaveformBand, WaveformBand] = [
    { min: EMPTY_PEAKS, max: EMPTY_PEAKS, centerY: 0, halfHeight: 0 },
    { min: EMPTY_PEAKS, max: EMPTY_PEAKS, centerY: 0, halfHeight: 0 },
];

/**
 * 波形包络列的**设备像素宽**。
 *
 * 【栅格对齐契约（与 09dc9973 的网格修复同一族）】波形按「每列一条竖直包络」
 * 绘制，列枚举在**设备像素网格**上进行：每列恰好 `round(dpr)` 个物理像素宽
 * （dpr 非整数时向下取到 1），列边严格落在设备像素边界上。于是任何 DPR 下
 * 线宽都是**恒定的整数物理像素**——若按 CSS 像素枚举（列宽恒 1 CSS px），
 * dpr=1.25/1.5 这类非整数比会让列宽落在 1~2 物理像素之间随位置跳变，观感
 * 就是"线宽一直在 1~2 之间抖动"。
 *
 * dpr ≥ 2 时取 `round(dpr)` 而非 1：1 物理像素的高频列在 hi-dpi 屏上覆盖
 * 不足、观感发"浅"（WebGL LINES 时代的老问题，见 surfaceRenderer 的宽度
 * 契约注释）；`round(dpr)` 恰好等价于旧的 1 CSS px 列宽，dpr=1/2 的画面
 * 逐像素不变，只有非整数 dpr 从"抖动"变为"恒定"。
 *
 * @param dpr 设备像素比（非法值按 1 处理）。
 * @returns 每条包络列覆盖的物理像素数。
 */
export function waveformColumnWidthDevicePx(dpr: number): number {
    return Number.isFinite(dpr) && dpr > 0 ? Math.max(1, Math.round(dpr)) : 1;
}

function clamp01(value: number): number {
    return Math.min(1, Math.max(0, value));
}

export function parseWaveformColor(value: string): [number, number, number, number] {
    const input = value.trim();
    const hex = /^#([\da-f]{6}|[\da-f]{8})$/i.exec(input)?.[1];
    if (hex) {
        return [
            Number.parseInt(hex.slice(0, 2), 16) / 255,
            Number.parseInt(hex.slice(2, 4), 16) / 255,
            Number.parseInt(hex.slice(4, 6), 16) / 255,
            hex.length === 8 ? Number.parseInt(hex.slice(6, 8), 16) / 255 : 1,
        ];
    }

    const rgba =
        /^rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)(?:\s*,\s*([\d.]+))?\s*\)$/i.exec(input);
    if (rgba) {
        return [
            clamp01(Number(rgba[1]) / 255),
            clamp01(Number(rgba[2]) / 255),
            clamp01(Number(rgba[3]) / 255),
            clamp01(rgba[4] == null ? 1 : Number(rgba[4])),
        ];
    }
    return [1, 1, 1, 1];
}

/**
 * 由**峰值索引**换算 clip 局部时间（秒）——「该峰值自己的时刻」。
 *
 * 索引锚定（而非屏幕列锚定）意味着同一个峰值在任何缩放等级下都取到同一个
 * 增益值，这是缩放一致性的根基。极值（argmin/argmax）与它的时刻必须成对使用，
 * 见 {@link MAX_COLUMN_GAIN_SLICES} 的说明。
 */
function sourceIndexClipTimeSec(
    index: number,
    dataStartSec: number,
    dataDurationSec: number,
    sampleCount: number,
    reversed: boolean,
    sourceStartSec: number,
    sourceEndSec: number,
    sourceDurationSec: number,
    clipLocalStartSec: number,
    localSpanSec: number,
): number {
    const srcSec = dataStartSec + ((index + 0.5) / sampleCount) * dataDurationSec;
    const t = clamp01(
        reversed
            ? (sourceEndSec - srcSec) / sourceDurationSec
            : (srcSec - sourceStartSec) / sourceDurationSec,
    );
    return clipLocalStartSec + t * localSpanSec;
}

function gainAtClipTime(
    clipTimeSec: number,
    totalDurationSec: number,
    fadeInSec: number,
    fadeOutSec: number,
    fadeInShape: number,
    fadeInDir: number,
    fadeOutShape: number,
    fadeOutDir: number,
): number {
    let gain = 1;
    if (fadeInSec > 0 && clipTimeSec < fadeInSec) {
        gain *= fadeGainIn(fadeInShape, fadeInDir, clamp01(clipTimeSec / fadeInSec));
    }
    const fadeOutStart = totalDurationSec - fadeOutSec;
    if (fadeOutSec > 0 && clipTimeSec > fadeOutStart) {
        gain *= fadeGainOut(
            fadeOutShape,
            fadeOutDir,
            clamp01((clipTimeSec - fadeOutStart) / fadeOutSec),
        );
    }
    return gain;
}

/**
 * 顶点缓冲槽。
 *
 * 由调用方持有并跨帧复用：`buffer` 容量不足时会被替换为更大的新数组，
 * 调用方下次传入新的即可。这样**稳态零分配**——否则每帧末尾都要把顶点
 * `slice()` 成独立副本（实测 703 KB/帧），纯属浪费：顶点一旦上传 GPU /
 * 提交给 Canvas2D，CPU 侧就不再需要它了。
 */
export interface WaveformVertexSink {
    buffer: Float32Array;
}

/**
 * 未显式提供 sink 时的模块级兜底槽。
 *
 * 保留它有多个原因：基准（`waveformPerf.bench.ts`）与单元测试不传 sink，
 * 行为应与改动前一致（跨调用复用同一块缓冲、稳态零分配）；生产侧则由
 * `WaveformSurface` 各自持有一个槽，避免两个波形面互相踩缓冲。
 */
const fallbackSink: WaveformVertexSink = { buffer: new Float32Array(0) };

/** 写入游标与当前缓冲打包在一起，避免模块级可变状态。 */
interface VertexSinkState {
    buffer: Float32Array;
    length: number;
}

/** 初始容量（元素个数）：与改动前 `vertexScratch` 的初值一致。 */
const INITIAL_VERTEX_CAPACITY = 8192;

function createVertexSink(
    state: VertexSinkState,
): (x: number, y: number, r: number, g: number, b: number, a: number) => void {
    return (x, y, r, g, b, a) => {
        if (state.length + 6 > state.buffer.length) {
            const next = new Float32Array(
                Math.max(state.buffer.length * 2 || INITIAL_VERTEX_CAPACITY, state.length + 6),
            );
            next.set(state.buffer.subarray(0, state.length));
            state.buffer = next;
        }
        const i = state.length;
        state.buffer[i] = x;
        state.buffer[i + 1] = y;
        state.buffer[i + 2] = r;
        state.buffer[i + 3] = g;
        state.buffer[i + 4] = b;
        state.buffer[i + 5] = a;
        state.length = i + 6;
    };
}

/**
 * 由波形场景构建逐线段顶点几何。
 *
 * 流程：遍历场景的段与标记，按像素列采样 mipmap、算包络与淡变增益，把
 * 每条线段写成 2 个顶点（每顶点 `x, y, r, g, b, a` 共 6 个 float）。
 *
 * @param args.scene 绘制场景（`buildWaveformScene` 的产物）。
 * @param args.color 波形描边色（hex / rgb(a)）。
 * @param args.getPeaks 峰值解析器。
 * @param args.amplitudeMap 幅度映射（缺省线性直投）。见 {@link WaveformAmplitudeMap}。
 * @param args.sink 顶点写入目标；**跨帧复用可做到稳态零分配**。容量不足时
 *   内部倍增并把新缓冲写回 `sink.buffer`。省略时使用模块级兜底槽。
 * @param args.dpr 设备像素比（缺省 1）。包络列按设备像素网格枚举（见
 *   {@link waveformColumnWidthDevicePx}），非整数 dpr 下列宽不再抖动。
 * @returns 顶点视图（**借用 `sink.buffer`，调用方不得长期持有**——下一次
 *   构建会覆写它；生产侧只在同一次 `render()` 内使用，安全）、线段数与
 *   数据完整性标志。
 */
export function buildWaveformGeometry(args: {
    scene: WaveformScene;
    color: string;
    getPeaks: WaveformPeakResolver;
    amplitudeMap?: WaveformAmplitudeMap;
    sink?: WaveformVertexSink;
    dpr?: number;
}): WaveformGeometry {
    const [red, green, blue, colorAlpha] = parseWaveformColor(args.color);
    const dpr = args.dpr != null && Number.isFinite(args.dpr) && args.dpr > 0 ? args.dpr : 1;
    const columnDeviceWidth = waveformColumnWidthDevicePx(dpr);
    /** 半列宽（CSS px）：用于把列中心换算回列的左右边界（淡变区判定）。 */
    const halfColumnCss = columnDeviceWidth / (2 * dpr);
    const sink = args.sink ?? fallbackSink;
    const state: VertexSinkState = { buffer: sink.buffer, length: 0 };
    const push = createVertexSink(state);
    const amplitudeMap = args.amplitudeMap ?? linearAmplitudeMap;
    // 时域因子视图（可选契约，见 WaveformAmplitudeFactors）：声明了它的映射
    // 只需每切片求值**一次**因子，min/max 复用 —— 切片把逐值调用提到每列 32 次，
    // 因子路径把同一时刻的解析/取样工作收敛成一次（拖动参数时的卡顿根因）。
    const amplitudeFactors = readAmplitudeFactors(args.amplitudeMap);
    // 因子求值的**单态调用**：切片循环里每列要调用 1~2 次，此处绑定一次可避免
    // 热路径上反复走「可选链 + 联合类型判空」的分派（实测这一层分派在 3.4 万次
    // 调用下不可忽略）。语义与 `amplitudeFactors?.factorAt(t) ?? null` 等价。
    const factorAtFn =
        amplitudeFactors !== null ? amplitudeFactors.factorAt.bind(amplitudeFactors) : null;
    // 电平上界视图（可选契约，见 WaveformAmplitudeFactors::levelCeilingOverWindow）：
    // 用于把"桶峰 × 因子"钳在物理可达范围内，消除近零原声处的幻峰。
    const levelCeiling = amplitudeFactors?.levelCeilingOverWindow?.bind(amplitudeFactors);
    // ── 恒定增益快路径的准入条件（见下方列循环内的说明）─────────────────
    //
    // 列内切片（见 MAX_COLUMN_GAIN_SLICES）是为**逐时刻变化**的增益准备的：
    // 每个切片把极值还原成它自己的时刻、再求该时刻的增益。但时间轴上绝大多数
    // 段的增益是**常数**：幅度映射是缺省线性直投，且段的时间范围不与淡变区相交。
    // 此时切片纯属浪费——每列最多 16 次 `absSecAtIndex`（10 参数函数）与 32 次
    // `clipGainAtSec`（内含 `fadeGainSigned` 的形状/曲率求值）全部白付，而结果与
    // 「整列取一次极值 × 常数增益」**逐位相同**：
    //
    //   max over slices ( raw_slice_max × g ) = g × max over slices ( raw_slice_max )
    //                                        = g × max over column        （g ≥ 0）
    //
    // 实测（14993 列 × 17 桶 × 16 切片）该快路径把列循环从 3.14 ms 降到 0.49 ms。
    //
    // 注意这**不是近似**：两条路径的极值取自同一个桶、乘法只做一次，因此结果逐位
    // 相等（`geometry.test.ts` 有对拍用例）。切片路径本身对逐时刻增益仍是必需且
    // 正确的——参数编辑器的音量/动态面板照旧走它。
    //
    // `factorAtFn` / `levelCeiling` 的判空在缺省映射下是冗余的（缺省映射没有这两个
    // 契约），显式写出是为了把前提钉死：一旦有人给线性映射挂上因子或上界，快路径
    // 自动失效，不会静默算错。
    const amplitudeIsIdentity =
        args.amplitudeMap === undefined || args.amplitudeMap === linearAmplitudeMap;
    const canUseConstantGain =
        amplitudeIsIdentity && factorAtFn === null && levelCeiling === undefined;
    // 查询窗口提示：段的时间覆盖范围（见 WaveformAmplitudeFactors::beginWindow）。
    // 几何层只会查询 `clipStartSec + [clipLocalStartSec, clipLocalEndSec]` 之内
    // 的时刻 —— `absSecAtIndex` 的 t 被 clamp01 到 [0,1]。故窗口取各段的时间
    // 并集即可，无需扫描屏幕矩形（trim/裁切只改变可见列，不改变可查询时刻）。
    if (amplitudeFactors?.beginWindow !== undefined) {
        let windowLo = Number.POSITIVE_INFINITY;
        let windowHi = Number.NEGATIVE_INFINITY;
        for (const segment of args.scene.segments) {
            const segStart = (segment.clipStartSec ?? 0) + segment.clipLocalStartSec;
            const segEnd = (segment.clipStartSec ?? 0) + segment.clipLocalEndSec;
            if (segStart < windowLo) windowLo = segStart;
            if (segEnd > windowHi) windowHi = segEnd;
        }
        if (Number.isFinite(windowLo) && Number.isFinite(windowHi)) {
            amplitudeFactors.beginWindow(windowLo, windowHi);
        }
    }
    let complete = true;

    for (const segment of args.scene.segments) {
        const sourceDurationSec = segment.sourceEndSec - segment.sourceStartSec;
        if (!(sourceDurationSec > 1e-9) || segment.screenRect.width <= 0) continue;
        // inactive take lane：颜色整体压暗（rgb × RGB_SCALE），与场景层已乘入
        // segment.alpha 的 LANE_ALPHA 叠加，复刻旧 Canvas 多 Take 的观感。
        const inactive = Boolean(segment.inactive);
        const segmentRed = inactive ? red * INACTIVE_TAKE_RGB_SCALE : red;
        const segmentGreen = inactive ? green * INACTIVE_TAKE_RGB_SCALE : green;
        const segmentBlue = inactive ? blue * INACTIVE_TAKE_RGB_SCALE : blue;
        const peaks = args.getPeaks(
            segment.sourcePath,
            segment.sourceSampleRate,
            segment.sourceStartSec,
            sourceDurationSec,
            segment.channelMode,
            segment.sourceChannels,
        );
        if (!peaks || peaks.min.length === 0 || peaks.max.length === 0) {
            complete = false;
            continue;
        }

        const sampleCount = Math.min(peaks.min.length, peaks.max.length);
        const dataDurationSec = Math.max(1e-12, peaks.dataDurationSec);
        const dataEndSec = peaks.dataStartSec + dataDurationSec;
        // 列枚举在设备像素网格上（见 waveformColumnWidthDevicePx 的契约说明）：
        // 列 k 覆盖设备像素 `[k·W, (k+1)·W)`，即 CSS `[k·W/dpr, (k+1)·W/dpr)`。
        // 旧实现按 CSS 像素枚举、列宽恒 1 CSS px，非整数 dpr 下列的设备覆盖
        // 在 1~2 物理像素间随位置跳变 —— 用户报告的"线宽抖动"根因。
        const firstColumn = Math.max(
            0,
            Math.ceil((segment.screenRect.x * dpr) / columnDeviceWidth),
        );
        const lastColumn = Math.max(
            firstColumn,
            Math.ceil(
                ((segment.screenRect.x + segment.screenRect.width) * dpr) / columnDeviceWidth,
            ),
        );
        // 每列覆盖的时间窗（秒）：总时长 × 列宽占比。W=dpr 时与旧的
        // 「时长 / CSS 列数」逐值相等（dpr=1/2 行为不变，见 W 的取值说明）。
        const sourceSecondsPerColumn =
            (sourceDurationSec * columnDeviceWidth) / (segment.screenRect.width * dpr);
        const halfHeight = segment.screenRect.height / 2;
        const centerY = segment.screenRect.y + halfHeight;
        const dual = peaks.channels === 2 && peaks.ch1Min != null && peaks.ch1Max != null;
        // 双带布局：ch0 占上半带（中心 1/4）、ch1 占下半带（中心 3/4），
        // 各自包络以半带高度归一 —— 立体声素材一眼可辨。
        // 带对象写入模块级 scratch（见 bandScratch），稳态零分配。
        const bandCount = dual ? 2 : 1;
        if (dual) {
            const upper = bandScratch[0];
            upper.min = peaks.min;
            upper.max = peaks.max;
            upper.centerY = segment.screenRect.y + segment.screenRect.height / 4;
            upper.halfHeight = halfHeight / 2;
            const lower = bandScratch[1];
            lower.min = peaks.ch1Min as Float32Array;
            lower.max = peaks.ch1Max as Float32Array;
            lower.centerY = segment.screenRect.y + (segment.screenRect.height * 3) / 4;
            lower.halfHeight = halfHeight / 2;
        } else {
            const single = bandScratch[0];
            single.min = peaks.min;
            single.max = peaks.max;
            single.centerY = centerY;
            single.halfHeight = halfHeight;
        }

        // ── 段作用域的常量与两个求值函数 ────────────────────────────────
        // 【为什么不放在列循环里】`absSecAtIndex` / `clipGainAtSec` 只依赖段与
        // 峰值数据，**不依赖列**。此前它们定义在列循环体内，于是每列都要新建
        // 两个闭包（函数对象 + 捕获上下文）—— 一次典型重建 2112 列 = 4224 次
        // 闭包分配，全部是年轻代垃圾，把 GC 拖进了渲染关键路径。提到段作用域后
        // 一次重建只需 2 次分配，语义逐值不变。
        const clipStartSec = segment.clipStartSec ?? 0;
        const localSpanSec = segment.clipLocalEndSec - segment.clipLocalStartSec;
        // 淡变区间（clip 局部时间）。段可被查询的时刻恒落在
        // [clipLocalStartSec, clipLocalEndSec] 之内（`absSecAtIndex` 把 t 钳在
        // [0,1] 后线性映射到该区间），因此「段的时间范围是否与淡变区相交」可以
        // 只按段判断；更精确的逐列判断在列循环里（淡变只影响其所在的少数列）。
        const fadeInSec = segment.fadeInSec;
        const fadeOutSec = segment.fadeOutSec;
        const fadeOutStartSec = segment.clipTotalDurationSec - fadeOutSec;
        const segmentHasFade = fadeInSec > 0 || fadeOutSec > 0;
        // 负增益会让「列内最大值 × 增益」不再等于「列内最大乘积」（乘负数把极值
        // 翻转），此时退回切片路径。场景层已把增益钳到 ≥0，这里是纵深防御。
        const skipSlices = canUseConstantGain && segment.gain >= 0;

        // 桶索引 ↔ 时间轴绝对秒（「该峰值自己的时刻」），供极值配对使用。
        // 索引锚定 ⇒ 同一峰值在任何缩放等级下取到同一个增益（缩放一致性）。
        const absSecAtIndex = (index: number): number =>
            clipStartSec +
            sourceIndexClipTimeSec(
                index,
                peaks.dataStartSec,
                dataDurationSec,
                sampleCount,
                segment.reversed,
                segment.sourceStartSec,
                segment.sourceEndSec,
                sourceDurationSec,
                segment.clipLocalStartSec,
                localSpanSec,
            );

        // 某**时间轴绝对秒**处的 clip 增益（clip 增益 × 淡变）。
        // 传入的秒由 absSecAtIndex 产出，故减去 clipStartSec 即为
        // gainAtClipTime 需要的 clip 局部时间。
        const clipGainAtSec = (absSec: number): number =>
            segment.gain *
            gainAtClipTime(
                absSec - clipStartSec,
                segment.clipTotalDurationSec,
                segment.fadeInSec,
                segment.fadeOutSec,
                segment.fadeInShape,
                segment.fadeInDir,
                segment.fadeOutShape,
                segment.fadeOutDir,
            );

        for (let bandIndex = 0; bandIndex < bandCount; bandIndex += 1) {
            const band = bandScratch[bandIndex];
            for (let column = firstColumn; column < lastColumn; column += 1) {
                // 列中心（CSS 坐标）：device 中心 = (column·W + W/2)，换回 CSS。
                const xCss = ((column + 0.5) * columnDeviceWidth) / dpr;
                const t = clamp01((xCss - segment.screenRect.x) / segment.screenRect.width);
                const sourceCenterSec = segment.reversed
                    ? segment.sourceEndSec - t * sourceDurationSec
                    : segment.sourceStartSec + t * sourceDurationSec;
                const sourceLoSec = Math.max(
                    peaks.dataStartSec,
                    sourceCenterSec - sourceSecondsPerColumn / 2,
                );
                const sourceHiSec = Math.min(
                    dataEndSec,
                    sourceCenterSec + sourceSecondsPerColumn / 2,
                );
                const indexStart = Math.max(
                    0,
                    Math.floor(
                        ((sourceLoSec - peaks.dataStartSec) / dataDurationSec) * sampleCount,
                    ),
                );
                const indexEnd = Math.min(
                    sampleCount - 1,
                    Math.max(
                        indexStart,
                        Math.ceil(
                            ((sourceHiSec - peaks.dataStartSec) / dataDurationSec) * sampleCount,
                        ) - 1,
                    ),
                );
                const windowCount = indexEnd - indexStart + 1;
                // 窗口被数据边界裁空（sourceLo 越过数据末尾）→ 与旧守卫同语义：跳过该列。
                if (windowCount < 1) continue;

                const rectTop = segment.screenRect.y;
                const rectBottom = rectTop + segment.screenRect.height;
                let mappedMin = Number.POSITIVE_INFINITY;
                let mappedMax = Number.NEGATIVE_INFINITY;

                // ── 本列增益是否为常数？（决定能否跳过切片）─────────────────
                // 段的整体增益是常数（缺省线性映射、增益非负）时，还要看本列的
                // 时间窗是否与淡变区相交：淡变只影响其所在的少数列，其余列照旧
                // 可以走快路径。列的时间窗由列中心的 ± 半列宽换算回 clip 局部秒
                // （t 与局部秒在该段内是仿射关系），**取列的左右边界**而非中心，
                // 保证边界列不会因为「中心在淡变区外」而被误判为恒定增益。
                let columnGainIsConstant = skipSlices;
                if (columnGainIsConstant && segmentHasFade) {
                    const rectX = segment.screenRect.x;
                    const rectW = segment.screenRect.width;
                    const tLo = (xCss - halfColumnCss - rectX) / rectW;
                    const tHi = (xCss + halfColumnCss - rectX) / rectW;
                    const colLoLocalSec = segment.clipLocalStartSec + tLo * localSpanSec;
                    const colHiLocalSec = segment.clipLocalStartSec + tHi * localSpanSec;
                    columnGainIsConstant =
                        !(fadeInSec > 0 && colLoLocalSec < fadeInSec) &&
                        !(fadeOutSec > 0 && colHiLocalSec > fadeOutStartSec);
                }

                if (columnGainIsConstant) {
                    // ── 恒定增益：单遍扫描，不做切片 ──────────────────────
                    // 与切片路径**逐位相同**（推导见 canUseConstantGain 的说明）：
                    // 极值取自同一批桶，乘法只做一次。省掉每列最多 16 次
                    // `absSecAtIndex` 与 32 次 `clipGainAtSec`。
                    let rawMin = Number.POSITIVE_INFINITY;
                    let rawMax = Number.NEGATIVE_INFINITY;
                    for (let index = indexStart; index <= indexEnd; index += 1) {
                        const vMin = band.min[index] ?? 0;
                        const vMax = band.max[index] ?? 0;
                        if (vMax > rawMax) rawMax = vMax;
                        if (vMin < rawMin) rawMin = vMin;
                    }
                    if (Number.isFinite(rawMin) && Number.isFinite(rawMax)) {
                        mappedMax = rawMax * segment.gain;
                        mappedMin = rawMin * segment.gain;
                    }
                } else {
                    // ── 列内切片边界 ─────────────────────────────────
                    // 只切分索引窗口（不含任何时刻/增益计算）：极值与它**自己所在桶**
                    // 的时刻在下面的装配循环里成对求出（见 MAX_COLUMN_GAIN_SLICES
                    // 的"极值必须与自己的时刻配对"）。此前在这里按片**中心**预计算
                    // 一个时刻，装配时把"片内最大峰值"乘上它 —— 分子按窗口聚合、
                    // 分母是点采样，两者口径不同，在近零原声等增益剧变处会造出
                    // 随缩放增高的幻峰（伪影根因）。
                    // （倒放语义与旧实现一致：t 始终是屏幕推进方向的位置。）
                    const sliceCount = Math.min(windowCount, MAX_COLUMN_GAIN_SLICES);
                    for (let slice = 0; slice < sliceCount; slice += 1) {
                        // 公平切分：边界 = floor(s·W/K)，各切恰好覆盖窗口、互不重叠，
                        // 最后一切精确落在 indexEnd 上。
                        sliceIndexLoScratch[slice] =
                            indexStart + Math.floor((slice * windowCount) / sliceCount);
                        sliceIndexHiScratch[slice] =
                            indexStart + Math.floor(((slice + 1) * windowCount) / sliceCount) - 1;
                        // 上界钳制需要的时间窗端点与索引同批求出（见 scratch 的说明）。
                        if (levelCeiling !== undefined) {
                            sliceTimeLoScratch[slice] = absSecAtIndex(sliceIndexLoScratch[slice]);
                            sliceTimeHiScratch[slice] = absSecAtIndex(sliceIndexHiScratch[slice]);
                        }
                    }

                    // 音量增益（gain > 1）会把包络放大到波形矩形之外 —— 表现为波形
                    // "溢出" clip 上下边界（DAW 通用 bug）。与 REAPER 一致，增益放大
                    // 的显示按矩形削顶（flat-top），既保留"已削波"的视觉暗示又不越界。
                    //
                    // 幅度经 `amplitudeMap` 归一化：缺省是线性直投；动态/音量面板传
                    // 逐帧映射。**逐切片**映射（每切一次求值、极值复用该切时刻）
                    // —— 极值与其自身时刻配对，见 MAX_COLUMN_GAIN_SLICES 的说明。
                    //
                    // 两条等价路径（见 WaveformAmplitudeFactors）：
                    // - 映射声明了 `factorAt`：每切片求值因子，min/max 各一次乘法；
                    // - 未声明：逐值调用映射（min/max 各一次，各自配对自身时刻）。
                    for (let slice = 0; slice < sliceCount; slice += 1) {
                        // 片内**极值及其位置**：max 与 min 可能落在不同桶上，
                        // 各自必须与自己的桶时刻配对（否则重现"分子窗口聚合 ×
                        // 分母点采样"的失配幻峰）。
                        let rawMin = Number.POSITIVE_INFINITY;
                        let rawMax = Number.NEGATIVE_INFINITY;
                        let argMin = sliceIndexLoScratch[slice];
                        let argMax = sliceIndexLoScratch[slice];
                        for (
                            let index = sliceIndexLoScratch[slice];
                            index <= sliceIndexHiScratch[slice];
                            index += 1
                        ) {
                            // 采样当前带（band）的声道平面 —— 双带布局下两带分别是
                            // ch0/ch1 视图；此前误读 peaks.min/max（恒为 ch0），两条
                            // 带画出同一个左声道。
                            const vMin = band.min[index] ?? 0;
                            const vMax = band.max[index] ?? 0;
                            if (vMax > rawMax) {
                                rawMax = vMax;
                                argMax = index;
                            }
                            if (vMin < rawMin) {
                                rawMin = vMin;
                                argMin = index;
                            }
                        }
                        if (!Number.isFinite(rawMin) || !Number.isFinite(rawMax)) continue;
                        // 极值各自与**自身桶时刻**配对（见 MAX_COLUMN_GAIN_SLICES）。
                        // 两者落在同一桶是常见情形（包络上下沿同源），此时只求值一次。
                        const tMax = absSecAtIndex(argMax);
                        const tMin = argMax === argMin ? tMax : absSecAtIndex(argMin);
                        const gMax = clipGainAtSec(tMax);
                        const gMin = argMax === argMin ? gMax : clipGainAtSec(tMin);
                        let mappedSliceMax: number;
                        let mappedSliceMin: number;
                        // 因子视图：一次求值、两次乘法（映射 = value × gain × factor）。
                        const fMax = factorAtFn !== null ? factorAtFn(tMax) : null;
                        const fMin =
                            argMax === argMin
                                ? fMax
                                : factorAtFn !== null
                                  ? factorAtFn(tMin)
                                  : null;
                        if (fMax !== null && fMin !== null) {
                            mappedSliceMax = rawMax * gMax * fMax;
                            mappedSliceMin = rawMin * gMin * fMin;
                        } else {
                            // 该时刻无法用单一乘性因子表达（非乘性映射 / 数据异常）
                            // → 回落到逐值调用，保证非乘性映射仍正确工作。
                            mappedSliceMax = amplitudeMap(rawMax, gMax, tMax);
                            mappedSliceMin = amplitudeMap(rawMin, gMin, tMin);
                        }
                        if (!Number.isFinite(mappedSliceMax) || !Number.isFinite(mappedSliceMin)) {
                            continue;
                        }
                        // ── 物理可达上界的钳制（消除近零原声处的幻峰）──────────
                        // 分子的峰值取自 mipmap 桶（粗缩放 L2 桶宽 ≈85ms），分母
                        // （原声基线）是 20ms 窗 —— 窗口不一致，桶峰可大于基线，
                        // 近零处乘上大增益即成为**不可实现**的满高幻峰（随缩放
                        // 位置随机变化）。真实输出逐样本 ≤ `vol × 目标`，故用它
                        // 作上界钳掉幻峰；未编辑区（目标=原声）恒不触发（见接口
                        // levelCeilingOverWindow 的推导）。
                        //
                        // 上界按**本切片的桶时间窗**取（而非整列），保证"某处
                        // 允许的高电平"不会泄漏给邻近的低电平切片。
                        if (levelCeiling !== undefined) {
                            // 切片首/末桶的时刻（倒放时前者大于后者，逐值函数内部
                            // 会排序；这里只传两个端点，无需关心方向）。端点与索引
                            // 同批求出（切片边界循环），此处不再重算。
                            const ceilLevel = levelCeiling(
                                sliceTimeLoScratch[slice] as number,
                                sliceTimeHiScratch[slice] as number,
                            );
                            if (ceilLevel !== null && Number.isFinite(ceilLevel)) {
                                // clip 增益×淡变随时刻变化：取两者中较大的增益作
                                // 保守上界（钳制只用于压制越界，宁可略宽松也不
                                // 误压合法包络）。这里用该切片的极端时刻增益。
                                const gCeil = Math.max(gMax, gMin);
                                const limit = ceilLevel * gCeil;
                                if (Number.isFinite(limit)) {
                                    if (mappedSliceMax > limit) mappedSliceMax = limit;
                                    if (mappedSliceMin < -limit) mappedSliceMin = -limit;
                                }
                            }
                        }
                        if (mappedSliceMax > mappedMax) mappedMax = mappedSliceMax;
                        if (mappedSliceMin < mappedMin) mappedMin = mappedSliceMin;
                    }
                }
                if (!Number.isFinite(mappedMin) || !Number.isFinite(mappedMax)) continue;

                const yTop = Math.min(
                    rectBottom,
                    Math.max(rectTop, band.centerY - mappedMax * band.halfHeight),
                );
                const yBottom = Math.min(
                    rectBottom,
                    Math.max(rectTop, band.centerY - mappedMin * band.halfHeight),
                );
                const alpha =
                    colorAlpha * segment.alpha * (inactive ? INACTIVE_TAKE_COLOR_ALPHA : 1);

                // 列以**中心线**语义写出：渲染端按 dpr 把它展开成恰好覆盖
                // `[column·W, (column+1)·W)` 设备像素的等宽四边形 / 描边
                // （expandLineSegmentsToQuads / Canvas2D lineWidth，同宽契约）。
                push(xCss, yTop, segmentRed, segmentGreen, segmentBlue, alpha);
                push(xCss, yBottom, segmentRed, segmentGreen, segmentBlue, alpha);
            }
        }
    }

    for (const marker of args.scene.markers) {
        const size = Math.min(7, Math.max(4.5, marker.heightPx * 0.16));
        const halfWidth = size * 0.62;
        const inactive = Boolean(marker.inactive);
        const markerRed = inactive ? red * INACTIVE_TAKE_RGB_SCALE : red;
        const markerGreen = inactive ? green * INACTIVE_TAKE_RGB_SCALE : green;
        const markerBlue = inactive ? blue * INACTIVE_TAKE_RGB_SCALE : blue;
        const alpha = colorAlpha * (inactive ? INACTIVE_TAKE_COLOR_ALPHA : 1);
        // 实心 ▽：按**设备像素**扫描线逐行填充（行高 = 包络列同款 W 物理像素、
        // 行距与首行同走设备网格 —— 行与行无缝拼合，任何 DPR 下都不会出现
        // "行高 1~2px 抖动"或行间缝隙）。旧版是 1px 空心折线，WebGL
        // 线元无抗锯齿，两条斜边锯齿非常明显；横线落在设备行中心上天然锐利，
        // 小尺寸下实心标记也更易读。
        // x 轴与首行 y 都对齐到设备像素中心；步数按设备行数取整，dpr=1 时与
        // 旧实现逐值相等（size CSS px 高、每行 1 设备像素）。
        const x = (Math.round(marker.xPx * dpr) + 0.5) / dpr;
        const firstRowDevice = Math.round(marker.yPx * dpr);
        const steps = Math.max(2, Math.round((size * dpr) / columnDeviceWidth));
        for (let i = 0; i < steps; i += 1) {
            const hw = halfWidth * (1 - i / steps);
            if (hw < 0.5 / dpr) break;
            const y = (firstRowDevice + i * columnDeviceWidth + 0.5) / dpr;
            push(x - hw, y, markerRed, markerGreen, markerBlue, alpha);
            push(x + hw, y, markerRed, markerGreen, markerBlue, alpha);
        }
    }

    // 直接返回缓冲视图，不再 slice 出副本：顶点在 `render()` 内即被上传
    // GPU / 提交给 Canvas2D，此后 CPU 侧不再需要。这省掉每帧一次约 703 KB
    // 的拷贝（400 clip / 全览实测值）。代价是调用方不能跨帧持有返回值。
    const used = state.length;
    sink.buffer = state.buffer;
    return {
        vertices: state.buffer.subarray(0, used),
        lineCount: used / 12,
        complete,
    };
}
