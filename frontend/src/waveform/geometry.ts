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
 * 每次的 `timeSec` 是该切片中心的**时间轴绝对时间**，`value` 是该切片时间窗内
 * 的原始峰值 min/max；对同一切片的 min / max 两次调用拿到**同一个** `timeSec`
 * （切片内增益恒定，包络上下沿同缩放，形状不失真）。峰值由此与**自身时刻**的
 * 增益配对 —— 若整列共用列中心的一次增益，峰值高度会随水平缩放乱跳（幻峰 /
 * 丢峰），缩放等级之间画出截然不同的波形。
 *
 * 【对称性】钩子对每一切片的 min / max 各调用一次，两者拿到同一个 `timeSec`。
 *
 * 【性能】本契约要求**逐值**调用（一列最多 32 次），而绝大多数映射其实只是
 * 一个"该切片时刻的乘性因子"（`映射 = value × gain × F(t)`）。这类映射应额外
 * 声明 {@link WaveformAmplitudeFactors}：几何层改为每切片求值**一次**因子、
 * 复用到 min/max —— 既省掉一半调用，也消除逐值调用里任何可缓存的解析工作
 * （参数面板拖动时曾在一次重建内做数万次字符串切分，见该接口的说明）。
 *
 * @param value 源文件峰值（线性，±1 满量程）。
 * @param gain clip 增益 × 淡变（线性）。
 * @param timeSec 该像素列对应的**时间轴绝对时间**（秒）。未知时为 null。
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
 * 【切片语义】把列的峰值索引窗口等分为至多 N 切，每切用**自己的中心索引时刻**
 * 采样增益并映射，列包络 = 各切映射结果的 max/min。索引锚定 ⇒ 每个峰值桶的
 * 可听高度是**缩放不变量**；粗列 = 细列的逐段 max ⇒ 缩放一致（细化收敛）。
 * N=16 时切片粒度在常见工程（3 分钟 / 全览）约等于自动化帧周期（≈5ms），
 * 视觉上已与逐帧精确一致；同时把每列的映射调用次数封顶，极端缩放下的
 * 每帧成本有界（≤ 列数 × N × 2 次调用）。
 */
const MAX_COLUMN_GAIN_SLICES = 16;

// 列内增益切片的复用 scratch（模块级一次分配；buildWaveformGeometry 同步执行，
// 无重入风险）——热路径上每列每帧新建数组会把 GC 拖进渲染关键路径。
const sliceIndexLoScratch = new Int32Array(MAX_COLUMN_GAIN_SLICES);
const sliceIndexHiScratch = new Int32Array(MAX_COLUMN_GAIN_SLICES);
const sliceTimeSecScratch = new Float64Array(MAX_COLUMN_GAIN_SLICES);
const sliceGainScratch = new Float64Array(MAX_COLUMN_GAIN_SLICES);

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
    const sink = args.sink ?? fallbackSink;
    const state: VertexSinkState = { buffer: sink.buffer, length: 0 };
    const push = createVertexSink(state);
    const amplitudeMap = args.amplitudeMap ?? linearAmplitudeMap;
    // 时域因子视图（可选契约，见 WaveformAmplitudeFactors）：声明了它的映射
    // 只需每切片求值**一次**因子，min/max 复用 —— 切片把逐值调用提到每列 32 次，
    // 因子路径把同一时刻的解析/取样工作收敛成一次（拖动参数时的卡顿根因）。
    const amplitudeFactors = readAmplitudeFactors(args.amplitudeMap);
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
        const bands: {
            min: Float32Array;
            max: Float32Array;
            centerY: number;
            halfHeight: number;
        }[] = dual
            ? [
                  {
                      min: peaks.min,
                      max: peaks.max,
                      centerY: segment.screenRect.y + segment.screenRect.height / 4,
                      halfHeight: halfHeight / 2,
                  },
                  {
                      min: peaks.ch1Min as Float32Array,
                      max: peaks.ch1Max as Float32Array,
                      centerY: segment.screenRect.y + (segment.screenRect.height * 3) / 4,
                      halfHeight: halfHeight / 2,
                  },
              ]
            : [{ min: peaks.min, max: peaks.max, centerY, halfHeight }];

        for (const band of bands) {
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

                // ── 列内增益切片 ─────────────────────────────────
                // 【源时间 → 时间轴时刻】切片时刻按**峰值索引**反推（源锚定）：
                // 索引 i 的源时间 = dataStartSec + ((i+0.5)/sampleCount)·数据时长，
                // 再经 reversed / playbackRate 对应的段内归一化位置换算成时间轴
                // 绝对时间。锚定到峰值而不是屏幕，保证"同一峰值 × 它自己时刻的
                // 增益"在任何缩放等级下是同一个数 —— 这是缩放一致性的根基。
                // （倒放语义与旧实现一致：t 始终是屏幕推进方向的位置。）
                const clipStartSec = segment.clipStartSec ?? 0;
                const localSpanSec = segment.clipLocalEndSec - segment.clipLocalStartSec;
                const sliceCount = Math.min(windowCount, MAX_COLUMN_GAIN_SLICES);
                for (let slice = 0; slice < sliceCount; slice += 1) {
                    // 公平切分：边界 = floor(s·W/K)，各切恰好覆盖窗口、互不重叠，
                    // 最后一切精确落在 indexEnd 上。
                    const lo = indexStart + Math.floor((slice * windowCount) / sliceCount);
                    const hi =
                        indexStart + Math.floor(((slice + 1) * windowCount) / sliceCount) - 1;
                    sliceIndexLoScratch[slice] = lo;
                    sliceIndexHiScratch[slice] = hi;
                    const centerIndex = (lo + hi) / 2;
                    const sliceSrcSec =
                        peaks.dataStartSec + ((centerIndex + 0.5) / sampleCount) * dataDurationSec;
                    const sliceT = clamp01(
                        segment.reversed
                            ? (segment.sourceEndSec - sliceSrcSec) / sourceDurationSec
                            : (sliceSrcSec - segment.sourceStartSec) / sourceDurationSec,
                    );
                    const clipTimeSec = segment.clipLocalStartSec + sliceT * localSpanSec;
                    sliceTimeSecScratch[slice] = clipStartSec + clipTimeSec;
                    sliceGainScratch[slice] =
                        segment.gain *
                        gainAtClipTime(
                            clipTimeSec,
                            segment.clipTotalDurationSec,
                            segment.fadeInSec,
                            segment.fadeOutSec,
                            segment.fadeInShape,
                            segment.fadeInDir,
                            segment.fadeOutShape,
                            segment.fadeOutDir,
                        );
                }

                // 音量增益（gain > 1）会把包络放大到波形矩形之外 —— 表现为波形
                // "溢出" clip 上下边界（DAW 通用 bug）。与 REAPER 一致，增益放大
                // 的显示按矩形削顶（flat-top），既保留"已削波"的视觉暗示又不越界。
                //
                // 幅度经 `amplitudeMap` 归一化：缺省是线性直投；动态/音量面板传
                // 逐帧映射。**逐切片**映射（每切一次求值、min/max 复用同一时刻）
                // —— 峰值与自身时刻的增益配对，见 MAX_COLUMN_GAIN_SLICES 的说明。
                //
                // 两条等价路径（见 WaveformAmplitudeFactors）：
                // - 映射声明了 `factorAt`：每切片求值一次因子，min/max 各一次乘法；
                // - 未声明：逐值调用映射（min/max 各一次，共享该切时刻）。
                const rectTop = segment.screenRect.y;
                const rectBottom = rectTop + segment.screenRect.height;
                let mappedMin = Number.POSITIVE_INFINITY;
                let mappedMax = Number.NEGATIVE_INFINITY;
                for (let slice = 0; slice < sliceCount; slice += 1) {
                    let rawMin = Number.POSITIVE_INFINITY;
                    let rawMax = Number.NEGATIVE_INFINITY;
                    for (
                        let index = sliceIndexLoScratch[slice];
                        index <= sliceIndexHiScratch[slice];
                        index += 1
                    ) {
                        // 采样当前带（band）的声道平面 —— 双带布局下两带分别是
                        // ch0/ch1 视图；此前误读 peaks.min/max（恒为 ch0），两条
                        // 带画出同一个左声道。
                        rawMin = Math.min(rawMin, band.min[index] ?? 0);
                        rawMax = Math.max(rawMax, band.max[index] ?? 0);
                    }
                    if (!Number.isFinite(rawMin) || !Number.isFinite(rawMax)) continue;
                    const gain = sliceGainScratch[slice];
                    const timeSec = sliceTimeSecScratch[slice];
                    let mappedSliceMax: number;
                    let mappedSliceMin: number;
                    const factor = amplitudeFactors?.factorAt(timeSec) ?? null;
                    if (factor !== null) {
                        // 乘性因子路径：一次求值、两次乘法（映射 =
                        // value × gain × factor）。
                        mappedSliceMax = rawMax * gain * factor;
                        mappedSliceMin = rawMin * gain * factor;
                    } else {
                        mappedSliceMax = amplitudeMap(rawMax, gain, timeSec);
                        mappedSliceMin = amplitudeMap(rawMin, gain, timeSec);
                    }
                    if (!Number.isFinite(mappedSliceMax) || !Number.isFinite(mappedSliceMin)) {
                        continue;
                    }
                    if (mappedSliceMax > mappedMax) mappedMax = mappedSliceMax;
                    if (mappedSliceMin < mappedMin) mappedMin = mappedSliceMin;
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
