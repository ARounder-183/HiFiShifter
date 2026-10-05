/**
 * 拖拽期间的**响度时域映射**（把"按下之前"的基线/曲线搬到"现在"的位置）。
 *
 * ## 问题
 *
 * 参数编辑器的波形乘数是 `volume(t) × dyn增益(t)`，而
 * `dyn增益(t) = 目标电平(t) / 原声电平基线(t)`。基线是 **clip 几何的函数**：
 * clip 从 0s 移到 3s，同一段素材的基线就从帧 `[0, 200)` 挪到 `[600, 800)`。
 *
 * 时间轴拖拽期间几何只在 Redux 里乐观更新（不写后端），响度快照却要等提交
 * （`paramsEpoch` 递增）才重取。于是整个手势期间「旧位置的基线 × 新位置的峰值」：
 * 响段高度乱跳，静段被"无内容淡出"压平或被放大成满高平台。这就是用户报告的
 * 「拖拽过程中波形有很大问题，松手才恢复」。
 *
 * ## 本模块的核心：**源坐标**，不是时间轴范围
 *
 * 上一版把映射表达成「旧时间轴范围 → 新时间轴范围」的仿射重采样。它能表达移动与
 * 拉伸，但表达不了 Slip（源窗口平移）与延伸/截短（内容揭示），而且它套用拉伸时用的是
 * 后端 `resample_curve` 的 `(旧长−1)/(新长−1)` 口径，而不是**音频内容的消费**口径
 * —— 二者仅在 Alt 拉伸下巧合一致。
 *
 * 后端的真相在 `pitch_clip::assemble_nonloop_pitch_from_window`：
 *
 * ```text
 * src_f = win_start_frame_f + i · rate      // i = clip 内的时间线帧下标
 * ```
 *
 * 即 **某时间线帧播放哪段素材，只由「消费锚点、起点、速率」决定，与 clip 多长无关**。
 * 记 `srcF(f) = anchorF + (f − startF) · rate`（帧域，可为负速率 = 倒放），要求同一段
 * 素材在新旧几何下的时间线帧 `f` / `g`：
 *
 * ```text
 * srcF_old(g) = srcF_new(f)
 * ⟹ g(f) = startF_old + ( anchorF_new − anchorF_old + (f − startF_new)·rate_new ) / rate_old
 * ```
 *
 * 这一个仿射公式**同时精确表达四种手势**，而且「延伸/截短 ⟹ 恒等」是被**推导出来的**：
 *
 * | 手势 | 几何变化 | 代入结果 |
 * |---|---|---|
 * | 移动 | 仅 `startF` 变 | `g(f) = f − Δ` 刚性平移 |
 * | Slip | 仅 `anchorF` 变 | `g(f) = f + Δ/anchor` 刚性平移 |
 * | 拉伸/缩短 | `rate` 变（源跨度不变） | 仿射缩放，全程有定义 |
 * | 延伸/截短 | `anchorF` 与 `startF` 同步变 | **`g(f) ≡ f`** —— 内容位置根本没动 |
 *
 * 最后一行正是用户报告的「延伸/截短看起来像拉伸/缩短」的根因：旧模型把 `lengthSec`
 * 的变化一律当作时间缩放，而物理真相是"露出/藏起"。
 *
 * ## 用户曲线走另一条口径（与后端提交路径一致）
 *
 * 后端只在 **移动**（`move_clips` 携带参数线）与 **拉伸**（`stretch_linked_params`）
 * 时搬移用户画的曲线，**裁切 / Slip / 增益 / 淡化 / 吸附偏移一律不动**曲线。因此
 * 曲线的映射按手势分类产出：移动 → 同一个平移仿射；拉伸 → `resample_curve` 的
 * `(旧长−1)/(新长−1)`（逐字镜像后端）；其余 → 恒等。
 *
 * ## 近似边界（已知且可接受）
 *
 * - 基线是**能量域融合**（`sqrt(Σ(电平ᵢ×增益ᵢ)²)`）后的曲线，前端只有融合结果、
 *   没有逐 clip 分量，因此映射是对**整条曲线**逐 clip 做的。单 clip 精确；多个 clip
 *   在新区间重叠时无法分解，是近似（错峰处高度略有偏差，松手由权威快照纠正）。
 * - **Loop 回绕**：消费函数带 `floor_mod`，但仿射映射仍然**精确** —— 仿射解出的
 *   `g` 使 `srcOld(g)` 与 `srcNew(f)` 的**未回绕**源位置相等，而 `rem_euclid` 施加在
 *   同一个值上；基线是源位置的周期函数（周期 D），故取模后的等价解给出同一个基线值。
 *   前提有三条，缺一不可：锚点与方向与后端一致（正放 `source_start`、倒放
 *   `min(source_end, D)` 且方向为负，见 `resolveClipConsumption`）；**锚点差按相位取**
 *   （`mod n` 的最小剩余，否则跨过媒体端点时会偏出整整一个周期）；**映射值按周期折回**
 *   旧范围（否则越界帧会被误判为"未知"）。后两条见 `createLoudnessGeometryWarp` 的
 *   段构造与 `mapBaselineFrame`。**媒体时长 D 未知时**才退化为正放仿射近似。
 * - 淡化的变化只影响"该帧是否计入能量"的门限（后端 `fade_weight_at > 0`），不影响
 *   电平位置；忽略。
 */

import { resolveClipContentDurationSec } from "../../../utils/loopRender";

/** 构造映射所需的最小 clip 几何（Redux `ClipInfo` 天然满足）。 */
export interface WarpClipGeometry {
    readonly id: string;
    readonly startSec: number;
    readonly lengthSec: number;
    readonly gain?: number | null;
    readonly sourceStartSec?: number | null;
    readonly sourceEndSec?: number | null;
    readonly playbackRate?: number | null;
    readonly reversed?: boolean;
    readonly loopEnabled?: boolean;
    /**
     * 媒体元数据：Loop 回绕周期 `D` 的唯一来源（与后端
     * `clip_source_media_duration_sec` 同一取值链，见 `resolveClipContentDurationSec`）。
     *
     * 【为什么必须有】Loop 的消费是 `floor_mod(锚点 ∓ 已消费, D)`：**没有 D 就无法
     * 表达回绕**，倒放的锚点更是 `min(source_end, D)`。缺这些字段时只能退化为
     * 正放仿射（见 `resolveClipConsumption` 的 Loop 分支）。
     */
    readonly durationSec?: number | null;
    readonly durationFrames?: number | null;
    readonly sourceSampleRate?: number | null;
    readonly sourcePath?: string | null;
    readonly midiNoteData?: ReadonlyArray<{ endSec: number }> | null;
}

/** 手势开始时的几何快照条目（与 `clipGeometryPreviewBus` 的快照同形）。 */
export interface WarpClipOrigin {
    readonly clipId: string;
    readonly startSec: number;
    readonly lengthSec: number;
    readonly gain: number;
    readonly sourceStartSec: number;
    readonly sourceEndSec: number;
    readonly playbackRate: number;
    readonly reversed: boolean;
    readonly loopEnabled: boolean;
    /** 媒体元数据（Loop 回绕周期 D）—— 见 {@link WarpClipGeometry}。 */
    readonly durationSec?: number | null;
    readonly durationFrames?: number | null;
    readonly sourceSampleRate?: number | null;
    readonly sourcePath?: string | null;
    readonly midiNoteData?: ReadonlyArray<{ endSec: number }> | null;
}

/** 基线帧落在「旧几何下该 clip 不覆盖」的区间：基线未知（不施加动态增益）。 */
export const WARP_BASELINE_UNKNOWN = -3;
/** 曲线帧落在「旧范围但不在任何新范围内」时的返回值：该帧恢复为 pad 值。 */
export const WARP_CURVE_PAD = -2;
/** 该帧不受任何映射影响（恒等）。 */
export const WARP_SEGMENT_IDENTITY = -1;

/**
 * 某个 clip 在某状态下的**内容消费参数**（本模块的原子概念）。
 *
 * 全部由后端同款公式导出（见 `map_clip_curve` / `clip_pitch_trim_window_sec` /
 * `clip_playback_window_sec`），因此"前端预览"与"后端重算"用同一个坐标系。
 */
export interface ClipConsumption {
    /** 时间线起点（帧）：`round(start_sec · 1000/fp)`。 */
    readonly startF: number;
    /** 时间线长度（帧）：`round(length_sec · 1000/fp)`。 */
    readonly lenF: number;
    /** 消费速率（源帧 / 时间线帧）；倒放为负。 */
    readonly rate: number;
    /** 消费锚点（源帧）：`T = startF` 时播放的源位置。 */
    readonly anchorF: number;
    readonly gain: number;
    /**
     * **回绕周期**（时间线帧数）：`> 0` = 该 clip 的基线是时间线帧的**周期函数**，
     * 周期为本值；`0` = 非周期（非 Loop，或 Loop 但媒体时长 `D` 未知）。
     *
     * 【为什么是时间线帧而不是源帧】源域周期是 `n = round(D·fps)`，而时间线每走一帧
     * 消费 `|rate|` 个源帧 ⇒ 时间线周期 = `n / |rate|`（见
     * {@link resolveClipConsumption} 的 Loop 分支）。
     *
     * 【它解决什么】`floor_mod` 的消费只依赖锚点的**相位**：锚点差 `Δ` 与 `Δ ± n`
     * 逐帧给出同一个源位置。同时，映射值偏离旧范围一个周期时，基线值**逐值不变**
     * —— 于是"把映射折回主值区间"与"直接取"等价。没有这一项，锚点一旦跨过媒体
     * 端点，整段基线会被误判为「未知」（见 {@link LoudnessGeometryWarp.baselineFrame}）。
     */
    readonly wrapFrames: number;
    /**
     * 映射是否**精确**（`false` = 该 clip 开了 Loop 但媒体时长 `D` 未知，只能按
     * 正放仿射近似，见文件头"近似边界"）。
     *
     * 【与 `wrapFrames` 的关系】Loop 且 `D` 已知时映射**精确**（`exact: true`），
     * 回绕由 `wrapFrames` 显式表达；两者不再互相蕴含。
     */
    readonly exact: boolean;
}

/**
 * 单个 clip 的时域映射段（构造期算好，查询期零换算）。
 *
 * 基线映射在本段内恒为 `g(f) = offset + slope · f`；`wrapFrames > 0` 时是
 * **周期函数**：映射值先折回 `[oldStartF, oldStartF + wrapFrames)`（见
 * {@link LoudnessGeometryWarp.baselineFrame}）。
 */
interface PreparedSegment {
    /** 该 clip 在新几何下的可见帧范围 `[newStartF, newEndF)`。 */
    readonly newStartF: number;
    readonly newEndF: number;
    /** 旧几何下该 clip 的可见帧范围（映射的有效域；越界 ⇒ 基线未知）。 */
    readonly oldStartF: number;
    readonly oldEndF: number;
    readonly slope: number;
    readonly offset: number;
    /** 该 clip 的增益比 `newGain / oldGain`（1 = 未变）。 */
    readonly gainScale: number;
    /**
     * 该段的回绕周期（时间线帧数）：`> 0` = 基线是帧的周期函数，映射值可折回
     * 主值区间而不改变取值；`0` = 非周期（越界即未知）。
     *
     * 取**旧几何**的周期：`g` 是"旧几何下的帧坐标"，周期自然也是旧几何下的
     * （`n / |rate_old|`，见 `ClipConsumption.wrapFrames`）。
     */
    readonly wrapFrames: number;
    /**
     * 是否允许查表的整数帧线性插值。
     *
     * 等价于"映射把整数帧送到整数帧且每帧走一格"，即 `slope === 1`（刚性平移）。
     * 平移 / 恒等（移动、Slip、延伸截短）满足；Alt 拉伸的仿射缩放不满足 ——
     * 缩放会把快照的分段线性折点搬进格内，线性插值会在折点两侧"抄近路"。
     *
     * **周期段（`wrapFrames > 0`）一律为 false**：折返会让映射在段内出现跳变，
     * 线性插值会跨过跳变造出两条路径都不存在的中间值。
     */
    readonly interpolable: boolean;
    /** 用户曲线的映射策略（已在构造期按 `lockParamLines` 降级，见 `CurvePolicy`）。 */
    readonly curve: CurvePolicy;
}

/**
 * 用户曲线的映射策略。
 *
 * 【为什么不能一律用基线的那条仿射】后端只在移动与拉伸时搬移用户曲线，裁切 / Slip
 * 一律不动。若不管手势一律搬，拖拽预览会凭空造出一个后端不会做的编辑 —— 松手后
 * 曲线"跳回去"，正是用户报告的延伸/截短与 Slip 的异常。
 */
type CurvePolicy =
    | { readonly kind: "none" }
    /** 移动：与基线同一条平移仿射。 */
    | { readonly kind: "move"; readonly slope: number; readonly offset: number }
    /** 拉伸：`resample_curve` 的 `(旧长−1)/(新长−1)`（逐字镜像后端）。 */
    | {
          readonly kind: "resample";
          readonly oldStartF: number;
          readonly oldCount: number;
          readonly newStartF: number;
          readonly newCount: number;
      };

/**
 * 把基线/曲线采样到"现在的位置"的只读视图。
 *
 * 全部方法都是 O(映射数) 的纯查询、零分配 —— 它们在波形几何重建的热路径上被逐帧
 * 调用（一次重建按窗口帧数 × 常数次），因此不接受闭包/对象分配。
 */
export interface LoudnessGeometryWarp {
    /**
     * 该帧应采样的**基线**帧（绝对帧，可为小数）。
     *
     * 返回 {@link WARP_BASELINE_UNKNOWN} = 该帧播放的素材在旧几何下不可见
     * （延伸/截短新露出的部分、Slip 带入的部分）—— 旧快照里没有它的电平，
     * 调用方应按"无基线"处理（动态增益恒 1），不得拿别处的基线顶替。
     */
    baselineFrame(frameF: number): number;
    /** 该帧基线的增益比（1 = 不变）。 */
    baselineScale(frameF: number): number;
    /**
     * 该帧应采样的**音量 / dyn 目标曲线**帧；{@link WARP_CURVE_PAD} 表示该帧已被
     * 恢复为 pad（volume → 1.0；dyn 目标 → 沿用原声）。
     */
    curveFrame(frameF: number): number;
    /** 该帧所属的基线映射段（{@link WARP_SEGMENT_IDENTITY} = 不受影响）。 */
    baselineSegment(frameF: number): number;
    /** 该帧所属的曲线映射段（同上；pad 帧返回 {@link WARP_CURVE_PAD}）。 */
    curveSegment(frameF: number): number;
    /**
     * 该帧的映射是否**允许整数帧线性插值**（false ⇒ 必须逐值求值）。
     *
     * 【为什么需要】查表把每个整数帧的取值预先算好、查询时在相邻两帧之间线性插值。
     * 这只在"映射把整数帧送到整数帧、且每帧恰好走一格"时与"在映射后的帧上直接取样"
     * 逐值等价（此时格内的快照折点恰好落在格端点上）。平移与恒等满足；仿射缩放
     * （Alt 拉伸）与曲线重采样（拉伸）不满足，整段回退逐值。
     */
    interpolableAt(frameF: number): boolean;
}

/** 浮点比较容差：线性增益。 */
const GAIN_EPSILON = 1e-4;
/** 浮点比较容差：消费速率。 */
const RATE_EPSILON = 1e-6;
/** 浮点比较容差：源帧锚点（帧域）。 */
const ANCHOR_EPSILON_FRAMES = 1e-6;

function finiteOr(value: number | null | undefined, fallback: number): number {
    return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

/**
 * 解析单个 clip 的消费参数（**纯函数**，本模块几何口径的唯一落点）。
 *
 * 与后端逐分支同构：
 * - 非 Loop 正放：窗口起点 `source_start_sec`，速率 `+playback_rate`；
 * - 非 Loop 倒放：消费窗口重定向为 `[se − len·rate, se]` 且输出整体翻转
 *   （`clip_pitch_trim_window_sec`），故锚点在 `source_end_sec` 侧、速率为负；
 * - Loop：锚点回绕（`floor_mod`）。媒体时长 `D` 已知时映射**精确**
 *   （`exact: true`，回绕由 `wrapFrames` 表达）；`D` 未知时退化为正放仿射
 *   （`exact: false`、`wrapFrames: 0`）。
 *
 * @param fps 每秒钟的帧数（`1000 / framePeriodMs`）。
 */
export function resolveClipConsumption(
    clip: WarpClipGeometry,
    fps: number,
): ClipConsumption | null {
    if (!(fps > 0)) return null;
    const startSec = Math.max(0, finiteOr(clip.startSec, 0));
    const lengthSec = Math.max(0, finiteOr(clip.lengthSec, 0));
    const rate = finiteOr(clip.playbackRate, 1);
    if (!(rate > 0)) return null;

    const startF = Math.round(startSec * fps);
    const lenF = Math.round(lengthSec * fps);
    const gain = finiteOr(clip.gain, 1);
    const sourceStartSec = finiteOr(clip.sourceStartSec, 0);
    const sourceEndSec = finiteOr(clip.sourceEndSec, sourceStartSec + lengthSec * rate);

    if (clip.loopEnabled === true) {
        // 后端 Loop 消费：**锚点回绕**，逐帧与音频渲染（`mixdown.rs` 的
        // `floor_mod(anchor ± f, D)`）、音高组装（`schedule.rs`）与曲线映射
        // （`pitch_clip::trim_and_resample_midi` 的 loop 分支）三处一致：
        //
        //   正放 idx(i) = floor_mod(round(source_start·fps) + round(i·rate), n)
        //   倒放 idx(i) = floor_mod(round(min(source_end, D)·fps) − 1 − round(i·rate), n)
        //
        // 其中 `i = f − startF`（clip 内时间线帧下标）、`n = round(D·fps)`。
        //
        // 【本轮修的是什么】此前这里一律返回"正放仿射"（锚点 `source_start`、
        // 速率 `+rate`），**完全忽略 `reversed`**。于是**倒放 + Loop** 的 clip 在
        // 拖拽期间基线被搬到错误相位（锚点在 source_start 侧、方向反了），松手后
        // 才被权威快照纠正 —— 正是"拖拽时略微有问题、松开鼠标才恢复正常"。
        //
        // 【为什么仿射仍然成立】`rem_euclid` 对两边施加在**同一个未回绕源位置**上
        // （仿射解出的 g 使 `srcOld(g)` 与 `srcNew(f)` 的未回绕值相等，回绕后自然
        // 相等），而基线是源位置的函数（周期 D）—— 因此 `k≠0` 的那些等价解与
        // `k=0` 给出同一个基线值。回绕不破坏仿射性；错的是锚点与方向。
        //
        // 【本分支只负责"锚点/方向"，回绕由消费方处理】这里返回的 `anchorF` 是**未
        // 取模**的源帧锚点（与后端 `round(source_start·fps)` 同口径）；"锚点差取相位"
        // 与"映射值折回"需要同时看到新旧两侧，故落在 `createLoudnessGeometryWarp` 里
        // （`wrapFrames` 是本分支提供的唯一输入）。
        const mediaTotalSec = resolveClipContentDurationSec(clip) ?? 0;
        if (mediaTotalSec > 0) {
            // 回绕周期：源域 `n = round(D·fps)`，时间线域除以 |rate|（见 wrapFrames）。
            const wrapFrames = Math.round(mediaTotalSec * fps) / Math.abs(rate);
            if (clip.reversed === true) {
                // 倒放锚点 = min(source_end, D)，且首帧消费 `anchor_r − 1`
                //（后端下标从 `anchor_r − 1 − consumed` 起算）。
                const anchorRSec = Math.min(sourceEndSec, mediaTotalSec);
                return {
                    startF,
                    lenF,
                    rate: -rate,
                    anchorF: Math.round(anchorRSec * fps) - 1,
                    gain,
                    wrapFrames,
                    exact: true,
                };
            }
            return {
                startF,
                lenF,
                rate,
                anchorF: Math.round(sourceStartSec * fps),
                gain,
                wrapFrames,
                exact: true,
            };
        }
        // 媒体时长未知：无法表达回绕，退化为正放仿射（与修复前的行为一致）。
        return {
            startF,
            lenF,
            rate,
            anchorF: sourceStartSec * fps,
            gain,
            wrapFrames: 0,
            exact: false,
        };
    }

    if (clip.reversed === true) {
        // 后端：win = [se − len·rate, se]，按升序消费后再整体翻转输出。
        // 翻转后 clip 内第 j 帧对应升序第 (lenF−1−j) 帧 ⇒ srcF(j) = 锚点 − j·rate。
        const winStartSec = sourceEndSec - lengthSec * rate;
        return {
            startF,
            lenF,
            rate: -rate,
            anchorF: winStartSec * fps + (lenF - 1) * rate,
            gain,
            wrapFrames: 0,
            exact: true,
        };
    }

    return {
        startF,
        lenF,
        rate,
        anchorF: sourceStartSec * fps,
        gain,
        wrapFrames: 0,
        exact: true,
    };
}

function readConsumption(
    clip: WarpClipGeometry,
    fps: number,
): ClipConsumption | null {
    const startSec = Math.max(0, finiteOr(clip.startSec, 0));
    const lengthSec = Math.max(0, finiteOr(clip.lengthSec, 0));
    const rate = finiteOr(clip.playbackRate, 1);
    if (!(rate > 0)) return null;
    // 时长/速率都合法才继续；`resolveClipConsumption` 负责其余口径。
    if (!Number.isFinite(startSec + lengthSec)) return null;
    return resolveClipConsumption(clip, fps);
}

/** 曲线重采样的帧映射（逐字镜像后端 `resample_curve` 的下标公式）。 */
function resampleFrame(policy: Extract<CurvePolicy, { kind: "resample" }>, frameF: number): number {
    const newMaxIdx = policy.newCount > 1 ? policy.newCount - 1 : 1;
    const oldMaxIdx = Math.max(1, policy.oldCount) - 1;
    const ratio = oldMaxIdx / newMaxIdx;
    return policy.oldStartF + (frameF - policy.newStartF) * ratio;
}

/**
 * 差分「按下时几何 → 当前几何」，构造时域映射视图。
 *
 * 只对**两侧都存在**的 clip 产出映射：手势期间新增的 clip 没有旧位置可搬
 * （它的基线贡献是"从无到有"，映射表达不了），删除的 clip 同理。
 *
 * @param origin 手势开始时的全量几何快照。
 * @param clips 当前的 clip 几何（应已按参数编辑器所属**根轨道组**过滤 —— 别的
 *   轨道组的 clip 与本组基线无关，把它们的映射算进来会错误地搬移本组基线）。
 * @param framePeriodMs 帧周期（毫秒），用于秒↔帧换算。
 * @param lockParamLines 「锁定参数线」是否开启（决定用户曲线是否跟着走）。
 * @returns 映射视图；没有任何有效映射时返回 `null`（调用方据此完全跳过该路径，
 *   与修复前逐像素一致）。
 */
export function createLoudnessGeometryWarp(args: {
    origin: readonly (WarpClipOrigin | WarpClipGeometry)[];
    clips: readonly WarpClipGeometry[];
    framePeriodMs: number;
    lockParamLines: boolean;
}): LoudnessGeometryWarp | null {
    const fp = args.framePeriodMs;
    if (!(fp > 0)) return null;
    const fps = 1000 / fp;

    const originById = new Map<string, WarpClipGeometry>();
    for (const entry of args.origin) {
        const id = "clipId" in entry ? entry.clipId : entry.id;
        originById.set(id, entry as WarpClipGeometry);
    }
    // 手势期间新增的 clip 没有旧位置可搬（其基线贡献"从无到有"，仿射表达不了）。
    if (originById.size === 0) return null;

    const segments: PreparedSegment[] = [];
    for (const clip of args.clips) {
        const before = originById.get(clip.id);
        if (before === undefined) continue;
        const oldC = readConsumption(before, fps);
        const newC = readConsumption(clip, fps);
        if (oldC === null || newC === null) continue;
        if (newC.lenF <= 0) continue;

        const newStartF = newC.startF;
        const newEndF = newC.startF + newC.lenF;
        const oldStartF = oldC.startF;
        const oldEndF = oldC.startF + oldC.lenF;

        const gainScale =
            Math.abs(oldC.gain) > 1e-6 && Math.abs(newC.gain - oldC.gain) > GAIN_EPSILON * Math.abs(oldC.gain)
                ? newC.gain / oldC.gain
                : 1;

        // 基线仿射：g(f) = startF_old + (anchor'_new − anchor_old + (f − startF_new)·rate_new) / rate_old
        //
        // ── Loop：锚点差必须按**相位**（mod 一个回绕周期）取 ────────────────────
        // Loop 的消费是 `floor_mod(锚点 ± 已消费, D)`，只依赖锚点的**相位**：`Δ` 与
        // `Δ ± n` 逐帧给出同一个源位置。而 `sourceStartSec` 是被 `computeSlipWindow`
        // 取过模的存储值（`slipWindow.ts:150-151`），逐帧连续的 Slip 会在某一帧让它
        // 从 `3.98` 跳到 `0.03` —— 直接相减得到 **−795 帧**，而真实相位差只有
        // **+5 帧**。偏出的那一个周期会把**每一帧**都送出旧范围，整段基线被判为
        // 「未知」（用户报告："Slip 时动态参数干扰波形，且与 Loop 位置有关"）。
        //
        // 取**绝对值最小的等价代表**（`(-n/2, n/2]`）：与"逐帧连续滑动"的直觉一致，
        // 也让映射值尽量落在旧范围内（回绕只需救回边缘的那几帧）。
        const periodic = oldC.wrapFrames > 0 && newC.wrapFrames > 0;
        let anchorDeltaF = newC.anchorF - oldC.anchorF;
        if (periodic) {
            // 源域周期 `n`：由旧几何的 wrapFrames 还原（`wrapFrames = n / |rate_old|`）。
            const n = Math.round(oldC.wrapFrames * Math.abs(oldC.rate));
            if (n > 0) {
                const r = ((anchorDeltaF % n) + n) % n;
                anchorDeltaF = r > n / 2 ? r - n : r;
            }
        }
        const slope = newC.rate / oldC.rate;
        const offset = oldC.startF + anchorDeltaF / oldC.rate - newStartF * slope;

        const rateChanged = Math.abs(newC.rate - oldC.rate) > RATE_EPSILON * Math.max(1, Math.abs(oldC.rate));
        const anchorChanged = Math.abs(newC.anchorF - oldC.anchorF) > ANCHOR_EPSILON_FRAMES;
        const lenChanged = oldC.lenF !== newC.lenF;
        const startChanged = oldC.startF !== newC.startF;

        // 用户曲线策略：镜像后端"什么手势会搬参数线"。
        //
        // 【为什么移动要额外要求"起点真的动了"】增益旋钮、淡化、吸附偏移都**不改**
        // 起点/长度/速率 —— 若把它们也归入"移动"，曲线会被凭空搬走，而后端对这些
        // 编辑根本不调用 `move_clips`。判据必须落在"时间轴范围真的平移了"上。
        let curve: CurvePolicy = { kind: "none" };
        if (args.lockParamLines) {
            if (rateChanged) {
                // 拉伸：后端用 `resample_curve`，范围按**秒取整后**的下标计算
                //（`stretch_linked_params_in_root_range` 对两端都取 round）。
                const oldStartFForCurve = Math.round(
                    Math.max(0, finiteOr(before.startSec, 0)) * fps,
                );
                const oldEndFForCurve = Math.round(
                    (Math.max(0, finiteOr(before.startSec, 0)) +
                        Math.max(0, finiteOr(before.lengthSec, 0))) *
                        fps,
                );
                curve = {
                    kind: "resample",
                    oldStartF: oldStartFForCurve,
                    oldCount: oldEndFForCurve - oldStartFForCurve,
                    newStartF: Math.round(Math.max(0, finiteOr(clip.startSec, 0)) * fps),
                    newCount:
                        Math.round(
                            (Math.max(0, finiteOr(clip.startSec, 0)) +
                                Math.max(0, finiteOr(clip.lengthSec, 0))) *
                                fps,
                        ) - Math.round(Math.max(0, finiteOr(clip.startSec, 0)) * fps),
                };
            } else if (!anchorChanged && !lenChanged && startChanged) {
                // 移动：后端 `move_clips` 携带参数线（纯平移）。
                curve = { kind: "move", slope, offset };
            }
            // 其余（裁切 / Slip / 增益 / 淡化 / 吸附偏移）：后端**不动**用户曲线。
        }

        // 完全无变化的 clip 不产出段：段索引参与查表的"段变了 ⇒ 不可插值"判定，
        // 凭空多出一段会让未受影响区域的列白白回退逐值（稳态也就此变慢）。
        //
        // 【为什么必须同时要求"范围也相同"】延伸/截短的基线映射虽然是**恒等**，
        // 但它的覆盖范围变了 —— 新露出的帧必须由段来宣告"基线未知"。若按映射恒等
        // 就跳过整段，那些帧会被当作"未覆盖"而退回恒等取样，等于拿旧快照在那一处
        // 的残留值（通常是 0）当基线，波形会被"无内容淡出"压平。
        const sameRange = oldC.startF === newC.startF && oldC.lenF === newC.lenF;
        const identityMapping =
            Math.abs(slope - 1) <= 1e-12 && Math.abs(offset) <= 1e-9;
        if (identityMapping && sameRange && gainScale === 1 && curve.kind === "none") continue;

        segments.push({
            newStartF,
            newEndF,
            oldStartF,
            oldEndF,
            slope,
            offset,
            gainScale,
            wrapFrames: periodic ? oldC.wrapFrames : 0,
            // 恒等（slope 1 + offset 0）与平移（slope 1）都可整数帧插值；
            // 缩放会把折点搬进格内，必须逐值。
            //
            // 【周期段为何也排除】回绕使映射在段内出现**折返跳变**（`g` 跨过
            // `oldEndF` 后折回 `oldStartF`），跨过该跳变的线性插值会造出两条路径
            // 都不存在的中间值 —— 与缩放把折点搬进格内是同一类错误。
            interpolable: Math.abs(slope - 1) <= 1e-9 && !periodic,
            curve,
        });
    }
    if (segments.length === 0) return null;

    /**
     * 该帧落在哪个新范围内。
     *
     * 【为什么从后往前找】后端按映射顺序**逐个写入**新范围，后到的覆盖先到的；
     * 因此重叠时"最后一段"才是权威结果。倒序查找即复刻这一语义。
     */
    const segmentAt = (frameF: number): number => {
        for (let i = segments.length - 1; i >= 0; i -= 1) {
            const seg = segments[i] as PreparedSegment;
            if (frameF >= seg.newStartF && frameF < seg.newEndF) return i;
        }
        return WARP_SEGMENT_IDENTITY;
    };

    /** 该帧是否落在某个**会搬曲线**段的旧范围内（= 被搬走、需恢复 pad）。 */
    const coveredByOldTimeRange = (frameF: number): boolean => {
        for (let i = segments.length - 1; i >= 0; i -= 1) {
            const seg = segments[i] as PreparedSegment;
            if (seg.curve.kind === "none") continue;
            if (frameF >= seg.oldStartF && frameF < seg.oldEndF) return true;
        }
        return false;
    };

    /** 曲线段身份（用于查表的"段变了 ⇒ 不可插值"判定）。 */
    const curveSegmentIndex = (seg: PreparedSegment): number =>
        seg.curve.kind === "none" ? WARP_SEGMENT_IDENTITY : segments.indexOf(seg);

    /**
     * 映射后的**基线取样帧**：落在旧范围外时，周期段先按回绕折回主值区间。
     *
     * 【为什么折回是**精确**的，而不是近似】Loop 的基线是**源位置的函数**，而源位置
     * 以 `D` 为周期回绕 ⇒ 基线作为时间线帧的函数，周期正是 `seg.wrapFrames`。
     * 映射值 `mapped` 与 `mapped ± k·wrapFrames` 对应的源位置**相差整数个周期**
     *（`srcOld(g + wrapFrames) = srcOld(g) ± n`），`rem_euclid` 下逐值相同 ⇒
     * 折回前后取到的基线值逐值相等。
     *
     * 【不折回的代价】锚点跨过媒体端点时（或 clip 长于一个周期、偏移累积到越界时）
     * **整段**帧都会被判为"未知"，动态增益整段退化为 1 —— 用户看到的正是
     * "Slip 时动态参数干扰波形、与 Loop 位置有关、松手才恢复"。
     *
     * 【折回后仍越界 ⇒ 真未知】旧 clip 覆盖的不足一个完整周期时，某些相位在旧几何下
     * 确实从未被消费（Slip / 延伸带入的新素材），此时如实宣告未知。
     */
    const mapBaselineFrame = (seg: PreparedSegment, frameF: number): number => {
        const mapped = seg.offset + seg.slope * frameF;
        if (mapped >= seg.oldStartF && mapped < seg.oldEndF) return mapped;
        if (!(seg.wrapFrames > 0)) return WARP_BASELINE_UNKNOWN;
        const u = mapped - seg.oldStartF;
        const folded = ((u % seg.wrapFrames) + seg.wrapFrames) % seg.wrapFrames;
        return folded < seg.oldEndF - seg.oldStartF
            ? seg.oldStartF + folded
            : WARP_BASELINE_UNKNOWN;
    };

    return {
        baselineFrame(frameF) {
            const index = segmentAt(frameF);
            if (index < 0) return frameF;
            return mapBaselineFrame(segments[index] as PreparedSegment, frameF);
        },
        baselineScale(frameF) {
            const index = segmentAt(frameF);
            if (index < 0) return 1;
            const seg = segments[index] as PreparedSegment;
            // 折回与基线帧同口径：折回后的帧仍在旧覆盖内 ⇒ 增益比同样成立。
            if (mapBaselineFrame(seg, frameF) === WARP_BASELINE_UNKNOWN) return 1;
            return seg.gainScale;
        },
        curveFrame(frameF) {
            const index = segmentAt(frameF);
            if (index >= 0) {
                const seg = segments[index] as PreparedSegment;
                if (seg.curve.kind === "none") return frameF;
                if (seg.curve.kind === "move") {
                    return seg.curve.offset + seg.curve.slope * frameF;
                }
                return resampleFrame(seg.curve, frameF);
            }
            // 旧范围里但没被任何新范围覆盖：该处曲线已被搬走 → 恢复 pad。
            return coveredByOldTimeRange(frameF) ? WARP_CURVE_PAD : frameF;
        },
        baselineSegment: segmentAt,
        curveSegment(frameF) {
            const index = segmentAt(frameF);
            if (index >= 0) return curveSegmentIndex(segments[index] as PreparedSegment);
            return coveredByOldTimeRange(frameF) ? WARP_CURVE_PAD : WARP_SEGMENT_IDENTITY;
        },
        interpolableAt(frameF) {
            const index = segmentAt(frameF);
            // 不受任何映射影响的帧就是恒等映射 —— 恒等当然可插值。
            if (index < 0) return true;
            const seg = segments[index] as PreparedSegment;
            // 曲线走重采样时，`lutContentFade` 之外的三个通道（vol/target/base）
            // 里 target/vol 按重采样映射取样，折点同样会落进格内 ⇒ 一并回退逐值。
            if (seg.curve.kind === "resample") return false;
            return seg.interpolable;
        },
    };
}
