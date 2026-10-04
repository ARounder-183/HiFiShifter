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
 * ## 本模块做什么
 *
 * 把「按下时的几何 → 当前几何」的差分表达成一组**范围映射**
 * （`旧范围 → 新范围`，与后端 `TimelineState::stretch_linked_params_in_root_range`
 * 的 `frame_mappings` 同一形状），再据此把快照上的基线/曲线**按帧重采样**到新位置。
 *
 * 三条语义全部**镜像后端**，因此拖拽期间的显示与松手后的权威结果一致：
 *
 * 1. **原声基线永远跟着几何走**（它就是几何推导出来的量）；
 * 2. **用户曲线（volume / dyn 目标）只在"锁定参数线"开启时才跟着走** ——
 *    与后端只在该开关打开时才调用 `stretch_linked_params_in_root_range` 一致；
 * 3. **旧范围中不被任何新范围覆盖的帧恢复为 pad 值** —— volume → 1.0，
 *    dyn 目标 → 哨兵（= 沿用原声，即增益 1）。与后端
 *    `uncovered_old_segments` + `automation_curve_pad_value` 同一口径。
 *
 * ## 重采样公式：逐字镜像后端
 *
 * 后端的 `resample_curve(values, target_len, false)` 用
 * `ratio = (old_len − 1) / (new_len − 1)` 做线性插值（长度 1 时用 1.0 兜底）。
 * 本模块用同一公式把「新帧」映射回「旧帧」，因此同一次拖拽在前端预览出的曲线与
 * 后端提交后的曲线**逐值同源**，松手不会跳变。
 *
 * ## 近似边界（已知且可接受）
 *
 * 基线是**能量域融合**后的曲线（`sqrt(Σ(电平ᵢ×增益ᵢ)²)`），前端只有融合结果、没有
 * 逐 clip 分量，因此映射是对**整条曲线**做的。于是：
 *
 * - 单 clip 拖拽、整组同位移拖拽：**精确**（所有分量整体平移）；
 * - 裁切 / 拉伸：**精确**（仿射重采样，与后端同公式）；
 * - 增益旋钮：单 clip 精确；多 clip 重叠时按比例缩放融合值，是近似；
 * - 被拖走的 clip 与**未移动**的 clip 在新区间重叠：融合关系改变，映射无法表达，
 *   是近似（该处的波形高度可能略有偏差，松手后由权威快照纠正）。
 *
 * 这些都是**拖拽预览**的误差，且只出现在重叠区；相比修复前的"整段错位"，量级完全不同。
 */

/** 差分所需的最小 clip 几何。 */
export interface WarpClipGeometry {
    readonly id: string;
    readonly startSec: number;
    readonly lengthSec: number;
    readonly gain?: number | null;
}

/** 手势开始时的几何快照条目（与 `clipGeometryPreviewBus` 的快照同形）。 */
export interface WarpClipOrigin {
    readonly clipId: string;
    readonly startSec: number;
    readonly lengthSec: number;
    readonly gain: number;
}

/** 一次「旧范围 → 新范围」映射（与后端 `StretchLinkedRangeSec` 同形 + 增益比）。 */
export interface LoudnessRangeMapping {
    readonly oldStartSec: number;
    readonly oldLengthSec: number;
    readonly newStartSec: number;
    readonly newLengthSec: number;
    /**
     * 该 clip 的增益比 `newGain / oldGain`（1 = 未变）。
     *
     * 基线含静态 clip 增益，因此增益旋钮拖拽也会让基线整段变化。单 clip 时
     * 精确（基线 ∝ 增益）；重叠时按比例缩放融合值，属已知近似。
     */
    readonly gainScale: number;
}

/** 曲线帧落在「旧范围但不在任何新范围内」时的返回值：该帧恢复为 pad 值。 */
export const WARP_CURVE_PAD = -2;
/** 该帧不受任何映射影响（恒等）。 */
export const WARP_SEGMENT_IDENTITY = -1;

/**
 * 把基线/曲线采样到"现在的位置"的只读视图。
 *
 * 全部方法都是 O(映射数) 的纯查询、零分配 —— 它们在波形几何重建的热路径上被逐帧
 * 调用（一次重建按窗口帧数 × 常数次），因此不接受闭包/对象分配。
 */
export interface LoudnessGeometryWarp {
    /** 该帧应采样的**基线**帧（绝对帧，可为小数）。 */
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
     * **逐值等价**（此时格内的快照折点恰好落在格端点上）。
     *
     * - 平移（旧长 == 新长）：映射是整数帧的刚性位移 ⇒ 等价；
     * - 裁切 / 拉伸（旧长 ≠ 新长）：映射把快照的分段线性折点搬进了格内 ⇒ 线性插值
     *   会在折点两侧"抄近路"，与逐值路径分叉。
     *
     * 因此后者整段回退逐值 —— 只影响被裁切 / 拉伸的那一个 clip 的区间，代价可控，
     * 而"两条路径逐值等价"这条不变量得以保持。
     */
    interpolableAt(frameF: number): boolean;
}

/** 已归一化为帧的映射段（构造期算好，查询期零换算）。 */
interface PreparedSegment {
    readonly newStartF: number;
    readonly newEndF: number;
    readonly oldStartF: number;
    /** 旧范围的结束帧（不含）；用于判定"旧范围里但没被任何新范围覆盖"的帧。 */
    readonly oldEndF: number;
    /** 旧侧的最大下标（= oldCount − 1），对应后端 `resample_curve` 的 `old_max_idx`。 */
    readonly oldMaxIdx: number;
    /** 新侧的最大下标；长度为 1 时取 1.0（与后端同款兜底，避免除零）。 */
    readonly newMaxIdx: number;
    readonly gainScale: number;
    /**
     * 该段是否允许查表的整数帧线性插值（见 {@link LoudnessGeometryWarp.interpolableAt}）。
     *
     * 等价于"映射把整数帧送到整数帧且每帧走一格"：`oldMaxIdx == newMaxIdx`（刚性
     * 平移）或 `oldMaxIdx == 0`（整段塌到一帧，格内没有折点）。
     */
    readonly interpolable: boolean;
    /**
     * 该段是否改变了**时域**（起点/长度）。
     *
     * 【为什么必须区分】只有时域变化才会让后端调用
     * `stretch_linked_params_in_root_range` 去搬移用户曲线；纯增益变化只影响基线。
     * 若不区分，增益旋钮拖拽会把 volume/dyn 曲线也一起"搬走"——凭空造出一个
     * 后端不会做的编辑。
     */
    readonly timeChanged: boolean;
}

/** 浮点比较容差：秒级几何。 */
const TIME_EPSILON_SEC = 1e-9;
/** 浮点比较容差：线性增益。 */
const GAIN_EPSILON = 1e-4;

/**
 * 差分出「按下时几何 → 当前几何」的范围映射。
 *
 * 只对**两侧都存在**的 clip 产出映射：手势期间新增的 clip 没有旧位置可搬
 * （它的基线贡献是"从无到有"，映射表达不了），删除的 clip 同理。
 *
 * @param origin 手势开始时的全量几何快照。
 * @param clips 当前的 clip 几何（应已按参数编辑器所属**根轨道组**过滤 —— 别的
 *   轨道组的 clip 与本组基线无关，把它们的映射算进来会错误地搬移本组基线）。
 */
export function diffClipGeometryMappings(
    origin: readonly WarpClipOrigin[],
    clips: readonly WarpClipGeometry[],
): LoudnessRangeMapping[] {
    if (origin.length === 0 || clips.length === 0) return [];
    const originById = new Map<string, WarpClipOrigin>();
    for (const entry of origin) originById.set(entry.clipId, entry);

    const out: LoudnessRangeMapping[] = [];
    for (const clip of clips) {
        const before = originById.get(clip.id);
        if (before === undefined) continue;

        const oldStartSec = Number(before.startSec) || 0;
        const oldLengthSec = Math.max(0, Number(before.lengthSec) || 0);
        const newStartSec = Number(clip.startSec) || 0;
        const newLengthSec = Math.max(0, Number(clip.lengthSec) || 0);

        const oldGain = Number.isFinite(before.gain) ? before.gain : 1;
        const newGain = Number.isFinite(clip.gain) ? Number(clip.gain) : 1;
        const gainScale =
            oldGain > 1e-6 && Math.abs(newGain - oldGain) > GAIN_EPSILON * oldGain
                ? newGain / oldGain
                : 1;

        const timeChanged =
            Math.abs(newStartSec - oldStartSec) > TIME_EPSILON_SEC ||
            Math.abs(newLengthSec - oldLengthSec) > TIME_EPSILON_SEC;
        if (!timeChanged && gainScale === 1) continue;

        out.push({
            oldStartSec,
            oldLengthSec,
            newStartSec,
            newLengthSec,
            gainScale,
        });
    }
    return out;
}

/**
 * 构造时域映射视图。
 *
 * @param mappings 由 {@link diffClipGeometryMappings} 产出的范围映射。
 * @param lockParamLines 「锁定参数线」是否开启（决定用户曲线是否跟着走）。
 * @param framePeriodMs 帧周期（毫秒），用于秒↔帧换算。
 * @returns 映射视图；没有任何有效映射时返回 `null`（调用方据此完全跳过该路径，
 *   与修复前逐像素一致）。
 */
export function createLoudnessGeometryWarp(args: {
    mappings: readonly LoudnessRangeMapping[];
    lockParamLines: boolean;
    framePeriodMs: number;
}): LoudnessGeometryWarp | null {
    const fp = args.framePeriodMs;
    if (!(fp > 0) || args.mappings.length === 0) return null;

    /** 秒 → 帧：与后端 `(sec * 1000.0 / fp).round()` 逐字一致（含负值钳零）。 */
    const toFrame = (sec: number): number => Math.round((Math.max(0, sec) * 1000) / fp);

    const segments: PreparedSegment[] = [];
    for (const mapping of args.mappings) {
        const oldStartF = toFrame(mapping.oldStartSec);
        const oldEndF = toFrame(mapping.oldStartSec + Math.max(0, mapping.oldLengthSec));
        const newStartF = toFrame(mapping.newStartSec);
        const newEndF = toFrame(mapping.newStartSec + Math.max(0, mapping.newLengthSec));
        // 空的新范围覆盖不到任何帧，搬过去也没有意义。
        if (newEndF <= newStartF) continue;

        // 与后端 `resample_curve` 同款的长度兜底：旧侧 `len − 1`，新侧 `len − 1`
        // （长度 1 时用 1.0，避免除零）。
        const oldCount = Math.max(1, oldEndF - oldStartF);
        const newCount = newEndF - newStartF;
        const oldMaxIdx = oldCount - 1;
        const newMaxIdx = newCount > 1 ? newCount - 1 : 1;
        segments.push({
            newStartF,
            newEndF,
            oldStartF,
            oldEndF,
            oldMaxIdx,
            newMaxIdx,
            gainScale: mapping.gainScale,
            // 整数帧 → 整数帧且每帧走一格（刚性平移），或整段塌到一帧（格内无折点）。
            interpolable: oldMaxIdx === newMaxIdx || oldMaxIdx === 0,
            timeChanged: oldStartF !== newStartF || oldCount !== newCount,
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

    /** 该帧是否落在某个**时域变化**段的旧范围内（= 被搬走、需恢复 pad）。 */
    const coveredByOldTimeRange = (frameF: number): boolean => {
        for (let i = segments.length - 1; i >= 0; i -= 1) {
            const seg = segments[i] as PreparedSegment;
            if (!seg.timeChanged) continue;
            if (frameF >= seg.oldStartF && frameF < seg.oldEndF) return true;
        }
        return false;
    };

    /** 新帧 → 旧帧（段内仿射；恒等段直接返回原帧）。 */
    const sourceFrameIn = (seg: PreparedSegment, frameF: number): number =>
        seg.oldStartF + (frameF - seg.newStartF) * (seg.oldMaxIdx / seg.newMaxIdx);

    return {
        baselineFrame(frameF) {
            const index = segmentAt(frameF);
            if (index < 0) return frameF;
            return sourceFrameIn(segments[index] as PreparedSegment, frameF);
        },
        baselineScale(frameF) {
            const index = segmentAt(frameF);
            if (index < 0) return 1;
            return (segments[index] as PreparedSegment).gainScale;
        },
        curveFrame(frameF) {
            if (!args.lockParamLines) return frameF;
            const index = segmentAt(frameF);
            if (index >= 0) {
                const seg = segments[index] as PreparedSegment;
                // 纯增益变化不搬移用户曲线（后端也不会）。
                return seg.timeChanged ? sourceFrameIn(seg, frameF) : frameF;
            }
            return coveredByOldTimeRange(frameF) ? WARP_CURVE_PAD : frameF;
        },
        baselineSegment: segmentAt,
        curveSegment(frameF) {
            if (!args.lockParamLines) return WARP_SEGMENT_IDENTITY;
            const index = segmentAt(frameF);
            if (index >= 0) {
                const seg = segments[index] as PreparedSegment;
                return seg.timeChanged ? index : WARP_SEGMENT_IDENTITY;
            }
            return coveredByOldTimeRange(frameF) ? WARP_CURVE_PAD : WARP_SEGMENT_IDENTITY;
        },
        interpolableAt(frameF) {
            const index = segmentAt(frameF);
            // 不受任何映射影响的帧就是恒等映射 —— 恒等当然可插值。
            if (index < 0) return true;
            return (segments[index] as PreparedSegment).interpolable;
        },
    };
}
