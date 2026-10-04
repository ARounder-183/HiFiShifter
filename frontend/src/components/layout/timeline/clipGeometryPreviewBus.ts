/**
 * 时间轴「clip 几何手势」总线。
 *
 * ## 为什么需要它
 *
 * 参数编辑器的波形画的是**可听结果**：`源峰值 × clip增益×淡化 × volume(t) × dyn增益(t)`。
 * 其中乘数里的 `dyn增益 = 目标电平 / 原声电平基线`，而**原声电平基线是 clip 几何的函数**
 * —— clip 从 0s 移到 3s，同一段素材的基线就从帧 `[0, 200)` 挪到 `[600, 800)`。
 *
 * 时间轴拖拽期间，clip 几何只在 **Redux 里乐观更新**（不写后端），而响度快照要等
 * `paramsEpoch` 递增（= 提交落库）才会重取。于是整个手势期间：
 *
 * - 波形**几何**已经跟到新位置（`useClipsPeaksForPianoRoll` 读的就是 Redux）；
 * - 波形**乘数**还停在按下之前的位置 —— 旧位置的基线配新位置的峰值。
 *
 * 在响段上表现为高度乱跳，在静段上（旧位置基线 ≈ 0）被"无内容淡出"压成 0 或反被放大成
 * 满高平台。这正是用户报告的「拖拽过程中波形有很大问题，松手才恢复」。
 *
 * ## 本模块的职责（只有一件事）
 *
 * 在几何手势开始时发布一份**全量 clip 几何快照**，结束时宣告手势已完成。消费方
 * （`loudnessGeometryWarp`）用它和当前几何做差分，得出**源坐标仿射映射**，
 * 从而在本地把基线/曲线搬到正确的位置 —— 不需要任何 IPC，也不需要等后端。
 *
 * 【为什么快照必须含源窗口与速率】Slip 只改 `sourceStartSec / sourceEndSec`
 * （起点/长度/速率全不变），而拉伸改的是 `playbackRate`。只记 `start/length`
 * 会让 Slip 被判定成"没有变化"、让延伸/截短被错当成拉伸 —— 见 `loudnessGeometryWarp`
 * 文件头的推导。
 *
 * ## 为什么是"全量快照"而不是"参与集合"
 *
 * 手势真正改动的 clip 不止参与集合：波纹跟随（ripple）会把后续 clip 一起平移，
 * 自动交叉淡化会改写相邻 clip 的淡化。全量快照 + 差分的口径与后端
 * `build_root_dyn_key`（它对**所有** clip 的几何取哈希）一致，不需要在前端复刻
 * 「这次手势影响了谁」的判断。
 *
 * ## 与其他模块的关系
 * - 发布方：`TimelinePanel` 的内核手势交互锁（`begin/endKernelGestureInteraction`
 *   —— 已有的一对"手势生命周期"回调，全部内核手势都经过它）。
 * - 消费方：`PianoRollPanel` → `loudnessGeometryWarp`。
 */

/** 手势开始时记录的单个 clip 几何（决定"哪段素材在哪个时刻被播放"的全部字段）。 */
export interface ClipGeometrySnapshot {
    readonly clipId: string;
    /** 时间轴起点（秒）。 */
    readonly startSec: number;
    /** 时间轴长度（秒）。 */
    readonly lengthSec: number;
    /**
     * 静态 clip 增益。
     *
     * 【为什么它在几何快照里】DYN 的原声基线**含**静态 clip 增益
     * （见 `pitch_analysis::dyn_analysis` 的"电平的物理口径"：用户听到的响度包含它）。
     * 因此增益旋钮拖拽同样会让基线整段变化，必须能表达为映射的一部分。
     */
    readonly gain: number;
    /**
     * 源窗口（秒）。
     *
     * 【为什么必须有】Slip 只改这两个字段（`slipWindow.ts`），起点/长度/速率全不变。
     * 上一版快照缺了它们，差分判定"没有变化"⇒ 映射为空 ⇒ Slip 期间波形仍是
     * 「旧内容的基线 × 新内容的峰值」。延伸/截短同样靠它们才能被识别为"揭示"而非"缩放"。
     */
    readonly sourceStartSec: number;
    readonly sourceEndSec: number;
    /** 播放速率（`playbackRate`，含 take 倍率）。拉伸会改它 ⇒ 必须参与映射。 */
    readonly playbackRate: number;
    readonly reversed: boolean;
    readonly loopEnabled: boolean;
}

/** 可从 clip 列表建立快照的最小字段集（Redux `ClipInfo` 天然满足）。 */
export interface ClipGeometrySource {
    readonly id: string;
    readonly startSec: number;
    readonly lengthSec: number;
    readonly gain?: number | null;
    readonly sourceStartSec?: number | null;
    readonly sourceEndSec?: number | null;
    readonly playbackRate?: number | null;
    readonly reversed?: boolean;
    readonly loopEnabled?: boolean;
}

let origin: readonly ClipGeometrySnapshot[] | null = null;
const listeners = new Set<() => void>();

function notify(): void {
    for (const listener of listeners) {
        try {
            listener();
        } catch {
            // 订阅者异常不应打断手势本身。
        }
    }
}

/**
 * 手势开始：发布几何快照。
 *
 * 必须在**第一次乐观写入之前**调用 —— 快照的语义是"按下时的几何"，一旦被手势改过，
 * 差分就退化成恒等，映射也就失去意义。调用点（`beginKernelGestureInteraction`）
 * 已满足这一时序：它在首个真实位移帧、且在任何预览派发之前执行。
 */
export function beginClipGeometryPreview(clips: readonly ClipGeometrySource[]): void {
    origin = clips.map((clip) => ({
        clipId: clip.id,
        startSec: Number(clip.startSec) || 0,
        lengthSec: Math.max(0, Number(clip.lengthSec) || 0),
        gain: Number.isFinite(clip.gain) ? Number(clip.gain) : 1,
        sourceStartSec: Number(clip.sourceStartSec) || 0,
        sourceEndSec: Number(clip.sourceEndSec) || 0,
        playbackRate: Number(clip.playbackRate) || 1,
        reversed: clip.reversed === true,
        loopEnabled: clip.loopEnabled === true,
    }));
    notify();
}

/**
 * 手势结束：宣告手势已完成（消费方进入"等权威快照追上"态）。
 *
 * 【为什么消费方**不**在这里立刻撤下映射】提交之后、权威快照回来之前，映射仍然是
 * 唯一正确的数据源（它把旧位置的基线搬到新位置，而这正是后端即将算出的结果）。
 * 立刻撤下会让波形在那一两帧里退回**旧位置的基线**——用户看到的就是"松手闪一下"。
 * 撤下的判据是**后端返回的基线所依据的几何变了**（`LoudnessSnapshot.baselineKey`），
 * 与快照落地落在同一次渲染里；本函数只负责宣告手势已经结束。
 */
export function endClipGeometryPreview(): void {
    if (origin === null) return;
    origin = null;
    notify();
}

/** 当前手势的几何快照；`null` = 没有进行中的几何手势。 */
export function getClipGeometryPreviewOrigin(): readonly ClipGeometrySnapshot[] | null {
    return origin;
}

/**
 * 订阅手势起止。
 *
 * 订阅时**不会**补发当前状态：调用方应先读一次
 * {@link getClipGeometryPreviewOrigin} 再订阅（避免订阅回调里做首次同步时
 * 触发一次多余的重渲染）。
 */
export function subscribeClipGeometryPreview(listener: () => void): () => void {
    listeners.add(listener);
    return () => {
        listeners.delete(listener);
    };
}
