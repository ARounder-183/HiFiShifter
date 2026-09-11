/**
 * 时间轴渲染内核 · clip 实例构建
 *
 * 【主要内容】
 * 把已投影到**内容坐标**的 clip 模型数组转成 GL 实例缓冲（每个 clip 25 个 float，
 * 布局见 `runtime/timelineClipGlRenderer`）：逐 clip 计算视觉样式、判定相邻分隔缝、
 * 写入实例数据；缓冲跨帧复用（按需倍增），避免每帧分配。
 *
 * 【作用】
 * 单 WebGL2 渲染器的 clip 块面由「一批实例 + 一次 draw call」绘制。几何以内容坐标
 * 产出、与视口无关，因此滚动帧只需更新 `u_viewOrigin` uniform——**不需要重建实例**。
 * 只有内容编辑 / 缩放 / 超出横向余量时才重新调用本构建器。
 *
 * 【与其他模块的关系】
 * - 上游：调用方用 `runtime/timelineCanvasModel.buildSparseClipRenderModel` 产出
 *   `TimelineCanvasClipModel[]`（本模块的 `ClipInstanceClip` 与其结构兼容），
 *   再交给本模块构建实例。
 * - 横向复用（**Spike 的明确例外**）：视觉样式复用 `runtime/timelineCanvasStyle`
 *   的 `buildTimelineClipVisualStyle`，实例写入复用 `runtime/timelineClipGlRenderer`
 *   的 `buildClipBodyInstance`——两者是既有已验证的绘制契约，重写会引入视觉漂移。
 * - 下游：GL 渲染器的 BOX 模式把实例缓冲上传并绘制。
 *
 * 【设计约束】
 * 1. 缓冲**跨帧复用**：`build()` 返回的 `instances` 是内部缓冲的切片视图（`count`
 *    之前的有效区间），调用方必须在下一次 `build()` 前消费完毕（上传 GPU 后即可）。
 * 2. 分隔缝只画 0.5px（与既有 Canvas2D / GL 路径一致）：右半边会被下一个 clip 覆盖，
 *    画满会产生 1px 粗缝。
 * 3. 编组激活的 clip 在既有实现中因外圈描边伸出矩形而退回 Canvas2D；Spike 无 2D
 *    兜底，此处**仍走 GL 画块面**（仅缺少外圈描边），差异记录在 Spike 报告中。
 */

import { buildClipBodyInstance, CLIP_INSTANCE_FLOATS } from "../../runtime/timelineClipGlRenderer";
import { buildTimelineClipVisualStyle } from "../../runtime/timelineCanvasStyle";

/**
 * 参与实例构建的最小 clip 字段集。
 *
 * 与 `runtime/timelineCanvasModel.TimelineCanvasClipModel` 结构兼容：调用方可直接
 * 传入后者，无需转换。
 */
export interface ClipInstanceClip {
    readonly id: string;
    readonly trackId: string;
    readonly name: string;
    readonly leftPx: number;
    readonly topPx: number;
    readonly widthPx: number;
    readonly heightPx: number;
    readonly headerHeightPx: number;
    readonly selected: boolean;
    readonly muted: boolean;
    readonly gain: number;
    readonly playbackRate: number;
    readonly groupId?: string;
    readonly trackColor?: string;
    readonly leadingOverlapPx?: number;
    readonly isMidiClip?: boolean;
}

/** 实例构建参数。 */
export interface ClipInstanceArgs {
    /** 内容坐标下的 clip 模型（通常来自 `buildSparseClipRenderModel`）。 */
    readonly clips: readonly ClipInstanceClip[];
    /** 主题模式：决定色块明度带与前景深浅方向。 */
    readonly darkMode: boolean;
    /** 字体族（样式计算用于名称宽度档）；缺省时由样式模块自行解析。 */
    readonly fontFamily?: string;
    /** 激活编组 id 集合（影响 badge / 描边样式）。 */
    readonly activeGroupIds?: ReadonlySet<string>;
    /** 禁用编组 id 列表。 */
    readonly disabledGroupIds?: readonly string[];
    /** 泳道底色（相邻 clip 之间的分隔缝颜色）。 */
    readonly seamColor: string;
}

/** 构建结果：实例缓冲 + 有效实例数。 */
export interface ClipInstances {
    /** 实例缓冲（长度为 `count × CLIP_INSTANCE_FLOATS` 的有效前缀）。 */
    readonly instances: Float32Array;
    /** 有效实例数。 */
    readonly count: number;
}

/** clip 实例构建器（持有可复用缓冲）。 */
export interface ClipInstanceBuilder {
    /**
     * 构建一帧的 clip 实例。
     *
     * @param args 构建参数。
     * @returns 实例缓冲与数量；缓冲在下一次 `build()` 时被覆写。
     */
    build(args: ClipInstanceArgs): ClipInstances;
}

/** 相邻 clip 判定容差（CSS px）：左缘与前一 clip 右缘在此容差内视为紧贴。 */
const SEAM_TOLERANCE_PX = 0.5;

/**
 * 计算需要绘制分隔缝的 clip id 集合。
 *
 * 判据（与既有 Canvas2D 路径一致）：同轨存在另一个 clip，其左缘 ≈ 本 clip 右缘。
 * 紧贴的两个 clip 之间画泳道底色细缝，避免被误认为连续的一块。
 *
 * 流程：按轨道分组 → 按左缘升序排序 → 对每个 clip 向后扫描（越过右缘即 break，
 * 首尾相接的常见排布下退化为 O(n)）。
 *
 * @param clips 内容坐标下的 clip 模型。
 * @returns 需要分隔缝的 clip id 集合。
 */
function computeSeamClipIds(clips: readonly ClipInstanceClip[]): Set<string> {
    const byTrack = new Map<string, Array<{ id: string; left: number; right: number }>>();
    for (const clip of clips) {
        const left = clip.leftPx;
        const right = clip.leftPx + Math.max(1, clip.widthPx);
        let list = byTrack.get(clip.trackId);
        if (list === undefined) {
            list = [];
            byTrack.set(clip.trackId, list);
        }
        list.push({ id: clip.id, left, right });
    }

    const seams = new Set<string>();
    for (const list of byTrack.values()) {
        if (list.length < 2) continue;
        list.sort((a, b) => a.left - b.left);
        for (let i = 0; i < list.length; i += 1) {
            const current = list[i];
            for (let j = i + 1; j < list.length; j += 1) {
                const next = list[j];
                // 升序 ⇒ 起点越过后缘之后不可能再紧贴，提前跳出内层。
                if (next.left > current.right + SEAM_TOLERANCE_PX) break;
                if (Math.abs(next.left - current.right) <= SEAM_TOLERANCE_PX) {
                    seams.add(current.id);
                    break;
                }
            }
        }
    }
    return seams;
}

/**
 * 创建 clip 实例构建器。
 *
 * 流程：
 * 1. 首次调用时按需分配实例缓冲（容量不足则倍增，避免每帧重新分配）；
 * 2. 逐 clip 计算视觉样式（复用既有样式模块）并写入实例；
 * 3. 返回缓冲与有效实例数。
 *
 * 特殊说明：
 * - 样式模块在 node 测试环境下走 DOM 兜底（`typeof document === "undefined"` 分支），
 *   因此本构建器可在无 DOM 环境单测。
 * - `isPitchAdjustment` 沿用既有渲染路径的传法（`isMidiClip`），保持视觉一致；
 *   该命名与语义的偏差属于既有代码的历史约定，不在本次 Spike 修正范围内。
 *
 * @returns 构建器实例；内部缓冲跨帧复用，须长生命周期持有。
 */
export function createClipInstanceBuilder(): ClipInstanceBuilder {
    let buffer = new Float32Array(0);

    return {
        build(args) {
            const clips = args.clips;
            const count = clips.length;
            const needed = count * CLIP_INSTANCE_FLOATS;
            if (buffer.length < needed) {
                // 倍增而非精确分配：滚动/缩放过程中窗口内 clip 数会小幅波动，
                // 每次精确分配会让 GC 压力回到"每帧一次大分配"。
                buffer = new Float32Array(Math.max(needed, buffer.length * 2));
            }

            const seamClipIds = computeSeamClipIds(clips);
            for (let index = 0; index < count; index += 1) {
                const clip = clips[index];
                const isGroupActive =
                    clip.groupId != null && args.activeGroupIds?.has(clip.groupId) === true;
                const isGroupDisabled =
                    clip.groupId != null &&
                    (args.disabledGroupIds?.includes(clip.groupId) ?? false);
                const style = buildTimelineClipVisualStyle({
                    widthPx: Math.max(1, clip.widthPx),
                    trackColor: clip.trackColor,
                    selected: clip.selected,
                    muted: clip.muted,
                    gain: clip.gain,
                    playbackRate: clip.playbackRate,
                    name: clip.name,
                    fontFamily: args.fontFamily,
                    isPitchAdjustment: clip.isMidiClip === true,
                    groupId: clip.groupId,
                    isGroupActive,
                    isGroupDisabled,
                    darkMode: args.darkMode,
                });
                buildClipBodyInstance(
                    buffer,
                    index,
                    clip,
                    style,
                    seamClipIds.has(clip.id) ? args.seamColor : null,
                );
            }

            return { instances: buffer, count };
        },
    };
}
