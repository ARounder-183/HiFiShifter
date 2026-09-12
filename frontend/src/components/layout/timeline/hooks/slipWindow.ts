/**
 * slipWindow.ts — Slip（内部偏移）拖拽的源窗口平移几何。
 *
 * 【主要内容】
 * 把「指针的水平位移」换算为 clip 的**源窗口**（`sourceStartSec` / `sourceEndSec`）
 * 平移量：长度与时间轴位置都不变，只挪动 clip 内部取用的素材区间。
 *
 * 【为什么单独成模块】
 * 这段逻辑原先只存在于旧实现的 `useSlipDrag` 内。渲染内核的「Alt + 拖 clip 中部 =
 * 调整内部偏移」需要**同一份**语义；两处各写一份会在倒放 / 循环 / 非内容承载
 * （MIDI 或空素材）这几条分支上分叉——而这些分支的差异只在听感/波形上体现，
 * 极难归因。
 *
 * 【三条分支（与旧实现逐条一致）】
 * 1. `loopEnabled`：窗口两端对内容时长取模环绕（floor_mod），与渲染/引擎的回绕
 *    映射一致；内容时长未知时保持平移（避免卡死）。
 * 2. 非 Loop **正放**且内容承载：窗口按派生模型给出 —— 终点 = 起点 + 长度 × 速率，
 *    越出媒体的部分渲染静音（前导/尾部对称无界）。
 * 3. 其余（非 Loop 倒放 / 非内容承载）：只做整体平移，保持跨度不变
 *    （倒放的 `sourceEndSec` 是反向锚点，跨度可合法大于 长度 × 速率）。
 *
 * 【方向语义】倒放的播放时间轴是镜像的，源窗口必须沿指针**反方向**平移。
 *
 * 【与其他模块的关系】
 * - 上游：`useSlipDrag`（旧实现 DOM 拖拽）与 `TimelinePanel`（内核手势）。
 * - 依赖：`utils/loopRender` 的内容时长解析、`features/session` 的 ClipInfo 类型。
 */

import type { SessionState } from "../../../../features/session/sessionSlice";
import { resolveClipContentDurationSec } from "../../../../utils/loopRender";
import type { BoundarySnapClip } from "../../../../utils/loopSnap";

/** slip 计算所需的 clip 视图（从 ClipInfo 提取，缺省值已归一化）。 */
export interface SlipClipView {
    readonly playbackRate: number;
    readonly sourceStartSec: number;
    readonly sourceEndSec: number;
    readonly lengthSec: number;
    readonly reversed: boolean;
    readonly loopEnabled: boolean;
    readonly isContentBearing: boolean;
    readonly contentDurSec: number | null;
}

/**
 * 从 SessionState 的 ClipInfo 提取 slip 所需字段（缺省值归一化）。
 *
 * 特殊说明：旧实现的拖拽起点需要**按下时**的这几个字段（倒放 / 循环 / 速率 /
 * 长度）做快照，因此本函数对外导出，避免调用方再抄一份归一化规则。
 */
export function readSlipClip(clip: SessionState["clips"][number]): SlipClipView {
    const playbackRate = Number(clip.playbackRate ?? 1) || 1;
    return {
        playbackRate: playbackRate > 0 && Number.isFinite(playbackRate) ? playbackRate : 1,
        sourceStartSec: Number(clip.sourceStartSec ?? 0) || 0,
        sourceEndSec: Number(clip.sourceEndSec ?? 0) || 0,
        lengthSec: Math.max(0, Number(clip.lengthSec ?? 0) || 0),
        reversed: !!clip.reversed,
        loopEnabled: !!clip.loopEnabled,
        isContentBearing:
            !!clip.sourcePath || !!(clip.midiNoteData && clip.midiNoteData.length > 0),
        contentDurSec: resolveClipContentDurationSec({
            sourcePath: clip.sourcePath,
            midiNoteData: clip.midiNoteData ?? null,
            durationFrames: clip.durationFrames,
            sourceSampleRate: clip.sourceSampleRate,
            durationSec: clip.durationSec,
        }),
    };
}

/**
 * 把 clip 转成「媒体边界吸附」视图（`nearestBoundarySnapOffsetSec` /
 * `slipBoundaryAlignedSides` 的入参）。
 *
 * 【为什么需要它】loop 边界吸附的候选族与「内容时长 D」的解析规则必须与
 * 波形/引擎一致：`D` 优先取 `durationFrames / sourceSampleRate`，回退
 * `durationSec`，音高参考块等无源媒体 Clip 走 `resolveClipContentDurationSec`
 * 的覆盖值。旧实现（`useSlipDrag`）与渲染内核各写一份会让两处对「D 是多少」
 * 产生分叉——症状只在 loop Clip 跨媒体边界时出现，极难归因。
 *
 * 特殊说明：吸附候选族只依赖**按下时**的几何（平移不变），因此调用方应在
 * 手势开始时快照一次，不要逐帧重建。
 *
 * @param clip 目标 clip（来自 `session.clips`）。
 * @returns 吸附函数所需的归一化视图 + `isContentBearing`（吸附参与条件之一：
 *   无源媒体的空 Clip 不参与边界吸附）。
 */
export function toBoundarySnapClip(
    clip: SessionState["clips"][number],
): BoundarySnapClip & { readonly isContentBearing: boolean } {
    const v = readSlipClip(clip);
    return {
        loopEnabled: v.loopEnabled,
        reversed: v.reversed,
        sourceStartSec: v.sourceStartSec,
        sourceEndSec: v.sourceEndSec,
        playbackRate: v.playbackRate,
        lengthSec: v.lengthSec,
        durationFrames: clip.durationFrames ?? null,
        sourceSampleRate: clip.sourceSampleRate ?? null,
        durationSec: clip.durationSec ?? null,
        contentDurationSec: v.contentDurSec,
        isContentBearing: v.isContentBearing,
    };
}

/**
 * 计算 slip 拖拽后的源窗口。
 *
 * 【参数约定（**必须严格遵守**，曾因此出过一次"方向反了"的线上缺陷）】
 * `deltaSec` 不是屏幕位移，而是**窗口平移量**：
 * - **正值 = 源窗口向素材后段平移**（`sourceStartSec` / `sourceEndSec` 同时**增大**）；
 * - 负值 = 向前段平移。
 *
 * ⚠️ 屏幕「向右拖」对应的是**负**窗口平移量（REAPER 语义：把内容往右推，
 * Clip 起点露出的就是更早的素材）。因此**从屏幕位移换算时必须取反号**：
 * - 旧实现 `useSlipDrag`：`desiredTotal = 起点指针 − 当前指针`（向右拖为负）✓
 * - 渲染内核 `handleKernelDragPreview`：内核给的是 `deltaSec`（正 = 向右拖），
 *   故须传 `-deltaSec`。
 *
 * 本函数内部只做「按约定平移 + 分支归一化」，不做任何屏幕坐标换算——把换算
 * 留在调用方是刻意的：两边的坐标域不同（内容坐标 vs 指针坐标），混在一处
 * 正是上次方向搞反的原因。
 *
 * 特殊说明：调用方应传入**当前**的 clip（旧实现逐帧读取 Redux 当前值做增量平移，
 * 而不是用「按下时的基准 + 累计位移」重算——后者在拖拽中被其它编辑改写时会跳变）。
 *
 * @param clip 当前 clip（来自 session.clips）。
 * @param deltaSec 窗口平移量（秒，正 = 向素材后段平移；见上方约定）。
 * @returns 新的源窗口；输入非法时返回 null（调用方跳过该帧）。
 */
export function computeSlipWindow(
    clip: SessionState["clips"][number],
    deltaSec: number,
): { sourceStartSec: number; sourceEndSec: number } | null {
    if (!Number.isFinite(deltaSec)) return null;
    const v = readSlipClip(clip);
    // 方向语义：倒放的播放时间轴是镜像的，源窗口必须沿指针反方向平移。
    const dir = v.reversed ? -1 : 1;
    const deltaSrcSec = deltaSec * v.playbackRate * dir;
    let nextSourceStart = v.sourceStartSec + deltaSrcSec;
    let nextSourceEnd = v.sourceEndSec + deltaSrcSec;

    if (v.loopEnabled) {
        // Loop（循环源）：窗口两端对内容时长取模环绕（floor_mod），与
        // 渲染/引擎的回绕映射一致；音高参考块的 D = 音符内容范围。
        if (v.contentDurSec != null && v.contentDurSec > 1e-9) {
            const mediaDur = v.contentDurSec;
            nextSourceStart = ((nextSourceStart % mediaDur) + mediaDur) % mediaDur;
            nextSourceEnd = ((nextSourceEnd % mediaDur) + mediaDur) % mediaDur;
        }
        // 内容时长未知：保持平移，避免卡死。
    } else if (v.isContentBearing && !v.reversed) {
        // 非 Loop 正放：派生窗口模型 —— 终点 = 起点 + 长度 × 速率；
        // 越出媒体的部分渲染静音（前导/尾部对称无界）。
        nextSourceEnd = nextSourceStart + v.lengthSec * v.playbackRate;
    }
    // 其余分支：只做整体平移（保持跨度）。

    return { sourceStartSec: nextSourceStart, sourceEndSec: nextSourceEnd };
}
