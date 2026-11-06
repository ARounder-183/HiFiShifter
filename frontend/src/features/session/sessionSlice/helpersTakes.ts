// Session Take / 源窗口 / 回滚快照 helper。
import type { TimelineClip, TimelineClipTake, TimelineState } from "../../../types/api";
import type { ClipInfo, ClipTakeInfo } from "../sessionTypes";
import { modEuclid, resolveLoopMediaDurationSec } from "../../../utils/loopRender";
import { clampGain } from "../../../utils/gainDb";
import { normalizeChannelMode } from "../../../utils/channelMode";
import { clamp } from "./helpersCore";
import { applyTimelineState } from "./helpersTimeline";
import type { SessionState } from "./types";
export function parseSelectClipRemoteArg(
    arg:
        | string
        | null
        | {
              clipId: string | null;
              preserveTrackFocus?: boolean;
          },
): { clipId: string | null; preserveTrackFocus: boolean } {
    if (typeof arg === "object" && arg !== null && "clipId" in arg) {
        return {
            clipId: arg.clipId,
            preserveTrackFocus: Boolean(arg.preserveTrackFocus),
        };
    }
    return {
        clipId: arg,
        preserveTrackFocus: false,
    };
}

export function parseTimelineClipTake(take: TimelineClipTake): ClipTakeInfo {
    return {
        id: take.id,
        name: take.name ?? "",
        gain: clampGain(Number(take.gain ?? 1)),
        sourcePath: take.source_path,
        sourcePathRelative: take.source_path_relative,
        durationSec: Number(take.duration_sec ?? 0) || undefined,
        durationFrames: take.duration_frames,
        sourceSampleRate: take.source_sample_rate,
        sourceStartSec: Number(take.source_start_sec ?? 0) || 0,
        sourceEndSec: Number.isFinite(Number(take.source_end_sec))
            ? Number(take.source_end_sec)
            : 0,
        playbackRate: take.playback_rate != null ? clamp(Number(take.playback_rate), 0.1, 10) : 1,
        reversed: Boolean(take.reversed),
        loopEnabled: Boolean(take.loop_enabled),
        channelMode: normalizeChannelMode(take.channel_mode),
        sourceChannels:
            Number.isFinite(Number(take.source_channels)) && Number(take.source_channels) > 0
                ? Number(take.source_channels)
                : undefined,
        midiNoteData: take.midi_note_data?.map((n) => ({
            startSec: n.start_sec,
            endSec: n.end_sec,
            note: n.note,
            velocity: n.velocity,
            channel: n.channel ?? 0,
        })),
        midiFillGaps: take.midi_fill_gaps ?? false,
        // 宿主的非线性拉伸标记 / take 包络：插件不渲染，但前端要据此把分割/拖边显式
        // 置为只读（见 `clipHasNonlinearStretch` / `clipHasTakeEnvelope`）。
        stretchMarkers: Array.isArray(take.stretch_markers) ? take.stretch_markers : undefined,
        envelopes: take.envelopes ?? undefined,
    };
}

export function parseClipTakes(clip: TimelineClip, flat: ClipInfo): ClipTakeInfo[] {
    if (Array.isArray(clip.takes) && clip.takes.length > 0) {
        return clip.takes.map(parseTimelineClipTake);
    }
    // 旧后端/过渡载荷：用 flat 投影合成一个 take。
    const clipRate = clamp(Number(clip.clip_playback_rate ?? 1) || 1, 0.1, 10);
    return [
        {
            id: clip.active_take_id ?? `${clip.id}_take_1`,
            name: flat.name,
            gain: flat.gain,
            sourcePath: flat.sourcePath,
            sourcePathRelative: undefined,
            durationSec: flat.durationSec,
            durationFrames: flat.durationFrames,
            sourceSampleRate: flat.sourceSampleRate,
            sourceStartSec: flat.sourceStartSec,
            sourceEndSec: flat.sourceEndSec,
            playbackRate: clamp(flat.playbackRate / clipRate, 0.1, 10),
            reversed: flat.reversed,
            loopEnabled: flat.loopEnabled,
            channelMode: flat.channelMode,
            sourceChannels: flat.sourceChannels,
            midiNoteData: flat.midiNoteData,
            midiFillGaps: flat.midiFillGaps ?? false,
        },
    ];
}

export function getClipRateMultiplier(clip: ClipInfo): number {
    const rate = Number(clip.clipPlaybackRate ?? 1);
    return Number.isFinite(rate) && rate > 0 ? rate : 1;
}

/**
 * 方向翻转（正放 ↔ 倒放）时保持**消费内容**不变的源窗口/锚点换算。
 *
 * 与后端 `flip_direction_source_window` 逐字段同口径，必须在改写
 * `reversed` **之前**调用（内部按"翻转前方向"读取消费窗口）：
 *   - 非 Loop 正放消费窗口 [ss, ss+span)、倒放 [se−span, se)。派生窗口
 *     模型下非锚定方向的存储字段可能陈旧，直接翻布尔会让消费内容跳变
 *     （如裁剪过的 Clip 倒放后播到陈旧 se 所指的文件末段）；
 *   - Loop 的字段承载回绕锚点（引擎正放自 mod(ss, D) 升、倒放自
 *     mod(se, D) 降），换算以原方向消费区间为准：翻为倒放锚定消费
 *     终点 mod(ss + span, D)（自该点下降覆盖原区间的镜像），翻为正放
 *     锚定消费起点 mod(se − span, D)。
 */
export function flipSourceWindowForDirection(
    fields: {
        reversed: boolean;
        loopEnabled: boolean;
        sourceStartSec: number;
        sourceEndSec: number;
    },
    spanSec: number,
    mediaTotalSec: number | null,
): void {
    if (fields.loopEnabled) {
        if (mediaTotalSec != null && mediaTotalSec > 0) {
            if (fields.reversed) {
                // 翻为正放：正放自锚点上升 → 新锚 = 原倒放消费起点。
                fields.sourceStartSec = modEuclid(fields.sourceEndSec - spanSec, mediaTotalSec);
            } else {
                // 翻为倒放：倒放自锚点下降 → 新锚 = 原正放消费终点。
                fields.sourceEndSec = modEuclid(fields.sourceStartSec + spanSec, mediaTotalSec);
            }
        } else if (fields.reversed) {
            // 媒体时长未知：退化为原始字段直算（引擎侧回绕兜底）。
            fields.sourceStartSec = fields.sourceEndSec - spanSec;
        } else {
            fields.sourceEndSec = fields.sourceStartSec + spanSec;
        }
        return;
    }
    if (fields.reversed) {
        // 翻为正放：ss := se − span（原倒放消费窗口的起点）。
        fields.sourceStartSec = fields.sourceEndSec - spanSec;
    } else {
        // 翻为倒放：se := ss + span（原正放消费窗口的终点）。
        fields.sourceEndSec = fields.sourceStartSec + spanSec;
    }
}

/** [`flipSourceWindowForDirection`] 的 Take 封装：span 按该 Take 的组合消费
 * 速率（Clip 倍率 × Take 速率）计算。 */
export function flipTakeSourceWindowForDirection(
    take: ClipTakeInfo,
    lengthSec: number,
    clipRate: number,
): void {
    const clipRateNum = Number.isFinite(clipRate) && clipRate > 1e-6 ? clipRate : 1;
    const takeRate =
        Number.isFinite(take.playbackRate) && take.playbackRate > 1e-6 ? take.playbackRate : 1;
    const span = Math.max(0, Number(lengthSec) || 0) * clipRateNum * takeRate;
    const mediaTotal = resolveLoopMediaDurationSec({
        durationFrames: take.durationFrames,
        sourceSampleRate: take.sourceSampleRate,
        durationSec: take.durationSec,
    });
    flipSourceWindowForDirection(take, span, mediaTotal > 0 ? mediaTotal : null);
}

export function updateActiveTakeFromFlat(clip: ClipInfo): void {
    const takes = clip.takes ?? [];
    const take = takes.find((entry) => entry.id === clip.activeTakeId) ?? takes[0];
    if (!take) return;
    take.gain = clip.gain;
    take.sourcePath = clip.sourcePath;
    take.sourcePathRelative = clip.sourcePathRelative;
    take.durationSec = clip.durationSec;
    take.durationFrames = clip.durationFrames;
    take.sourceSampleRate = clip.sourceSampleRate;
    take.sourceStartSec = clip.sourceStartSec;
    take.sourceEndSec = clip.sourceEndSec;
    take.playbackRate = clamp(clip.playbackRate / getClipRateMultiplier(clip), 0.1, 10);
    take.reversed = clip.reversed;
    take.channelMode = clip.channelMode;
    take.sourceChannels = clip.sourceChannels;
    take.loopEnabled = clip.loopEnabled;
    take.midiNoteData = clip.midiNoteData;
    take.midiFillGaps = clip.midiFillGaps;
}

/**
 * 与后端 `patch_clip_state` 的“同步编辑所有 Take”语义对齐：开启该设置时，
 * 内容级乐观编辑（增益/源窗口/速率/倒放/Loop/声道模式）需要镜像到全部 Take，
 * 否则多 Take 展示下 inactive lane 的拖拽预览与后端口径短暂分叉
 * （fulfilled 快照才会收敛）。`playbackRate` 平铺值是组合有效速率，
 * 写入各 Take 前按倍率反推 —— 与后端 from_clip 口径一致。
 *
 */
export function updateTakesFromFlatWithSync(state: SessionState, clip: ClipInfo): void {
    updateActiveTakeFromFlat(clip);
    if (!state.syncEditsAcrossTakes) return;
    const takes = clip.takes ?? [];
    if (takes.length <= 1) return;
    const rateMultiplier = getClipRateMultiplier(clip);
    for (const take of takes) {
        take.gain = clip.gain;
        take.sourceStartSec = clip.sourceStartSec;
        take.sourceEndSec = clip.sourceEndSec;
        take.playbackRate = clamp(clip.playbackRate / rateMultiplier, 0.1, 10);
        take.reversed = clip.reversed;
        take.channelMode = clip.channelMode;
        take.sourceChannels = clip.sourceChannels;
        take.loopEnabled = clip.loopEnabled;
    }
}

export function applyActiveTakeToFlat(clip: ClipInfo, take: ClipTakeInfo): void {
    clip.gain = take.gain;
    clip.sourcePath = take.sourcePath;
    clip.sourcePathRelative = take.sourcePathRelative;
    clip.durationSec = take.durationSec;
    clip.durationFrames = take.durationFrames;
    clip.sourceSampleRate = take.sourceSampleRate;
    clip.sourceStartSec = take.sourceStartSec;
    clip.sourceEndSec = take.sourceEndSec;
    clip.playbackRate = getClipRateMultiplier(clip) * take.playbackRate;
    clip.reversed = take.reversed;
    clip.channelMode = take.channelMode;
    clip.sourceChannels = take.sourceChannels;
    clip.loopEnabled = take.loopEnabled;
    clip.midiNoteData = take.midiNoteData;
    clip.midiNoteCount = take.midiNoteData?.length;
    clip.midiFillGaps = take.midiFillGaps ?? false;
}

/**
 * 应用权威时间轴，但保留前端当前播放光标。
 *
 * 后端 `TimelineState.playhead_sec` 只在显式 seek / transport 操作时同步；
 * 普通 take 管理命令返回的全量快照可能携带旧值。切换/管理 take 不应该
 * 改变播放位置，因此这里在覆写后恢复本地光标。
 */
/**
 * DAW 惯例：分割后选中右段、取消左段。
 *
 * 左段继承被分割 clip 的原 id（选择因此"残留"在左段上），右段为后端新建
 * clip，经 payload.created_clip_ids 按输入顺序返回。这里把单选/多选迁移到
 * 右段：
 * - 多选中未被分割的成员保持选中；
 * - 原单选若属于被分割的 clip，则映射到其对应右段（输入与右段按顺序
 *   一一对应；数量不一致时回退到第一个右段）。
 */
export function applySplitSelection(
    state: SessionState,
    action: { meta: { arg: { clipId?: string; clipIds?: string[] } } },
    payload: { created_clip_ids?: string[] | null } & TimelineState,
) {
    const prevSelectedClipId = state.selectedClipId;
    const prevMultiSelectedClipIds = state.multiSelectedClipIds;
    applyTimelineState(state, payload, { force: true });

    const arg = action.meta.arg ?? {};
    const argIds = arg.clipIds ?? (arg.clipId ? [arg.clipId] : []);
    const splitOriginals = new Set(argIds);
    const rightIds = (payload.created_clip_ids ?? []).filter((id) =>
        state.clips.some((clip) => clip.id === id),
    );
    if (rightIds.length === 0) {
        return;
    }
    state.multiSelectedClipIds = [
        ...prevMultiSelectedClipIds.filter((id) => !splitOriginals.has(id)),
        ...rightIds,
    ];
    // 分割把右段设为选中，属动作驱动（用户没表达"整组拖动"）。
    state.multiSelectionIntentional = false;
    if (prevSelectedClipId && splitOriginals.has(prevSelectedClipId)) {
        const selectedIndex = argIds.indexOf(prevSelectedClipId);
        state.selectedClipId =
            selectedIndex >= 0 && argIds.length === rightIds.length
                ? rightIds[selectedIndex]
                : rightIds[0];
    }
}

export function applyTimelineStatePreservingPlayhead(
    state: SessionState,
    timeline: TimelineState,
): void {
    const playheadSec = state.playheadSec;
    applyTimelineState(state, timeline, { force: true });
    state.playheadSec = Math.max(0, Number(playheadSec ?? 0) || 0);
}

// ── Take 乐观切换的回滚支持 ─────────────────────────────────────────────
// pending 阶段会本地切换 activeTakeId 并物化平铺投影；后端失败（rejected 或
// fulfilled ok:false）时若无回滚，UI 显示与音频引擎播放的内容将持续分裂，
// 直到下一次全量 fetchTimeline。这里按 clipId 记录切换前快照供恢复。

export type TakeOptimisticFlat = Pick<
    ClipInfo,
    | "gain"
    | "sourcePath"
    | "sourcePathRelative"
    | "durationSec"
    | "durationFrames"
    | "sourceSampleRate"
    | "sourceStartSec"
    | "sourceEndSec"
    | "playbackRate"
    | "clipPlaybackRate"
    | "reversed"
    | "loopEnabled"
    | "midiNoteData"
    | "midiNoteCount"
    | "midiFillGaps"
>;

export const pendingTakeRollbacks = new Map<
    string,
    { activeTakeId: string | undefined; flat: TakeOptimisticFlat }
>();

/**
 * take 快照要浅拷贝的字段。
 *
 * 【为什么带 `satisfies`】`TakeOptimisticFlat` 是 `Pick<ClipInfo, …>`，键集合
 * 必须与它一致。原先只靠一句注释声称"由类型约束保证"，而下面那行
 * `@ts-expect-error` 恰恰**禁用**了这个保证 —— `ClipInfo` 上某键改名或删除时，
 * 回滚会静默写入 `undefined` 而不是编译失败。`satisfies` 把这条不变量交还给
 * 编译器：键写错、写重、或指向已不存在的字段，都会在编译期报错。
 */
export const TAKE_FLAT_KEYS = [
    "gain",
    "sourcePath",
    "sourcePathRelative",
    "durationSec",
    "durationFrames",
    "sourceSampleRate",
    "sourceStartSec",
    "sourceEndSec",
    "playbackRate",
    "clipPlaybackRate",
    "reversed",
    "loopEnabled",
    "midiNoteData",
    "midiNoteCount",
    "midiFillGaps",
] as const satisfies readonly (keyof TakeOptimisticFlat)[];

export function snapshotTakeValue(value: unknown): unknown {
    // captureTakeRollback 在 Immer producer 内运行：对象/数组字段（如
    // midiNoteData）读到的是 draft 代理，存进模块级 Map 后会在 producer
    // 结束时被吊销；回滚时把吊销代理写回 state 会让 Immer finalization
    // 抛 "Cannot perform 'get' on a proxy that has been revoked"。
    // 这里做一次脱壳拷贝（take 快照字段均为 JSON 数据）。
    if (value === null || typeof value !== "object") return value;
    // Immer draft 是 Proxy，structuredClone 会抛 DataCloneError；普通对象则走
    // structuredClone（比 JSON 往返快，且保留 undefined / Date 等 JSON 会丢的值）。
    try {
        return structuredClone(value);
    } catch {
        return JSON.parse(JSON.stringify(value)) as unknown;
    }
}

export function captureTakeRollback(state: SessionState, clipIds: readonly string[]): void {
    for (const clipId of clipIds) {
        const clip = state.clips.find((entry) => entry.id === clipId);
        if (!clip) continue;
        // 同一 clip 已有待回滚快照时保留**最早**的那份：快速连续切换两次且
        // 都失败时，第二次失败后仍能回到最初的 take，而不是中间态。
        if (pendingTakeRollbacks.has(clipId)) continue;
        const flat = {} as TakeOptimisticFlat;
        for (const key of TAKE_FLAT_KEYS) {
            // 【为什么这里是一次显式放宽，而不是 `@ts-expect-error`】TS 不做
            // "联合键"的相关推理：`flat[key]` 与 `clip[key]` 在同一联合键下各自
            // 展开成联合类型，赋值要求逐成员可赋值，编译器无法关联二者。
            // `@ts-expect-error` 会连同**键拼错**这类真错误一起吞掉；这里只把目标
            // 类型放宽成 `Record<string, unknown>`，键的合法性仍由上面的
            // `satisfies` 在编译期守住。
            (flat as Record<string, unknown>)[key] = snapshotTakeValue(clip[key]);
        }
        pendingTakeRollbacks.set(clipId, { activeTakeId: clip.activeTakeId, flat });
    }
}

export function restoreTakeRollback(state: SessionState, clipIds: readonly string[]): void {
    for (const clipId of clipIds) {
        const snapshot = pendingTakeRollbacks.get(clipId);
        if (!snapshot) continue;
        pendingTakeRollbacks.delete(clipId);
        const clip = state.clips.find((entry) => entry.id === clipId);
        if (!clip) continue;
        clip.activeTakeId = snapshot.activeTakeId;
        Object.assign(clip, snapshot.flat);
    }
}

export function clearTakeRollback(clipIds: readonly string[]): void {
    for (const clipId of clipIds) {
        pendingTakeRollbacks.delete(clipId);
    }
}

/**
 * take 族异步操作的「过期响应守卫」。
 *
 * 与 `setClipStateRemote.fulfilled` 的 `_latestEditRequestId` 同一机制、同一动机：
 * 后端命令在线程池上并发执行，响应可能乱序到达；用户打开/新建工程时
 * `openProject/newProject.pending` 会把 `_latestEditRequestId` 置空作废一切在途
 * 编辑响应。take 族的权威回包是**整份 TimelineState 覆写**，不做守卫的话：
 * (a) 工程切换在途的 take 响应会把旧工程的时间线整体压回新工程上；
 * (b) 快速连续切换时旧响应后到，把 UI 回滚到过期 take（后端却持更新状态）。
 * 每个 take 族 thunk 在 `.pending` 登记 `requestId`，`.fulfilled`/`.rejected`
 * 只采纳仍是最新的那份。
 */
export function takeRequestIsStale(state: SessionState, requestId: string): boolean {
    return state._latestEditRequestId !== requestId;
}

/**
 * 将后端返回的 TimelineState 全量覆写到前端 Redux state。
 *
 * @param force  默认 false。当 `_interactionLockCount > 0`（用户正在拖动/滑动等连续交互）
 *               且 force 为 false 时，函数直接 return，跳过全量覆写，避免过期后端快照覆盖
 *               前端乐观更新导致的 UI 闪烁。
 *               对于"权威操作"（open/new/import/undo/redo/save/fetchTimeline 等），
 *               应传 `force: true` 以确保状态一定被应用。
 *               `adoptPlayhead: true` 时采纳载荷的 playhead_sec —— 仅限工程生命
 *               周期载入（打开/新建/导入）与撤销/重做（检查点快照即回退/
 *               恢复后的后端实际光标，视觉编辑点必须跟随）等播放头权威流程。
 *               编辑类命令不得传入：它们携带的 playhead_sec 是编辑未触及的
 *               旧值（播放期间停留在本次播放的起始位置），采纳会让光标跳变。
 */
