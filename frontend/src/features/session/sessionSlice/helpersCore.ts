// Session 核心乐观编辑 helper（轨道/Clip/自动交叉淡化）。
import type { TimelineState } from "../../../types/api";
import type { ClipInfo, ClipFormantMorph, TrackInfo } from "../sessionTypes";
import { resolveLoopMediaDurationSec } from "../../../utils/loopRender";
import { clampGain } from "../../../utils/gainDb";
import { normalizeChannelMode } from "../../../utils/channelMode";
import { markProjectDirty } from "../sessionDirtyState";
import type { ClipColor, SessionState } from "./types";
import { MAX_TRACK_VOLUME } from "./settings";
import { applyTimelineState } from "./helpersTimeline";
import {
    getClipRateMultiplier,
    flipSourceWindowForDirection,
    updateTakesFromFlatWithSync,
} from "./helpersTakes";
export function clamp(value: number, minValue: number, maxValue: number): number {
    return Math.min(maxValue, Math.max(minValue, value));
}

/**
 * 从被拒绝的 thunk 里取出**对用户有意义**的原因。
 *
 * 【为什么不能直接用 `action.error.message`】`createAsyncThunk` 配
 * `rejectWithValue` 时，真正的原因在 `action.payload`，而 `action.error.message`
 * 恒为字面量 `"Rejected"`。于是状态栏显示"错误：Rejected" —— 用户拿不到任何可
 * 行动的信息（这正是"点`显示速度映射`得到 错误：Rejected"的来源）。
 */
export function rejectedMessage(action: {
    payload?: unknown;
    error?: { message?: string };
}): string {
    const payload = action.payload;
    if (typeof payload === "string" && payload.trim() !== "") return payload;
    if (payload !== null && typeof payload === "object" && "message" in payload) {
        const message = (payload as { message?: unknown }).message;
        if (typeof message === "string" && message.trim() !== "") return message;
    }
    const message = action.error?.message;
    // `"Rejected"` 不是原因，只是 RTK 的占位符 —— 拿它当消息等于什么都没说。
    if (typeof message === "string" && message !== "" && message !== "Rejected") return message;
    return "Request failed";
}

/**
 * 导入 clip 的 id：由源路径**确定性**导出。
 *
 * 【为什么不能随机】reducer 必须是纯函数。卫星窗口会把广播动作在本地 reducer
 * 上**重放**（见 `features/dock/detachBridge.ts`），若 id 在 reducer 里现算，
 * 主窗口与卫星窗口会为同一个动作算出不同的 id，`session.clips` 从此永久发散
 * （快照只在卫星窗口启动时对齐一次）。由路径导出则两边必然一致。
 *
 * 【为什么不会撞 id】`upsertImportedClip` 开头就按 `sourcePath` 早退，同一路径
 * 不会二次插入；不同路径再叠上长度参与哈希，实际碰撞概率可忽略。
 */
export function importedClipId(audioPath: string): string {
    // 32 位 FNV-1a。
    let hash = 0x811c9dc5;
    for (let index = 0; index < audioPath.length; index += 1) {
        hash ^= audioPath.charCodeAt(index);
        hash = Math.imul(hash, 0x01000193);
    }
    return `clip_import_${(hash >>> 0).toString(36)}_${audioPath.length.toString(36)}`;
}

export function basenameFromPath(path: string): string {
    return path.split(/[\\/]/).filter(Boolean).pop() ?? "Audio.wav";
}

export function normalizeClipColor(color: string | undefined): ClipColor {
    if (color === "blue") return "blue";
    if (color === "violet") return "violet";
    if (color === "amber") return "amber";
    if (color === "cyan") return "cyan";
    return "emerald";
}

/**
 * Auto-crossfade logic applied directly in a reducer (no dispatch needed).
 * For each clip in `movedIds`, detect overlaps with same-track clips and set
 * auto fade in/out to the crossfade-eligible overlap duration.
 *
 * A clip fully contained inside another (start/end coincident or inside) is
 * never a valid crossfade relationship, so its auto fade is cleared instead.
 */
export function applyAutoCrossfadeInReducer(state: SessionState, movedIds: string[]) {
    if (!state.autoCrossfadeEnabled || movedIds.length === 0) return;

    const movedSet = new Set(movedIds);
    const trackClipsMap: Record<string, ClipInfo[]> = {};
    const clipMap: Record<string, ClipInfo> = {};

    // 一次性建立轨道分组和全局 ID 索引，时间复杂度 O(N)
    for (const clip of state.clips) {
        clipMap[clip.id] = clip;
        if (!trackClipsMap[clip.trackId]) trackClipsMap[clip.trackId] = [];
        trackClipsMap[clip.trackId].push(clip);
    }

    // raw* 用于判断哪一侧被触碰；fade*Overlaps 只保存可产生自动交叉淡化的值。
    const rawFadeInOverlaps = new Map<string, number>();
    const rawFadeOutOverlaps = new Map<string, number>();
    const fadeInOverlaps = new Map<string, number>();
    const fadeOutOverlaps = new Map<string, number>();

    const isContained = (aStart: number, aEnd: number, bStart: number, bEnd: number): boolean => {
        const eps = 1e-9;
        return aStart >= bStart - eps && aEnd <= bEnd + eps;
    };

    for (const id of movedIds) {
        // O(1) 直接获取，消除多余的 find 遍历
        const clip = clipMap[id];
        if (!clip) continue;
        const clipStart = Number(clip.startSec);
        const clipEnd = clipStart + Number(clip.lengthSec);

        const sameTrack = (trackClipsMap[clip.trackId] || []).filter((c) => c.id !== id);

        for (const other of sameTrack) {
            const otherStart = Number(other.startSec);
            const otherEnd = otherStart + Number(other.lengthSec);
            const overlapStart = Math.max(clipStart, otherStart);
            const overlapEnd = Math.min(clipEnd, otherEnd);
            const overlap = overlapEnd - overlapStart;
            if (overlap <= 0.001) continue;

            const eligible =
                isContained(clipStart, clipEnd, otherStart, otherEnd) ||
                isContained(otherStart, otherEnd, clipStart, clipEnd)
                    ? 0
                    : overlap;

            if (clipStart <= otherStart) {
                rawFadeOutOverlaps.set(id, Math.max(rawFadeOutOverlaps.get(id) ?? 0, overlap));
                fadeOutOverlaps.set(id, Math.max(fadeOutOverlaps.get(id) ?? 0, eligible));
                rawFadeInOverlaps.set(
                    other.id,
                    Math.max(rawFadeInOverlaps.get(other.id) ?? 0, overlap),
                );
                fadeInOverlaps.set(other.id, Math.max(fadeInOverlaps.get(other.id) ?? 0, eligible));
            } else {
                rawFadeInOverlaps.set(id, Math.max(rawFadeInOverlaps.get(id) ?? 0, overlap));
                fadeInOverlaps.set(id, Math.max(fadeInOverlaps.get(id) ?? 0, eligible));
                rawFadeOutOverlaps.set(
                    other.id,
                    Math.max(rawFadeOutOverlaps.get(other.id) ?? 0, overlap),
                );
                fadeOutOverlaps.set(
                    other.id,
                    Math.max(fadeOutOverlaps.get(other.id) ?? 0, eligible),
                );
            }
        }
    }

    const allClipIds = new Set([
        ...rawFadeInOverlaps.keys(),
        ...rawFadeOutOverlaps.keys(),
        ...movedIds,
    ]);

    for (const clipId of allClipIds) {
        // O(1) 直接获取，消除多余的 find 遍历
        const clip = clipMap[clipId];
        if (!clip) continue;

        const rawIn = rawFadeInOverlaps.get(clipId);
        const rawOut = rawFadeOutOverlaps.get(clipId);

        if (rawIn !== undefined) {
            clip.autoFadeInSec = Math.max(0, fadeInOverlaps.get(clipId) ?? 0);
        } else if (movedSet.has(clipId)) {
            // 新导入/新建 clip 没有合法交叉淡化重叠时，自动 fade 应为 0。
            clip.autoFadeInSec = 0;
        }
        if (rawOut !== undefined) {
            clip.autoFadeOutSec = Math.max(0, fadeOutOverlaps.get(clipId) ?? 0);
        } else if (movedSet.has(clipId)) {
            clip.autoFadeOutSec = 0;
        }
    }
}

export function mapTimelineTracks(tracks: TimelineState["tracks"]): TrackInfo[] {
    return tracks.map((track) => ({
        id: track.id,
        name: track.name,
        parentId: track.parent_id ?? null,
        depth: track.depth ?? 0,
        childTrackIds: track.child_track_ids ?? [],
        muted: Boolean(track.muted),
        solo: Boolean(track.solo),
        volume: clamp(Number(track.volume ?? 1), 0, MAX_TRACK_VOLUME),

        composeEnabled: Boolean(track.compose_enabled),
        pitchAnalysisAlgo: String(track.pitch_analysis_algo ?? "nsf_hifigan_onnx"),
        color: track.color || undefined,
    }));
}

export function applyTimelineTracksOnly(state: SessionState, timeline: TimelineState) {
    state.tracks = mapTimelineTracks(timeline.tracks);
    state.selectedTrackId = timeline.selected_track_id;
}

/** 建轨回执的形状（插件自己的轨道与宿主轨道共用）。 */
export type AddedTrackPayload = {
    ok?: boolean;
    project?: { dirty?: boolean };
} & TimelineState;

/**
 * 增删轨道共用：标脏 + 交互锁期间只同步轨道列表。
 *
 * 【为什么要标脏】增删轨道是工程结构变更：不标脏的话"加了轨道 → 新建工程"会静默
 * 丢弃（与 Tempo Map 同款缺口）。
 * 【为什么交互锁期间只同步轨道】拖拽中整包应用后端快照会覆盖前端的 clip 乐观位置，
 * 产生闪烁。
 */
export function applyAddedTrack(state: SessionState, payload: AddedTrackPayload): void {
    if (!payload.ok) {
        return;
    }
    markProjectDirty(state.project);
    if (state._interactionLockCount > 0) {
        applyTimelineTracksOnly(state, payload);
        return;
    }
    applyTimelineState(state, payload, { force: true });
}

/**
 * 把音高曲线快照按"当前仍存在的 clip"过滤后写回。
 *
 * applyTimelineState 会剪掉快照中缺失 clip 的曲线；但曲线是前端独有的检测
 * 结果、并不随任何后端载荷返回，所以调用方必须先浅拷贝快照、应用快照后再
 * 恢复。若直接整包恢复，被删除 clip 的曲线（midiCurve 可能很长）会永久滞留。
 * 这里只保留仍在 state.clips 中的 id，其余随快照剪枝一起丢弃。
 *
 * 快照必须是浅拷贝（`{ ...state.clipPitchCurves }`）：在 Immer producer 内
 * 直接持有 draft 代理会被随后的 prune 透过修改，"恢复"等于写回已删空的 map。
 */
export function restoreClipPitchCurves(
    state: SessionState,
    snapshot: SessionState["clipPitchCurves"],
): void {
    const survivingClipIds = new Set(state.clips.map((clip) => clip.id));
    const restored: SessionState["clipPitchCurves"] = {};
    for (const [clipId, curve] of Object.entries(snapshot)) {
        if (survivingClipIds.has(clipId)) {
            restored[clipId] = curve;
        }
    }
    state.clipPitchCurves = restored;
}

export function applyTimelineStatePreservingPitchVisuals(
    state: SessionState,
    timeline: TimelineState,
) {
    const currentParamsEpoch = state.paramsEpoch;
    // 必须捕获浅拷贝快照：在 Immer producer 内直接持有 state.clipPitchCurves
    // 得到的是 draft 代理，applyTimelineState 随后的 prune（delete 缺失 clip）
    // 会透过代理可见，"恢复"赋值等于把已删空的 draft 原样写回。
    const currentClipPitchCurves = { ...state.clipPitchCurves };
    applyTimelineState(state, timeline, { force: true });
    state.paramsEpoch = currentParamsEpoch;
    restoreClipPitchCurves(state, currentClipPitchCurves);
}

export function applyOptimisticTrackState(
    state: SessionState,
    payload: {
        trackId: string;
        muted?: boolean;
        solo?: boolean;
        volume?: number;
        composeEnabled?: boolean;
        pitchAnalysisAlgo?: string;
        color?: string;
        name?: string;
    },
) {
    const track = state.tracks.find((entry) => entry.id === payload.trackId);
    if (!track) return;
    if (payload.muted !== undefined) {
        track.muted = Boolean(payload.muted);
    }
    if (payload.solo !== undefined) {
        track.solo = Boolean(payload.solo);
    }
    if (payload.volume !== undefined) {
        track.volume = clamp(Number(payload.volume), 0, MAX_TRACK_VOLUME);
    }
    if (payload.composeEnabled !== undefined) {
        track.composeEnabled = Boolean(payload.composeEnabled);
    }
    if (payload.pitchAnalysisAlgo !== undefined) {
        track.pitchAnalysisAlgo = String(payload.pitchAnalysisAlgo || track.pitchAnalysisAlgo);
    }
    if (payload.color !== undefined) {
        track.color = payload.color || undefined;
    }
    if (payload.name !== undefined) {
        track.name = String(payload.name || track.name);
    }
}

export function applyOptimisticClipState(
    state: SessionState,
    payload: {
        clipId: string;
        name?: string;
        color?: string;
        startSec?: number;
        lengthSec?: number;
        /** Take 级增益（take 面板）。 */
        gain?: number;
        /** Item 级增益（时间轴旋钮）。 */
        itemGain?: number;
        muted?: boolean;
        sourceStartSec?: number;
        sourceEndSec?: number;
        playbackRate?: number;
        clipPlaybackRate?: number;
        reversed?: boolean;
        loopEnabled?: boolean;
        snapOffsetSec?: number;
        fadeInSec?: number;
        fadeOutSec?: number;
        fadeInShape?: number;
        fadeOutShape?: number;
        fadeInDir?: number;
        fadeOutDir?: number;
        fadeInS?: number;
        fadeOutS?: number;
        autoFadeInSec?: number;
        autoFadeOutSec?: number;
        formantMorph?: ClipFormantMorph;
    },
) {
    const clip = state.clips.find((entry) => entry.id === payload.clipId);
    if (!clip) return;
    if (payload.name !== undefined) {
        clip.name = String(payload.name || clip.name);
    }
    if (payload.color !== undefined) {
        clip.color = normalizeClipColor(payload.color);
    }
    if (payload.startSec !== undefined) {
        clip.startSec = Math.max(0, Number(payload.startSec) || 0);
    }
    if (payload.lengthSec !== undefined) {
        clip.lengthSec = Math.max(0, Number(payload.lengthSec) || 0);
        // trim 改写长度而未携带 snapOffset 时同步下钳（与后端 patch_clip_state
        // 口径一致），避免残留 offset > length 的"幻影吸附目标"。
        if (payload.snapOffsetSec === undefined) {
            clip.snapOffsetSec = Math.min(Math.max(0, clip.snapOffsetSec), clip.lengthSec);
        }
    }
    if (payload.itemGain !== undefined) {
        clip.clip_gain = clampGain(Number(payload.itemGain));
    }
    if (payload.gain !== undefined) {
        clip.gain = clampGain(Number(payload.gain));
    }
    if (payload.muted !== undefined) {
        clip.muted = Boolean(payload.muted);
    }
    if (payload.sourceStartSec !== undefined) {
        // 与 sourceEndSec 同一口径：**不得钳制到 ≥0**（Slip 向左滑出媒体起点
        // ⇒ 负起点 = 前导静音）。用 `Number.isFinite` 而非 `Number(x) || 0`：
        // 后者虽保留负数，却会把 NaN 静默变成 0，与对侧不一致。
        const value = Number(payload.sourceStartSec);
        clip.sourceStartSec = Number.isFinite(value) ? value : clip.sourceStartSec;
    }
    if (payload.sourceEndSec !== undefined) {
        // 不得钳制到 ≥0：倒放 Clip 的消费窗口锚定 se，se<0（整窗在媒体
        // 下方的静音段）与 se>D（前导静音）都是合法状态。
        const value = Number(payload.sourceEndSec);
        clip.sourceEndSec = Number.isFinite(value) ? value : clip.sourceEndSec;
    }
    if (payload.playbackRate !== undefined) {
        clip.playbackRate = clamp(Number(payload.playbackRate), 0.1, 10);
    }
    if (payload.clipPlaybackRate !== undefined) {
        const nextClipRate = clamp(Number(payload.clipPlaybackRate) || 1, 0.1, 10);
        const takes = clip.takes ?? [];
        const activeTake = takes.find((entry) => entry.id === clip.activeTakeId) ?? takes[0];
        const previousTakeRate = activeTake
            ? activeTake.playbackRate
            : clamp(clip.playbackRate / getClipRateMultiplier(clip), 0.1, 10);
        clip.clipPlaybackRate = nextClipRate;
        clip.playbackRate = clamp(nextClipRate * previousTakeRate, 0.1, 10);
    }
    if (payload.reversed !== undefined) {
        // 方向翻转（且本请求未显式指定源窗口）时镜像后端 flip_direction_source_window
        // 的换算，保持消费窗口不变 —— 权威载荷到达前波形不跳变。
        if (
            payload.reversed !== clip.reversed &&
            payload.sourceStartSec === undefined &&
            payload.sourceEndSec === undefined
        ) {
            const rate =
                Number.isFinite(clip.playbackRate) && clip.playbackRate > 1e-6
                    ? clip.playbackRate
                    : 1;
            const mediaTotal = resolveLoopMediaDurationSec({
                durationFrames: clip.durationFrames,
                sourceSampleRate: clip.sourceSampleRate,
                durationSec: clip.durationSec,
            });
            flipSourceWindowForDirection(
                clip,
                Math.max(0, clip.lengthSec) * rate,
                mediaTotal > 0 ? mediaTotal : null,
            );
        }
        clip.reversed = Boolean(payload.reversed);
    }
    if (payload.loopEnabled !== undefined) {
        clip.loopEnabled = Boolean(payload.loopEnabled);
    }
    if (payload.snapOffsetSec !== undefined) {
        const offset = Math.max(0, Number(payload.snapOffsetSec) || 0);
        // 与后端一致：偏移点必须落在 [0, length] 内才可见、可吸附。
        clip.snapOffsetSec = Math.min(offset, clip.lengthSec);
    }
    if (payload.fadeInSec !== undefined) {
        clip.fadeInSec = Math.max(0, Number(payload.fadeInSec) || 0);
    }
    if (payload.fadeOutSec !== undefined) {
        clip.fadeOutSec = Math.max(0, Number(payload.fadeOutSec) || 0);
    }
    if (payload.fadeInShape !== undefined) {
        clip.fadeInShape = payload.fadeInShape;
    }
    if (payload.fadeOutShape !== undefined) {
        clip.fadeOutShape = payload.fadeOutShape;
    }
    if (payload.fadeInDir !== undefined) {
        clip.fadeInDir = Math.min(1, Math.max(-1, Number(payload.fadeInDir) || 0));
    }
    if (payload.fadeOutDir !== undefined) {
        clip.fadeOutDir = Math.min(1, Math.max(-1, Number(payload.fadeOutDir) || 0));
    }
    if (payload.fadeInS !== undefined) {
        clip.fadeInS = Math.min(1, Math.max(-1, Number(payload.fadeInS) || 0));
    }
    if (payload.fadeOutS !== undefined) {
        clip.fadeOutS = Math.min(1, Math.max(-1, Number(payload.fadeOutS) || 0));
    }
    if (payload.autoFadeInSec !== undefined) {
        clip.autoFadeInSec = Math.max(0, Number(payload.autoFadeInSec) || 0);
    }
    if (payload.autoFadeOutSec !== undefined) {
        clip.autoFadeOutSec = Math.max(0, Number(payload.autoFadeOutSec) || 0);
    }
    if (payload.formantMorph !== undefined) {
        clip.formantMorph = payload.formantMorph ? { ...payload.formantMorph } : undefined;
    }
    updateTakesFromFlatWithSync(state, clip);

    const clipEnd = clip.startSec + clip.lengthSec;
    if (clipEnd > state.projectSec) {
        state.projectSec = Math.ceil(clipEnd);
    }
}

export function applyOptimisticBulkClipState(
    state: SessionState,
    updates: Array<{
        clipId: string;
        /** Take 级增益（take 面板）。 */
        gain?: number;
        /** Item 级增益（时间轴旋钮）。 */
        itemGain?: number;
        muted?: boolean;
        startSec?: number;
        lengthSec?: number;
        sourceStartSec?: number;
        sourceEndSec?: number;
        snapOffsetSec?: number;
        clipPlaybackRate?: number;
        fadeInSec?: number;
        fadeOutSec?: number;
        fadeInShape?: number;
        fadeInDir?: number;
        fadeOutShape?: number;
        fadeOutDir?: number;
        fadeInS?: number;
        fadeOutS?: number;
        autoFadeInSec?: number;
        autoFadeOutSec?: number;
        reversed?: boolean;
        loopEnabled?: boolean;
        /** 声道模式 0..=4（对齐 REAPER CHANMODE）。 */
        channelMode?: number;
    }>,
) {
    // 多选批量拖拽每帧都会带 K 个 update 进来，逐 update 全量 find 是
    // O(K·N)；先建一次 id 索引把复杂度压回 O(N+K)。
    const clipsById = new Map(state.clips.map((c) => [c.id, c]));
    for (const update of updates) {
        const clip = clipsById.get(update.clipId);
        if (!clip) continue;
        if (update.itemGain !== undefined) {
            clip.clip_gain = clampGain(Number(update.itemGain));
        }
        if (update.gain !== undefined) {
            clip.gain = clampGain(Number(update.gain));
        }
        if (update.muted !== undefined) {
            clip.muted = Boolean(update.muted);
        }
        if (update.startSec !== undefined) {
            clip.startSec = Math.max(0, Number(update.startSec) || 0);
        }
        if (update.lengthSec !== undefined) {
            clip.lengthSec = Math.max(0, Number(update.lengthSec) || 0);
            // trim 改写长度而未携带 snapOffset 时同步下钳（与后端 patch_clip_state
            // 口径一致），避免残留 offset > length 的"幻影吸附目标"。
            if (update.snapOffsetSec === undefined) {
                clip.snapOffsetSec = Math.min(Math.max(0, clip.snapOffsetSec), clip.lengthSec);
            }
        }
        if (update.snapOffsetSec !== undefined) {
            clip.snapOffsetSec = Math.min(
                Math.max(0, Number(update.snapOffsetSec) || 0),
                clip.lengthSec,
            );
        }
        if (update.clipPlaybackRate !== undefined) {
            // 与 applyOptimisticClipState 同口径：写 Clip 级倍率，组合有效
            // 速率 = Clip 级 × active take 速率。
            const nextClipRate = clamp(Number(update.clipPlaybackRate) || 1, 0.1, 10);
            const takes = clip.takes ?? [];
            const activeTake = takes.find((entry) => entry.id === clip.activeTakeId) ?? takes[0];
            const previousTakeRate = activeTake
                ? activeTake.playbackRate
                : clamp(clip.playbackRate / getClipRateMultiplier(clip), 0.1, 10);
            clip.clipPlaybackRate = nextClipRate;
            clip.playbackRate = clamp(nextClipRate * previousTakeRate, 0.1, 10);
        }
        if (update.sourceStartSec !== undefined) {
            // 同 setClipSourceRange：不得钳制到 ≥0（Slip 负起点 = 前导静音）。
            const value = Number(update.sourceStartSec);
            clip.sourceStartSec = Number.isFinite(value) ? value : clip.sourceStartSec;
        }
        if (update.sourceEndSec !== undefined) {
            // 同 setClipSourceRange：不得钳制到 ≥0（倒放窗口合法含负值）。
            const value = Number(update.sourceEndSec);
            clip.sourceEndSec = Number.isFinite(value) ? value : clip.sourceEndSec;
        }
        if (update.fadeInSec !== undefined) {
            clip.fadeInSec = Math.max(0, Number(update.fadeInSec) || 0);
        }
        if (update.fadeOutSec !== undefined) {
            clip.fadeOutSec = Math.max(0, Number(update.fadeOutSec) || 0);
        }
        if (update.fadeInShape !== undefined) {
            clip.fadeInShape = update.fadeInShape;
        }
        if (update.fadeInDir !== undefined) {
            clip.fadeInDir = Math.min(1, Math.max(-1, Number(update.fadeInDir) || 0));
        }
        if (update.fadeOutShape !== undefined) {
            clip.fadeOutShape = update.fadeOutShape;
        }
        if (update.fadeOutDir !== undefined) {
            clip.fadeOutDir = Math.min(1, Math.max(-1, Number(update.fadeOutDir) || 0));
        }
        if (update.fadeInS !== undefined) {
            clip.fadeInS = Math.min(1, Math.max(-1, Number(update.fadeInS) || 0));
        }
        if (update.fadeOutS !== undefined) {
            clip.fadeOutS = Math.min(1, Math.max(-1, Number(update.fadeOutS) || 0));
        }
        if (update.autoFadeInSec !== undefined) {
            clip.autoFadeInSec = Math.max(0, Number(update.autoFadeInSec) || 0);
        }
        if (update.autoFadeOutSec !== undefined) {
            clip.autoFadeOutSec = Math.max(0, Number(update.autoFadeOutSec) || 0);
        }
        if (update.reversed !== undefined) {
            // 方向翻转（且本请求未显式指定源窗口）时镜像后端换算，保持
            // 消费窗口不变 —— 权威载荷到达前波形不跳变。
            if (
                update.reversed !== clip.reversed &&
                update.sourceStartSec === undefined &&
                update.sourceEndSec === undefined
            ) {
                const rate =
                    Number.isFinite(clip.playbackRate) && clip.playbackRate > 1e-6
                        ? clip.playbackRate
                        : 1;
                const mediaTotal = resolveLoopMediaDurationSec({
                    durationFrames: clip.durationFrames,
                    sourceSampleRate: clip.sourceSampleRate,
                    durationSec: clip.durationSec,
                });
                flipSourceWindowForDirection(
                    clip,
                    Math.max(0, clip.lengthSec) * rate,
                    mediaTotal > 0 ? mediaTotal : null,
                );
            }
            clip.reversed = Boolean(update.reversed);
        }
        if (update.loopEnabled !== undefined) {
            clip.loopEnabled = Boolean(update.loopEnabled);
        }
        if (update.channelMode !== undefined) {
            clip.channelMode = normalizeChannelMode(update.channelMode);
        }
        updateTakesFromFlatWithSync(state, clip);
    }
}
