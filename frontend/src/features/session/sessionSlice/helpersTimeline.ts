// Session 时间线快照应用与导入 upsert helper。
import type { TimelineClip, TimelineState } from "../../../types/api";
import type { GridSize } from "../sessionTypes";
import { clampGain } from "../../../utils/gainDb";
import { normalizeChannelMode } from "../../../utils/channelMode";
import { SCALE_KEYS } from "../../../utils/musicalScales";
import type { ScaleLike } from "../../../utils/musicalScales";
import { sanitizeCustomScalePreset } from "../../../utils/customScales";
import { clampDenominator, fromBackendTempoMap, TEMPO_DENOMINATORS } from "../../../utils/tempoMap";
import type { SessionState, StretchAlgorithmOption } from "./types";
import {
    applyTimelineTracksOnly,
    normalizeClipColor,
    clamp,
    importedClipId,
    basenameFromPath,
} from "./helpersCore";
import { parseClipTakes } from "./helpersTakes";
import { VALID_GRID_SIZES } from "./settings";
export function applyTimelineState(
    state: SessionState,
    timeline: TimelineState,
    opts?: { force?: boolean; preserveProjectNotes?: boolean; adoptPlayhead?: boolean },
) {
    // 交互锁守卫：拖动/滑动期间跳过非强制的全量覆写
    if (state._interactionLockCount > 0 && !opts?.force) {
        return;
    }

    applyTimelineTracksOnly(state, timeline);

    // 静音检测预览覆盖层锚定在旧时间线上，任何全量快照应用即失效。
    // 请求 id 必须随之作废：否则在途分析的 fulfilled 只比对 id（ id 未变），
    // 会把按**旧剪辑布局**算出的静音段重新画回时间线（甚至画到切除后的新剪辑上）
    // —— 与 `setSilencePreview` 手动清空时的口径一致。
    state.silencePreviewSegments = null;
    state._silencePreviewRequestId = null;

    // 旧 clips 的 id 索引：playback_rate 缺失时的回退查询需要 O(1) 命中，
    // 否则 per-clip find 会让千级 clip 工程的全量快照应用退化为 O(n²)。
    const oldClipsById = new Map(state.clips.map((c) => [c.id, c]));

    state.clips = timeline.clips.map((clip: TimelineClip) => {
        const parsed = {
            id: clip.id,
            trackId: clip.track_id,
            name: clip.name,
            startSec: Number(clip.start_sec ?? 0),
            lengthSec: Math.max(0.0, Number(clip.length_sec ?? 1)),
            color: normalizeClipColor(clip.color),
            sourcePath: clip.source_path,
            sourcePathRelative: clip.source_path_relative,
            durationSec: Number(clip.duration_sec ?? 0) || undefined,
            durationFrames: clip.duration_frames,
            sourceSampleRate: clip.source_sample_rate,
            // Take 级增益（active take 投影）。
            gain: clampGain(Number(clip.gain ?? 1)),
            // Item 级增益（时间轴旋钮）。后端未提供时按 1.0 —— 注意 0 是合法静音值，
            // 不能用 `|| 1` 兜底。
            clip_gain: clampGain(Number(clip.clip_gain ?? 1)),
            muted: Boolean(clip.muted),
            // Allow negative sourceStartSec to represent leading silence (slip-edit past source start).
            sourceStartSec: Number(clip.source_start_sec ?? 0) || 0,
            sourceEndSec: (() => {
                const raw = Number(clip.source_end_sec);
                // 仅当字段**缺失/非法**时才回退默认值。
                // 注意两点：
                // 1. 不得 Math.max(0, …)：倒放 Clip 的消费窗口锚定 se，
                //    se<0（整窗在媒体下方的静音段）是合法状态 —— 钳零会把
                //    分割出的静音段凭空拉回媒体域；
                // 2. 不得把 se==0 当作"到媒体末尾"的旧哨兵改写为 duration：
                //    该哨兵语义已在后端 open_project 迁移中展开为真实时长，
                //    此处的二次改写会摧毁合法的 0/负值窗口。
                if (!Number.isFinite(raw)) {
                    return (
                        Number(clip.duration_sec ?? 0) || Math.max(0, Number(clip.length_sec ?? 1))
                    );
                }
                return raw;
            })(),
            playbackRate:
                clip.playback_rate != null
                    ? clamp(Number(clip.playback_rate), 0.1, 10)
                    : (oldClipsById.get(clip.id)?.playbackRate ?? 1),
            clipPlaybackRate: clamp(Number(clip.clip_playback_rate ?? 1) || 1, 0.1, 10),
            reversed: Boolean(clip.reversed),
            channelMode: normalizeChannelMode(clip.channel_mode),
            sourceChannels:
                Number.isFinite(Number(clip.source_channels)) && Number(clip.source_channels) > 0
                    ? Number(clip.source_channels)
                    : undefined,
            loopEnabled: Boolean(clip.loop_enabled),
            hostMedia: typeof clip.host_media === "string" ? clip.host_media : undefined,
            hostMediaReason:
                typeof clip.host_media_reason === "string" ? clip.host_media_reason : undefined,
            // SnapOffset（吸附偏移）：旧工程缺失时自动补齐为 0。
            snapOffsetSec: Math.max(0, Number(clip.snap_offset_sec ?? 0) || 0),
            fadeInSec: Math.max(0, Number(clip.fade_in_sec ?? 0)),
            hostFades: clip.host_fades,
            fadeOutSec: Math.max(0, Number(clip.fade_out_sec ?? 0)),
            // 后端未提供形状字段时（极旧开发版载荷），与新建默认一致取快起。
            fadeInShape: Number.isFinite(Number(clip.fade_in_shape))
                ? Number(clip.fade_in_shape)
                : 1,
            fadeOutShape: Number.isFinite(Number(clip.fade_out_shape))
                ? Number(clip.fade_out_shape)
                : 1,
            fadeInDir: Math.min(1, Math.max(-1, Number(clip.fade_in_dir ?? 0) || 0)),
            fadeOutDir: Math.min(1, Math.max(-1, Number(clip.fade_out_dir ?? 0) || 0)),
            // S 轴没有 kernel 字段（REAPER ≥7.81 独有），从宿主读数装饰里取；
            // legacy 宿主与独立 App 没有这根轴 ⇒ 0。
            fadeInS:
                clip.host_fades?.curve_mode === "continuous"
                    ? Math.min(1, Math.max(-1, Number(clip.host_fades.in_s ?? 0) || 0))
                    : 0,
            fadeOutS:
                clip.host_fades?.curve_mode === "continuous"
                    ? Math.min(1, Math.max(-1, Number(clip.host_fades.out_s ?? 0) || 0))
                    : 0,
            autoFadeInSec: Math.max(0, Number(clip.auto_fade_in_sec ?? 0) || 0),
            autoFadeOutSec: Math.max(0, Number(clip.auto_fade_out_sec ?? 0) || 0),
            formantMorph: clip.formant_morph
                ? {
                      enabled: Boolean(clip.formant_morph.enabled),
                      targetF1Hz: Number(clip.formant_morph.target_f1_hz ?? 800),
                      targetF2Hz: Number(clip.formant_morph.target_f2_hz ?? 1400),
                      strength: Number(clip.formant_morph.strength ?? 0.5),
                  }
                : undefined,
            midiNoteCount: clip.midi_note_count,
            midiNoteData: clip.midi_note_data?.map(
                (n: {
                    start_sec: number;
                    end_sec: number;
                    note: number;
                    velocity: number;
                    channel?: number;
                }) => ({
                    startSec: n.start_sec,
                    endSec: n.end_sec,
                    note: n.note,
                    velocity: n.velocity,
                    channel: n.channel ?? 0,
                }),
            ),
            midiFillGaps: clip.midi_fill_gaps ?? false,
            groupId: clip.group_id ?? undefined,
        };

        const takes = parseClipTakes(clip, parsed);
        return {
            ...parsed,
            takes,
            activeTakeId:
                clip.active_take_id && takes.some((take) => take.id === clip.active_take_id)
                    ? clip.active_take_id
                    : (takes[0]?.id ?? `${clip.id}_take_1`),
        };
    });
    // 剪枝孤儿条目：先建一次 Set（O(clips)），逐条 O(1) 查询 —— `.some()` 内层
    // 线性扫描是 O(entries × clips)，本函数在每个编辑 / 撤销 / 保存回包上都会跑，
    // 千级 clip 下那是每次数百万次比较。
    const availableClipIds = new Set(state.clips.map((clip) => clip.id));
    state.clipFormantStatus = Object.fromEntries(
        Object.entries(state.clipFormantStatus).filter(([clipId]) => availableClipIds.has(clipId)),
    ) as Record<string, "ready" | "rebuilding" | "failed">;
    state.clipFormantAnalysis = Object.fromEntries(
        Object.entries(state.clipFormantAnalysis).filter(([clipId]) =>
            availableClipIds.has(clipId),
        ),
    );
    if (
        state.clipFormantToolWindow.clipId &&
        !availableClipIds.has(state.clipFormantToolWindow.clipId)
    ) {
        state.clipFormantToolWindow.open = false;
        state.clipFormantToolWindow.clipId = null;
    }

    state.selectedTrackId = timeline.selected_track_id;
    state.selectedClipId = timeline.selected_clip_id;
    // 与 Tempo Map 变化点一致的 BPM 范围（10-960）。
    state.bpm = clamp(Number(timeline.bpm ?? state.bpm), 10, 960);
    // 播放头归传输层（轮询 / seek / stop_audio / 显式跳转）所有。编辑命令
    // 返回的全量快照携带的 playhead_sec 是编辑未触及的旧值（播放期间停留
    // 在本次播放的起始位置），无条件采纳会让光标跳回起点——尤其在播放中
    // 编辑、或引擎等待重渲染短暂报告未播放时。只有显式 opt-in 的
    // 工程生命周期载入流程才采纳。需要移动光标的操作（如粘贴跳转
    // pasteEndSec）在本函数之后显式设置，不受影响。
    if (opts?.adoptPlayhead) {
        state.playheadSec = Math.max(0, Number(timeline.playhead_sec ?? 0));
    }
    state.projectSec = Math.max(4, Number(timeline.project_sec ?? state.projectSec));
    state.disabledGroupIds = Array.isArray(timeline.disabled_group_ids)
        ? [...timeline.disabled_group_ids]
        : [];

    const project = timeline.project as
        | {
              name?: string;
              path?: string | null;
              dirty?: boolean;
              recent?: string[];
              notes_markdown?: string;
              base_scale?: string;
              use_custom_scale?: boolean;
              custom_scale?: {
                  id?: string;
                  name?: string;
                  notes?: number[];
              } | null;
              beats_per_bar?: number;
              time_signature_denominator?: number;
              grid_size?: string;
              stretch_algorithm_override?: StretchAlgorithmOption | null;
              hifigan_mel_stretch_override?: boolean | null;
              save_undo_history?: boolean;
          }
        | undefined;
    if (project) {
        const nextBaseScaleRaw = String(project.base_scale ?? state.project.baseScale);
        const nextBaseScale = (SCALE_KEYS as readonly string[]).includes(nextBaseScaleRaw)
            ? (nextBaseScaleRaw as typeof state.project.baseScale)
            : "C";
        const nextBeatsPerBar = clamp(
            Number(project.beats_per_bar ?? state.project.beatsPerBar),
            1,
            32,
        );
        const nextTimeSignatureDenominator = (TEMPO_DENOMINATORS as readonly number[]).includes(
            Number(project.time_signature_denominator),
        )
            ? Number(project.time_signature_denominator)
            : clampDenominator(state.project.timeSignatureDenominator);
        const nextGridSizeRaw = String(project.grid_size ?? state.project.gridSize);
        const nextGridSize = VALID_GRID_SIZES.has(nextGridSizeRaw as GridSize)
            ? (nextGridSizeRaw as GridSize)
            : "1/4";

        state.project = {
            name: String(project.name ?? state.project.name ?? "Untitled"),
            path: project.path === undefined ? state.project.path : (project.path ?? null),
            dirty: Boolean(project.dirty),
            recent: Array.isArray(project.recent) ? project.recent : state.project.recent,
            // 记事本：只有**该步本身是记事本编辑**时后端才会带回文本。
            //
            // 早期实现在每次撤销/重做都用 `project.notes_markdown` 覆盖前端值，
            // 而后端只在保存时才看到记事本 —— 于是"记事本有内容时为任意操作
            // 撤销都会清空它"。现在后端在 `HistoryRecord` 上记录了每步的记事
            // 本（`None` = 与记事本无关的一步），撤销/重做据此**只在确实跨过
            // 记事本编辑时才改**。
            //
            // 不跨记事本编辑时保留前端现场值：那正是用户正在编辑的内容。
            notesMarkdown:
                // 撤销/重做时后端在载荷顶层带回**目标那一步**的记事本内容
                // （后端把它记在 HistoryRecord 上）。没有带回就绝不覆盖前端
                // 现场值 —— 早期实现一律用 `project.notes_markdown` 覆盖，
                // 而后端只在保存时看到记事本，于是"记事本有内容时为任意操作
                // 撤销都会清空它"。
                typeof timeline.notes_markdown === "string"
                    ? timeline.notes_markdown
                    : opts?.preserveProjectNotes === false
                      ? String(project.notes_markdown ?? "")
                      : state.project.notesMarkdown,
            baseScale: nextBaseScale,
            useCustomScale: Boolean(project.use_custom_scale),
            customScale: project.custom_scale
                ? sanitizeCustomScalePreset(project.custom_scale)
                : null,
            beatsPerBar: nextBeatsPerBar,
            timeSignatureDenominator: nextTimeSignatureDenominator,
            gridSize: nextGridSize,
            stretchAlgorithmOverride:
                project.stretch_algorithm_override === undefined
                    ? state.project.stretchAlgorithmOverride
                    : (project.stretch_algorithm_override ?? null),
            hifiganMelStretchOverride:
                project.hifigan_mel_stretch_override === undefined
                    ? state.project.hifiganMelStretchOverride
                    : (project.hifigan_mel_stretch_override ?? null),
            saveUndoHistory:
                project.save_undo_history === undefined
                    ? state.project.saveUndoHistory
                    : Boolean(project.save_undo_history),
        };
        state.beats = nextBeatsPerBar;
        state.grid = nextGridSize;
    }

    // Tempo Map（后端载荷始终带 tempo_map 字段；旧后端无此字段时保持现值）。
    // 在工程元数据之后解析：初始点缺失时用工程音阶物化（初始点即工程基准记录）。
    const rawTempoMap = (timeline as unknown as { tempo_map?: unknown }).tempo_map;
    if (rawTempoMap !== undefined) {
        const projectScale: ScaleLike | null =
            state.project.useCustomScale && state.project.customScale
                ? state.project.customScale.notes
                : state.project.baseScale;
        state.tempoMap = fromBackendTempoMap(rawTempoMap, state.bpm, state.beats || 4, {
            projectScale: projectScale ?? undefined,
            projectScaleName: state.project.useCustomScale
                ? (state.project.customScale?.name ?? undefined)
                : undefined,
            projectDenominator: state.project.timeSignatureDenominator,
        });
        if (state.tempoMap) {
            // 0 位置点与工程基准 BPM/拍号保持一致（后端同步保证）。
            const first = state.tempoMap.points[0];
            state.bpm = clamp(first.bpm, 10, 960);
            const firstSig = first.timeSignature ?? { numerator: 4, denominator: 4 };
            state.beats = Math.min(32, Math.max(1, Math.round(firstSig.numerator)));
            state.project.timeSignatureDenominator = clampDenominator(firstSig.denominator);
        }
    }

    // availableClipIds 已在上方 formant 剪枝处构建（同一次剪枝共用），此处直接复用。
    // 清理已删除 clip 的音高曲线数据，避免 PianoRoll 残留已删除 clip 的 detectedPitchCurve
    for (const clipId of Object.keys(state.clipPitchCurves)) {
        if (!availableClipIds.has(clipId)) {
            delete state.clipPitchCurves[clipId];
        }
    }
    // 清理已删除 clip 的多选 ID，避免删除轨道组后残留无效的 clip 引用
    if (state.multiSelectedClipIds.length > 0) {
        state.multiSelectedClipIds = state.multiSelectedClipIds.filter((id) =>
            availableClipIds.has(id),
        );
    }

    const nextPitchRanges: Record<string, { min: number; max: number }> = {};
    for (const clip of timeline.clips) {
        nextPitchRanges[clip.id] = clip.pitch_range ?? { min: -24, max: 24 };
    }
    state.clipPitchRanges = nextPitchRanges;

    // Any timeline refresh may change pitch analysis inputs and therefore param curves.
    state.paramsEpoch = (Number(state.paramsEpoch) || 0) + 1;
}

export function upsertImportedClip(
    state: SessionState,
    audioPath: string,
    meta?: {
        durationSec?: number;
        pitchRange?: { min: number; max: number };
    },
) {
    const existing = state.clips.find((clip) => clip.sourcePath === audioPath);
    if (existing) {
        state.selectedClipId = existing.id;
        if (meta?.pitchRange) {
            state.clipPitchRanges[existing.id] = meta.pitchRange;
        }
        return;
    }

    const targetTrackId = state.tracks[0]?.id ?? "track_imported";
    if (!state.tracks[0]) {
        state.tracks.push({
            id: targetTrackId,
            name: "Imported",
            muted: false,
            solo: false,
            volume: 1,

            composeEnabled: false,
            pitchAnalysisAlgo: "nsf_hifigan_onnx",
        });
    }

    const maxEndSec = state.clips.reduce(
        (maxSec, clip) => Math.max(maxSec, clip.startSec + clip.lengthSec),
        0,
    );
    const startSec = Math.max(0, Math.ceil(maxEndSec));
    const newClipId = importedClipId(audioPath);
    const lengthSec = Math.max(1, meta?.durationSec ?? 4);
    state.clips.push({
        id: newClipId,
        trackId: targetTrackId,
        name: basenameFromPath(audioPath),
        startSec,
        lengthSec,
        color: "emerald",
        sourcePath: audioPath,
        durationSec: meta?.durationSec,
        gain: 1,
        muted: false,
        sourceStartSec: 0,
        sourceEndSec: meta?.durationSec ?? lengthSec,
        playbackRate: 1,
        reversed: false,
        // 乐观创建的导入 Clip：Loop 跟随"为新的音频块启用循环"设置
        //（默认开启；后端权威载荷返回后会覆盖该值）。
        loopEnabled: state.loopNewClipsEnabled !== false,
        channelMode: 0,
        sourceChannels: undefined,
        snapOffsetSec: 0,
        fadeInSec: 0,
        fadeOutSec: 0,
        // 新建默认曲线 = 快起（REAPER Fast Start；dir 锚点 0）。
        fadeInShape: 1,
        fadeOutShape: 1,
        fadeInDir: 0,
        fadeOutDir: 0,
    });
    state.selectedClipId = newClipId;
    state.playheadSec = startSec;
    state.clipPitchRanges[newClipId] = meta?.pitchRange ?? {
        min: -24,
        max: 24,
    };
    // 导入后自动扩展工程边界
    const clipEnd = startSec + lengthSec;
    if (clipEnd > state.projectSec) {
        state.projectSec = Math.ceil(clipEnd);
    }
}
