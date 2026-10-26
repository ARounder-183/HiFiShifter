import type { TimelineState } from "../../../../types/api";
import type { GridSize } from "../../sessionTypes";
import {
    saveProjectAsRemote,
    saveProjectRemote,
    saveProjectToPathRemote,
    setProjectSaveUndoHistoryRemote,
    setProjectBaseScaleRemote,
    setProjectCustomScaleRemote,
    setProjectStretchSettingsRemote,
    setProjectTimelineSettingsRemote,
} from "../../thunks/projectThunks";
import { SCALE_KEYS } from "../../../../utils/musicalScales";
import type { ScaleLike } from "../../../../utils/musicalScales";
import { sanitizeCustomScalePreset } from "../../../../utils/customScales";
import type { TempoMap } from "../../../../utils/tempoMap";
import {
    clampDenominator,
    isSameTempoMap,
    scaleLikeToScaleData,
    TEMPO_DENOMINATORS,
} from "../../../../utils/tempoMap";
import { setTempoMapRemote } from "../../thunks/tempoMapThunks";
import { markProjectDirty } from "../../sessionDirtyState";
import { applyTimelineState } from "../helpersTimeline";
import { restoreClipPitchCurves, clamp } from "../helpersCore";
import { VALID_GRID_SIZES } from "../settings";
import { setRejected } from "./shared";
import type { ActionReducerMapBuilder } from "@reduxjs/toolkit";
import type { SessionState, StretchAlgorithmOption } from "../types";

export function registerProject(builder: ActionReducerMapBuilder<SessionState>) {
    builder
        .addCase(saveProjectRemote.fulfilled, (state, action) => {
            const payload = action.payload as {
                ok?: boolean;
                canceled?: boolean;
                versionConflict?: boolean;
                path?: string;
                existingVersion?: number;
                currentVersion?: number;
                existingIsNewer?: boolean;
                timeline?: TimelineState & { ok?: boolean };
                tracks?: unknown;
                clips?: unknown;
            };
            if (payload?.versionConflict) {
                state.saveVersionConflictDialog = {
                    path: payload.path as string,
                    existingVersion: Number(payload.existingVersion ?? 0),
                    currentVersion: Number(payload.currentVersion ?? 0),
                    existingIsNewer: Boolean(payload.existingIsNewer),
                };
                state.status = "Save version confirmation required";
                return;
            }
            if (payload?.ok && payload?.canceled) {
                state.status = "Save canceled";
                return;
            }

            // 保留播放头位置和音高曲线，避免 UI 跳变
            const currentPlayheadSec = state.playheadSec;
            const currentParamsEpoch = state.paramsEpoch;
            // 浅拷贝快照：draft 代理会被 applyTimelineState 的 prune 透过修改（见 applyTimelineStatePreservingPitchVisuals）
            const currentClipPitchCurves = { ...state.clipPitchCurves };

            if (payload?.ok && payload?.timeline?.ok) {
                applyTimelineState(state, payload.timeline as TimelineState, { force: true });
                state.playheadSec = currentPlayheadSec;
                state.paramsEpoch = currentParamsEpoch;
                restoreClipPitchCurves(state, currentClipPitchCurves);
                state.status = "Project saved";
                return;
            }

            if (payload?.ok && payload?.tracks && payload?.clips) {
                applyTimelineState(state, payload as TimelineState, { force: true });
                state.playheadSec = currentPlayheadSec;
                state.paramsEpoch = currentParamsEpoch;
                restoreClipPitchCurves(state, currentClipPitchCurves);
                state.status = "Project saved";
                return;
            }

            if (payload?.ok) {
                state.project.dirty = false;
                state.status = "Project saved";
                return;
            }

            state.status = "Save failed";
        })

        .addCase(saveProjectAsRemote.fulfilled, (state, action) => {
            const payload = action.payload as {
                ok?: boolean;
                canceled?: boolean;
                versionConflict?: boolean;
                path?: string;
                existingVersion?: number;
                currentVersion?: number;
                existingIsNewer?: boolean;
                timeline?: TimelineState & { ok?: boolean };
                tracks?: unknown;
                clips?: unknown;
            };
            if (payload?.versionConflict) {
                state.saveVersionConflictDialog = {
                    path: payload.path as string,
                    existingVersion: Number(payload.existingVersion ?? 0),
                    currentVersion: Number(payload.currentVersion ?? 0),
                    existingIsNewer: Boolean(payload.existingIsNewer),
                };
                state.status = "Save version confirmation required";
                return;
            }
            if (payload?.ok && payload?.canceled) {
                state.status = "Save As canceled";
                return;
            }

            // 保留播放头位置和音高曲线，避免 UI 跳变
            const currentPlayheadSec = state.playheadSec;
            const currentParamsEpoch = state.paramsEpoch;
            // 浅拷贝快照：draft 代理会被 applyTimelineState 的 prune 透过修改（见 applyTimelineStatePreservingPitchVisuals）
            const currentClipPitchCurves = { ...state.clipPitchCurves };

            if (payload?.ok && payload?.timeline?.ok) {
                applyTimelineState(state, payload.timeline as TimelineState, { force: true });
                state.playheadSec = currentPlayheadSec;
                state.paramsEpoch = currentParamsEpoch;
                restoreClipPitchCurves(state, currentClipPitchCurves);
                state.status = "Project saved";
                return;
            }

            if (payload?.ok && payload?.tracks && payload?.clips) {
                applyTimelineState(state, payload as TimelineState, { force: true });
                state.playheadSec = currentPlayheadSec;
                state.paramsEpoch = currentParamsEpoch;
                restoreClipPitchCurves(state, currentClipPitchCurves);
                state.status = "Project saved";
                return;
            }

            if (payload?.ok) {
                state.project.dirty = false;
                state.status = "Project saved";
                return;
            }

            state.status = "Save As failed";
        })

        .addCase(saveProjectToPathRemote.fulfilled, (state, action) => {
            const payload = action.payload as {
                ok?: boolean;
                canceled?: boolean;
                versionConflict?: boolean;
                path?: string;
                existingVersion?: number;
                currentVersion?: number;
                existingIsNewer?: boolean;
                timeline?: TimelineState & { ok?: boolean };
                tracks?: unknown;
                clips?: unknown;
            };
            if (payload?.versionConflict) {
                // 理论上 force 保存不会再冲突；兜底再次弹出确认框。
                state.saveVersionConflictDialog = {
                    path: payload.path as string,
                    existingVersion: Number(payload.existingVersion ?? 0),
                    currentVersion: Number(payload.currentVersion ?? 0),
                    existingIsNewer: Boolean(payload.existingIsNewer),
                };
                state.status = "Save version confirmation required";
                return;
            }
            if (payload?.ok && payload?.canceled) {
                state.status = "Save canceled";
                return;
            }

            // 保留播放头位置和音高曲线，避免 UI 跳变
            const currentPlayheadSec = state.playheadSec;
            const currentParamsEpoch = state.paramsEpoch;
            // 浅拷贝快照：draft 代理会被 applyTimelineState 的 prune 透过修改（见 applyTimelineStatePreservingPitchVisuals）
            const currentClipPitchCurves = { ...state.clipPitchCurves };

            if (payload?.ok && payload?.timeline?.ok) {
                applyTimelineState(state, payload.timeline as TimelineState, { force: true });
                state.playheadSec = currentPlayheadSec;
                state.paramsEpoch = currentParamsEpoch;
                restoreClipPitchCurves(state, currentClipPitchCurves);
                state.status = "Project saved";
                return;
            }

            if (payload?.ok && payload?.tracks && payload?.clips) {
                applyTimelineState(state, payload as TimelineState, { force: true });
                state.playheadSec = currentPlayheadSec;
                state.paramsEpoch = currentParamsEpoch;
                restoreClipPitchCurves(state, currentClipPitchCurves);
                state.status = "Project saved";
                return;
            }

            if (payload?.ok) {
                state.project.dirty = false;
                state.status = "Project saved";
                return;
            }

            state.status = "Save failed";
        })
        .addCase(saveProjectToPathRemote.rejected, (state) => {
            state.status = "Save failed";
        })

        .addCase(setProjectSaveUndoHistoryRemote.fulfilled, (state, action) => {
            const payload = action.payload as {
                ok?: boolean;
                project?: { save_undo_history?: boolean; dirty?: boolean };
            };
            if (!payload.ok || !payload.project) return;
            if (payload.project.save_undo_history != null) {
                state.project.saveUndoHistory = Boolean(payload.project.save_undo_history);
            }
            if (payload.project.dirty != null) {
                state.project.dirty = Boolean(payload.project.dirty);
            }
        })
        .addCase(setProjectSaveUndoHistoryRemote.rejected, setRejected)

        .addCase(setProjectBaseScaleRemote.fulfilled, (state, action) => {
            const payload = action.payload as {
                ok?: boolean;
                project?: {
                    base_scale?: string;
                    use_custom_scale?: boolean;
                    custom_scale?: {
                        id?: string;
                        name?: string;
                        notes?: number[];
                    } | null;
                    dirty?: boolean;
                };
            };
            if (!payload.ok) {
                return;
            }
            const next = payload.project?.base_scale;
            if (next && (SCALE_KEYS as readonly string[]).includes(next)) {
                state.project.baseScale = next as typeof state.project.baseScale;
            }
            if (typeof payload.project?.use_custom_scale === "boolean") {
                state.project.useCustomScale = payload.project.use_custom_scale;
            }
            if (payload.project?.custom_scale != null) {
                state.project.customScale = payload.project.custom_scale
                    ? sanitizeCustomScalePreset(payload.project.custom_scale)
                    : null;
            }
            if (typeof payload.project?.dirty === "boolean") {
                state.project.dirty = payload.project.dirty;
            }
            // 后端会把工程音阶同步到 Tempo Map 初始点（初始点即工程基准记录）：
            // 前端镜像该同步，避免 effectiveScaleAtSec 等 UI 计算使用过期音阶
            // （钢琴卷帘高亮、音高吸附会与音频渲染不一致）。
            if (state.tempoMap && state.tempoMap.points.length > 0) {
                const projectScale: ScaleLike | null =
                    state.project.useCustomScale && state.project.customScale
                        ? state.project.customScale.notes
                        : state.project.baseScale;
                state.tempoMap = {
                    ...state.tempoMap,
                    points: [
                        {
                            ...state.tempoMap.points[0],
                            scale: scaleLikeToScaleData(
                                projectScale ?? undefined,
                                state.project.useCustomScale
                                    ? (state.project.customScale?.name ?? undefined)
                                    : undefined,
                            ),
                        },
                        ...state.tempoMap.points.slice(1),
                    ],
                };
            }
            // 工程音阶变化会影响子轨道“度数差”等依赖音阶的渲染，触发参数曲线/渲染缓存失效。
            state.paramsEpoch = (Number(state.paramsEpoch) || 0) + 1;
        })

        .addCase(setProjectCustomScaleRemote.fulfilled, (state, action) => {
            const payload = action.payload as {
                ok?: boolean;
                project?: {
                    use_custom_scale?: boolean;
                    custom_scale?: {
                        id?: string;
                        name?: string;
                        notes?: number[];
                    } | null;
                    dirty?: boolean;
                };
            };
            if (!payload.ok) return;
            if (typeof payload.project?.use_custom_scale === "boolean") {
                state.project.useCustomScale = payload.project.use_custom_scale;
            }
            if (payload.project?.custom_scale != null) {
                state.project.customScale = payload.project.custom_scale
                    ? sanitizeCustomScalePreset(payload.project.custom_scale)
                    : null;
            }
            if (typeof payload.project?.dirty === "boolean") {
                state.project.dirty = payload.project.dirty;
            }
            // 与 setProjectBaseScaleRemote 一致：镜像后端对 Tempo Map 初始点
            // 音阶的同步（初始点即工程基准记录）。
            if (state.tempoMap && state.tempoMap.points.length > 0) {
                const projectScale: ScaleLike | null =
                    state.project.useCustomScale && state.project.customScale
                        ? state.project.customScale.notes
                        : state.project.baseScale;
                state.tempoMap = {
                    ...state.tempoMap,
                    points: [
                        {
                            ...state.tempoMap.points[0],
                            scale: scaleLikeToScaleData(
                                projectScale ?? undefined,
                                state.project.useCustomScale
                                    ? (state.project.customScale?.name ?? undefined)
                                    : undefined,
                            ),
                        },
                        ...state.tempoMap.points.slice(1),
                    ],
                };
            }
            // 工程音阶变化会影响子轨道“度数差”等依赖音阶的渲染，触发参数曲线/渲染缓存失效。
            state.paramsEpoch = (Number(state.paramsEpoch) || 0) + 1;
        })

        .addCase(setProjectTimelineSettingsRemote.fulfilled, (state, action) => {
            const payload = action.payload as {
                ok?: boolean;
                project?: {
                    beats_per_bar?: number;
                    time_signature_denominator?: number;
                    grid_size?: string;
                    dirty?: boolean;
                };
            };
            if (!payload.ok) {
                return;
            }
            const beats = clamp(Number(payload.project?.beats_per_bar ?? state.beats), 1, 32);
            const denominator = (TEMPO_DENOMINATORS as readonly number[]).includes(
                Number(payload.project?.time_signature_denominator),
            )
                ? Number(payload.project?.time_signature_denominator)
                : clampDenominator(state.project.timeSignatureDenominator);
            const gridRaw = String(payload.project?.grid_size ?? state.grid);
            const valid = VALID_GRID_SIZES.has(gridRaw as GridSize);
            const grid = (valid ? gridRaw : "1/4") as GridSize;

            state.beats = beats;
            state.grid = grid;
            state.project.beatsPerBar = beats;
            state.project.timeSignatureDenominator = denominator;
            state.project.gridSize = grid;
            if (typeof payload.project?.dirty === "boolean") {
                state.project.dirty = payload.project.dirty;
            }
        })

        .addCase(setTempoMapRemote.fulfilled, (state, action) => {
            const payload = action.payload as {
                ok?: boolean;
                project?: { dirty?: boolean };
            } & TimelineState;
            if (!payload.ok || !payload.tracks || !payload.clips) {
                return;
            }
            // 【迟到的回声必须整份忽略】与 BPM 路径同一类问题：拖拽 / 滚轮手势
            // 期间请求会积压，旧地图的回声在乐观值前进之后才到达。若照单应用，
            // 网格会被反复拽回旧锚点（`Tempo Map 变化点` 重塑网格，锚点一变整条
            // 网格就平移），与用户报告的"抽搐"完全一致。判据：**当前地图仍等于
            // 本次请求送出的地图**（没有被更新的写入覆盖）才采纳。
            const requestedMap = action.meta.arg as TempoMap | null;
            if (!isSameTempoMap(state.tempoMap, requestedMap)) {
                return;
            }
            // 后端为权威来源：应用完整快照（含 tempo_map 与工程基准值）。
            //
            // 【幂等回声不产生新引用】快照会用一个**新的** tempoMap 对象替换引用，
            // 而标尺刻度的 memo 依赖该引用：滚轮 / 拖拽这类连续手势里，每一次
            // 提交都会因此多出一次全量重算（标尺"抽搐"的放大器之一）。后端没有
            // 改动地图时（本次提交的内容就是它已有的），保留旧引用即可 ——
            // 其余字段照常应用，不牺牲任何同步语义。
            const previousTempoMap = state.tempoMap;
            applyTimelineState(state, payload, { force: true });
            const tempoMapChanged =
                !previousTempoMap ||
                !state.tempoMap ||
                !isSameTempoMap(previousTempoMap, state.tempoMap);
            if (!tempoMapChanged && previousTempoMap) {
                state.tempoMap = previousTempoMap;
            }
            // 显式触发后台预渲染：Tempo Map 音阶变化会影响子轨道“度数差”等
            // 依赖音阶的渲染。applyTimelineState 已使 paramsEpoch 递增，
            // App 层据此调用 startBackgroundRender（与工程音阶变更路径
            // setProjectBaseScaleRemote.fulfilled 保持一致）；此处再显式递增，
            // 确保该触发不依赖 applyTimelineState 的内部实现细节。
            //
            // 地图未变（纯回声）时不再递增：没有任何音阶变化需要重渲染。
            if (tempoMapChanged) {
                state.paramsEpoch = (Number(state.paramsEpoch) || 0) + 1;
            }
            /*
             * 【为什么必须在**接受分支**里标脏，而不是用 `tempoMapChanged`】
             * BPM / 拍号的编辑是乐观更新：提交前本地已 `setTempoMap(nextMap)`，
             * 后端回声与本地完全一致 —— 上面的"纯回声"判据在此恒为 false，
             * 拿它当脏标记条件等于永不标脏（这正是本处曾经的缺口：改 BPM 后
             * "新建工程"不询问、静默丢弃）。后端若在响应里回报 `project.dirty`
             * 就以它为准（与基础音阶/时间线设置那几条一致），否则视为已修改。
             */
            if (typeof payload.project?.dirty === "boolean") {
                state.project.dirty = payload.project.dirty;
            } else {
                markProjectDirty(state.project);
            }
            state.status = "Tempo map updated";
        })
        .addCase(setTempoMapRemote.rejected, setRejected)

        .addCase(setProjectStretchSettingsRemote.fulfilled, (state, action) => {
            const payload = action.payload as {
                ok?: boolean;
                project?: {
                    stretch_algorithm_override?: StretchAlgorithmOption | null;
                    hifigan_mel_stretch_override?: boolean | null;
                    dirty?: boolean;
                };
            };
            if (!payload.ok) {
                return;
            }
            if (payload.project?.stretch_algorithm_override !== undefined) {
                state.project.stretchAlgorithmOverride =
                    payload.project.stretch_algorithm_override ?? null;
            }
            if (payload.project?.hifigan_mel_stretch_override !== undefined) {
                state.project.hifiganMelStretchOverride =
                    payload.project.hifigan_mel_stretch_override ?? null;
            }
            if (typeof payload.project?.dirty === "boolean") {
                state.project.dirty = payload.project.dirty;
            }
        });
}
