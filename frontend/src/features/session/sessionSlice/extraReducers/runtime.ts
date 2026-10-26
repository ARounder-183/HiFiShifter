import type { TimelineState } from "../../../../types/api";
import type {
    DragDirection,
    DrawDragDirection,
    PitchSnapUnit,
    TimelineSnapSettings,
    TimeUnit,
    TimeUnitChoice,
    SilenceDetectSettings,
    ToolMode,
} from "../../sessionTypes";
import { TOOL_MODES, normalizeSplitTransitionCurve } from "../../sessionTypes";
import { normalizeSearchSettings } from "../../../search/searchSettings";
import {
    migrateLegacyMediaOnly,
    normalizeFileBrowserViewOptions,
} from "../../../fileBrowser/fileBrowserViewOptions";
import { normalizeFolderImportOptions } from "../../../fileBrowser/folderImportOptions";
import {
    clearRenderCacheRemote,
    clearWaveformCacheRemote,
    loadUiSettings,
    refreshRuntime,
} from "../../thunks/runtimeThunks";
import {
    DEFAULT_CHANNEL_IMPORT_POLICY,
    DEFAULT_PEN_INPUT_SETTINGS,
    DEFAULT_RENDER_CACHE_SETTINGS,
} from "../../../../services/api/settings";
import { SCALE_KEYS } from "../../../../utils/musicalScales";
import { normalizeParamAxisUnits } from "../../../../components/layout/pianoRoll/paramAxisUnits";
import type { CustomScalePreset } from "../../../../utils/customScales";
import { sanitizeCustomScalePreset } from "../../../../utils/customScales";
import {
    MAX_VIBRATO_PRESETS,
    isBuiltinVibratoPresetId,
    sanitizeVibratoPreset,
    BASELINE_MODES,
} from "../../../vibrato/vibratoPresets";
import type { BaselineMode, VibratoPreset } from "../../../vibrato/vibratoTypes";
import { applyTimelineState } from "../helpersTimeline";
import { clamp } from "../helpersCore";
import {
    VALID_TIME_UNITS,
    readLegacyMediaOnly,
    normalizeTimelineSnapSettings,
    createDefaultTimelineSnapSettings,
    applyToolMode,
} from "../settings";
import { setPending, setRejected } from "./shared";
import type { ActionReducerMapBuilder } from "@reduxjs/toolkit";
import type { SessionState, StretchAlgorithmOption } from "../types";

export function registerRuntime(builder: ActionReducerMapBuilder<SessionState>) {
    builder
        .addCase(refreshRuntime.pending, (state) => setPending(state, "Refreshing runtime..."))
        .addCase(refreshRuntime.fulfilled, (state, action) => {
            state.busy = false;
            if ((action.payload as { ok?: boolean }).ok) {
                const payload = action.payload as {
                    device: string;
                    model_loaded: boolean;
                    audio_loaded: boolean;
                    has_synthesized: boolean;
                    is_playing?: boolean;
                    playback_target?: string | null;
                    timeline?: TimelineState;
                    gpu_backend?: string;
                };
                const nextIsPlaying = payload.is_playing ?? false;
                if (nextIsPlaying !== state.runtime.isPlaying) {
                    // 播放状态被权威刷新翻转时推进传输纪元，使在途播放轮询
                    // 的迟到响应（采样于翻转前的另一侧）被丢弃。
                    state._transportEpoch = (Number(state._transportEpoch) || 0) + 1;
                }
                state.runtime = {
                    device: payload.device,
                    modelLoaded: payload.model_loaded,
                    audioLoaded: payload.audio_loaded,
                    hasSynthesized: payload.has_synthesized,
                    isPlaying: nextIsPlaying,
                    playbackWaitingForRender: false,
                    playbackTarget: payload.playback_target ?? null,
                    playbackPositionSec: state.runtime.playbackPositionSec,
                    playbackDurationSec: state.runtime.playbackDurationSec,
                    gpuBackend: payload.gpu_backend ?? "",
                };
                if (payload.timeline) {
                    applyTimelineState(state, payload.timeline, { force: true });
                }
                state.status = "Runtime updated";
            } else {
                state.status = "Runtime update failed";
            }
            state.lastResult = action.payload;
        })
        .addCase(refreshRuntime.rejected, setRejected)

        .addCase(clearWaveformCacheRemote.pending, (state) =>
            setPending(state, "Clearing waveform cache..."),
        )
        .addCase(clearWaveformCacheRemote.fulfilled, (state, action) => {
            state.busy = false;
            const payload = action.payload as {
                ok?: boolean;
                removed_files?: number;
                removed_bytes?: number;
                dir?: string;
            };
            if (payload.ok) {
                const n = Number(payload.removed_files ?? 0) || 0;
                state.status = `Waveform cache cleared (${n} files)`;
            } else {
                state.status = "Clear waveform cache failed";
            }
        })
        .addCase(clearWaveformCacheRemote.rejected, setRejected)

        .addCase(clearRenderCacheRemote.pending, (state) =>
            setPending(state, "Clearing render cache..."),
        )
        .addCase(clearRenderCacheRemote.fulfilled, (state, action) => {
            state.busy = false;
            const payload = action.payload as {
                ok?: boolean;
                removedFiles?: number;
                error?: string;
            };
            if (payload.ok) {
                const n = Number(payload.removedFiles ?? 0) || 0;
                state.status = `Render cache cleared (${n} entries)`;
            } else {
                state.status = "Clear render cache failed";
            }
        })
        .addCase(clearRenderCacheRemote.rejected, setRejected)

        .addCase(loadUiSettings.fulfilled, (state, action) => {
            const s = action.payload;
            state.autoCrossfadeEnabled = s.autoCrossfade;
            state.showAllTakes = Boolean(s.showAllTakes ?? true);
            state.syncEditsAcrossTakes = Boolean(s.syncEditsAcrossTakes ?? true);
            state.loopNewClipsEnabled = s.loopNewClips ?? true;
            state.splitTransitionEnabled = Boolean(s.splitTransitionEnabled ?? true);
            state.splitTransitionMode = s.splitTransitionMode === "fade" ? "fade" : "overlap";
            state.splitTransitionDurationUnit =
                s.splitTransitionDurationUnit === "percent" ? "percent" : "seconds";
            const splitDuration = Number(s.splitTransitionDurationSec);
            if (Number.isFinite(splitDuration)) {
                state.splitTransitionDurationSec = clamp(splitDuration, 0.001, 10);
            }
            const splitPercent = Number(s.splitTransitionDurationPercent);
            if (Number.isFinite(splitPercent)) {
                state.splitTransitionDurationPercent = clamp(splitPercent, 0.01, 100);
            }
            const splitCurve = s.splitTransitionCurve as string;
            // 旧命名曲线在加载期迁移为新版预设 id；未知值回退 "keep"
            //（保留原 Clip 曲线，不再改写）。
            state.splitTransitionCurve = normalizeSplitTransitionCurve(splitCurve);
            state.splitTransitionOverlapCrossfade =
                s.splitTransitionOverlapCrossfade === "always" ? "always" : "auto";
            const loadedSnapEnabled = s.snapEnabled ?? s.gridSnap ?? true;
            state.snapEnabled = loadedSnapEnabled;
            if (s.timelineSnap && typeof s.timelineSnap === "object") {
                state.timelineSnap = normalizeTimelineSnapSettings(
                    createDefaultTimelineSnapSettings(),
                    s.timelineSnap as Partial<TimelineSnapSettings>,
                );
                state.timelineSnap.enabled = Boolean(loadedSnapEnabled);
                state.snapEnabled = state.timelineSnap.enabled;
            }
            if (s.tempoMapVisible != null) {
                state.tempoMapVisible = Boolean(s.tempoMapVisible);
            }
            const loadedPrimaryUnit = s.primaryTimeUnit as unknown;
            if (VALID_TIME_UNITS.has(loadedPrimaryUnit as TimeUnit)) {
                state.primaryTimeUnit = loadedPrimaryUnit as TimeUnit;
            }
            const loadedSecondaryUnit = s.secondaryTimeUnit as unknown;
            if (
                loadedSecondaryUnit === "none" ||
                VALID_TIME_UNITS.has(loadedSecondaryUnit as TimeUnit)
            ) {
                state.secondaryTimeUnit = loadedSecondaryUnit as TimeUnitChoice;
            }
            const loadedSpacing = Number(s.rulerLabelSpacingPx);
            if (Number.isFinite(loadedSpacing)) {
                state.rulerLabelSpacingPx = Math.max(40, Math.min(320, Math.round(loadedSpacing)));
            }
            if (s.showPlayheadTimeInTrackHeader != null) {
                state.showPlayheadTimeInTrackHeader = Boolean(s.showPlayheadTimeInTrackHeader);
            }
            if (s.paramEditorSyncTimeline != null) {
                state.paramEditorSyncTimeline = Boolean(s.paramEditorSyncTimeline);
            }
            state.pitchSnapEnabled = s.pitchSnap;
            // Validate pitchSnapUnit
            const validUnits: PitchSnapUnit[] = ["semitone", "scale"];
            state.pitchSnapUnit = validUnits.includes(s.pitchSnapUnit as PitchSnapUnit)
                ? (s.pitchSnapUnit as PitchSnapUnit)
                : "semitone";
            // Validate pitchSnapScale
            state.pitchSnapScale = (SCALE_KEYS as readonly string[]).includes(
                s.pitchSnapScale ?? "C",
            )
                ? ((s.pitchSnapScale ?? "C") as typeof state.pitchSnapScale)
                : "C";
            // Load pitch snap tolerance (cents) if present in saved settings
            if (s.pitchSnapToleranceCents != null) {
                state.pitchSnapToleranceCents = clamp(
                    Number(s.pitchSnapToleranceCents) || 0,
                    0,
                    1000,
                );
            }
            state.playheadZoomEnabled = s.playheadZoom;
            if (s.autoScroll != null) state.autoScrollEnabled = s.autoScroll;
            if (s.paramEditorSeekPlayhead != null)
                state.paramEditorSeekPlayheadEnabled = Boolean(s.paramEditorSeekPlayhead);
            if (s.paramEditorTimelineClickSelectTrack != null) {
                state.paramEditorTimelineClickSelectTrackEnabled = Boolean(
                    s.paramEditorTimelineClickSelectTrack,
                );
            }
            if (s.showClipboardPreview != null) state.showClipboardPreview = s.showClipboardPreview;
            if (s.showParamValuePopup != null)
                state.showParamValuePopup = Boolean(s.showParamValuePopup);
            // 纵轴单位映射：手改坏配置（未知参数 / 非法取值）在这里被过滤掉。
            state.paramAxisUnits = normalizeParamAxisUnits(s.paramAxisUnits);
            if (s.scaleHighlightMode != null)
                state.scaleHighlightMode = s.scaleHighlightMode === "always" ? "always" : "off";
            if (s.ignoreGrouping != null) state.ignoreGrouping = Boolean(s.ignoreGrouping);
            const loadedRippleMode = s.rippleMode;
            if (loadedRippleMode === "track" || loadedRippleMode === "all") {
                state.rippleMode = loadedRippleMode;
            } else {
                state.rippleMode = "off";
            }
            if (s.lockParamLines != null) state.lockParamLinesEnabled = Boolean(s.lockParamLines);
            if (s.metronomeEnabled != null) state.metronomeEnabled = Boolean(s.metronomeEnabled);
            if (s.metronomeGain != null) {
                const gain = Number(s.metronomeGain);
                if (Number.isFinite(gain)) state.metronomeGain = Math.min(1, Math.max(0, gain));
            }
            if (s.metronomeMode != null) {
                state.metronomeMode =
                    s.metronomeMode === "beat" || s.metronomeMode === "bar"
                        ? s.metronomeMode
                        : "grid";
            }
            if (s.metronomeAccent != null) state.metronomeAccent = Boolean(s.metronomeAccent);
            if (s.metronomeSound != null) {
                state.metronomeSound =
                    s.metronomeSound === "woodblock" || s.metronomeSound === "beep"
                        ? s.metronomeSound
                        : "click";
            }
            if (s.silenceDetectOptions != null) {
                const o = s.silenceDetectOptions as Partial<SilenceDetectSettings>;
                state.silenceDetectOptions = { ...state.silenceDetectOptions, ...o };
            }
            if (s.search != null) state.searchSettings = normalizeSearchSettings(s.search);
            /*
             * 视图选项：先补一次旧 localStorage 的迁移，再归一化。
             *
             * 迁移只在配置里**没有** `mediaOnly` 时生效（见
             * `migrateLegacyMediaOnly`）—— 否则用户改过新设置后，旧键还在，
             * 下次启动又被打回旧值。
             */
            state.fileBrowserView = normalizeFileBrowserViewOptions(
                migrateLegacyMediaOnly(s.fileBrowser, readLegacyMediaOnly()),
            );
            state.folderImportOptions = normalizeFolderImportOptions(s.folderImport);
            if (Array.isArray(s.fileBrowserFavorites)) {
                state.fileBrowserFavorites = s.fileBrowserFavorites
                    .filter((path: unknown): path is string => typeof path === "string")
                    .map((path: string) => path.trim())
                    .filter((path: string) => path.length > 0);
            }
            if (s.quickSearchAutoNormalize != null)
                state.quickSearchAutoNormalizeEnabled = Boolean(s.quickSearchAutoNormalize);
            if (s.saveUndoHistoryByDefault != null)
                state.saveUndoHistoryByDefault = Boolean(s.saveUndoHistoryByDefault);
            if (Array.isArray(s.visibleReferenceRootTrackIds)) {
                state.visibleReferenceRootTrackIds = s.visibleReferenceRootTrackIds
                    .filter((id: unknown): id is string => typeof id === "string")
                    .map((id: string) => id.trim())
                    .filter((id: string) => id.length > 0);
            }
            const defaultStretchAlgorithm = s.defaultStretchAlgorithm;
            if (
                defaultStretchAlgorithm != null &&
                ["linear", "signalsmith", "soundtouch"].includes(defaultStretchAlgorithm)
            ) {
                state.defaultStretchAlgorithm = defaultStretchAlgorithm as StretchAlgorithmOption;
            }
            if (s.defaultHifiganMelStretch != null) {
                state.defaultHifiganMelStretch = Boolean(s.defaultHifiganMelStretch);
            }
            if (s.ortEp != null) {
                const normalized = String(s.ortEp).toLowerCase();
                state.ortEp = ["auto", "cpu", "gpu"].includes(normalized) ? normalized : "auto";
            }
            if (s.ortDeviceKey !== undefined) {
                const val = s.ortDeviceKey;
                state.ortDeviceKey = val == null ? null : String(val);
            }
            if (s.autoBackgroundRender != null) {
                state.autoBackgroundRender = Boolean(s.autoBackgroundRender);
            }
            if (s.renderCache) {
                state.renderCache = {
                    ...DEFAULT_RENDER_CACHE_SETTINGS,
                    ...s.renderCache,
                };
            }
            if (s.channelImportPolicy) {
                state.channelImportPolicy = {
                    ...DEFAULT_CHANNEL_IMPORT_POLICY,
                    ...s.channelImportPolicy,
                };
            }
            if (s.penInput) {
                state.penInput = {
                    ...DEFAULT_PEN_INPUT_SETTINGS,
                    ...s.penInput,
                };
            }
            const selectDir = s.selectDragDirection;
            if (selectDir != null && ["free", "x-only", "y-only"].includes(selectDir)) {
                state.selectDragDirection = selectDir as DragDirection;
            }
            const drawDir = s.drawDragDirection;
            if (drawDir != null && ["free", "x-only"].includes(drawDir)) {
                state.drawDragDirection = drawDir as DrawDragDirection;
            }
            const lineVibratoDir = s.lineVibratoDragDirection;
            if (lineVibratoDir != null && ["free", "x-only"].includes(lineVibratoDir)) {
                state.lineVibratoDragDirection = lineVibratoDir as DrawDragDirection;
            }
            const smoothness = s.smoothnessPercent ?? s.edgeSmoothnessPercent;
            if (smoothness != null) {
                state.edgeSmoothnessPercent = clamp(Number(smoothness) || 0, 0, 100);
            }
            if (Array.isArray(s.customScalePresets)) {
                state.customScalePresets = s.customScalePresets.map((preset: unknown) =>
                    sanitizeCustomScalePreset(preset as Partial<CustomScalePreset>),
                );
            }
            // 逐项规整：手改过的配置不能让内核拿到越界或缺失的字段。
            if (Array.isArray(s.vibratoPresets)) {
                state.vibratoPresets = s.vibratoPresets
                    .map((preset: unknown) => sanitizeVibratoPreset(preset as VibratoPreset))
                    .filter((preset) => !isBuiltinVibratoPresetId(preset.id))
                    .slice(0, MAX_VIBRATO_PRESETS);
            }
            if (typeof s.activeVibratoPresetId === "string" && s.activeVibratoPresetId) {
                state.activeVibratoPresetId = s.activeVibratoPresetId;
            }
            // 摆放方式：只认已知取值（手改配置里的野值一律忽略，回落默认）。
            if (
                typeof s.vibratoBaseline === "string" &&
                (BASELINE_MODES as readonly string[]).includes(s.vibratoBaseline)
            ) {
                state.vibratoBaseline = s.vibratoBaseline as BaselineMode;
            }
            // 停用名单：只收字符串，去重；未知 id 留着无害（过滤时按 id 比对）。
            if (Array.isArray(s.disabledVibratoPresetIds)) {
                state.disabledVibratoPresetIds = [
                    ...new Set(
                        s.disabledVibratoPresetIds.filter(
                            (id: unknown): id is string => typeof id === "string" && id.length > 0,
                        ),
                    ),
                ];
            }
            // 系统预设的自定义顺序：同样只收字符串；缺项 / 无效项由
            // `effectiveBuiltinPresetOrder` 兜底，这里不必校验成员合法性。
            if (Array.isArray(s.builtinVibratoPresetOrder)) {
                state.builtinVibratoPresetOrder = s.builtinVibratoPresetOrder.filter(
                    (id: unknown): id is string => typeof id === "string" && id.length > 0,
                );
            }
            // 上次使用的工具：只认已知取值（手改配置里的野值一律忽略，回落默认）。
            // 放在最后：`applyToolMode` 会一并派生 `toolModeGroup` / `drawToolMode`，
            // 后面不该再有分支去改这三个字段。
            if (
                typeof s.paramEditorTool === "string" &&
                (TOOL_MODES as readonly string[]).includes(s.paramEditorTool)
            ) {
                applyToolMode(state, s.paramEditorTool as ToolMode);
            }
        });
}
