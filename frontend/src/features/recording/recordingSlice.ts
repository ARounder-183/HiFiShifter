import { createAsyncThunk, createSlice, type PayloadAction } from "@reduxjs/toolkit";
import type { RootState } from "../../app/store";
import {
    DEFAULT_RECORDING_SETTINGS,
    type RecordingDeviceInfo,
    type RecordingFinishedInfo,
    type RecordingMeterPayload,
    type RecordingSettings,
} from "../../services/api/recording";
import { webApi } from "../../services/webviewApi";
import {
    playOriginal,
    stopAudioPlayback,
} from "../session/thunks/transportThunks";
import { applyTimelinePayload } from "../session/sessionSlice";

const delay = (ms: number) => new Promise<void>((resolve) => window.setTimeout(resolve, ms));

interface RecordingSliceState {
    settings: RecordingSettings;
    settingsLoaded: boolean;
    devices: RecordingDeviceInfo[];
    devicesLoaded: boolean;
    active: boolean;
    busy: boolean;
    countdownSec: number;
    countdownRemaining: number;
    countdownCancelRequested: boolean;
    elapsedSec: number;
    level: number;
    peak: number;
    outputPath: string | null;
    startSec: number | null;
    error: string | null;
    lastResult: RecordingFinishedInfo | null;
}

const initialState: RecordingSliceState = {
    settings: DEFAULT_RECORDING_SETTINGS,
    settingsLoaded: false,
    devices: [],
    devicesLoaded: false,
    active: false,
    busy: false,
    countdownSec: 0,
    countdownRemaining: 0,
    countdownCancelRequested: false,
    elapsedSec: 0,
    level: 0,
    peak: 0,
    outputPath: null,
    startSec: null,
    error: null,
    lastResult: null,
};

export const loadRecordingSettings = createAsyncThunk(
    "recording/loadSettings",
    async (_, { dispatch, getState }) => {
        const current = (getState() as RootState).recording;
        if (current.settingsLoaded) return current.settings;
        const settings = await webApi.getRecordingSettings();
        dispatch(recordingSlice.actions.settingsLoaded(settings));
        return settings;
    },
);

export const saveRecordingSettings = createAsyncThunk(
    "recording/saveSettings",
    async (settings: RecordingSettings, { dispatch, rejectWithValue }) => {
        const result = await webApi.saveRecordingSettings(settings);
        if (!result.ok || !result.settings) {
            return rejectWithValue("recording_error_save_settings");
        }
        dispatch(recordingSlice.actions.settingsLoaded(result.settings));
        return result.settings;
    },
);

export const loadRecordingDevices = createAsyncThunk(
    "recording/loadDevices",
    async (_, { dispatch, getState }) => {
        const current = (getState() as RootState).recording;
        if (current.devicesLoaded && current.devices.length > 0) {
            return current.devices;
        }
        const result = await webApi.getRecordingDevices();
        const devices = result.devices ?? [];
        dispatch(recordingSlice.actions.devicesLoaded(devices));
        return devices;
    },
);

export const startRecordingFlow = createAsyncThunk(
    "recording/startFlow",
    async (_, { dispatch, getState, rejectWithValue }) => {
        const current = (getState() as RootState).recording;
        if (current.active || current.busy) {
            return rejectWithValue("recording_error_already_active");
        }

        dispatch(recordingSlice.actions.setBusy(true));
        dispatch(recordingSlice.actions.clearError());
        try {
            const settings = current.settings;
            if (settings.countdownSec > 0) {
                dispatch(recordingSlice.actions.beginCountdown(settings.countdownSec));
                for (let remaining = settings.countdownSec; remaining > 0; remaining -= 1) {
                    await delay(1000);
                    const afterTick = (getState() as RootState).recording;
                    if (afterTick.countdownCancelRequested) {
                        dispatch(recordingSlice.actions.cancelCountdown());
                        return { ok: true as const, cancelled: true as const };
                    }
                    dispatch(recordingSlice.actions.tickCountdown());
                }
            }

            const session = (getState() as RootState).session;
            const startSec = Number(session.playheadSec ?? 0);
            const result = await webApi.startRecording(startSec);
            if (!result.ok) {
                const error = result.error ?? "recording_error_start_failed";
                dispatch(recordingSlice.actions.recordingFailed(error));
                return rejectWithValue(error);
            }

            dispatch(
                recordingSlice.actions.recordingStarted({
                    startSec: Number(result.startSec ?? startSec),
                    outputPath: result.outputPath ?? null,
                }),
            );

            // 录音与时间轴播放同时开始：从当前播放光标处播放背景音乐 / 伴奏。
            await dispatch(playOriginal()).unwrap();
            return { ok: true as const, cancelled: false as const };
        } catch (err) {
            const message =
                err instanceof Error && err.message ? err.message : "recording_error_start_failed";
            dispatch(recordingSlice.actions.recordingFailed(message));
            return rejectWithValue(message);
        } finally {
            dispatch(recordingSlice.actions.setBusy(false));
        }
    },
);

export const stopRecordingFlow = createAsyncThunk(
    "recording/stopFlow",
    async (_, { dispatch, getState, rejectWithValue }) => {
        const current = (getState() as RootState).recording;
        if (!current.active) {
            return rejectWithValue("recording_error_not_active");
        }
        dispatch(recordingSlice.actions.setBusy(true));
        try {
            const result = await webApi.stopRecording();
            if (!result.ok) {
                const error = result.error ?? "recording_error_stop_failed";
                dispatch(recordingSlice.actions.recordingFailed(error));
                return rejectWithValue(error);
            }
            if (result.timeline) {
                dispatch(applyTimelinePayload(result.timeline));
            }
            dispatch(recordingSlice.actions.recordingStopped(result.recording ?? null));
            // 录音结束后时间轴播放已停止，同步前端播放状态。
            void dispatch(stopAudioPlayback());
            return result;
        } catch (err) {
            const message =
                err instanceof Error && err.message ? err.message : "recording_error_stop_failed";
            dispatch(recordingSlice.actions.recordingFailed(message));
            // 后端 stop_recording 一旦被调用，会话即已结束；即使导入/封口失败也清除前端录音态。
            dispatch(recordingSlice.actions.recordingStopped(null));
            return rejectWithValue(message);
        } finally {
            dispatch(recordingSlice.actions.setBusy(false));
        }
    },
);

export const cancelRecordingCountdown = createAsyncThunk(
    "recording/cancelCountdown",
    async (_, { dispatch }) => {
        dispatch(recordingSlice.actions.requestCancelCountdown());
    },
);

const recordingSlice = createSlice({
    name: "recording",
    initialState,
    reducers: {
        settingsLoaded(state, action: PayloadAction<RecordingSettings>) {
            state.settings = action.payload;
            state.settingsLoaded = true;
        },
        devicesLoaded(state, action: PayloadAction<RecordingDeviceInfo[]>) {
            state.devices = action.payload;
            state.devicesLoaded = true;
        },
        setBusy(state, action: PayloadAction<boolean>) {
            state.busy = action.payload;
        },
        clearError(state) {
            state.error = null;
        },
        recordingFailed(state, action: PayloadAction<string>) {
            state.error = action.payload;
            state.countdownRemaining = 0;
        },
        beginCountdown(state, action: PayloadAction<number>) {
            state.countdownSec = Math.max(1, Math.floor(action.payload));
            state.countdownRemaining = state.countdownSec;
            state.countdownCancelRequested = false;
        },
        tickCountdown(state) {
            state.countdownRemaining = Math.max(0, state.countdownRemaining - 1);
        },
        requestCancelCountdown(state) {
            state.countdownCancelRequested = true;
        },
        cancelCountdown(state) {
            state.countdownCancelRequested = false;
            state.countdownRemaining = 0;
        },
        recordingStarted(
            state,
            action: PayloadAction<{ startSec: number; outputPath: string | null }>,
        ) {
            state.active = true;
            state.startSec = action.payload.startSec;
            state.outputPath = action.payload.outputPath;
            state.elapsedSec = 0;
            state.level = 0;
            state.peak = 0;
            state.error = null;
            state.countdownRemaining = 0;
        },
        updateMeter(state, action: PayloadAction<RecordingMeterPayload>) {
            if (!state.active) return;
            state.elapsedSec = Number(action.payload.elapsedSec ?? 0);
            state.level = Number(action.payload.level ?? 0);
            state.peak = Number(action.payload.peak ?? 0);
        },
        recordingStopped(state, action: PayloadAction<RecordingFinishedInfo | null>) {
            state.active = false;
            state.lastResult = action.payload;
            state.elapsedSec = action.payload?.durationSec ?? 0;
            state.level = 0;
            state.peak = 0;
            state.countdownRemaining = 0;
            state.countdownCancelRequested = false;
        },
    },
});

export const {
    settingsLoaded,
    devicesLoaded,
    setBusy,
    clearError,
    recordingFailed,
    beginCountdown,
    tickCountdown,
    requestCancelCountdown,
    cancelCountdown,
    recordingStarted,
    updateMeter,
    recordingStopped,
} = recordingSlice.actions;

export default recordingSlice.reducer;
