import type { TimelineResult } from "../../types/api";

import { invoke } from "../invoke";

export interface RecordingSettings {
    sourceDevice: string;
    sampleRate: number;
    bitDepth: 16 | 24 | 32;
    channels: 1 | 2;
    inputGainDb: number;
    monitorEnabled: boolean;
    monitorGainDb: number;
    countdownSec: number;
    autoNormalize: boolean;
    autoStopAtSelectionEnd: boolean;
    pathTemplate: string;
}

export interface RecordingDeviceInfo {
    id: string;
    name: string;
    kind: string;
    isDefault: boolean;
    isLoopback: boolean;
}

export interface RecordingStatePayload {
    active: boolean;
    elapsedSec: number;
    level: number;
    peak: number;
    startSec?: number | null;
    outputPath?: string | null;
    error?: string | null;
}

export interface RecordingMeterPayload {
    active: boolean;
    elapsedSec: number;
    level: number;
    peak: number;
}

export interface RecordingFinishedInfo {
    startSec: number;
    durationSec: number;
    sampleRate: number;
    channels: number;
    peak: number;
    outputPath: string;
}

export const DEFAULT_RECORDING_SETTINGS: RecordingSettings = {
    sourceDevice: "default",
    sampleRate: 48_000,
    bitDepth: 24,
    channels: 2,
    inputGainDb: 0,
    monitorEnabled: false,
    monitorGainDb: 0,
    countdownSec: 0,
    autoNormalize: false,
    autoStopAtSelectionEnd: false,
    pathTemplate:
        "<ProjectFolder>/HiFiShifter Record/%Y-%m-%d-%H-%M-%S.wav",
};

export const recordingApi = {
    getSettings: () => invoke<RecordingSettings>("get_recording_settings"),

    saveSettings: (settings: RecordingSettings) =>
        invoke<{ ok: boolean; settings?: RecordingSettings }>(
            "save_recording_settings",
            settings,
        ),

    getDevices: () =>
        invoke<{ ok: boolean; devices?: RecordingDeviceInfo[] }>("get_recording_devices"),

    startRecording: (startSec: number) =>
        invoke<{
            ok: boolean;
            startSec?: number;
            outputPath?: string;
            error?: string;
        }>("start_recording", startSec),

    stopRecording: () =>
        invoke<{
            ok: boolean;
            timeline?: TimelineResult;
            recording?: RecordingFinishedInfo;
            error?: string;
        }>("stop_recording"),

    getState: () => invoke<RecordingStatePayload>("get_recording_state"),
};

