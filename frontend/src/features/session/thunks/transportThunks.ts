import { createAsyncThunk } from "@reduxjs/toolkit";
import { webApi } from "../../../services/webviewApi";

import { setMetronomeConfig, persistUiSettings } from "../sessionSlice";
import type { SessionState } from "../sessionSlice";

/**
 * 更新节拍器配置并即时应用到引擎：
 * 1. 更新 Redux（UI 立即反映）；2. `set_metronome`（引擎原子配置 + 按细分
 * 模式重建响点表 + 持久化到 UiSettings）；3. 同步一份到通用设置持久化，
 * 防止后续任意设置保存把节拍器字段回写成陈旧值。
 */
export const updateMetronome = createAsyncThunk(
    "session/updateMetronome",
    async (
        payload: Partial<{
            metronomeEnabled: boolean;
            metronomeGain: number;
            metronomeMode: "grid" | "beat" | "bar";
            metronomeSound: "click" | "woodblock" | "beep";
            metronomeAccent: boolean;
        }>,
        { dispatch, getState },
    ) => {
        dispatch(setMetronomeConfig(payload));
        const s = (getState() as { session: SessionState }).session;
        try {
            await webApi.setMetronome({
                enabled: s.metronomeEnabled,
                gain: s.metronomeGain,
                mode: s.metronomeMode,
                accent: s.metronomeAccent,
                sound: s.metronomeSound,
            });
        } finally {
            void dispatch(persistUiSettings());
        }
    },
);

export const fetchTimeline = createAsyncThunk("session/fetchTimeline", async () => {
    return webApi.getTimelineState();
});

export const stopAudioPlayback = createAsyncThunk(
    "session/stopAudioPlayback",
    async (options: { restoreAnchor?: boolean } | void, { getState }) => {
        const restoreAnchor = Boolean(
            (options as { restoreAnchor?: boolean } | undefined)?.restoreAnchor,
        );
        const state = getState() as { session: SessionState };
        const positionSec = Number(state.session.runtime.playbackPositionSec ?? 0);
        const wasPlaying = Boolean(
            state.session.runtime.isPlaying ||
            positionSec > 1e-4 ||
            Math.abs(
                Number(state.session.playheadSec ?? 0) -
                    Number(state.session.playbackAnchorSec ?? 0),
            ) > 1e-4,
        );
        const anchorSec = state.session.playbackAnchorSec;
        const result = await webApi.stopAudio();
        // Only restore when this stop action actually interrupted active playback.
        if (restoreAnchor && wasPlaying && anchorSec !== undefined && anchorSec !== null) {
            await webApi.setTransport({ playheadSec: anchorSec });
        }
        return { ...result, restoreAnchor, wasPlaying, anchorSec };
    },
);

export const seekPlayhead = createAsyncThunk("session/seekPlayhead", async (sec: number) => {
    return webApi.setTransport({ playheadSec: sec });
});

export const updateTransportBpm = createAsyncThunk(
    "session/updateTransportBpm",
    async (bpm: number) => {
        return webApi.setTransport({ bpm });
    },
);

export const syncPlaybackState = createAsyncThunk("session/syncPlaybackState", async () => {
    return webApi.getPlaybackState();
});

export const playOriginal = createAsyncThunk("session/playOriginal", async (_, { getState }) => {
    const state = getState() as { session: SessionState };
    const anchorSec = state.session.playheadSec;
    // Ensure backend transport is in sync before starting playback.
    await webApi.setTransport({ playheadSec: anchorSec });
    const result = await webApi.playOriginal(0);
    return {
        ...result,
        clipId: null,
        anchorSec,
    };
});
