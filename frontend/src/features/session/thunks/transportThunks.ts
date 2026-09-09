import { createAsyncThunk } from "@reduxjs/toolkit";
import { webApi } from "../../../services/webviewApi";

import { setMetronomeConfig, persistUiSettings } from "../sessionSlice";
import type { SessionState } from "../sessionSlice";

/**
 * 更新节拍器配置并即时应用到引擎：
 * 1. 更新 Redux（UI 立即反映）；2. `set_metronome`（引擎原子配置 + 按细分
 * 模式重建响点表，并写回后端设置缓存）；3. 磁盘持久化**去抖**走通用
 * `save_ui_settings` 通道 —— 音量滑杆 / 滚轮细调是高频手势，逐次全量
 * 配置写盘（读-合并-写-备份 ≈ 8 次文件操作）既拖慢命令线程，也会与
 * 并发的其他设置保存互相踩踏。
 */
// 引擎写入串行化链：Tauri 命令在线程池上并发执行，滚轮连发的多次
// set_metronome 若不排序，引擎可能停在较旧的增益/模式上。
let metronomeInvokeChain: Promise<void> = Promise.resolve();
// 磁盘持久化去抖计时器（trailing edge：停止操作后统一落盘一次）。
let metronomePersistTimer: ReturnType<typeof setTimeout> | undefined;

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
        const snapshot = {
            enabled: s.metronomeEnabled,
            gain: s.metronomeGain,
            mode: s.metronomeMode,
            accent: s.metronomeAccent,
            sound: s.metronomeSound,
        };
        const run = metronomeInvokeChain.then(async () => {
            await webApi.setMetronome(snapshot);
        });
        // 引擎同步失败不阻塞 UI：保留乐观状态，由下一次操作或去抖持久化
        // 时自然收敛（节拍器是非关键路径）。
        metronomeInvokeChain = run.catch(() => undefined);
        try {
            await run;
        } catch (err) {
            console.error("[metronome] engine sync failed", err);
        }
        if (metronomePersistTimer != null) clearTimeout(metronomePersistTimer);
        metronomePersistTimer = setTimeout(() => {
            metronomePersistTimer = undefined;
            void dispatch(persistUiSettings());
        }, 500);
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
