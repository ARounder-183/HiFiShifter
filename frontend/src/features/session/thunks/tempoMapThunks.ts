/**
 * Tempo Map 后端同步 thunk。
 *
 * 前端为 Tempo Map 的唯一编辑入口；提交时把完整 TempoMap 结构发送给后端，
 * 后端持久化、同步工程基准 BPM/拍号，并在音阶部分发生变化时失效渲染缓存。
 */
import { createAsyncThunk } from "@reduxjs/toolkit";
import { timelineApi } from "../../../services/api/timeline";
import type { TempoMap } from "../../../utils/tempoMap";
import { toBackendTempoMap } from "../../../utils/tempoMap";
import { fetchTimeline } from "./transportThunks";

export const setTempoMapRemote = createAsyncThunk(
    "session/setTempoMapRemote",
    async (tempoMap: TempoMap | null, { dispatch, rejectWithValue }) => {
        try {
            return await timelineApi.setTimelineTempoMap(toBackendTempoMap(tempoMap));
        } catch (err) {
            // 后端为权威来源：提交失败时回滚乐观更新 —— 重新拉取时间线快照，
            // 避免 UI 停留在“已编辑但未持久化”的 Tempo Map 上（与后端永久分叉）。
            void dispatch(fetchTimeline());
            return rejectWithValue(err instanceof Error ? err.message : "tempo_map_commit_failed");
        }
    },
);
