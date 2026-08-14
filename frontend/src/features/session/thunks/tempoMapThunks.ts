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

export const setTempoMapRemote = createAsyncThunk(
    "session/setTempoMapRemote",
    async (tempoMap: TempoMap | null) => {
        return timelineApi.setTimelineTempoMap(toBackendTempoMap(tempoMap));
    },
);
