import { configureStore } from "@reduxjs/toolkit";
import sessionReducer from "../features/session/sessionSlice";
import fileBrowserReducer from "../features/fileBrowser/fileBrowserSlice";
import keybindingsReducer from "../features/keybindings/keybindingsSlice";
import notebookReducer from "../features/notebook/notebookSlice";
import recordingReducer from "../features/recording/recordingSlice";

export const store = configureStore({
    reducer: {
        session: sessionReducer,
        fileBrowser: fileBrowserReducer,
        keybindings: keybindingsReducer,
        notebook: notebookReducer,
        recording: recordingReducer,
    },
    middleware: (getDefaultMiddleware) =>
        getDefaultMiddleware({
            // session 切片包含 Tauri 后端返回的大体积纯数据 payload（波形数组、
            // 参数曲线），序列化检查只在 dev 下运行，但整切片逐字段深检在高频
            // 播放轮询下开销可观，故按路径豁免。新增非序列化字段前请三思。
            serializableCheck: {
                ignoredPaths: ["session"],
                ignoredActions: ["session/setTimelineState"],
            },
            // 注意：此前豁免的 "session.timeline" 并不存在于 SessionState
            // （状态是 tracks/clips 等平铺字段），属死配置，已移除。
        }),
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
