import { combineReducers, configureStore } from "@reduxjs/toolkit";
import sessionReducer from "../features/session/sessionSlice";
import dockReducer from "../features/dock/dockSlice";
import fileBrowserReducer from "../features/fileBrowser/fileBrowserSlice";
import keybindingsReducer, {
    keybindingsPersistenceMiddleware,
} from "../features/keybindings/keybindingsSlice";
import notebookReducer from "../features/notebook/notebookSlice";
import recordingReducer from "../features/recording/recordingSlice";
import {
    BRIDGE_SNAPSHOT_ACTION,
    createStoreBridgeMiddleware,
    installBridge,
    isSatelliteWindow,
} from "../features/dock/detachBridge";

const rootReducer = combineReducers({
    session: sessionReducer,
    // 停靠布局独立成切片：它不参与工程会话（不写工程文件、不进工程撤销），
    // 放进 session 只会让它跟着 `persistUiSettings` 的大 payload 一起流动，
    // 还会与工程切换耦合。
    dock: dockReducer,
    fileBrowser: fileBrowserReducer,
    keybindings: keybindingsReducer,
    notebook: notebookReducer,
    recording: recordingReducer,
});

export type RootState = ReturnType<typeof rootReducer>;

/**
 * 根 reducer：额外接受"应用跨窗口快照"这一个动作。
 *
 * 【为什么需要】独立窗口启动时要把主窗口的状态整份搬过来（见 `detachBridge`）。
 * 逐切片 replay 动作既慢又不完整；一次整体替换最简单，也最不容易出错 —— 快照就是
 * 权威状态本身，不存在"合并"语义。
 */
function appReducer(state: RootState | undefined, action: unknown): RootState {
    if (
        action !== null &&
        typeof action === "object" &&
        (action as { type?: unknown }).type === BRIDGE_SNAPSHOT_ACTION
    ) {
        return (action as { payload: RootState }).payload;
    }
    return rootReducer(state, action as Parameters<typeof rootReducer>[1]);
}

/**
 * 创建应用 store。
 *
 * 【为什么是工厂】独立窗口需要**自己的** store 实例（另一个 JS 上下文），但 reducer
 * 与中间件必须与主窗口完全一致 —— 两边跑同一套 reducer，动作复制才能保证状态收敛。
 */
export function createAppStore() {
    const satellite = isSatelliteWindow();
    const store = configureStore({
        reducer: appReducer,
        middleware: (getDefaultMiddleware) =>
            getDefaultMiddleware({
                // session 切片包含 Tauri 后端返回的大体积纯数据 payload（波形数组、
                // 参数曲线），序列化检查只在 dev 下运行，但整切片逐字段深检在高频
                // 播放轮询下开销可观，故按路径豁免。新增非序列化字段前请三思。
                serializableCheck: {
                    ignoredPaths: ["session"],
                    ignoredActions: ["session/setTimelineState", BRIDGE_SNAPSHOT_ACTION],
                },
                // 注意：此前豁免的 "session.timeline" 并不存在于 SessionState
                // （状态是 tracks/clips 等平铺字段），属死配置，已移除。
            })
                .prepend(keybindingsPersistenceMiddleware.middleware)
                .concat(createStoreBridgeMiddleware(satellite ? "satellite" : "main")),
    });

    // 接线跨窗口状态桥。主窗口应答快照请求，卫星窗口请求并应用快照。
    const teardown = installBridge({
        role: satellite ? "satellite" : "main",
        getState: () => store.getState(),
        dispatch: (action) => store.dispatch(action as never),
        onSnapshot: (state) => {
            store.dispatch({ type: BRIDGE_SNAPSHOT_ACTION, payload: state } as never);
        },
    });

    return { store, teardown };
}

/** 主窗口的全局 store（各模块直接 import 它，因此必须保持单例）。 */
export const store = createAppStore().store;

export type AppDispatch = typeof store.dispatch;
