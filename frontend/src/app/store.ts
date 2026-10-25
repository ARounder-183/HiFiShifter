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
 *
 * 【为什么按切片合并而不是整体替换】快照来自主窗口（可能因重载与卫星存在版本
 * 偏差）：整体替换会让缺切片的快照把本窗口对应切片置成 `undefined` 并一路渗到
 * UI。`{ ...state, ...payload }` 对快照里**存在**的切片仍是权威替换，缺的切片
 * 则保留本窗口 reducer 的当前值。
 */
/**
 * 快照切片的最小运行时校验：只接受「对象形态」的已知切片。
 *
 * 【为什么只做这么少】快照来自另一个窗口的 JSON，可能存在版本偏差（主窗口
 * 重载后旧卫星仍在线）。旧实现 `{ ...state, ...payload } as RootState` 把任何
 * 畸形载荷（null / 数组 / 未知结构）原样并入，错误以"某个面板渲染炸掉"的形态
 * 深处爆发。逐字段深校验又不现实（session 切片巨大）。折中：逐切片检查"是纯
 * 对象"，不认识的**键**不并入 —— 版本偏差降级为"新字段本窗口暂时没有"，
 * 与缺切片同一语义。
 */
function mergeSnapshot(state: RootState | undefined, payload: object): RootState {
    const merged: Record<string, unknown> = { ...(state as Record<string, unknown>) };
    let accepted = 0;
    for (const [key, slice] of Object.entries(payload)) {
        if (slice !== null && typeof slice === "object" && !Array.isArray(slice)) {
            merged[key] = slice;
            accepted += 1;
        }
    }
    if (accepted === 0) {
        return rootReducer(state, { type: "@@hsBridgeUnusableSnapshot" });
    }
    return merged as RootState;
}

function appReducer(state: RootState | undefined, action: unknown): RootState {
    if (
        action !== null &&
        typeof action === "object" &&
        (action as { type?: unknown }).type === BRIDGE_SNAPSHOT_ACTION
    ) {
        const payload = (action as { payload?: Partial<RootState> }).payload;
        if (!payload || typeof payload !== "object") {
            return rootReducer(state, { type: "@@hsBridgeEmptySnapshot" });
        }
        return mergeSnapshot(state, payload);
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
                    ignoredPaths: [
                        "session",
                        // 目录列表与搜索结果：一次列出两万个文件时，这两个数组
                        // 本身就是两万个对象。dev 下两条检查中间件**每个 action**
                        // 都要深走一遍状态 —— 播放时每秒几十个 action，于是一次
                        // "打开一个大目录"就让整个应用卡住（不只是面板）。
                        // 数据来自后端 IPC 的 JSON，天生可序列化，豁免不损失什么。
                        "fileBrowser.entries",
                        "fileBrowser.searchResults",
                    ],
                    ignoredActions: [BRIDGE_SNAPSHOT_ACTION],
                },
                // 同一组路径也要豁免不可变性检查：它同样是 dev 下的**每 action 深走**。
                immutableCheck: {
                    ignoredPaths: ["session", "fileBrowser.entries", "fileBrowser.searchResults"],
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
            // 【必须标记 hsRemote】否则这个动作会经桥中间件被广播回主窗口，而主窗口
            // 的根 reducer 会用它整体覆盖状态 —— 一次 detach 就把主窗口回滚到卫星
            // 请求快照那一刻（播放进度、布局等全部丢失）。
            store.dispatch({
                type: BRIDGE_SNAPSHOT_ACTION,
                payload: state,
                meta: { hsRemote: true },
            } as never);
        },
    });

    return { store, teardown };
}

/** 主窗口的全局 store（各模块直接 import 它，因此必须保持单例）。 */
export const store = createAppStore().store;

export type AppDispatch = typeof store.dispatch;
