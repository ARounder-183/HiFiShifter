import { useEffect } from "react";

import { useAppDispatch } from "../app/hooks";
import { setHistoryDepths } from "../features/session/sessionSlice";
import { timelineApi } from "../services/api";

/**
 * 撤销/重做可用性镜像（栈深度）。
 *
 * - 挂载时同步一次 `get_history_state`：开发热重载 / 从命令行打开工程时，
 *   早于监听器注册的事件不会丢失；
 * - 此后由后端 `history_state` 事件驱动 —— 每次打点（新检查点会清空重做栈）、
 *   清空历史（新建 / 打开工程）、撤销、重做都会广播。
 *
 * 菜单项置灰与快捷键的前置判断都读这份镜像，无需轮询。
 */
export function useHistoryStateListener(): void {
    const dispatch = useAppDispatch();

    useEffect(() => {
        let disposed = false;

        void timelineApi
            .getHistoryState()
            .then((result) => {
                if (disposed || !result || result.ok === false) return;
                dispatch(
                    setHistoryDepths({
                        undoDepth: Number(result.undoDepth) || 0,
                        redoDepth: Number(result.redoDepth) || 0,
                    }),
                );
            })
            .catch(() => {
                // 非 Tauri 环境（浏览器 / pywebview）下安全忽略。
            });

        return () => {
            disposed = true;
        };
    }, [dispatch]);

    useEffect(() => {
        let disposed = false;
        let unlisten: (() => void) | null = null;

        async function setup() {
            try {
                const mod = await import("@tauri-apps/api/event");
                unlisten = await mod.listen<{ undoDepth?: number; redoDepth?: number }>(
                    "history_state",
                    (event) => {
                        if (disposed) return;
                        const payload = event.payload;
                        if (!payload) return;
                        dispatch(
                            setHistoryDepths({
                                undoDepth: Number(payload.undoDepth) || 0,
                                redoDepth: Number(payload.redoDepth) || 0,
                            }),
                        );
                    },
                );
                // cleanup 可能发生在 await resolve 之前：已卸载则立即反注册。
                if (disposed && unlisten) {
                    unlisten();
                    unlisten = null;
                }
            } catch {
                // 非 Tauri 环境（浏览器 / pywebview）下安全忽略。
            }
        }

        void setup();
        return () => {
            disposed = true;
            if (unlisten) unlisten();
        };
    }, [dispatch]);
}
