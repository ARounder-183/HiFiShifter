import { useCallback, useEffect } from "react";

import { useAppDispatch } from "../app/hooks";
import { setHistoryState } from "../features/session/sessionSlice";
import { timelineApi } from "../services/api";
import type { HistoryStateResult } from "../types/api";

/** `history_state` 事件的载荷（与 `get_history_state` 同构）。 */
type HistoryStatePayload = Partial<Omit<HistoryStateResult, "ok">> & { ok?: boolean };

/**
 * 撤销/重做可用性 + 「操作记录」镜像。
 *
 * - 挂载时同步一次 `get_history_state`：开发热重载 / 从命令行打开工程时，
 *   早于监听器注册的事件不会丢失；
 * - 此后由后端 `history_state` 事件驱动 —— 每次打点（新检查点会清空重做栈）、
 *   清空历史（新建 / 打开工程）、撤销、重做、跳转都会广播。
 *
 * 撤销/重做按钮与菜单项的置灰、快捷键的前置判断、「操作记录」窗口的内容
 * 都读这份镜像，无需轮询。
 */
export function useHistoryStateListener(): void {
    const dispatch = useAppDispatch();

    const applyPayload = useCallback(
        (payload: HistoryStatePayload | null | undefined) => {
            if (!payload) return;
            dispatch(
                setHistoryState({
                    undoDepth: Number(payload.undoDepth) || 0,
                    redoDepth: Number(payload.redoDepth) || 0,
                    records: Array.isArray(payload.records) ? payload.records : undefined,
                }),
            );
        },
        [dispatch],
    );

    useEffect(() => {
        let disposed = false;

        void timelineApi
            .getHistoryState()
            .then((result) => {
                if (disposed || !result || result.ok === false) return;
                applyPayload(result);
            })
            .catch(() => {
                // 非 Tauri 环境（浏览器 / pywebview）下安全忽略。
            });

        return () => {
            disposed = true;
        };
    }, [applyPayload]);

    useEffect(() => {
        let disposed = false;
        let unlisten: (() => void) | null = null;

        async function setup() {
            try {
                const mod = await import("@tauri-apps/api/event");
                unlisten = await mod.listen<HistoryStatePayload>("history_state", (event) => {
                    if (disposed) return;
                    applyPayload(event.payload);
                });
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
    }, [applyPayload]);
}
