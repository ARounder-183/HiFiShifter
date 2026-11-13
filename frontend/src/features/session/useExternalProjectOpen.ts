/**
 * macOS 从访达打开工程的入口：先注册监听，再报告前端就绪，最后消费冷启动路径。
 *
 * 【为什么顺序不能反】Rust 在 frontend_ready 之后才会把 LaunchServices 的
 * Opened 事件直接 emit 给前端；此前的路径留在 pending。若先 consume 再 listen，
 * 冷启动时 Rust 可能已经 emit 到空房间，路径既不在 pending 也没人接收 —— 用户
 * 看到的就是空白工程。先 listen → markFrontendReady → consume 覆盖两种时序：
 *
 * - 冷启动：Opened 早于 ready → pending，consume 取走；
 * - 热打开：ready 已 true → emit，监听器已就绪，立即打开；
 * - 监听失败：仍然尝试 consume，至少不丢冷启动路径。
 */

import { useEffect, useRef } from "react";

import { projectApi } from "../../services/api/project";
import { listen } from "../../services/hostEvents";
import { isPluginMode } from "../../services/hostDetect";
import { detectExternalActionKindFromPath, type ExternalFileActionKind } from "./projectOpenEvents";

export const EXTERNAL_PROJECT_OPEN_EVENT = "hs:external-project-open";

interface ExternalProjectOpenPayload {
    path?: unknown;
}

/**
 * @param handler 收到路径时的动作回调（通常是 App 的 handleExternalFileAction）。
 *   用 ref 保存最新实现，effect 只注册一次，避免项目脏标记变化导致重复监听/消费。
 */
export function useExternalProjectOpen(
    handler: (kind: ExternalFileActionKind, path: string) => void,
): void {
    const handlerRef = useRef(handler);
    useEffect(() => {
        handlerRef.current = handler;
    }, [handler]);

    useEffect(() => {
        let disposed = false;
        let unlisten: (() => void) | null = null;

        const openIfSupported = (rawPath: unknown): void => {
            const path = String(rawPath ?? "").trim();
            if (!path) return;
            const kind = detectExternalActionKindFromPath(path);
            if (kind === null) return;
            handlerRef.current(kind, path);
        };

        async function setup(): Promise<void> {
            // 插件形态没有独立的 LaunchServices 文档打开入口，也不应调用 App 命令。
            if (isPluginMode()) return;
            try {
                unlisten = await listen<ExternalProjectOpenPayload>(
                    EXTERNAL_PROJECT_OPEN_EVENT,
                    (event) => {
                        if (!disposed) openIfSupported(event.payload?.path);
                    },
                );
                if (disposed) {
                    unlisten();
                    unlisten = null;
                    return;
                }

                // 监听已就绪，才允许 Rust 把后续 Opened 直接 emit 过来。
                try {
                    await projectApi.markFrontendReady();
                } catch {
                    // 旧后端没有该命令：consume 仍然可用，不应阻断启动。
                }

                const result = await projectApi.consumeStartupProjectPath();
                if (!disposed) openIfSupported(result?.path);
            } catch {
                // 非 Tauri / 事件 API 不可用：仍尝试消费一次，避免冷启动路径丢失。
                try {
                    const result = await projectApi.consumeStartupProjectPath();
                    if (!disposed) openIfSupported(result?.path);
                } catch {
                    // 浏览器预览等环境没有后端，安全忽略。
                }
            }
        }

        void setup();

        return () => {
            disposed = true;
            if (unlisten) unlisten();
        };
    }, []);
}
