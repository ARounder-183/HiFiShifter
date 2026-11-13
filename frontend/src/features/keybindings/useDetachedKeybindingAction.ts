/**
 * 独立窗口的快捷键执行器。
 *
 * useKeybindings 只在窗口内识别「按了什么」；真正的副作用大多属于主窗口
 * （播放、工程、轨道、导出…）。这里把命令分成三类：
 *
 * 1. 本窗口就能完成的编辑：参数块 / 参数线 / 编辑器本地 Clipboard 事件；
 * 2. 时间轴操作：通过 windowEventBridge 送回 main；
 * 3. 其余动作：同样送回 main 的 handleKeybindingAction，由它按 sourceSurface
 *    路由；这样独立窗口不会拥有一套与主窗口分叉的业务实现。
 *
 * 布局中只有 maximize 在卫星窗口内有意义；toggleFloat / newPanel / dissolve
 * 会把窗体移出本窗口或改变主窗口的布局，明确拒绝并让调用方提示。
 */

import { useCallback } from "react";

import { useAppDispatch, useAppStore } from "../../app/hooks";
import { webApi } from "../../services/webviewApi";
import { getActiveSurface } from "../uiFocus/focusSurface";
import { maximizeActive } from "../dock/dockApi";
import { emitWindowEvent, KEYBINDING_ACTION_EVENT } from "../dock/windowEventBridge";
import { resolveWindowLabel } from "../dock/detachBridge";
import {
    ACTION_TO_EDIT_OP,
    PARAM_BLOCK_ACTION_TO_OP,
    resolveCopyCutRoute,
    resolveEditOpRoute,
    resolveParamBlockEditChannel,
    resolvePasteRoute,
} from "./focusRouting";
import type { ActionId } from "./types";
import type { HostKeyStroke } from "../../services/hostKeyRelay";

function dispatchLocal(name: string, detail: unknown): void {
    window.dispatchEvent(new CustomEvent(name, { detail }));
}

export function useDetachedKeybindingAction(args: {
    onUnavailable: () => void;
}): (actionId: ActionId, stroke: HostKeyStroke) => void {
    const { onUnavailable } = args;
    const dispatch = useAppDispatch();
    const appStore = useAppStore();

    return useCallback(
        (actionId: ActionId, stroke: HostKeyStroke) => {
            const blockOp = PARAM_BLOCK_ACTION_TO_OP[actionId];
            if (blockOp) {
                const channel = resolveParamBlockEditChannel(
                    getActiveSurface(),
                    appStore.getState().session,
                );
                if (channel) dispatchLocal(channel, { op: blockOp, stroke });
                return;
            }

            const editOp = ACTION_TO_EDIT_OP[actionId];
            if (editOp) {
                const session = appStore.getState().session;
                const surface = getActiveSurface();
                if (editOp === "paste") {
                    void (async () => {
                        let kind: string | null = null;
                        try {
                            kind = (await webApi.clipboardKind()).kind ?? null;
                        } catch {
                            // 探测失败按外来源/空处理，与主窗口一致。
                        }
                        const channel = resolvePasteRoute(kind, surface);
                        if (channel === "hifi:timelineEditOp") {
                            emitWindowEvent("main", channel, { op: "paste" });
                        } else if (channel) {
                            dispatchLocal(channel, { op: "paste" });
                        }
                    })();
                    return;
                }
                if (editOp === "copy" || editOp === "cut") {
                    const channel = resolveCopyCutRoute({
                        surface,
                        clipSelectionActive:
                            session.multiSelectedClipIds.length > 0 || !!session.selectedClipId,
                        paramSelectionActive: session.paramSelectionActive,
                        selectionContext: session.selectionContext,
                    });
                    if (channel === "hifi:timelineEditOp") {
                        emitWindowEvent("main", channel, { op: editOp });
                    } else if (channel) {
                        dispatchLocal(channel, { op: editOp });
                    }
                    return;
                }
                if (
                    editOp === "addClipsToParamSelection" ||
                    editOp === "removeClipsFromParamSelection"
                ) {
                    dispatchLocal("hifi:editOp", { op: editOp });
                    return;
                }
                const channel = resolveEditOpRoute(surface, editOp, session.toolMode);
                if (channel === "hifi:timelineEditOp") {
                    emitWindowEvent("main", channel, { op: editOp });
                } else if (channel) {
                    dispatchLocal(channel, { op: editOp });
                }
                return;
            }

            if (actionId === "edit.pasteVocalShifter") {
                dispatchLocal("hifi:editOp", { op: "pasteVocalShifter" });
                return;
            }

            switch (actionId) {
                case "layout.maximize":
                    maximizeActive(dispatch);
                    return;
                case "layout.toggleFloat":
                case "layout.focusNext":
                case "layout.focusPrev":
                case "layout.newPanel":
                case "layout.dissolvePanel":
                    onUnavailable();
                    return;
                default:
                    break;
            }

            // 其余动作统一回主窗口执行。sourceSurface 让参数编辑器相关命令
            // 仍按卫星窗口的实际焦点裁决，而不是主窗口时间轴。
            emitWindowEvent("main", KEYBINDING_ACTION_EVENT, {
                actionId,
                stroke,
                surface: getActiveSurface(),
                sourceWindow: resolveWindowLabel(),
            });
        },
        [appStore, dispatch, onUnavailable],
    );
}
