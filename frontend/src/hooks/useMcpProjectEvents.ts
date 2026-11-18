import { useEffect } from "react";
import { useAppDispatch, useAppStore } from "../app/hooks";
import { fetchTimeline, seekPlayhead } from "../features/session/thunks/transportThunks";
import {
    bumpParamsEpoch,
    setEditParam,
    setPlayheadSec,
    setSelectedClip,
    setToolMode,
    setVisibleReferenceRootTrackIds,
} from "../features/session/sessionSlice";
import { selectTrackRemote } from "../features/session/thunks/timelineThunks";
import type { EditParam } from "../features/session/sessionSlice";
import type { ToolMode } from "../features/session/sessionTypes";
import { mcpApi } from "../services/api/mcp";
import { isPluginMode } from "../services/hostCapabilities";
import { listen } from "../services/hostEvents";

/** MCP edits use the ordinary backend timeline reducer and parameter refresh path. */
export function useMcpProjectEvents(): void {
    const dispatch = useAppDispatch();
    const store = useAppStore();
    useEffect(() => {
        if (isPluginMode()) return;
        let disposed = false;
        let dirty = false;
        let running = false;
        let unlisten: (() => void) | undefined;
        let unlistenView: (() => void) | undefined;
        let viewChain = Promise.resolve();
        async function refresh() {
            dirty = true;
            if (running) return;
            running = true;
            try {
                while (dirty && !disposed) {
                    if (store.getState().session._interactionLockCount > 0) break;
                    dirty = false;
                    const view = await mcpApi.view();
                    if (disposed || !view.ok) break;
                    if (store.getState().session._interactionLockCount > 0) {
                        dirty = true;
                        break;
                    }
                    dispatch(fetchTimeline.fulfilled(view.timeline, "mcp-refresh", undefined));
                    dispatch(bumpParamsEpoch());
                    await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
                    if (!disposed) await mcpApi.acknowledgeView(view.revision);
                }
            } catch {
                // Browser/dev pages have no native MCP service. Do not disrupt normal editing.
            } finally {
                running = false;
            }
        }
        const unsubscribeStore = store.subscribe(() => {
            if (
                dirty &&
                !running &&
                !disposed &&
                store.getState().session._interactionLockCount === 0
            )
                void refresh();
        });
        void listen("mcp_project_changed", () => void refresh())
            .then((stop) => {
                if (disposed) stop();
                else {
                    unlisten = stop;
                    void refresh();
                }
            })
            .catch(() => undefined);
        type ViewCommand = {
            request_id: string;
            request: {
                selected_track_id?: string;
                selected_clip_id?: string;
                active_param?: string;
                reference_root_track_ids?: string[];
                tool_mode?: ToolMode;
                focus_start_sec?: number;
                focus_end_sec?: number;
                seek_playhead?: boolean;
            };
        };
        void listen<ViewCommand>("mcp_view_request", (event) => {
            viewChain = viewChain
                .then(async () => {
                    if (disposed) return;
                    const { request_id, request } = event.payload;
                    try {
                        if (store.getState().session._interactionLockCount > 0) {
                            await mcpApi.replyView(request_id, { ok: false, code: "VIEW_BUSY" });
                            return;
                        }
                        if (request.selected_track_id)
                            await dispatch(selectTrackRemote(request.selected_track_id)).unwrap();
                        if (request.selected_clip_id)
                            dispatch(setSelectedClip(request.selected_clip_id));
                        if (request.active_param)
                            dispatch(setEditParam(request.active_param as EditParam));
                        if (request.reference_root_track_ids)
                            dispatch(
                                setVisibleReferenceRootTrackIds(request.reference_root_track_ids),
                            );
                        if (request.tool_mode) dispatch(setToolMode(request.tool_mode));
                        if (
                            request.focus_start_sec !== undefined ||
                            request.focus_end_sec !== undefined
                        ) {
                            window.dispatchEvent(
                                new CustomEvent("hifi:focusTimelineRange", {
                                    detail: {
                                        startSec: request.focus_start_sec,
                                        endSec: request.focus_end_sec,
                                    },
                                }),
                            );
                        }
                        if (request.seek_playhead && request.focus_start_sec !== undefined) {
                            dispatch(setPlayheadSec(request.focus_start_sec));
                            await dispatch(seekPlayhead(request.focus_start_sec));
                        }
                        await new Promise<void>((resolve) =>
                            requestAnimationFrame(() => resolve()),
                        );
                        if (disposed) return;
                        const s = store.getState().session;
                        await mcpApi.replyView(request_id, {
                            ok: true,
                            selected_track_id: s.selectedTrackId,
                            selected_clip_id: s.selectedClipId,
                            active_param: s.editParam,
                            reference_root_track_ids: s.visibleReferenceRootTrackIds,
                            tool_mode: s.toolMode,
                            playhead_sec: s.playheadSec,
                            focus_start_sec: request.focus_start_sec ?? null,
                            focus_end_sec: request.focus_end_sec ?? null,
                        });
                    } catch (error) {
                        await mcpApi.replyView(request_id, {
                            ok: false,
                            code: "VIEW_UPDATE_FAILED",
                            message: String(error),
                        });
                    }
                })
                .catch(() => undefined);
        })
            .then((stop) => {
                if (disposed) stop();
                else unlistenView = stop;
            })
            .catch(() => undefined);
        return () => {
            disposed = true;
            unsubscribeStore();
            unlisten?.();
            unlistenView?.();
        };
    }, [dispatch, store]);
}
