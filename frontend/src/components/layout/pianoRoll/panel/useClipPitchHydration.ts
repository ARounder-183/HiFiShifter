/**
 * 参数编辑器的 Clip 源音高参考线懒取。
 *
 * 主窗口通过 clip_pitch_data 事件拿到曲线；detachBridge 的快照刻意剔除
 * clipPitchCurves（每 clip 数百到上千个浮点）。卫星窗口晚打开时事件早已
 * 发生，若不在这里补一次，参考线会永久空白。
 */

import { useEffect, useRef } from "react";

import type { AppDispatch } from "../../../../app/store";
import type { ClipInfo } from "../../../../features/session/sessionTypes";
import { setClipPitchData } from "../../../../features/session/sessionSlice";
import { paramsApi } from "../../../../services/api";

export function useClipPitchHydration(args: {
    dispatch: AppDispatch;
    editParam: string;
    clips: readonly ClipInfo[];
    session: { showClipPitchOverlay: boolean; clipPitchCurves: Record<string, unknown> };
}): void {
    const { dispatch, editParam, clips, session } = args;
    const { showClipPitchOverlay, clipPitchCurves } = session;
    const requestedRef = useRef(new Set<string>());

    useEffect(() => {
        if (editParam !== "pitch" || !showClipPitchOverlay) return;
        const missing = [...new Set(clips.map((clip) => clip.id))].filter(
            (clipId) => !clipPitchCurves[clipId] && !requestedRef.current.has(clipId),
        );
        if (missing.length === 0) return;
        for (const clipId of missing) requestedRef.current.add(clipId);
        void paramsApi
            .getClipPitchData(missing)
            .then((result) => {
                if (!result?.ok) return;
                for (const payload of result.payloads ?? []) {
                    if (!payload?.clip_id || !Array.isArray(payload.midi_curve)) continue;
                    dispatch(
                        setClipPitchData({
                            clipId: payload.clip_id,
                            curveStartSec: Number(payload.curve_start_sec ?? 0),
                            midiCurve: payload.midi_curve,
                            framePeriodMs: Number(payload.frame_period_ms ?? 5),
                        }),
                    );
                }
            })
            .catch(() => {
                // 后端分析未完成 / 旧版本没有该命令：事件推送路径仍可用。
                for (const clipId of missing) requestedRef.current.delete(clipId);
            });
    }, [clipPitchCurves, clips, dispatch, editParam, showClipPitchOverlay]);
}
