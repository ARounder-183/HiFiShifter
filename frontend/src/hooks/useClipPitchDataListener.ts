import { useEffect } from "react";
import { useAppDispatch } from "../app/hooks";
import { setClipPitchData } from "../features/session/sessionSlice";

/**
 * 后端 `clip_pitch_data` 事件的 payload 结构，
 * 与 Rust 侧 `ClipPitchDataPayload` 对应。
 */
interface ClipPitchDataPayload {
    clip_id: string;
    curve_start_sec: number;
    midi_curve: number[];
    frame_period_ms: number;
}

/**
 * 监听后端推送的 `clip_pitch_data` 事件，将 per-clip MIDI 曲线存入 Redux store。
 *
 * 应在应用顶层（App.tsx）挂载一次。
 */
export function useClipPitchDataListener(): void {
    const dispatch = useAppDispatch();

    useEffect(() => {
        let disposed = false;
        let unlisten: (() => void) | null = null;

        async function setup() {
            try {
                const mod = await import("@tauri-apps/api/event");
                unlisten = await mod.listen<ClipPitchDataPayload>("clip_pitch_data", (event) => {
                    if (disposed) return;
                    const p = event.payload;
                    if (!p?.clip_id || !Array.isArray(p.midi_curve)) return;
                    dispatch(
                        setClipPitchData({
                            clipId: p.clip_id,
                            curveStartSec: Number(p.curve_start_sec ?? 0),
                            midiCurve: p.midi_curve as number[],
                            framePeriodMs: Number(p.frame_period_ms ?? 5),
                        }),
                    );
                });
                if (disposed && unlisten) {
                    unlisten();
                }
            } catch {
                // 非 Tauri 环境（浏览器/pywebview）下安全忽略。
            }
        }

        void setup();

        return () => {
            disposed = true;
            if (unlisten) unlisten();
        };
    }, [dispatch]);
}
