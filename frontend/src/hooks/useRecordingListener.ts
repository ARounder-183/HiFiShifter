import { useEffect } from "react";
import { useAppDispatch } from "../app/hooks";
import { updateMeter } from "../features/recording/recordingSlice";
import type { RecordingMeterPayload } from "../services/api/recording";

/**
 * 监听后端录音电平/计时事件，并同步到 Redux。
 */
export function useRecordingListener(): void {
    const dispatch = useAppDispatch();

    useEffect(() => {
        let disposed = false;
        let unlisten: (() => void) | null = null;

        async function setup() {
            try {
                const mod = await import("@tauri-apps/api/event");
                unlisten = await mod.listen<RecordingMeterPayload>("recording-meter", (event) => {
                    if (disposed) return;
                    const payload = event.payload;
                    if (!payload || typeof payload.elapsedSec !== "number") return;
                    dispatch(updateMeter(payload));
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
