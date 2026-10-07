import { useEffect } from "react";
import { useAppDispatch } from "../app/hooks";
import { setClipFormantStatus } from "../features/session/sessionSlice";

interface ClipFormantStatusPayload {
    clip_id?: string;
    clipId?: string;
    status?: "ready" | "rebuilding" | "failed";
}

export function useClipFormantStatusListener(): void {
    const dispatch = useAppDispatch();

    useEffect(() => {
        let disposed = false;
        let unlisten: (() => void) | null = null;

        async function setup() {
            try {
                const mod = window.__HFS_PLUGIN_BOOTSTRAP__
                    ? await import("../services/hostEvents")
                    : await import("@tauri-apps/api/event");
                unlisten = await mod.listen<ClipFormantStatusPayload>(
                    "clip_formant_status",
                    (event) => {
                        if (disposed) return;
                        const payload = event.payload ?? {};
                        const clipId =
                            typeof payload.clipId === "string"
                                ? payload.clipId
                                : typeof payload.clip_id === "string"
                                  ? payload.clip_id
                                  : "";
                        const status = payload.status;
                        if (!clipId || !status) return;
                        dispatch(setClipFormantStatus({ clipId, status }));
                    },
                );
                if (disposed && unlisten) {
                    unlisten();
                }
            } catch {
                // Safe no-op outside Tauri runtime.
            }
        }

        void setup();
        return () => {
            disposed = true;
            if (unlisten) unlisten();
        };
    }, [dispatch]);
}
