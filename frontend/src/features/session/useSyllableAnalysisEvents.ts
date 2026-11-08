import { useEffect } from "react";
import { useAppDispatch } from "../../app/hooks";
import { listen } from "../../services/hostEvents";
import { applyClipSyllableAnalysis } from "./sessionSlice";

/** Shared by App, plugin and detached windows; metadata never reloads curves. */
export function useSyllableAnalysisEvents() {
    const dispatch = useAppDispatch();
    useEffect(() => {
        let disposed = false;
        const subscription = listen<Parameters<typeof applyClipSyllableAnalysis>[0]>(
            "clip_syllables_updated",
            (event) => {
                if (!disposed) dispatch(applyClipSyllableAnalysis(event.payload));
            },
        );
        return () => {
            disposed = true;
            void subscription.then((off) => off()).catch(() => undefined);
        };
    }, [dispatch]);
}
