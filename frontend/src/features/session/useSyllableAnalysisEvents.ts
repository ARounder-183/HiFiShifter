import { useEffect } from "react";
import { useAppDispatch } from "../../app/hooks";
import { listen } from "../../services/hostEvents";
import { applyClipSyllableAnalysis } from "./sessionSlice";

/** Mounted once in the main App/plugin shell; detached windows receive the
 * resulting session action through detachBridge (avoiding duplicate dispatch). */
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
