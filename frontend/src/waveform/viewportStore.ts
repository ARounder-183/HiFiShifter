import type {
    WaveformViewportPatch,
    WaveformViewportSnapshot,
    WaveformViewportStore,
} from "./types.ts";

const VIEWPORT_FIELDS = [
    "scrollLeftPx",
    "pxPerSec",
    "widthPx",
    "heightPx",
    "devicePixelRatio",
] as const;

function validValue(field: (typeof VIEWPORT_FIELDS)[number], value: number): boolean {
    if (!Number.isFinite(value)) return false;
    if (field === "scrollLeftPx") return value >= 0;
    return value > 0;
}

function freezeSnapshot(snapshot: WaveformViewportSnapshot): WaveformViewportSnapshot {
    return Object.freeze(snapshot);
}

export function createWaveformViewportStore(
    initial: WaveformViewportSnapshot,
): WaveformViewportStore {
    let snapshot = freezeSnapshot({
        revision: Math.max(0, Math.floor(initial.revision)),
        scrollLeftPx: validValue("scrollLeftPx", initial.scrollLeftPx) ? initial.scrollLeftPx : 0,
        pxPerSec: validValue("pxPerSec", initial.pxPerSec) ? initial.pxPerSec : 1,
        widthPx: validValue("widthPx", initial.widthPx) ? initial.widthPx : 1,
        heightPx: validValue("heightPx", initial.heightPx) ? initial.heightPx : 1,
        devicePixelRatio: validValue("devicePixelRatio", initial.devicePixelRatio)
            ? initial.devicePixelRatio
            : 1,
    });
    const listeners = new Set<() => void>();

    return {
        getSnapshot: () => snapshot,
        subscribe(listener) {
            listeners.add(listener);
            return () => listeners.delete(listener);
        },
        set(patch: WaveformViewportPatch) {
            let changed = false;
            const next = { ...snapshot };

            for (const field of VIEWPORT_FIELDS) {
                const value = patch[field];
                if (value === undefined || !validValue(field, value)) continue;
                if (Object.is(next[field], value)) continue;
                next[field] = value;
                changed = true;
            }

            if (!changed) return snapshot;
            snapshot = freezeSnapshot({ ...next, revision: snapshot.revision + 1 });
            for (const listener of [...listeners]) listener();
            return snapshot;
        },
    };
}
