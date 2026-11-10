import type { SnapCandidateKind, SnapResult } from "./timelineSnapping";

/**
 * Surface-independent snap guide bus.
 *
 * The timeline and the parameter editor share one immutable model, one colour
 * table and one device-pixel line implementation. Entries carry a surface and
 * markers carry an axis so a timeline clip drag can never paint a line in the
 * piano roll (and vice versa).
 */

export const SNAP_HIGHLIGHT_GROUP = "timeline";
export const TIMELINE_TRACK_SNAP_GROUP = "timelineTrackSnap";

export type SnapGuideSurface = "timeline" | "pianoRoll" | "all";
export type SnapGuideAxis = "time" | "value" | "row";
export type SnapHighlightRole = "target" | "source";

export type SnapHighlightKind =
    | "grid"
    | "clipStart"
    | "clipEnd"
    | "snapOffset"
    | "sourceStart"
    | "sourceEnd"
    | "selection"
    | "cursor"
    | "sampleRate"
    | "loopBoundary"
    | "track";

export interface SnapGuideMarker {
    readonly role: SnapHighlightRole;
    /**
     * Time in seconds for axis="time"; parameter value for axis="value";
     * row-boundary index for axis="row".
     */
    readonly position: number;
    readonly axis: SnapGuideAxis;
    /** Timeline row owner; null means a full-height guide. */
    readonly trackId: string | null;
    readonly clipId: string | null;
}

export interface SnapGuideEntry {
    readonly id: string;
    readonly kind: SnapHighlightKind;
    readonly markers: readonly SnapGuideMarker[];
    readonly surface: SnapGuideSurface;
}

export type SnapHighlightMarker = SnapGuideMarker;
export type SnapHighlightEntry = SnapGuideEntry;

export interface SnapGuideSnapshot {
    readonly rev: number;
    readonly entries: readonly SnapGuideEntry[];
}

const EMPTY_SNAPSHOT: SnapGuideSnapshot = { rev: 0, entries: [] };

let snapshot: SnapGuideSnapshot = EMPTY_SNAPSHOT;
let rev = 0;
const listeners = new Set<() => void>();

function commit(entries: SnapGuideEntry[]): void {
    rev += 1;
    snapshot = { rev, entries };
    for (const listener of listeners) listener();
}

export function subscribeSnapHighlight(listener: () => void): () => void {
    listeners.add(listener);
    return () => {
        listeners.delete(listener);
    };
}

export function getSnapHighlightSnapshot(): SnapGuideSnapshot {
    return snapshot;
}

export function getSnapGuideEntriesForSurface(
    surface: Exclude<SnapGuideSurface, "all">,
): readonly SnapGuideEntry[] {
    return snapshot.entries.filter((entry) => entry.surface === surface || entry.surface === "all");
}

export function publishSnapHighlights(
    group: string,
    entries: readonly SnapGuideEntry[],
    surface: SnapGuideSurface = "timeline",
): void {
    const prefix = group + "\u0000";
    const nextEntries = snapshot.entries.filter((entry) => !entry.id.startsWith(prefix));
    for (const entry of entries) {
        nextEntries.push({ ...entry, surface, id: prefix + entry.id });
    }
    commit(nextEntries);
}

export function clearSnapHighlights(group?: string): void {
    if (!group) {
        if (snapshot.entries.length === 0) return;
        commit([]);
        return;
    }
    const prefix = group + "\u0000";
    const nextEntries = snapshot.entries.filter((entry) => !entry.id.startsWith(prefix));
    if (nextEntries.length === snapshot.entries.length) return;
    commit(nextEntries);
}

export interface SnapHighlightSourceSpec {
    position?: number;
    sec?: number;
    trackId?: string | null;
    clipId?: string | null;
}

export function buildCandidateHighlightEntry(args: {
    id?: string;
    kind: SnapHighlightKind;
    sec: number;
    targetTrackId?: string | null;
    targetClipId?: string | null;
    sources?: readonly SnapHighlightSourceSpec[];
    axis?: SnapGuideAxis;
    surface?: SnapGuideSurface;
}): SnapGuideEntry {
    const axis = args.axis ?? "time";
    const markers: SnapGuideMarker[] = [
        {
            role: "target",
            position: args.sec,
            axis,
            trackId: args.targetTrackId ?? null,
            clipId: args.targetClipId ?? null,
        },
    ];
    for (const source of args.sources ?? []) {
        markers.push({
            role: "source",
            position: source.position ?? source.sec ?? args.sec,
            axis,
            trackId: source.trackId ?? null,
            clipId: source.clipId ?? null,
        });
    }
    return {
        id: args.id ?? "primary",
        kind: args.kind,
        markers,
        surface: args.surface ?? "timeline",
    };
}

/**
 * Publish only the snapped target. Passing the raw pointer as a source would
 * draw a second, misleading line.
 */
export function publishSingleTargetSnapHighlight(args: {
    readonly group: string;
    readonly result: SnapResult;
    readonly id?: string;
    readonly surface?: SnapGuideSurface;
}): void {
    if (!args.result.snapped || args.result.candidate === null) {
        clearSnapHighlights(args.group);
        return;
    }
    const candidateSec = args.result.candidate.sec;
    if (!Number.isFinite(candidateSec)) {
        console.warn("[snapGuide] ignored non-finite snap candidate", candidateSec);
        clearSnapHighlights(args.group);
        return;
    }
    publishSnapHighlights(
        args.group,
        [
            buildCandidateHighlightEntry({
                ...(args.id === undefined ? {} : { id: args.id }),
                kind: snapHighlightKindFromCandidate(args.result.candidate.kind),
                sec: candidateSec,
                targetTrackId: args.result.candidate.trackId ?? null,
                targetClipId: args.result.candidate.clipId ?? null,
                sources: [],
            }),
        ],
        args.surface ?? "timeline",
    );
}

/** Horizontal timeline guide at a track-row boundary. */
export function publishTimelineTrackSnapHighlight(args: {
    readonly rowIndex: number;
    readonly trackId: string | null;
}): void {
    if (!Number.isFinite(args.rowIndex)) {
        clearSnapHighlights(TIMELINE_TRACK_SNAP_GROUP);
        return;
    }
    publishSnapHighlights(
        TIMELINE_TRACK_SNAP_GROUP,
        [
            {
                id: "track-boundary",
                kind: "track",
                surface: "timeline",
                markers: [
                    {
                        role: "target",
                        axis: "row",
                        position: Math.round(args.rowIndex),
                        trackId: args.trackId,
                        clipId: null,
                    },
                ],
            },
        ],
        "timeline",
    );
}

/** Value-axis guide used by pitch snapping in the parameter editor. */
export function buildValueGuideEntry(args: {
    readonly id: string;
    readonly kind: SnapHighlightKind;
    readonly value: number;
}): SnapGuideEntry {
    return {
        id: args.id,
        kind: args.kind,
        surface: "pianoRoll",
        markers: [
            {
                role: "target",
                position: args.value,
                axis: "value",
                trackId: null,
                clipId: null,
            },
        ],
    };
}

export function buildLoopBoundaryHighlightEntry(args: {
    id?: string;
    secs: readonly number[];
    trackId: string | null;
    clipId: string | null;
}): SnapGuideEntry {
    const markers: SnapGuideMarker[] = args.secs.map((sec, index) => ({
        role: index === 0 ? "target" : "source",
        position: sec,
        axis: "time",
        trackId: args.trackId,
        clipId: args.clipId,
    }));
    if (markers.length === 1) {
        markers.push({ ...markers[0], role: "source" });
    }
    return {
        id: args.id ?? "primary",
        kind: "loopBoundary",
        markers,
        surface: "timeline",
    };
}

export function snapHighlightKindFromCandidate(
    kind: SnapCandidateKind,
): Exclude<SnapHighlightKind, "loopBoundary"> {
    return kind;
}
