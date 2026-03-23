/**
 * Tempo Map utility module.
 *
 * A TempoMap is an ordered list of TempoPoints. Each point defines the BPM and
 * time-signature that apply from its tick position until the next point.
 * The first point is always at positionTicks = 0.
 *
 * All conversions are based on absolute ticks (resolution = ticksPerBeat).
 */

// ────────────────────────────────────────────────────────────────────────────
// Types
// ────────────────────────────────────────────────────────────────────────────

export interface TempoPoint {
    id: string;
    /** Absolute position in ticks (tick 0 = start of timeline) */
    positionTicks: number;
    /** Beats per minute from this point onward */
    bpm: number;
    /** Time-signature numerator (e.g. 3 for 3/4) */
    numerator: number;
    /** Time-signature denominator (e.g. 4 for 3/4) */
    denominator: number;
}

export interface TempoMap {
    /** Tick resolution, e.g. 480 ticks per quarter-note */
    ticksPerBeat: number;
    /** Sorted by positionTicks; first point must be at 0 */
    points: TempoPoint[];
}

export interface BarBeatTick {
    bar: number;   // 1-based
    beat: number;  // 1-based within bar
    tick: number;  // remaining ticks within beat
}

export interface GridLine {
    sec: number;
    kind: "bar" | "beat" | "sub";
    label?: string;  // e.g. "5.1"
}

// ────────────────────────────────────────────────────────────────────────────
// Constants
// ────────────────────────────────────────────────────────────────────────────

export const DEFAULT_TICKS_PER_BEAT = 480;

// ────────────────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────────────────

let _idCounter = 0;
export function createTempoPointId(): string {
    return `tp_${Date.now()}_${++_idCounter}`;
}

/**
 * Create a default (single-point) tempo map from a global BPM + beats-per-bar.
 */
export function createDefaultTempoMap(
    bpm: number,
    beatsPerBar: number,
): TempoMap {
    return {
        ticksPerBeat: DEFAULT_TICKS_PER_BEAT,
        points: [
            {
                id: createTempoPointId(),
                positionTicks: 0,
                bpm: Math.max(10, Math.min(300, bpm)),
                numerator: Math.max(1, Math.min(32, Math.round(beatsPerBar))),
                denominator: 4,
            },
        ],
    };
}

// ────────────────────────────────────────────────────────────────────────────
// Core conversions
// ────────────────────────────────────────────────────────────────────────────

/** Seconds per tick at a given BPM for a given resolution. */
function secPerTick(bpm: number, ticksPerBeat: number): number {
    return 60 / (bpm * ticksPerBeat);
}

/**
 * Find the index of the TempoPoint that governs `ticks`.
 * Points are sorted ascending. We want the last point whose positionTicks <= ticks.
 */
function segmentIndexForTicks(points: TempoPoint[], ticks: number): number {
    let lo = 0;
    let hi = points.length - 1;
    while (lo < hi) {
        const mid = (lo + hi + 1) >> 1;
        if (points[mid].positionTicks <= ticks) lo = mid;
        else hi = mid - 1;
    }
    return lo;
}

/**
 * Convert absolute ticks → seconds.
 */
export function ticksToSeconds(ticks: number, map: TempoMap): number {
    const { points, ticksPerBeat: tpb } = map;
    let sec = 0;
    let remainingTicks = ticks;

    for (let i = 0; i < points.length; i++) {
        const pt = points[i];
        const nextTicks =
            i + 1 < points.length ? points[i + 1].positionTicks : Infinity;
        const segTicks = nextTicks - pt.positionTicks;

        if (remainingTicks <= 0) break;

        if (remainingTicks <= segTicks) {
            sec += remainingTicks * secPerTick(pt.bpm, tpb);
            remainingTicks = 0;
        } else {
            sec += segTicks * secPerTick(pt.bpm, tpb);
            remainingTicks -= segTicks;
        }
    }

    return sec;
}

/**
 * Convert seconds → absolute ticks.
 */
export function secondsToTicks(seconds: number, map: TempoMap): number {
    const { points, ticksPerBeat: tpb } = map;
    let remainingSec = seconds;
    let totalTicks = 0;

    for (let i = 0; i < points.length; i++) {
        const pt = points[i];
        const nextTicks =
            i + 1 < points.length ? points[i + 1].positionTicks : Infinity;
        const segTicks = nextTicks - pt.positionTicks;
        const segSec = segTicks * secPerTick(pt.bpm, tpb);

        if (remainingSec <= 0) break;

        if (remainingSec <= segSec || segSec === Infinity) {
            totalTicks += remainingSec / secPerTick(pt.bpm, tpb);
            remainingSec = 0;
        } else {
            totalTicks += segTicks;
            remainingSec -= segSec;
        }
    }

    return totalTicks;
}

/**
 * Convert absolute ticks → bar:beat:tick (all 1-based for bar/beat).
 */
export function ticksToBarBeatTick(ticks: number, map: TempoMap): BarBeatTick {
    const { points, ticksPerBeat: tpb } = map;
    let remaining = ticks;
    let bar = 1;

    for (let i = 0; i < points.length; i++) {
        const pt = points[i];
        const nextTicks =
            i + 1 < points.length ? points[i + 1].positionTicks : Infinity;
        const segTicks = Math.min(remaining, nextTicks - pt.positionTicks);

        if (segTicks <= 0 && remaining <= 0) break;

        const ticksPerBar = tpb * pt.numerator * (4 / pt.denominator);
        const fullBars = Math.floor(segTicks / ticksPerBar);
        const leftover = segTicks - fullBars * ticksPerBar;

        if (remaining <= segTicks || nextTicks === Infinity) {
            // We end within this segment
            const myTicks = remaining;
            const myBars = Math.floor(myTicks / ticksPerBar);
            const afterBars = myTicks - myBars * ticksPerBar;
            const beatTicks = tpb * (4 / pt.denominator);
            const myBeats = Math.floor(afterBars / beatTicks);
            const myLeftover = afterBars - myBeats * beatTicks;
            return {
                bar: bar + myBars,
                beat: 1 + myBeats,
                tick: Math.round(myLeftover),
            };
        }

        bar += fullBars;
        if (leftover > 1e-9) {
            // Partial bar at segment boundary – count as extra bar
            bar += 1;
        }
        remaining -= segTicks;
    }

    return { bar, beat: 1, tick: 0 };
}

/**
 * Get the TempoPoint that governs a given second position.
 */
export function getTempoAtSec(sec: number, map: TempoMap): TempoPoint {
    const ticks = secondsToTicks(sec, map);
    const idx = segmentIndexForTicks(map.points, ticks);
    return map.points[idx];
}

/**
 * Get the TempoPoint that governs a given tick position.
 */
export function getTempoAtTicks(ticks: number, map: TempoMap): TempoPoint {
    const idx = segmentIndexForTicks(map.points, ticks);
    return map.points[idx];
}

// ────────────────────────────────────────────────────────────────────────────
// Grid line generation
// ────────────────────────────────────────────────────────────────────────────

/**
 * Generate grid lines for a visible time range [startSec, endSec].
 * `subdivision` is expressed in beats (e.g. 0.5 for 1/8 note grid).
 */
export function getGridLines(
    startSec: number,
    endSec: number,
    map: TempoMap,
    subdivision: number,
): GridLine[] {
    const { points, ticksPerBeat: tpb } = map;
    const lines: GridLine[] = [];

    const startTicks = secondsToTicks(Math.max(0, startSec), map);
    const endTicks = secondsToTicks(endSec, map);

    // Walk through tempo segments that overlap [startTicks, endTicks]
    let bar = 1;
    let tickCursor = 0;

    for (let i = 0; i < points.length; i++) {
        const pt = points[i];
        const nextSegStart =
            i + 1 < points.length ? points[i + 1].positionTicks : Infinity;

        const ticksPerBar = tpb * pt.numerator * (4 / pt.denominator);
        const beatTicks = tpb * (4 / pt.denominator);
        const subTicks = tpb * subdivision;

        // Count bars from segment start to tickCursor if tickCursor is ahead
        if (tickCursor < pt.positionTicks) {
            tickCursor = pt.positionTicks;
        }

        // Align tickCursor to the sub-grid within this segment
        const segRelStart = tickCursor - pt.positionTicks;
        const firstSub = Math.ceil(segRelStart / subTicks) * subTicks;
        let t = pt.positionTicks + firstSub;

        // Compute bar count entering this segment
        if (i > 0) {
            // Recalculate bar number up to this segment start
            bar = ticksToBarBeatTick(pt.positionTicks, map).bar;
        }

        const segEnd = Math.min(nextSegStart, endTicks + subTicks);

        while (t <= segEnd) {
            if (t >= startTicks && t <= endTicks) {
                const sec = ticksToSeconds(t, map);
                const relTick = t - pt.positionTicks;
                const barRel = Math.floor(relTick / ticksPerBar);
                const inBar = relTick - barRel * ticksPerBar;

                let kind: GridLine["kind"];
                let label: string | undefined;

                if (Math.abs(inBar) < 0.5) {
                    kind = "bar";
                    label = `${bar + barRel}.1`;
                } else if (Math.abs(inBar % beatTicks) < 0.5) {
                    kind = "beat";
                } else {
                    kind = "sub";
                }
                lines.push({ sec, kind, label });
            }

            t += subTicks;
            if (t > endTicks + subTicks) break;
        }

        tickCursor = Math.min(nextSegStart, endTicks + subTicks);
        if (tickCursor >= endTicks + subTicks) break;
    }

    return lines;
}

// ────────────────────────────────────────────────────────────────────────────
// Snap helpers
// ────────────────────────────────────────────────────────────────────────────

/**
 * Snap a seconds position to the nearest grid line.
 * `subdivision` is in beats (e.g. 0.5 for 1/8).
 */
export function snapSecToGrid(
    sec: number,
    map: TempoMap,
    subdivision: number,
): number {
    const ticks = secondsToTicks(sec, map);
    const subTicks = map.ticksPerBeat * subdivision;
    const snapped = Math.round(ticks / subTicks) * subTicks;
    return ticksToSeconds(snapped, map);
}

// ────────────────────────────────────────────────────────────────────────────
// Bar generation (for TimeRuler)
// ────────────────────────────────────────────────────────────────────────────

export interface TimeRulerBar {
    sec: number;
    label: string;
}

/**
 * Generate bar labels (like "1.1", "2.1", …) for the entire project duration.
 * Each bar's position is in seconds.
 */
export function generateBars(
    projectDurationSec: number,
    map: TempoMap,
    visibleStartSec?: number,
    visibleEndSec?: number,
): TimeRulerBar[] {
    const { points, ticksPerBeat: tpb } = map;
    const bars: TimeRulerBar[] = [];
    const totalTicks = secondsToTicks(projectDurationSec, map);

    let barNumber = 1;
    let tickCursor = 0;

    for (let i = 0; i < points.length; i++) {
        const pt = points[i];
        const nextSegStart =
            i + 1 < points.length ? points[i + 1].positionTicks : Infinity;
        const segEnd = Math.min(nextSegStart, totalTicks);

        const ticksPerBar = tpb * pt.numerator * (4 / pt.denominator);

        if (tickCursor < pt.positionTicks) {
            tickCursor = pt.positionTicks;
        }

        // Align to bar boundary within this segment
        const segRel = tickCursor - pt.positionTicks;
        const firstBarRel = Math.ceil(segRel / ticksPerBar) * ticksPerBar;
        let t = pt.positionTicks + firstBarRel;

        while (t <= segEnd) {
            const sec = ticksToSeconds(t, map);

            // Visibility culling (optional)
            if (
                visibleStartSec == null ||
                visibleEndSec == null ||
                (sec >= visibleStartSec - 5 && sec <= visibleEndSec + 5)
            ) {
                const bbt = ticksToBarBeatTick(t, map);
                bars.push({ sec, label: `${bbt.bar}.1` });
            }

            barNumber++;
            t += ticksPerBar;
        }

        tickCursor = segEnd;
    }

    return bars;
}

// ────────────────────────────────────────────────────────────────────────────
// BPM at position (for playback_rate calculation)
// ────────────────────────────────────────────────────────────────────────────

/**
 * Compute the effective playback_rate for a clip, given:
 * - the BPM at the clip's position in the current tempo map
 * - the BPM at the clip's position in the *old* tempo map (before editing)
 * - the clip's current playback rate
 */
export function computeNewPlaybackRate(
    oldBpm: number,
    newBpm: number,
    currentRate: number,
): number {
    if (oldBpm <= 0 || !Number.isFinite(oldBpm)) return currentRate;
    if (newBpm <= 0 || !Number.isFinite(newBpm)) return currentRate;
    return (newBpm / oldBpm) * currentRate;
}

/**
 * Given a TempoMap serialised from the backend (snake_case), convert to
 * front-end TempoMap.
 */
export function fromBackendTempoMap(
    data: {
        ticks_per_beat: number;
        points: Array<{
            id: string;
            position_ticks: number;
            bpm: number;
            numerator: number;
            denominator: number;
        }>;
    } | null | undefined,
    fallbackBpm: number,
    fallbackBeatsPerBar: number,
): TempoMap {
    if (!data || !data.points || data.points.length === 0) {
        return createDefaultTempoMap(fallbackBpm, fallbackBeatsPerBar);
    }
    return {
        ticksPerBeat: data.ticks_per_beat || DEFAULT_TICKS_PER_BEAT,
        points: data.points.map((p) => ({
            id: p.id,
            positionTicks: p.position_ticks,
            bpm: p.bpm,
            numerator: p.numerator,
            denominator: p.denominator,
        })),
    };
}

/**
 * Convert front-end TempoMap to backend (snake_case) format.
 */
export function toBackendTempoMap(map: TempoMap) {
    return {
        ticks_per_beat: map.ticksPerBeat,
        points: map.points.map((p) => ({
            id: p.id,
            position_ticks: p.positionTicks,
            bpm: p.bpm,
            numerator: p.numerator,
            denominator: p.denominator,
        })),
    };
}
