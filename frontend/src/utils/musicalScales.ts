/**
 * Musical scale key signatures for pitch snap feature.
 * SCALE_KEYS: key name (e.g. "C", "Db", ...)
 * SCALE_LABELS: human-readable label (e.g. "C / Am", "D♭ / B♭m")
 *
 * Note: SCALE_LABELS 不是 i18n key，而是直接用于显示的本地化友好标签。
 */
export const SCALE_KEYS = [
    "C",
    "Db",
    "D",
    "Eb",
    "E",
    "F",
    "Gb",
    "G",
    "Ab",
    "A",
    "Bb",
    "B",
] as const;

export type ScaleKey = (typeof SCALE_KEYS)[number];
export type ScaleLike = ScaleKey | readonly number[];

/**
 * Human-readable label for each scale (major/minor pair).
 * e.g. "C" => "C / Am"
 */
export const SCALE_LABELS: Record<ScaleKey, string> = {
    C: "C / Am",
    Db: "D\u266D / B\u266Dm",
    D: "D / Bm",
    Eb: "E\u266D / Cm",
    E: "E / C\u266Fm",
    F: "F / Dm",
    Gb: "G\u266D / E\u266Dm",
    G: "G / Em",
    Ab: "A\u266D / Fm",
    A: "A / F\u266Fm",
    Bb: "B\u266D / Gm",
    B: "B / G\u266Fm",
};

/**
 * MIDI note numbers in one octave that belong to each major scale.
 * 0 = C, 1 = C#/Db, ... 11 = B
 */
export const SCALE_NOTES: Record<ScaleKey, number[]> = {
    C: [0, 2, 4, 5, 7, 9, 11],
    Db: [1, 3, 5, 6, 8, 10, 0],
    D: [2, 4, 6, 7, 9, 11, 1],
    Eb: [3, 5, 7, 8, 10, 0, 2],
    E: [4, 6, 8, 9, 11, 1, 3],
    F: [5, 7, 9, 10, 0, 2, 4],
    Gb: [6, 8, 10, 11, 1, 3, 5],
    G: [7, 9, 11, 0, 2, 4, 6],
    Ab: [8, 10, 0, 1, 3, 5, 7],
    A: [9, 11, 1, 2, 4, 6, 8],
    Bb: [10, 0, 2, 3, 5, 7, 9],
    B: [11, 1, 3, 4, 6, 8, 10],
};

function normalizePitchClasses(notes: readonly number[]): number[] {
    const unique = new Set<number>();
    for (const n of notes) {
        if (!Number.isFinite(n)) continue;
        const pc = ((Math.round(n) % 12) + 12) % 12;
        unique.add(pc);
    }
    const out = Array.from(unique).sort((a, b) => a - b);
    return out.length > 0 ? out : [...SCALE_NOTES.C];
}

export function isScaleKey(value: string): value is ScaleKey {
    return (SCALE_KEYS as readonly string[]).includes(value);
}

export function normalizeCustomScaleNotes(notes: readonly number[]): number[] {
    return normalizePitchClasses(notes);
}

export function resolveScaleNotes(scale: ScaleLike): number[] {
    if (Array.isArray(scale)) {
        return normalizePitchClasses(scale);
    }
    const key = scale as ScaleKey;
    return SCALE_NOTES[key] ?? SCALE_NOTES.C;
}

/**
 * Snap a MIDI note number to the nearest note in the given scale.
 */
export function snapToScale(midiNote: number, scale: ScaleLike): number {
    const notes = resolveScaleNotes(scale);
    // const pitchClass = ((midiNote % 12) + 12) % 12; // 已删除未使用变量
    const octave = Math.floor(midiNote / 12);

    let bestDist = Infinity;
    let bestNote = midiNote;
    for (const n of notes) {
        // Check same octave and adjacent octaves
        for (const oct of [octave - 1, octave, octave + 1]) {
            const candidate = oct * 12 + n;
            const dist = Math.abs(candidate - midiNote);
            if (dist < bestDist) {
                bestDist = dist;
                bestNote = candidate;
            }
        }
    }
    return bestNote;
}

/**
 * Snap a MIDI note to the nearest semitone (round to integer).
 */
export function snapToSemitone(midiNote: number): number {
    return Math.round(midiNote);
}

type ScaleDegreeAnchor = {
    absDegree: number;
    midi: number;
};

function floorDiv(a: number, b: number): number {
    return Math.floor(a / b);
}

function positiveMod(a: number, b: number): number {
    return ((a % b) + b) % b;
}

/**
 * Convert user-facing degree input to scale-step shift.
 * Music theory convention:
 * - 0 / 1 / -1 => 0 step
 * - +3 => +2 steps
 * - -3 => -2 steps
 * - +8 => +7 steps (one octave)
 * - -8 => -7 steps (one octave)
 *
 * Fractional mapping keeps ratio between neighboring valid degree labels:
 * - +3.5 => +2.5 steps
 * - -2.25 => -1.25 steps
 */
export function degreeInputToScaleSteps(inputDegrees: number): number {
    if (!Number.isFinite(inputDegrees)) return 0;
    if (inputDegrees > 1) return inputDegrees - 1;
    if (inputDegrees < -1) return inputDegrees + 1;
    return 0;
}

/**
 * Convert internal degree-step shift back to user-facing music-theory degree labels.
 *
 * Internal steps intentionally avoid +/-1 in user display:
 * - 0 => 0
 * - +2 => +3
 * - -1 => -2
 */
export function scaleStepsToDegreeDisplay(degreeSteps: number): number {
    if (!Number.isFinite(degreeSteps)) return 0;
    if (degreeSteps > 0) return degreeSteps + 1;
    if (degreeSteps < 0) return degreeSteps - 1;
    return 0;
}

/**
 * Build ascending scale semitone offsets for one octave cycle relative to C.
 * For non-C keys, wrapped notes are lifted by +12 to keep degree order monotonic.
 * Example Db major raw [1,3,5,6,8,10,0] => [1,3,5,6,8,10,12]
 */
function orderedScaleSemitoneOffsets(scale: ScaleLike): number[] {
    const raw = resolveScaleNotes(scale);
    if (raw.length === 0) return [];
    const out: number[] = [];
    let prev = -Infinity;
    for (const pcRaw of raw) {
        let v = ((pcRaw % 12) + 12) % 12;
        while (v <= prev) v += 12;
        out.push(v);
        prev = v;
    }
    return out;
}

function scaleDegreeToMidi(absDegree: number, scale: ScaleLike): number {
    const offsets = orderedScaleSemitoneOffsets(scale);
    const degreeCount = offsets.length;
    if (degreeCount === 0) return 0;

    const targetOct = floorDiv(absDegree, degreeCount);
    const targetIdx = positiveMod(absDegree, degreeCount);
    return targetOct * 12 + offsets[targetIdx];
}

function scaleDegreeToMidiFractional(absDegree: number, scale: ScaleLike): number {
    if (!Number.isFinite(absDegree)) return 0;
    const lowerDegree = Math.floor(absDegree);
    const frac = absDegree - lowerDegree;
    if (frac <= 1e-9) {
        return scaleDegreeToMidi(lowerDegree, scale);
    }
    const lowerMidi = scaleDegreeToMidi(lowerDegree, scale);
    const upperMidi = scaleDegreeToMidi(lowerDegree + 1, scale);
    return lowerMidi + (upperMidi - lowerMidi) * frac;
}

function getScaleDegreeAnchorsAroundMidi(
    midi: number,
    scale: ScaleLike,
): { lower: ScaleDegreeAnchor; upper: ScaleDegreeAnchor; ratio: number } {
    const offsets = orderedScaleSemitoneOffsets(scale);
    const degreeCount = offsets.length;
    if (degreeCount === 0) {
        return {
            lower: { absDegree: 0, midi },
            upper: { absDegree: 0, midi },
            ratio: 0,
        };
    }

    const baseOct = Math.floor(midi / 12);
    let lower: ScaleDegreeAnchor | null = null;
    let upper: ScaleDegreeAnchor | null = null;

    for (let oct = baseOct - 3; oct <= baseOct + 3; oct++) {
        for (let i = 0; i < degreeCount; i++) {
            const candidateMidi = oct * 12 + offsets[i];
            const candidate: ScaleDegreeAnchor = {
                absDegree: oct * degreeCount + i,
                midi: candidateMidi,
            };
            if (candidateMidi <= midi) {
                if (lower == null || candidateMidi > lower.midi) {
                    lower = candidate;
                }
            }
            if (candidateMidi >= midi) {
                if (upper == null || candidateMidi < upper.midi) {
                    upper = candidate;
                }
            }
        }
    }

    const safeLower = lower ?? { absDegree: 0, midi };
    const safeUpper = upper ?? safeLower;
    const span = safeUpper.midi - safeLower.midi;
    const ratio = span <= 1e-9 ? 0 : (midi - safeLower.midi) / span;

    return {
        lower: safeLower,
        upper: safeUpper,
        ratio: Math.max(0, Math.min(1, ratio)),
    };
}

function nearestScaleAnchor(
    midi: number,
    scale: ScaleLike,
): { absDegree: number; baseMidi: number; residual: number } {
    const offsets = orderedScaleSemitoneOffsets(scale);
    const degreeCount = offsets.length;
    const baseOct = Math.floor(midi / 12);

    let bestDist = Number.POSITIVE_INFINITY;
    let bestAbsDegree = 0;
    let bestBaseMidi = midi;

    for (let oct = baseOct - 3; oct <= baseOct + 3; oct++) {
        for (let i = 0; i < degreeCount; i++) {
            const candidate = oct * 12 + offsets[i];
            const dist = Math.abs(midi - candidate);
            if (dist < bestDist || (dist === bestDist && candidate < bestBaseMidi)) {
                bestDist = dist;
                bestAbsDegree = oct * degreeCount + i;
                bestBaseMidi = candidate;
            }
        }
    }

    return {
        absDegree: bestAbsDegree,
        baseMidi: bestBaseMidi,
        residual: midi - bestBaseMidi,
    };
}

/**
 * Transpose a MIDI pitch by scale-degree steps while preserving microtonal residual.
 * degreeSteps is internal step shift (already converted from user input if needed).
 */
export function transposePitchByScaleSteps(
    midi: number,
    degreeSteps: number,
    scale: ScaleLike,
): number {
    if (!Number.isFinite(midi) || !Number.isFinite(degreeSteps)) return midi;
    if (degreeSteps === 0) return midi;

    const stepShift = degreeSteps;
    const { lower, upper, ratio } = getScaleDegreeAnchorsAroundMidi(midi, scale);

    const targetLowerMidi = scaleDegreeToMidiFractional(lower.absDegree + stepShift, scale);
    const targetUpperMidi = scaleDegreeToMidiFractional(upper.absDegree + stepShift, scale);

    if (Math.abs(upper.midi - lower.midi) <= 1e-9) {
        return targetLowerMidi;
    }

    return targetLowerMidi + (targetUpperMidi - targetLowerMidi) * ratio;
}

/**
 * Calculate degree-step delta between two MIDI positions on the same scale.
 * Used for interactive drag-transpose by degrees.
 */
export function scaleStepDeltaBetween(fromMidi: number, toMidi: number, scale: ScaleLike): number {
    if (!Number.isFinite(fromMidi) || !Number.isFinite(toMidi)) return 0;
    const from = nearestScaleAnchor(fromMidi, scale);
    const to = nearestScaleAnchor(toMidi, scale);
    return to.absDegree - from.absDegree;
}
