import { normalizeCustomScaleNotes } from "./musicalScales";

export interface CustomScalePreset {
    id: string;
    name: string;
    notes: number[];
}

export const CHROMATIC_NOTE_LABELS = [
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
    "G#",
    "A",
    "A#",
    "B",
] as const;

export function createCustomScaleId(): string {
    return `custom_${Math.random().toString(36).slice(2, 10)}`;
}

export function sanitizeCustomScalePreset(
    input: Partial<CustomScalePreset>,
): CustomScalePreset {
    const name = String(input.name ?? "").trim() || "Custom Scale";
    const id = String(input.id ?? "").trim() || createCustomScaleId();
    const notes = normalizeCustomScaleNotes(input.notes ?? [0, 2, 4, 5, 7, 9, 11]);
    return { id, name, notes };
}

export function formatScaleNotes(notes: readonly number[]): string {
    const pcs = normalizeCustomScaleNotes(notes);
    return pcs.map((pc) => CHROMATIC_NOTE_LABELS[pc]).join(" ");
}
