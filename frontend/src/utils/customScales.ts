import { translateOutsideReact } from "../i18n/I18nProvider";
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

export function sanitizeCustomScalePreset(input: Partial<CustomScalePreset>): CustomScalePreset {
    // 未命名音阶的默认名在**落盘**时生成（与 hifiClipBlock 的 defaultClipTitle 同一
    // 情形）：调用点没有组件上下文，走组件外的 translateOutsideReact，与界面语言
    // 同源 —— 此前硬编码英文，菜单里非英语语系会看到 "Custom Scale"。
    const name =
        String(input.name ?? "").trim() || translateOutsideReact("custom_scale_default_name");
    const id = String(input.id ?? "").trim() || createCustomScaleId();
    const notes = normalizeCustomScaleNotes(input.notes ?? [0, 2, 4, 5, 7, 9, 11]);
    return { id, name, notes };
}

export function formatScaleNotes(notes: readonly number[]): string {
    const pcs = normalizeCustomScaleNotes(notes);
    return pcs.map((pc) => CHROMATIC_NOTE_LABELS[pc]).join(" ");
}
