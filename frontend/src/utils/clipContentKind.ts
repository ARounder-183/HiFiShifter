/**
 * Clip 内容类型：音频 / 音高参考 / 普通 MIDI Take。
 *
 * 这个判据以前散落在各个组件里，而且大多数只检查 midiNoteCount，导致
 * MIDI Take（midiTakeNoteCount）在 header 命中、提示、菜单里被当成音频。
 * 统一到这里，role 永远优先于旧镜像字段。
 */
export type ClipContentKind = "audio" | "pitch_reference" | "midi_take";

export interface ClipContentFields {
    midiRole?: "midi_take" | "pitch_reference" | string | null;
    midiNoteCount?: number | null;
    midiTakeNoteCount?: number | null;
}

export function resolveClipContentKind(
    clip: ClipContentFields | null | undefined,
): ClipContentKind {
    if (!clip) return "audio";
    if (clip.midiRole === "pitch_reference") return "pitch_reference";
    if (clip.midiRole === "midi_take") return "midi_take";
    // 旧载荷没有 role：有 midi_take 音符计数即新 MIDI Take；否则旧
    // midi_note_data 镜像按历史语义就是音高参考。
    if (clip.midiTakeNoteCount != null) return "midi_take";
    if (clip.midiNoteCount != null) return "pitch_reference";
    return "audio";
}

export function isMidiContentClip(clip: ClipContentFields | null | undefined): boolean {
    return resolveClipContentKind(clip) !== "audio";
}

export function isPitchReferenceClip(clip: ClipContentFields | null | undefined): boolean {
    return resolveClipContentKind(clip) === "pitch_reference";
}

export function isMidiTakeClip(clip: ClipContentFields | null | undefined): boolean {
    return resolveClipContentKind(clip) === "midi_take";
}
