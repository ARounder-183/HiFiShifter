export function hasFileDrag(dt: DataTransfer): boolean {
    if (!dt) return false;
    if (dt.files && dt.files.length > 0) return true;
    const types = Array.from(dt.types ?? []);
    if (types.includes("Files")) return true;
    const items = Array.from(dt.items ?? []);
    return items.some((it) => it.kind === "file");
}

export function extractLocalFilePath(dt: DataTransfer): { path: string; name: string } | null {
    type MaybePathFile = File & { path?: string };

    const itemFile = Array.from(dt.items ?? [])
        .find((it) => it.kind === "file")
        ?.getAsFile() as MaybePathFile | null;
    const file = (dt.files?.[0] as MaybePathFile | undefined) ?? itemFile;

    const directPath = String(file?.path ?? "").trim();
    if (directPath) {
        return {
            path: directPath,
            name: String(file?.name ?? directPath),
        };
    }

    const uriList = String(dt.getData("text/uri-list") ?? "").trim();
    if (uriList) {
        const first = uriList
            .split(/\r?\n/)
            .map((line) => line.trim())
            .find((line) => line && !line.startsWith("#"));
        if (first) {
            try {
                const url = new URL(first);
                if (url.protocol === "file:") {
                    let p = decodeURIComponent(url.pathname);
                    if (/^\/[A-Za-z]:\//.test(p)) p = p.slice(1);
                    if (p) {
                        return {
                            path: p,
                            name: String(file?.name ?? p),
                        };
                    }
                }
            } catch {
                // ignore
            }
        }
    }

    const text = String(dt.getData("text/plain") ?? "").trim();
    if (text && (text.includes("\\") || /^[A-Za-z]:\\/.test(text))) {
        return {
            path: text,
            name: String(file?.name ?? text),
        };
    }

    return null;
}

export function isProjectFilePath(path: string | null | undefined): boolean {
    const normalized = String(path ?? "").trim();
    if (!normalized) return false;
    return /\.(hshp|hsp|json)$/i.test(normalized);
}

export function isReaperProjectFilePath(path: string | null | undefined): boolean {
    const normalized = String(path ?? "").trim();
    if (!normalized) return false;
    return /\.rpp$/i.test(normalized);
}

export function isVocalShifterProjectFilePath(path: string | null | undefined): boolean {
    const normalized = String(path ?? "").trim();
    if (!normalized) return false;
    return /\.(vshp|vsp)$/i.test(normalized);
}

const AUDIO_FILE_RE =
    /\.(wav|flac|mp3|ogg|oga|opus|aac|m4a|aif|aiff|wma|ac3|eac3|ape|wv|mp2|mpa|dts|amr)$/i;
const VIDEO_FILE_RE =
    /\.(mp4|m4v|mov|mkv|webm|avi|flv|wmv|ts|mts|m2ts|vob|mpg|mpeg|3gp|3g2|ogv|rm|rmvb)$/i;

export function isVideoFilePath(path: string | null | undefined): boolean {
    const normalized = String(path ?? "").trim();
    if (!normalized) return false;
    return VIDEO_FILE_RE.test(normalized);
}

/** @deprecated 使用 isMediaFilePath（音频 + 视频容器） */
export function isAudioFilePath(path: string | null | undefined): boolean {
    const normalized = String(path ?? "").trim();
    if (!normalized) return false;
    return AUDIO_FILE_RE.test(normalized) || VIDEO_FILE_RE.test(normalized);
}

export function isMediaFilePath(path: string | null | undefined): boolean {
    return isAudioFilePath(path);
}

export function isMidiFilePath(path: string | null | undefined): boolean {
    const normalized = String(path ?? "").trim();
    if (!normalized) return false;
    return /\.(mid|midi)$/i.test(normalized);
}

export type ExternalPathActionKind =
    "openProject" | "importReaper" | "importVocalShifter" | "importAudio" | "importMidi";

export function detectExternalPathAction(
    path: string | null | undefined,
): ExternalPathActionKind | null {
    const normalized = String(path ?? "").trim();
    if (!normalized) return null;
    if (isProjectFilePath(normalized)) return "openProject";
    if (isReaperProjectFilePath(normalized)) return "importReaper";
    if (isVocalShifterProjectFilePath(normalized)) return "importVocalShifter";
    if (isAudioFilePath(normalized)) return "importAudio";
    if (isMidiFilePath(normalized)) return "importMidi";
    return null;
}

export function findFirstProjectFilePath(paths: Array<string | null | undefined>): string | null {
    for (const raw of paths) {
        const normalized = String(raw ?? "").trim();
        if (isProjectFilePath(normalized)) return normalized;
    }
    return null;
}

export function findFirstExternalPathAction(
    paths: Array<string | null | undefined>,
): { path: string; kind: ExternalPathActionKind } | null {
    for (const raw of paths) {
        const normalized = String(raw ?? "").trim();
        const kind = detectExternalPathAction(normalized);
        if (kind) {
            return { path: normalized, kind };
        }
    }
    return null;
}
