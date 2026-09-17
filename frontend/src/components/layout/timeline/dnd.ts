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
    // 含备份工程文件（.hshp-bak / .hsp-bak）：格式与 .hshp/.hsp 相同，
    // 系统级拖放/启动参数打开备份文件时按打开工程处理。
    return /\.(hshp|hsp|hshp-bak|hsp-bak|json)$/i.test(normalized);
}

export function isReaperProjectFilePath(path: string | null | undefined): boolean {
    const normalized = String(path ?? "").trim();
    if (!normalized) return false;
    return /\.(rpp|rpp-bak)$/i.test(normalized);
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
    // `smf`（Standard MIDI File）与后端的 `SUPPORTED_MIDI_EXTS` 保持一致：
    // 后端接受 mid / midi / smf，前端少认一种会让这类文件在拖放时被当成"未知扩展名"
    // 而走音频导入分支（随后在解码阶段失败）。
    return /\.(mid|midi|smf)$/i.test(normalized);
}

export type ExternalPathActionKind =
    | "openProject"
    | "importReaper"
    | "importVocalShifter"
    | "importAudio"
    | "importMidi";

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

/**
 * 拖放**准入判据**：该路径是否属于本应用接受的类型。
 *
 * 【这是一个真实缺陷的根因所对应的修复点】
 * 拖放此前**接受任意扩展名**并一律按音频导入：`detectExternalPathAction` 对未知
 * 扩展名返回 `null`，而各 drop 分支只对"已知的**非音频**种类"（工程 / REAPER /
 * VocalShifter / MIDI）做了特殊处理，`null` 于是**穿透**到 `importAudioAtPosition`
 * 的默认分支，被当成 Clip 添加。后端只做内容嗅探（Symphonia），于是任何能被解码的
 * 文件——甚至扩展名完全无关的文件——都会被接受。
 *
 * 本函数把"是否接受"收成**唯一判据**：`detectExternalPathAction(path) !== null`，
 * 即路径必须落在下面几类之一：
 * 1. 媒体文件（音频 + 视频容器，见 {@link isMediaFilePath}）；
 * 2. 本工程文件（`.hshp` / `.hsp` / `.json`），**含** `-bak` 备份后缀
 *    （见 {@link isProjectFilePath}）；
 * 3. 本项目可导入的外部工程格式（REAPER `.rpp`、VocalShifter `.vshp` / `.vsp`），
 *    含 REAPER 的 `-bak` 备份；
 * 4. MIDI 文件（`.mid` / `.midi` / `.smf`）。
 *
 * 其余一律拒绝拖放（不建 Clip、不弹预览）。
 *
 * 特殊说明：判据建立在 `detectExternalPathAction` 之上而不是另写一份正则——两者一旦
 * 分叉，就会出现"预览接受但落下被拒"（或反之）这类只能靠肉眼发现的不一致。
 *
 * @param path 文件路径（可为空 / null）。
 * @returns 是否接受该路径的拖放。
 */
export function isAcceptedDropPath(path: string | null | undefined): boolean {
    return detectExternalPathAction(path) !== null;
}

/**
 * DOM `File` 对象的准入判据（浏览器 HTML5 拖放且拿不到本地路径时的兜底）。
 *
 * 【为什么需要单独一个入口】浏览器有时只提供 `File` 对象而没有 `path` 属性，
 * 此时无法用扩展名正则直接判——只能综合 `file.name` 与 `file.type`（MIME）。
 * 此前该分支**完全不做校验**就把文件按音频导入。
 *
 * 判据（任一成立即接受）：
 * - `name` 的扩展名落在 {@link isAcceptedDropPath} 的白名单内；
 * - `type` 是 `audio/*` 或 `video/*`（MIME 明确表态；MIDI 的标准 MIME 也属于
 *   `audio/*`）。
 *
 * 特殊说明：`type` 为空字符串是常态（系统不知道 MIME），因此**不能**据此拒绝，
 * 必须回落到扩展名判定。
 *
 * @param file 浏览器提供的文件对象（只读 `name` / `type`）。
 * @returns 是否接受该文件的拖放。
 */
export function isAcceptedDropFile(
    file: { name?: string; type?: string } | null | undefined,
): boolean {
    if (!file) return false;
    const name = String(file.name ?? "").trim();
    if (name && isAcceptedDropPath(name)) return true;
    const mime = String(file.type ?? "").trim().toLowerCase();
    if (mime.startsWith("audio/") || mime.startsWith("video/")) return true;
    return false;
}

/** 一批拖放路径按"落下后应做什么"分类的结果。 */
export interface DroppedPathPartition {
    /** 工程文件（打开 / 导入工程）；取第一个即可。 */
    readonly projectPath: string | null;
    /** MIDI 文件（导入为 MIDI clip）。 */
    readonly midiPaths: readonly string[];
    /** 媒体文件（按音频导入为 Clip）。 */
    readonly mediaPaths: readonly string[];
    /** 不接受、已丢弃的路径（仅用于日志 / 诊断）。 */
    readonly rejectedPaths: readonly string[];
}

/**
 * 把一批拖放路径按准入判据分类。
 *
 * 【为什么"多文件"也要分类而不是整体转发】此前多文件分支把**全部**路径直接交给
 * `importMultipleAudioAtPosition`，于是混在其中的 MIDI / 工程 / 未知文件都会被当作
 * 音频导入（前两者解析失败、后者被后端内容嗅探放行成 Clip）。分类后各类走各自的
 * 正确处理路径，未知类型整体丢弃。
 *
 * 特殊说明 1：工程文件只取**第一个**——一次拖放多个工程没有明确语义，打开多个会互相
 * 覆盖当前会话。
 * 特殊说明 2：媒体文件的相对顺序保留原样（用户拖放顺序即期望的落点顺序）。
 *
 * @param paths 拖放路径序列（可为空）。
 * @returns 分类结果；`rejectedPaths` 仅含未通过准入判据的项。
 */
export function partitionDroppedPaths(
    paths: readonly (string | null | undefined)[],
): DroppedPathPartition {
    let projectPath: string | null = null;
    const midiPaths: string[] = [];
    const mediaPaths: string[] = [];
    const rejectedPaths: string[] = [];

    for (const raw of paths) {
        const normalized = String(raw ?? "").trim();
        if (!normalized) continue;
        const kind = detectExternalPathAction(normalized);
        if (kind === null) {
            rejectedPaths.push(normalized);
            continue;
        }
        if (kind === "openProject" || kind === "importReaper" || kind === "importVocalShifter") {
            if (projectPath === null) projectPath = normalized;
            continue;
        }
        if (kind === "importMidi") {
            midiPaths.push(normalized);
            continue;
        }
        mediaPaths.push(normalized);
    }

    return { projectPath, midiPaths, mediaPaths, rejectedPaths };
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
