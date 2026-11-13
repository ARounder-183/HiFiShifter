/**
 * 时间轴手势起点的 Clip 几何投影（纯函数）。
 *
 * 本模块不引用 window / Redux store / Tauri：发布方在时间轴手势开始时调用它，
 * 产出的数据作为 interaction.beginClipGeometryPreview 的 payload 经 detachBridge
 * 复制到卫星窗口。消费方（参数编辑器）因此不再依赖模块级单例。
 */

import {
    toClipSourceGeometry,
    type ClipGeometrySnapshot,
    type ClipSourceGeometry,
} from "../../../features/session/clipGeometryTypes";

export interface ClipGeometryPreviewData {
    origin: ClipGeometrySnapshot[];
    clips: ClipSourceGeometry[];
}

/**
 * 把当前 clip 列表投影成手势快照。
 *
 * 必须在任何乐观写入之前调用；函数内部逐字段深拷贝，调用方之后修改 Redux
 * 对象（Immer 产生新对象，但数组/嵌套对象仍可能被复用）不会污染快照。
 */
export function buildClipGeometryPreview(
    clips: readonly ClipSourceGeometry[],
): ClipGeometryPreviewData {
    const projected = clips.map(toClipSourceGeometry);
    return {
        clips: projected,
        origin: projected.map((clip) => ({
            clipId: clip.id,
            startSec: Number(clip.startSec) || 0,
            lengthSec: Math.max(0, Number(clip.lengthSec) || 0),
            gain: Number.isFinite(clip.gain) ? Number(clip.gain) : 1,
            sourceStartSec: Number(clip.sourceStartSec) || 0,
            sourceEndSec: Number(clip.sourceEndSec) || 0,
            playbackRate: Number(clip.playbackRate) || 1,
            reversed: clip.reversed === true,
            loopEnabled: clip.loopEnabled === true,
            durationSec: Number.isFinite(clip.durationSec) ? Number(clip.durationSec) : null,
            durationFrames: Number.isFinite(clip.durationFrames)
                ? Number(clip.durationFrames)
                : null,
            sourceSampleRate: Number.isFinite(clip.sourceSampleRate)
                ? Number(clip.sourceSampleRate)
                : null,
            sourcePath: typeof clip.sourcePath === "string" ? clip.sourcePath : null,
            midiNoteData:
                clip.midiNoteData?.map((note) => ({ endSec: Number(note.endSec) || 0 })) ?? null,
        })),
    };
}

/** 生成本手势的唯一 token（只用于丢弃迟到的 end，不参与任何业务语义）。 */
export function createClipGeometryPreviewToken(): string {
    const cryptoApi = globalThis.crypto;
    if (typeof cryptoApi?.randomUUID === "function") return cryptoApi.randomUUID();
    return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
}
