import type { ClipInfo } from "../../../features/session/sessionTypes";
import type { WaveformSceneClip } from "../../../waveform/sceneBuilder";

export interface TakeLaneLayout {
    takeId: string;
    index: number;
    count: number;
    /** 相对 Clip body 顶部的像素位置 */
    top: number;
    height: number;
    inactive: boolean;
}

export const MIN_TAKE_LANE_HEIGHT_PX = 14;

/** 计算一个 Clip 的全部波形 lane；空间不足或非多 Take 时返回 null。 */
export function resolveTakeLaneLayouts(
    clip: ClipInfo,
    showAllTakes: boolean,
    bodyHeightPx: number,
): TakeLaneLayout[] | null {
    if (!showAllTakes) return null;
    // 只为音频 take 建 lane（MIDI take 无波形）。判定依据是 take 集合本身
    // 而非 flat sourcePath —— 混合 MIDI/audio take 的 Clip（active 为 MIDI）
    // flat 投影无源路径，但其余音频 take 仍应可展开、可点击切换。
    const takes = (clip.takes ?? []).filter((take) => Boolean(take.sourcePath));
    if (takes.length <= 1) return null;
    if (takes.length * MIN_TAKE_LANE_HEIGHT_PX > bodyHeightPx + 1e-6) return null;

    const laneHeight = Math.max(MIN_TAKE_LANE_HEIGHT_PX, Math.floor(bodyHeightPx / takes.length));
    return takes.map((take, index) => ({
        takeId: take.id,
        index,
        count: takes.length,
        top: index * laneHeight,
        height:
            index === takes.length - 1
                ? Math.max(MIN_TAKE_LANE_HEIGHT_PX, bodyHeightPx - index * laneHeight)
                : laneHeight,
        inactive: take.id !== clip.activeTakeId,
    }));
}

/** 根据事件相对 Clip body 的本地 Y 坐标命中 inactive take。 */
export function hitInactiveTakeLane(
    clip: ClipInfo,
    showAllTakes: boolean,
    bodyHeightPx: number,
    localY: number,
): TakeLaneLayout | null {
    const lanes = resolveTakeLaneLayouts(clip, showAllTakes, bodyHeightPx);
    if (!lanes) return null;
    // 负值来自 Clip 头部/边框区域（body 上方），不得映射进 lane 0 ——
    // 否则点击头部空白会误触 inactive take 切换。
    if (!(localY >= 0)) return null;
    // 半开区间 [top, top+height)：±1px 双侧容差会让相邻 lane 边界处
    // 同时命中两个区间，find() 恒取上面的 lane，与视觉分界线不符。
    const clampedY = Math.min(bodyHeightPx - 1e-6, localY);
    const lane = lanes.find(
        (entry) => entry.inactive && clampedY >= entry.top && clampedY < entry.top + entry.height,
    );
    return lane ?? null;
}

/** Clip → 共享波形场景 clip（active-take 投影）；无音频源路径返回 null。 */
export function clipToSceneClip(clip: ClipInfo): WaveformSceneClip | null {
    if (!clip.sourcePath) return null;
    return sceneClipProjection(clip, clip.sourcePath);
}

function sceneClipProjection(clip: ClipInfo, sourcePath: string): WaveformSceneClip {
    return {
        id: clip.id,
        sourcePath,
        startSec: clip.startSec,
        lengthSec: clip.lengthSec,
        sourceStartSec: clip.sourceStartSec,
        sourceEndSec: clip.sourceEndSec,
        durationSec: clip.durationSec,
        durationFrames: clip.durationFrames,
        sourceSampleRate: clip.sourceSampleRate,
        playbackRate: clip.playbackRate,
        reversed: clip.reversed,
        loopEnabled: clip.loopEnabled,
        gain: clip.gain,
        muted: clip.muted,
        fadeInSec: clip.fadeInSec,
        fadeOutSec: clip.fadeOutSec,
        autoFadeInSec: clip.autoFadeInSec,
        autoFadeOutSec: clip.autoFadeOutSec,
        fadeInShape: Number.isFinite(clip.fadeInShape) ? clip.fadeInShape : 0,
        fadeInDir: clip.fadeInDir ?? 0,
        fadeOutShape: Number.isFinite(clip.fadeOutShape) ? clip.fadeOutShape : 0,
        fadeOutDir: clip.fadeOutDir ?? 0,
    };
}

/**
 * 把多 Take Clip 展开为每音频 Take 一个场景 clip：各 lane 携带相对 Clip body
 * 顶部的竖直子区间（laneTopPx/laneHeightPx）与 inactive 压暗标记。开关关闭、
 * 非多音频 Take 或行波形带放不下全部 lane 时返回 null，调用方回退为单
 * active-take 投影渲染。
 *
 * active take 消费 Clip 上的乐观投影（slip / trim 的实时预览写在 flat
 * sourceStartSec/sourceEndSec 上），inactive take 用 take 自身窗口；播放倍率
 * 按 clipPlaybackRate × take.playbackRate 组合（与后端合成一致）。合成 id
 * （clip::take::takeId）使 lane 不命中 leadingOverlap 高亮表，与旧 Canvas
 * 多 Take 实现一致。
 */
export function expandClipToTakeSceneClips(
    clip: ClipInfo,
    showAllTakes: boolean,
    bodyHeightPx: number,
): WaveformSceneClip[] | null {
    const layouts = resolveTakeLaneLayouts(clip, showAllTakes, bodyHeightPx);
    if (!layouts) return null;

    const takes = (clip.takes ?? []).filter((take) => Boolean(take.sourcePath));
    const clipRate =
        Number.isFinite(clip.clipPlaybackRate) && (clip.clipPlaybackRate ?? 0) > 0
            ? Number(clip.clipPlaybackRate)
            : 1;

    const expanded: WaveformSceneClip[] = [];
    for (const [index, take] of takes.entries()) {
        const layout = layouts[index];
        if (!layout || !take.sourcePath) continue;
        const isActiveTake = take.id === clip.activeTakeId;
        const rate = clipRate * (Number.isFinite(take.playbackRate) ? take.playbackRate : 1);
        expanded.push({
            ...sceneClipProjection(clip, take.sourcePath),
            id: `${clip.id}::take::${take.id}`,
            durationSec: take.durationSec,
            durationFrames: take.durationFrames,
            sourceSampleRate: take.sourceSampleRate,
            gain: take.gain,
            sourceStartSec: isActiveTake ? clip.sourceStartSec : take.sourceStartSec,
            sourceEndSec: isActiveTake ? clip.sourceEndSec : take.sourceEndSec,
            playbackRate: Number.isFinite(rate) && rate > 0.1 ? Math.min(10, rate) : 1,
            reversed: take.reversed,
            loopEnabled: take.loopEnabled,
            laneTopPx: layout.top,
            laneHeightPx: layout.height,
            inactive: layout.inactive,
        });
    }
    return expanded.length > 0 ? expanded : null;
}
