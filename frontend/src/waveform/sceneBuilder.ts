import {
    modEuclid,
    resolveLoopMediaDurationSec,
    resolvePlaybackWindowSec,
} from "../utils/loopRender.ts";

export interface WaveformSceneClip {
    id: string;
    sourcePath: string;
    startSec: number;
    lengthSec: number;
    sourceStartSec: number;
    sourceEndSec: number;
    durationSec?: number;
    durationFrames?: number;
    sourceSampleRate?: number;
    playbackRate: number;
    reversed: boolean;
    loopEnabled: boolean;
    gain: number;
    muted: boolean;
    fadeInSec: number;
    fadeOutSec: number;
    autoFadeInSec?: number;
    autoFadeOutSec?: number;
    /** REAPER 风格淡入形状 id（同 ClipInfo.fadeInShape，见 reaperFade.ts）。 */
    fadeInShape: number;
    fadeInDir: number;
    /** REAPER 风格淡出形状 id（语义同 fadeInShape）。 */
    fadeOutShape: number;
    fadeOutDir: number;
    /** 多 Take 展开：该 lane 相对行波形带（body）顶部的竖直偏移；未设置时用行波形带。 */
    laneTopPx?: number;
    /** 多 Take 展开：该 lane 的高度；未设置时用行波形带高度。 */
    laneHeightPx?: number;
    /** inactive take lane：波形整体压暗（几何层按此调暗颜色与透明度）。 */
    inactive?: boolean;
}

export interface WaveformSceneRow {
    topPx: number;
    waveformTopPx: number;
    waveformHeightPx: number;
    clips: readonly WaveformSceneClip[];
    leadingOverlapSecByClipId?: Readonly<Record<string, number>>;
}

export interface WaveformSceneSegment {
    clipId: string;
    sourcePath: string;
    sourceSampleRate: number;
    sourceStartSec: number;
    sourceEndSec: number;
    clipLocalStartSec: number;
    clipLocalEndSec: number;
    clipTotalDurationSec: number;
    screenRect: { x: number; y: number; width: number; height: number };
    reversed: boolean;
    gain: number;
    fadeInSec: number;
    fadeOutSec: number;
    /** REAPER 风格淡入形状 id（同 WaveformSceneClip，见 reaperFade.ts）。 */
    fadeInShape: number;
    fadeInDir: number;
    fadeOutShape: number;
    fadeOutDir: number;
    alpha: number;
    /** inactive take lane：几何层据此压暗顶点颜色。 */
    inactive?: boolean;
}

export interface WaveformSceneMarker {
    clipId: string;
    timelineSec: number;
    xPx: number;
    yPx: number;
    heightPx: number;
    kind: "loop" | "media-boundary";
    /** inactive take lane：几何层据此压暗标记颜色。 */
    inactive?: boolean;
}

export interface WaveformScene {
    segments: WaveformSceneSegment[];
    markers: WaveformSceneMarker[];
}

interface SourceTile {
    localStartSec: number;
    durationSec: number;
    sourceStartSec: number;
    sourceEndSec: number;
}

/**
 * inactive take lane 的压暗合成（复刻旧 Canvas 多 Take 实现的两段系数）：
 * 场景层先把 segment alpha 乘 LANE_ALPHA，几何层再把顶点颜色 rgb 乘
 * RGB_SCALE、颜色 alpha 乘 COLOR_ALPHA —— 总透明度 ≈ 0.61，与旧
 * darkenWaveformStroke + globalAlpha 的叠加观感一致。
 */
export const INACTIVE_TAKE_LANE_ALPHA = 0.78;
export const INACTIVE_TAKE_RGB_SCALE = 0.42;
export const INACTIVE_TAKE_COLOR_ALPHA = 0.78;

function finitePositive(value: number, fallback: number): number {
    return Number.isFinite(value) && value > 1e-6 ? value : fallback;
}

function effectiveFade(auto: number | undefined, manual: number): number {
    const automatic = Number(auto ?? 0);
    return automatic > 0 ? automatic : Math.max(0, Number(manual) || 0);
}

function validLocalInterval(
    tile: SourceTile,
    reversed: boolean,
    playbackRate: number,
    mediaDurationSec: number,
    localStartSec: number,
    localEndSec: number,
): [number, number] | null {
    if (!(mediaDurationSec > 0)) return null;

    const domainStart = reversed
        ? tile.localStartSec + (tile.sourceEndSec - mediaDurationSec) / playbackRate
        : tile.localStartSec + (0 - tile.sourceStartSec) / playbackRate;
    const domainEnd = reversed
        ? tile.localStartSec + tile.sourceEndSec / playbackRate
        : tile.localStartSec + (mediaDurationSec - tile.sourceStartSec) / playbackRate;
    const start = Math.max(localStartSec, Math.min(domainStart, domainEnd));
    const end = Math.min(localEndSec, Math.max(domainStart, domainEnd));
    return end > start + 1e-9 ? [start, end] : null;
}

function sourceRangeForLocal(
    tile: SourceTile,
    reversed: boolean,
    playbackRate: number,
    localStartSec: number,
    localEndSec: number,
): [number, number] {
    if (reversed) {
        return [
            tile.sourceEndSec - (localEndSec - tile.localStartSec) * playbackRate,
            tile.sourceEndSec - (localStartSec - tile.localStartSec) * playbackRate,
        ];
    }
    return [
        tile.sourceStartSec + (localStartSec - tile.localStartSec) * playbackRate,
        tile.sourceStartSec + (localEndSec - tile.localStartSec) * playbackRate,
    ];
}

export function buildWaveformScene(args: {
    viewportStartSec: number;
    viewportEndSec: number;
    pxPerSec: number;
    widthPx: number;
    /** 行 topPx 所在坐标系（内容绝对）到画布坐标系的竖直偏移
     * （= scrollTopPx）。波形面画布视口锚定，须按此平移后才与 DOM
     * 内容层在竖直滚动中同帧对齐。 */
    viewportTopPx?: number;
    rows: readonly WaveformSceneRow[];
}): WaveformScene {
    const segments: WaveformSceneSegment[] = [];
    const markers: WaveformSceneMarker[] = [];
    const pxPerSec = finitePositive(args.pxPerSec, 1);
    const viewportStartSec = Number.isFinite(args.viewportStartSec) ? args.viewportStartSec : 0;
    const viewportEndSec = Math.max(viewportStartSec, args.viewportEndSec);
    const widthPx = Math.max(1, args.widthPx);

    const viewportTopPx = Number.isFinite(args.viewportTopPx) ? (args.viewportTopPx ?? 0) : 0;

    for (const row of args.rows) {
        // 行 topPx 为内容绝对坐标：减去视口顶端得到画布坐标。
        const rowTopCanvasPx = row.topPx - viewportTopPx;
        for (const clip of row.clips) {
            if (!clip.sourcePath || !(clip.lengthSec > 1e-9)) continue;
            const clipEndSec = clip.startSec + clip.lengthSec;
            const visibleStartSec = Math.max(clip.startSec, viewportStartSec);
            const visibleEndSec = Math.min(clipEndSec, viewportEndSec);
            if (visibleEndSec <= visibleStartSec) continue;

            const mediaDurationSec = resolveLoopMediaDurationSec(clip);
            if (!(mediaDurationSec > 1e-9)) continue;
            const playbackRate = finitePositive(clip.playbackRate, 1);
            const reversed = Boolean(clip.reversed);
            const loopEnabled = Boolean(clip.loopEnabled);
            const sourceStartSec = Number(clip.sourceStartSec) || 0;
            const sourceEndSec = Number(clip.sourceEndSec) || mediaDurationSec;
            const window = resolvePlaybackWindowSec({
                loopEnabled,
                reversed,
                sourceStartSec,
                sourceEndSec,
                playbackRate,
                lengthSec: clip.lengthSec,
            });
            const visibleLocalStartSec = visibleStartSec - clip.startSec;
            const visibleLocalEndSec = visibleEndSec - clip.startSec;
            // 多 Take lane 覆盖：laneTopPx 相对行波形带（body）顶部；未展开的
            // clip 两个覆盖都缺省，正好退回行波形带。
            const bandTopPx = row.waveformTopPx + (clip.laneTopPx ?? 0);
            const bandHeightPx = Math.max(1, clip.laneHeightPx ?? row.waveformHeightPx);
            const tiles: SourceTile[] = [];

            if (!loopEnabled) {
                tiles.push({
                    localStartSec: 0,
                    durationSec: clip.lengthSec,
                    sourceStartSec: window.winStartSec,
                    sourceEndSec: window.winEndSec,
                });
            } else {
                const anchorForward = modEuclid(sourceStartSec, mediaDurationSec);
                const anchorReverse = modEuclid(sourceEndSec, mediaDurationSec);
                const headDurationSec =
                    (reversed ? anchorReverse : mediaDurationSec - anchorForward) / playbackRate;
                const periodSec = mediaDurationSec / playbackRate;

                if (headDurationSec > 1e-9 && visibleLocalStartSec < headDurationSec) {
                    tiles.push({
                        localStartSec: 0,
                        durationSec: headDurationSec,
                        sourceStartSec: reversed ? 0 : anchorForward,
                        sourceEndSec: reversed ? anchorReverse : mediaDurationSec,
                    });
                }

                const firstPeriod = Math.max(
                    0,
                    Math.floor((visibleLocalStartSec - headDurationSec - 1e-9) / periodSec),
                );
                for (
                    let localStartSec = headDurationSec + firstPeriod * periodSec, guard = 0;
                    localStartSec < visibleLocalEndSec - 1e-9 && guard < 4096;
                    localStartSec += periodSec, guard += 1
                ) {
                    tiles.push({
                        localStartSec,
                        durationSec: periodSec,
                        sourceStartSec: 0,
                        sourceEndSec: mediaDurationSec,
                    });
                }

                const firstMarker = Math.max(
                    0,
                    Math.ceil((visibleLocalStartSec - headDurationSec - 1e-9) / periodSec),
                );
                for (
                    let markerLocalSec = headDurationSec + firstMarker * periodSec, guard = 0;
                    markerLocalSec < Math.min(clip.lengthSec, visibleLocalEndSec) - 1e-9 &&
                    guard < 4096;
                    markerLocalSec += periodSec, guard += 1
                ) {
                    if (markerLocalSec <= 1e-9) continue;
                    markers.push({
                        clipId: clip.id,
                        timelineSec: clip.startSec + markerLocalSec,
                        xPx: (clip.startSec + markerLocalSec - viewportStartSec) * pxPerSec,
                        yPx: rowTopCanvasPx + bandTopPx,
                        heightPx: bandHeightPx,
                        kind: "loop",
                        inactive: Boolean(clip.inactive),
                    });
                }
            }

            const leadingOverlapSec = Math.max(
                0,
                Math.min(
                    clip.lengthSec,
                    Number(row.leadingOverlapSecByClipId?.[clip.id] ?? 0) || 0,
                ),
            );
            const baseAlpha =
                (clip.muted ? 0.4 : 1) * (clip.inactive ? INACTIVE_TAKE_LANE_ALPHA : 1);
            const fadeInSec = effectiveFade(clip.autoFadeInSec, clip.fadeInSec);
            const fadeOutSec = effectiveFade(clip.autoFadeOutSec, clip.fadeOutSec);

            for (const tile of tiles) {
                const tileEndSec = tile.localStartSec + tile.durationSec;
                const clippedLocalStart = Math.max(tile.localStartSec, visibleLocalStartSec);
                const clippedLocalEnd = Math.min(tileEndSec, visibleLocalEndSec);
                const valid = validLocalInterval(
                    tile,
                    reversed,
                    playbackRate,
                    mediaDurationSec,
                    clippedLocalStart,
                    clippedLocalEnd,
                );
                if (!valid) continue;

                const boundaries = [valid[0]];
                if (leadingOverlapSec > valid[0] + 1e-9 && leadingOverlapSec < valid[1] - 1e-9) {
                    boundaries.push(leadingOverlapSec);
                }
                boundaries.push(valid[1]);

                for (let index = 0; index + 1 < boundaries.length; index += 1) {
                    const localStartSec = boundaries[index];
                    const localEndSec = boundaries[index + 1];
                    const [pieceSourceStartSec, pieceSourceEndSec] = sourceRangeForLocal(
                        tile,
                        reversed,
                        playbackRate,
                        localStartSec,
                        localEndSec,
                    );
                    const x = (clip.startSec + localStartSec - viewportStartSec) * pxPerSec;
                    const right = (clip.startSec + localEndSec - viewportStartSec) * pxPerSec;
                    const clippedX = Math.max(0, x);
                    const clippedRight = Math.min(widthPx, right);
                    if (clippedRight <= clippedX) continue;

                    segments.push({
                        clipId: clip.id,
                        sourcePath: clip.sourcePath,
                        sourceSampleRate: finitePositive(clip.sourceSampleRate ?? 44100, 44100),
                        sourceStartSec: Math.max(0, pieceSourceStartSec),
                        sourceEndSec: Math.min(mediaDurationSec, pieceSourceEndSec),
                        clipLocalStartSec: localStartSec,
                        clipLocalEndSec: localEndSec,
                        clipTotalDurationSec: clip.lengthSec,
                        screenRect: {
                            x: clippedX,
                            y: rowTopCanvasPx + bandTopPx,
                            width: clippedRight - clippedX,
                            height: bandHeightPx,
                        },
                        reversed,
                        gain: Number.isFinite(clip.gain) ? Math.max(0, clip.gain) : 1,
                        fadeInSec,
                        fadeOutSec,
                        fadeInShape: Number.isFinite(clip.fadeInShape) ? clip.fadeInShape : 0,
                        fadeInDir: clip.fadeInDir ?? 0,
                        fadeOutShape: Number.isFinite(clip.fadeOutShape) ? clip.fadeOutShape : 0,
                        fadeOutDir: clip.fadeOutDir ?? 0,
                        alpha:
                            localStartSec < leadingOverlapSec - 1e-9 ? baseAlpha * 0.5 : baseAlpha,
                        inactive: Boolean(clip.inactive),
                    });
                }
            }

            if (!loopEnabled) {
                for (const boundarySec of [0, mediaDurationSec]) {
                    const localSec = reversed
                        ? (window.winEndSec - boundarySec) / playbackRate
                        : (boundarySec - window.winStartSec) / playbackRate;
                    if (
                        localSec <= visibleLocalStartSec + 1e-9 ||
                        localSec >= visibleLocalEndSec - 1e-9 ||
                        localSec <= 1e-9 ||
                        localSec >= clip.lengthSec - 1e-9
                    ) {
                        continue;
                    }
                    markers.push({
                        clipId: clip.id,
                        timelineSec: clip.startSec + localSec,
                        xPx: (clip.startSec + localSec - viewportStartSec) * pxPerSec,
                        yPx: rowTopCanvasPx + bandTopPx,
                        heightPx: bandHeightPx,
                        kind: "media-boundary",
                        inactive: Boolean(clip.inactive),
                    });
                }
            }
        }
    }

    return { segments, markers };
}
