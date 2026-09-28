/**
 * 波形场景构建：把 clip 元数据投影为「屏幕矩形 + 源时间区间」的绘制段。
 *
 * 【主要内容】按当前视口裁剪每个 clip 的可见部分，按 loop 展开成若干
 * source tile，再产出 `WaveformSceneSegment[]`（屏幕矩形 + 对应源音频区间）
 * 与 `WaveformSceneMarker[]`（loop 边界 / 媒体边界标记）。
 *
 * 【作用】波形几何层（`geometry.ts`）只消费本文件产出的屏幕矩形，不再接触
 * clip 的时间语义；loop、reverse、playbackRate、多 take lane 的复杂性全部
 * 收敛在这里。
 *
 * 【与其他模块的关系】
 * - 上游：`WaveformSurface` 在每次绘制时用 `TimelineAxis` 调用
 *   `buildWaveformScene()`；`TimelineWaveformSurface` / `PianoRollWaveformSurface`
 *   负责组装 `WaveformSceneRow[]`。
 * - 横向：**所有时间↔像素换算必须走 `timelineAxis.ts`**。本文件使用视口
 *   坐标系（`secToViewportPx`），与 clip 体画布的内容坐标系相差一个
 *   `scrollLeftPx`，二者由 axis 保证严格同源。
 * - 下游：`geometry.ts` 按像素列采样 mipmap 生成顶点。
 */

import {
    modEuclid,
    resolveLoopMediaDurationSec,
    resolvePlaybackWindowSec,
} from "../utils/loopRender.ts";
import {
    secToViewportPx,
    viewportEndSec as axisViewportEndSec,
    viewportStartSec as axisViewportStartSec,
    type TimelineAxis,
} from "../components/layout/renderKernel/timelineAxis.ts";

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
    /** Take 声道模式（0..=4，对齐 REAPER CHANMODE）；缺省 0（正常/按源）。 */
    channelMode?: number;
    /** 源文件声道数（未知时缺省 0，由峰值数据自身的 channels 兜底）。 */
    sourceChannels?: number;
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
    /**
     * 该 Clip 在**时间轴**上的起点（秒）。
     *
     * 作用：几何层据此把像素列换算成时间轴绝对时间（`clipStartSec + clipLocal…`），
     * `amplitudeMap` 需要它来按时间取逐帧参数（例如动态增益）。缺省 0。
     */
    clipStartSec?: number;
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
    /** Take 声道模式（0..=4，对齐 REAPER CHANMODE）。 */
    channelMode: number;
    /** 源文件声道数（未知时 0）。 */
    sourceChannels: number;
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

/**
 * 时间区间复用的 scratch（模块级一次分配）。
 *
 * 【为什么需要】`validLocalInterval` / `sourceRangeForLocal` 每个 clip、每个
 * tile 都要调用，若各自返回一个 `[start, end]` 元组数组，400 clip 的全览重建
 * 会产生约 1200 次短命数组分配 —— 全部是年轻代垃圾，落在渲染关键路径上。
 * 改为「返回布尔 / 无返回 + 把结果写进调用方给的 out」后稳态零分配，
 * 语义逐值不变。`buildWaveformScene` 同步执行，无重入风险。
 */
const validIntervalScratch = new Float64Array(2);
const sourceRangeScratch = new Float64Array(2);
/** 段边界（最多 3 个：有效区间两端 + 可选的前导重叠点）。 */
const boundaryScratch = new Float64Array(3);
/** 媒体边界标记的两个位置（0 与媒体时长）。 */
const mediaBoundaryScratch = new Float64Array(2);

function effectiveFade(auto: number | undefined, manual: number): number {
    const automatic = Number(auto ?? 0);
    return automatic > 0 ? automatic : Math.max(0, Number(manual) || 0);
}

/**
 * 求 tile 在给定本地区间内的**有效**子区间（落在源数据域内的那一段）。
 *
 * 结果写入 `out`（长度 ≥ 2），返回是否有效。
 *
 * @param out 结果写入目标：`out[0] = start`、`out[1] = end`。
 * @returns 有效区间存在时为 true；false 时 `out` 未被写入。
 */
function validLocalInterval(
    tile: SourceTile,
    reversed: boolean,
    playbackRate: number,
    mediaDurationSec: number,
    localStartSec: number,
    localEndSec: number,
    out: Float64Array,
): boolean {
    if (!(mediaDurationSec > 0)) return false;

    const domainStart = reversed
        ? tile.localStartSec + (tile.sourceEndSec - mediaDurationSec) / playbackRate
        : tile.localStartSec + (0 - tile.sourceStartSec) / playbackRate;
    const domainEnd = reversed
        ? tile.localStartSec + tile.sourceEndSec / playbackRate
        : tile.localStartSec + (mediaDurationSec - tile.sourceStartSec) / playbackRate;
    const start = Math.max(localStartSec, Math.min(domainStart, domainEnd));
    const end = Math.min(localEndSec, Math.max(domainStart, domainEnd));
    if (!(end > start + 1e-9)) return false;
    out[0] = start;
    out[1] = end;
    return true;
}

/**
 * 把 tile 内的本地区间投影回**源文件**区间。
 *
 * 结果写入 `out`（长度 ≥ 2）：`out[0] = sourceStart`、`out[1] = sourceEnd`。
 */
function sourceRangeForLocal(
    tile: SourceTile,
    reversed: boolean,
    playbackRate: number,
    localStartSec: number,
    localEndSec: number,
    out: Float64Array,
): void {
    if (reversed) {
        out[0] = tile.sourceEndSec - (localEndSec - tile.localStartSec) * playbackRate;
        out[1] = tile.sourceEndSec - (localStartSec - tile.localStartSec) * playbackRate;
        return;
    }
    out[0] = tile.sourceStartSec + (localStartSec - tile.localStartSec) * playbackRate;
    out[1] = tile.sourceStartSec + (localEndSec - tile.localStartSec) * playbackRate;
}

/**
 * 构建波形绘制场景。
 *
 * 流程：
 * 1. 由 axis 取出视口的秒级窗口，用于可见性裁剪；
 * 2. 逐 clip 求可见区间，按是否 loop 展开成 source tile；
 * 3. 每个 tile 再按 leading overlap 切段，投影为屏幕矩形输出；
 * 4. loop / 媒体边界额外产出 marker。
 *
 * 特殊说明：
 * - 输出的 `screenRect` 使用**视口坐标系**（已减 scrollLeftPx），因为波形
 *   画布是 sticky 的。与 clip 体画布的内容坐标系通过同一个 axis 保持同源，
 *   禁止在本文件内自行做 `sec * pxPerSec - scrollLeft` 之类的换算。
 * - 视口秒区间**只用于裁剪**，不得再乘回 pxPerSec（那会退化成「先除后乘」，
 *   重新引入与 clip / 网格的不等价）。
 *
 * @param args.axis 统一坐标投影（唯一的时间↔像素来源）。
 * @param args.widthPx 画布宽度（CSS 像素），用于把段裁剪到画布内。
 * @param args.viewportTopPx 行 topPx（内容绝对）到画布坐标的竖直偏移
 *        （= scrollTopPx），保证竖直滚动时与 DOM 内容层同帧对齐。
 * @param args.rows 轨道行场景数据（clip 列表与波形带几何）。
 * @returns 供 `buildWaveformGeometry` 消费的段与标记。
 */
export function buildWaveformScene(args: {
    axis: TimelineAxis;
    widthPx: number;
    /** 行 topPx 所在坐标系（内容绝对）到画布坐标系的竖直偏移
     * （= scrollTopPx）。波形面画布视口锚定，须按此平移后才与 DOM
     * 内容层在竖直滚动中同帧对齐。 */
    viewportTopPx?: number;
    rows: readonly WaveformSceneRow[];
}): WaveformScene {
    const segments: WaveformSceneSegment[] = [];
    const markers: WaveformSceneMarker[] = [];
    const axis = args.axis;
    const viewportStartSec = axisViewportStartSec(axis);
    const viewportEndSec = axisViewportEndSec(axis);
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
                    if (guard === 4095 && localStartSec + periodSec < visibleLocalEndSec - 1e-9) {
                        // 超短周期 × 长循环 × 极缩放可能超出 4096 个周期的上限。
                        // 简单截断会让剩余范围整段空白（旧实现有退化兜底）；
                        // 剩余周期都已亚像素级，退化为覆盖余下区间的单片近似，
                        // 视觉上是一段致密块，好过整段空白。
                        tiles.push({
                            localStartSec: localStartSec + periodSec,
                            durationSec: visibleLocalEndSec - (localStartSec + periodSec),
                            sourceStartSec: 0,
                            sourceEndSec: mediaDurationSec,
                        });
                    }
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
                        xPx: secToViewportPx(axis, clip.startSec + markerLocalSec),
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
                if (
                    !validLocalInterval(
                        tile,
                        reversed,
                        playbackRate,
                        mediaDurationSec,
                        clippedLocalStart,
                        clippedLocalEnd,
                        validIntervalScratch,
                    )
                ) {
                    continue;
                }
                const validLo = validIntervalScratch[0];
                const validHi = validIntervalScratch[1];

                // 边界最多 3 个：有效区间左端、可选的前导重叠点、有效区间右端。
                // 用 scratch 而非数组字面量 + push（每个 tile 一次分配）。
                const hasLeadingOverlap =
                    leadingOverlapSec > validLo + 1e-9 && leadingOverlapSec < validHi - 1e-9;
                boundaryScratch[0] = validLo;
                if (hasLeadingOverlap) {
                    boundaryScratch[1] = leadingOverlapSec;
                    boundaryScratch[2] = validHi;
                } else {
                    boundaryScratch[1] = validHi;
                }
                const boundaryCount = hasLeadingOverlap ? 3 : 2;

                for (let index = 0; index + 1 < boundaryCount; index += 1) {
                    const localStartSec = boundaryScratch[index];
                    const localEndSec = boundaryScratch[index + 1];
                    const x = secToViewportPx(axis, clip.startSec + localStartSec);
                    const right = secToViewportPx(axis, clip.startSec + localEndSec);
                    const clippedX = Math.max(0, x);
                    const clippedRight = Math.min(widthPx, right);
                    if (clippedRight <= clippedX) continue;

                    const localSpan = localEndSec - localStartSec;
                    let drawnLocalStartSec = localStartSec;
                    let drawnLocalEndSec = localEndSec;
                    if (localSpan > 1e-9 && (clippedX > x + 1e-9 || clippedRight < right - 1e-9)) {
                        // 屏幕矩形被视口裁剪：本地区间必须按同一比例裁剪。
                        // 几何层的像素↔源时间映射（sourceSecondsPerPixel =
                        // 源时长 / screenRect.width）与淡化包络映射（clipLocal
                        // 区间 / t）都基于裁剪后的宽度 —— 不一致裁剪会把波形
                        // 拉伸（相对音频平移/错缩）且包络位置漂移。上游的
                        // sec 级可见性裁剪通常已覆盖，本分支是浮点误差与未来
                        // 放宽裁剪时的守护。
                        const pxPerLocalSec = (right - x) / localSpan;
                        drawnLocalStartSec =
                            localStartSec + (clippedX - x) / Math.max(1e-9, pxPerLocalSec);
                        drawnLocalEndSec =
                            localEndSec - (right - clippedRight) / Math.max(1e-9, pxPerLocalSec);
                    }
                    sourceRangeForLocal(
                        tile,
                        reversed,
                        playbackRate,
                        drawnLocalStartSec,
                        drawnLocalEndSec,
                        sourceRangeScratch,
                    );
                    const pieceSourceStartSec = sourceRangeScratch[0];
                    const pieceSourceEndSec = sourceRangeScratch[1];

                    segments.push({
                        clipId: clip.id,
                        sourcePath: clip.sourcePath,
                        sourceSampleRate: finitePositive(clip.sourceSampleRate ?? 44100, 44100),
                        sourceStartSec: Math.max(0, pieceSourceStartSec),
                        sourceEndSec: Math.min(mediaDurationSec, pieceSourceEndSec),
                        // 时间轴起点：几何层据此把像素列还原成绝对时间，
                        // 供逐帧参数（动态增益）采样使用。
                        clipStartSec: clip.startSec,
                        clipLocalStartSec: drawnLocalStartSec,
                        clipLocalEndSec: drawnLocalEndSec,
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
                        channelMode: clip.channelMode ?? 0,
                        sourceChannels: clip.sourceChannels ?? 0,
                        inactive: Boolean(clip.inactive),
                    });
                }
            }

            if (!loopEnabled) {
                // 媒体边界标记：源数据的起点与终点在 clip 本地时间上的位置。
                // 位置列表用 scratch 而非数组字面量 `[0, mediaDurationSec]`
                // ——后者每个非 loop clip 每帧分配一次。
                mediaBoundaryScratch[0] = 0;
                mediaBoundaryScratch[1] = mediaDurationSec;
                for (let boundaryIndex = 0; boundaryIndex < 2; boundaryIndex += 1) {
                    const boundarySec = mediaBoundaryScratch[boundaryIndex];
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
                        xPx: secToViewportPx(axis, clip.startSec + localSec),
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
