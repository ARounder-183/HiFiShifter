/**
 * WaveformTrackCanvas - 轨道级波形 Canvas 组件（v3 rAF+invalidate 架构）
 *
 * 核心思想：每条轨道只有一个 Canvas，负责绘制该轨道上所有可见 clip 的波形。
 * 相比之前「每 clip 一个 Canvas」的方案，大幅减少 Canvas 上下文数量（从 O(clip) 降为 O(track)）。
 *
 * v3 性能优化（对齐 PianoRoll 架构）：
 *   - rAF + invalidate() 帧合并：同一帧内多次 invalidate 只绘制一次
 *   - 高频参数（viewportStartSec / pxPerSec / viewportEndSec）存 ref，避免 React re-render 触发重绘
 *   - 数据获取切换为 getInterleavedSlice + renderWaveform per-pixel 聚合（与 PianoRoll 完全一致）
 *   - 离屏 Canvas 缓存：每个 clip 先绘制到离屏 Canvas，再 drawImage 到主 Canvas
 *
 * 渲染流程：
 *   1. Canvas 物理宽度 = viewportWidthPx，固定不变
 *   2. Canvas 通过 left = viewportStartSec * pxPerSec 定位在视口左边缘
 *   3. 遍历所有可见 clip，对每个 clip：
 *      a. waveformMipmapStore.getInterleavedSlice() 获取原始 interleaved 数据（不 resample）
 *      b. applyGainsToPeaks 应用增益/淡入淡出（带 buffer 复用池）
 *      c. 在离屏 Canvas 上调用 renderWaveform() 绘制波形
 *      d. ctx.drawImage() 将离屏结果绘制到主 Canvas
 *
 * 数据流（v3 架构）：
 *   waveformMipmapStore.getInterleavedSlice() → interleaved Float32Array → applyGainsToPeaks → renderWaveform（离屏） → drawImage
 */

import React from "react";
import type { ClipInfo } from "../../features/session/sessionTypes";
import { waveformMipmapStore } from "../../utils/waveformMipmapStore";
import { timelineViewportBus } from "../../utils/timelineViewportBus";
import {
    applyGainsToPeaks,
    releaseGainBuffer,
    renderWaveform,
    type WaveformRenderParams,
} from "../../utils/waveformRenderer";
import {
    drawLoopMarkers,
    modEuclid,
    resolveLoopMediaDurationSec,
    resolvePlaybackWindowSec,
} from "../../utils/loopRender";
import { resolveTakeLaneLayouts, type TakeLaneLayout } from "../layout/timeline/takeLanes";
import {
    wfDiag_frameStart,
    wfDiag_frameEnd,
    wfDiag_invalidateBus,
    wfDiag_invalidateMipmap,
    wfDiag_invalidateProps,
    wfDiag_dataHit,
    wfDiag_dataMissNull,
    wfDiag_dataMissShort,
} from "../../utils/waveformDebug";
// ========================================
// 局部 Buffer 复用池
// ========================================
const _downsamplePool: Float32Array[] = [];
const POOL_MAX = 8;
const LEADING_OVERLAP_ALPHA = 0.5;

/** darkenWaveformStroke 的结果缓存：绘制循环内每个 inactive lane 都会调用，
 * 输入色板在会话内高度重复，按输入字符串缓存避免每帧正则解析。 */
const _darkenCache = new Map<string, string>();

/** 将 inactive take 的波形颜色压暗；支持常用 CSS 颜色格式。 */
function darkenWaveformStroke(color: string): string {
    const cached = _darkenCache.get(color);
    if (cached !== undefined) return cached;
    const result = computeDarkenedWaveformStroke(color);
    if (_darkenCache.size > 64) _darkenCache.clear();
    _darkenCache.set(color, result);
    return result;
}

function computeDarkenedWaveformStroke(color: string): string {
    const hex = color.trim().match(/^#([0-9a-f]{3}|[0-9a-f]{6})$/i);
    if (hex) {
        let value = hex[1];
        if (value.length === 3) {
            value = value
                .split("")
                .map((ch) => ch + ch)
                .join("");
        }
        const r = Math.round(Number.parseInt(value.slice(0, 2), 16) * 0.42);
        const g = Math.round(Number.parseInt(value.slice(2, 4), 16) * 0.42);
        const b = Math.round(Number.parseInt(value.slice(4, 6), 16) * 0.42);
        return `rgba(${r}, ${g}, ${b}, 0.78)`;
    }

    const rgba = color.match(
        /^rgba?\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*(?:,\s*([0-9.]+)\s*)?\)$/i,
    );
    if (rgba) {
        const r = Math.round(Number(rgba[1]) * 0.42);
        const g = Math.round(Number(rgba[2]) * 0.42);
        const b = Math.round(Number(rgba[3]) * 0.42);
        const a = rgba[4] == null ? 1 : Number(rgba[4]);
        return `rgba(${r}, ${g}, ${b}, ${(a * 0.72).toFixed(3)})`;
    }
    return color;
}

function acquireDownsampleBuffer(minLen: number): Float32Array {
    for (let i = 0; i < _downsamplePool.length; i++) {
        if (_downsamplePool[i].buffer.byteLength / 4 >= minLen) {
            const buf = _downsamplePool[i];
            _downsamplePool.splice(i, 1);
            return new Float32Array(buf.buffer, 0, minLen);
        }
    }
    return new Float32Array(minLen);
}

function releaseDownsampleBuffer(buf: Float32Array): void {
    if (buf.length > 0 && _downsamplePool.length < POOL_MAX) {
        _downsamplePool.push(new Float32Array(buf.buffer));
    }
}

export interface WaveformTrackCanvasProps {
    /** 当前轨道上的完整 clip 列表，由组件内部按视口自行过滤以保持引用稳定 */
    clips: ClipInfo[];
    /** 每个 clip 左侧前导重叠时长（秒），用于重叠区等权可视化混合 */
    leadingOverlapSecByClipId?: Readonly<Record<string, number>>;
    /** 轨道高度（像素），包含 header 和 padding */
    trackHeight: number;
    /** 波形区域的 top 偏移（跳过 clip header 部分） */
    waveformTop: number;
    /** 波形区域高度 */
    waveformHeight: number;
    /** 每秒像素数 */
    pxPerSec: number;
    /** 视口宽度（CSS 像素），Canvas 物理宽度固定为此值 */
    viewportWidthPx: number;
    /** 视口起始时间（秒） */
    viewportStartSec: number;
    /** 视口结束时间（秒） */
    viewportEndSec: number;
    /** 波形描边颜色 */
    strokeColor: string;
    /** 描边宽度 */
    strokeWidth?: number;
    /** 在垂直空间足够时，显示一个 Clip 内的全部音频 Take。 */
    showAllTakes?: boolean;
    /** 多 Take lane 之间的分界线颜色。 */
    takeSeparatorColor?: string;
}

type WaveformRenderClip = ClipInfo & {
    __takeLane?: TakeLaneLayout;
    /** 展开 lane 前的真实 Clip id（lane 的 id 带 `::take::` 后缀，不能用于
     * leadingOverlapSecByClipId 等以真实 clip id 为键的查表）。 */
    __originClipId?: string;
};

function expandClipsForTakeLanes(
    clips: ClipInfo[],
    showAllTakes: boolean,
    availableHeightPx: number,
): WaveformRenderClip[] {
    const output: WaveformRenderClip[] = [];
    for (const clip of clips) {
        const layouts = resolveTakeLaneLayouts(clip, showAllTakes, availableHeightPx);
        if (!layouts) {
            output.push(clip);
            continue;
        }

        for (const [index, take] of (clip.takes ?? [])
            .filter((take) => Boolean(take.sourcePath))
            .entries()) {
            const layout = layouts[index];
            if (!layout) continue;
            const clipRate =
                Number.isFinite(clip.clipPlaybackRate) && (clip.clipPlaybackRate ?? 0) > 0
                    ? Number(clip.clipPlaybackRate)
                    : 1;
            // Slip / trim / gain 的乐观更新写在 Clip 的 active-take 投影上；
            // 渲染 expanded lane 时必须让 active take 消费这份最新投影，
            // 否则多 Take 展示下拖拽预览会停留在旧窗口/旧增益
            // （增益拖拽逐帧只更新 flat clip.gain，不写 take.gain）。
            const isActiveTake = take.id === clip.activeTakeId;
            const effectiveTake = isActiveTake
                ? {
                      ...take,
                      sourceStartSec: clip.sourceStartSec,
                      sourceEndSec: clip.sourceEndSec,
                      gain: clip.gain,
                  }
                : take;
            output.push({
                ...clip,
                id: `${clip.id}::take::${take.id}`,
                __originClipId: clip.id,
                sourcePath: effectiveTake.sourcePath,
                durationSec: effectiveTake.durationSec,
                durationFrames: effectiveTake.durationFrames,
                sourceSampleRate: effectiveTake.sourceSampleRate,
                gain: effectiveTake.gain,
                sourceStartSec: effectiveTake.sourceStartSec,
                sourceEndSec: effectiveTake.sourceEndSec,
                playbackRate:
                    Number.isFinite(clipRate * effectiveTake.playbackRate) &&
                    clipRate * effectiveTake.playbackRate > 0.1
                        ? Math.min(10, clipRate * effectiveTake.playbackRate)
                        : 1,
                reversed: effectiveTake.reversed,
                loopEnabled: effectiveTake.loopEnabled,
                midiNoteData: undefined,
                midiNoteCount: undefined,
                __takeLane: layout,
            });
        }
    }
    return output;
}

export const WaveformTrackCanvas = React.memo(
    function WaveformTrackCanvas(props: WaveformTrackCanvasProps) {
        const {
            clips,
            waveformTop,
            waveformHeight,
            viewportWidthPx,
            strokeColor,
            strokeWidth = 1,
            showAllTakes = true,
            takeSeparatorColor,
        } = props;

        // ========================================
        // refs：高频变化的参数存 ref，避免 React re-render
        // ========================================
        const canvasRef = React.useRef<HTMLCanvasElement | null>(null);
        const lastLevelByClipRef = React.useRef<Record<string, 0 | 1 | 2>>({});
        const rafRef = React.useRef<number | null>(null);
        // 缓存 canvas 上次物理尺寸，避免设置 canvas.width/height 属性导致强制清空
        const lastCanvasDimsRef = React.useRef({ w: 0, h: 0 });

        // 高频参数用 ref 存储，避免依赖数组变化触发 useLayoutEffect
        const pxPerSecRef = React.useRef(props.pxPerSec);
        const viewportStartSecRef = React.useRef(props.viewportStartSec);
        const viewportEndSecRef = React.useRef(props.viewportEndSec);
        const clipsRef = React.useRef(clips);
        const leadingOverlapSecByClipIdRef = React.useRef(props.leadingOverlapSecByClipId ?? {});
        const waveformHeightRef = React.useRef(waveformHeight);
        const strokeColorRef = React.useRef(strokeColor);
        const strokeWidthRef = React.useRef(strokeWidth);
        const viewportWidthPxRef = React.useRef(viewportWidthPx);
        const showAllTakesRef = React.useRef(showAllTakes);
        const takeSeparatorColorRef = React.useRef(takeSeparatorColor);
        /** take-lane 展开结果缓存：输入三元组未变时复用上一帧对象。 */
        const expandedTakeLanesCacheRef = React.useRef<{
            clips: ClipInfo[];
            showAllTakes: boolean;
            height: number;
            result: WaveformRenderClip[];
        } | null>(null);

        // 同步 ref
        pxPerSecRef.current = props.pxPerSec;
        viewportStartSecRef.current = props.viewportStartSec;
        viewportEndSecRef.current = props.viewportEndSec;
        clipsRef.current = clips;
        leadingOverlapSecByClipIdRef.current = props.leadingOverlapSecByClipId ?? {};
        waveformHeightRef.current = waveformHeight;
        strokeColorRef.current = strokeColor;
        strokeWidthRef.current = strokeWidth;
        viewportWidthPxRef.current = viewportWidthPx;
        showAllTakesRef.current = showAllTakes;
        takeSeparatorColorRef.current = takeSeparatorColor;

        // ========================================
        // invalidate + rAF 帧合并（与 PianoRoll 完全一致）
        // 同一帧内无论有多少次 invalidate 调用，只执行一次绘制
        // ========================================
        const drawRef = React.useRef<() => void>(() => {});

        const invalidate = React.useCallback(() => {
            if (rafRef.current != null) return; // 已有待执行帧，跳过
            rafRef.current = requestAnimationFrame(() => {
                rafRef.current = null;
                drawRef.current();
            });
        }, []);

        // ========================================
        // 核心绘制函数（存入 drawRef，由 invalidate 调度）
        // ========================================
        drawRef.current = () => {
            wfDiag_frameStart();
            const canvas = canvasRef.current;
            if (!canvas) return;

            // ========================================
            // 性能诊断探针（通过 localStorage 开关）
            // 开启: localStorage.setItem('hifishifter.debugWaveformPerf', '1')
            // 关闭: localStorage.removeItem('hifishifter.debugWaveformPerf')
            // ========================================
            const __perfDebug =
                typeof window !== "undefined" &&
                window.localStorage?.getItem("hifishifter.debugWaveformPerf") === "1";
            const __t0 = __perfDebug ? performance.now() : 0;
            let __tSetup = 0;
            const __clipTimings: {
                name: string;
                sliceMs: number;
                downsampleMs: number;
                gainMs: number;
                renderMs: number;
                drawImageMs: number;
                interleavedLen: number;
                visibleWidthPx: number;
                downsampledTo: number;
            }[] = [];

            const currentPxPerSec = pxPerSecRef.current;
            const currentViewportStartSec = viewportStartSecRef.current;
            const currentViewportEndSec = viewportEndSecRef.current;
            const currentLeadingOverlapSecByClipId = leadingOverlapSecByClipIdRef.current;
            const currentWaveformHeight = waveformHeightRef.current;
            const currentStrokeColor = strokeColorRef.current;
            const currentStrokeWidth = strokeWidthRef.current;
            const currentViewportWidthPx = viewportWidthPxRef.current;
            const currentShowAllTakes = showAllTakesRef.current;
            // take-lane 展开是纯函数且开销随 clip×take 数增长；滚动/缩放时
            // 每帧重算属于纯分配 churn。按输入三元组 memo 化（clips 数组引用
            // 在 Redux 更新时才会变化，与数据一致性天然对齐）。
            const expandedCache = expandedTakeLanesCacheRef.current;
            let currentClips: WaveformRenderClip[];
            if (
                expandedCache !== null &&
                expandedCache.clips === clipsRef.current &&
                expandedCache.showAllTakes === currentShowAllTakes &&
                expandedCache.height === currentWaveformHeight
            ) {
                currentClips = expandedCache.result;
            } else {
                currentClips = expandClipsForTakeLanes(
                    clipsRef.current,
                    currentShowAllTakes,
                    currentWaveformHeight,
                );
                expandedTakeLanesCacheRef.current = {
                    clips: clipsRef.current,
                    showAllTakes: currentShowAllTakes,
                    height: currentWaveformHeight,
                    result: currentClips,
                };
            }
            const displayW = Math.max(1, Math.ceil(currentViewportWidthPx));
            const baseDisplayHeight = currentWaveformHeight;
            let displayH = baseDisplayHeight;
            // 每帧解析一次主题相关的默认分界线颜色（避免循环内反复读 DOM）。
            const defaultSeparatorColor = document.documentElement.classList.contains("dark")
                ? "rgba(255, 255, 255, 0.16)"
                : "rgba(0, 0, 0, 0.18)";

            // 取消限制 dpr 为 1
            const dpr = window.devicePixelRatio || 1;
            // 用 Math.round 代替 Math.floor，消除浮点累积误差导致的帧间尺寸振荡
            const internalW = Math.max(1, Math.round(displayW * dpr));
            const internalH = Math.max(1, Math.round(baseDisplayHeight * dpr));

            // 仅当物理尺寸真正变化时才设置 canvas.width/height（设置即清空画布）
            const lastDims = lastCanvasDimsRef.current;
            const dimsChanged = lastDims.w !== internalW || lastDims.h !== internalH;
            if (dimsChanged) {
                canvas.width = internalW;
                canvas.height = internalH;
                lastCanvasDimsRef.current = { w: internalW, h: internalH };
            }

            const ctx = canvas.getContext("2d");
            if (!ctx) return;

            // 尺寸可能未变，但每个 take lane 需要独立的垂直变换。
            const scaleX = internalW / Math.max(1, displayW);
            const scaleY = internalH / Math.max(1, baseDisplayHeight);
            ctx.setTransform(scaleX, 0, 0, scaleY, 0, 0);

            ctx.clearRect(0, 0, displayW, displayH);

            // 级别提示键清理：已删除/不再渲染的 clip 的 `${path}::${id}` 键
            // 若不清理会随会话无限累积（每 clip 一个小条目，长会话可感知）。
            {
                const liveKeys = new Set<string>();
                for (const c of currentClips) {
                    if (c.sourcePath) liveKeys.add(`${c.sourcePath}::${c.id}`);
                }
                const levelMap = lastLevelByClipRef.current;
                for (const key of Object.keys(levelMap)) {
                    if (!liveKeys.has(key)) delete levelMap[key];
                }
            }

            // CSS 尺寸也只在变化时写入，避免触发不必要的 layout
            if (canvas.style.width !== `${displayW}px`) canvas.style.width = `${displayW}px`;
            if (canvas.style.height !== `${displayH}px`) canvas.style.height = `${displayH}px`;

            if (__perfDebug) __tSetup = performance.now() - __t0;

            for (const rawClip of currentClips) {
                const clip = rawClip as WaveformRenderClip;
                const takeLane = clip.__takeLane;
                displayH = takeLane?.height ?? baseDisplayHeight;
                ctx.setTransform(scaleX, 0, 0, scaleY, 0, (takeLane?.top ?? 0) * scaleY);
                const strokeColorForClip = takeLane?.inactive
                    ? darkenWaveformStroke(currentStrokeColor)
                    : currentStrokeColor;
                if (!clip.sourcePath) continue;
                // 判断：只带 durationFrames+sourceSampleRate 的 clip 同样可渲染，
                // 与 piano-roll 消费端保持一致。mediaDur 同时作为后续所有
                // "媒体时长"消费点的具体数值（替代旧 durationSec 守卫的收窄）。
                const mediaDur = resolveLoopMediaDurationSec({
                    durationFrames: clip.durationFrames,
                    sourceSampleRate: clip.sourceSampleRate,
                    durationSec: clip.durationSec,
                });
                if (!(mediaDur > 1e-6)) continue;

                const clipStartSec = clip.startSec;
                const clipEndSec = clipStartSec + clip.lengthSec;

                // clip 与视口的交集
                const visStartSec = Math.max(clipStartSec, currentViewportStartSec);
                const visEndSec = Math.min(clipEndSec, currentViewportEndSec);
                if (visEndSec <= visStartSec) continue;

                // 统一使用浮点像素坐标，避免多重 round 导致的帧间抖动
                const viewportStartPx = currentViewportStartSec * currentPxPerSec;
                const clipStartPx = clipStartSec * currentPxPerSec;
                const clipEndPx = clipEndSec * currentPxPerSec;
                const visLeftPx = Math.max(0, clipStartPx - viewportStartPx);
                const visRightPx = Math.min(displayW, clipEndPx - viewportStartPx);
                if (visRightPx <= visLeftPx) continue;
                // 速率净化：NaN/非正值按 1.0（Math.max(1e-6, NaN) === NaN，
                // 会沿 headDur/bodyDur/取窗链传播成整条 clip 不渲染）。
                const pr =
                    Number.isFinite(clip.playbackRate) && clip.playbackRate > 1e-6
                        ? clip.playbackRate
                        : 1;
                const sourceStartSec = Number(clip.sourceStartSec ?? 0) || 0;

                // 计算源文件时间范围
                const sampleRate = clip.sourceSampleRate || 44100;
                const spp = Math.max(1, Math.round(sampleRate / currentPxPerSec));
                const levelKey = `${clip.sourcePath}::${clip.id}`;
                const previousLevel = lastLevelByClipRef.current[levelKey];
                const stableLevel = waveformMipmapStore.selectLevelStable(spp, previousLevel);
                lastLevelByClipRef.current[levelKey] = stableLevel;

                // 消费窗口模型（与后端 clip_playback_window_sec 一致）：
                //   正放 win = [ss, ss+len·r)；倒放 win = [se−len·r, se)。
                // 倒放的 sourceStartSec 只是历史/编辑字段，不参与取窗 ——
                // 否则 trim/延伸写入的域外锚点会让波形与音频错位
                //（该有声处被画成空白 / 空白处画满波形）。
                const isLoop = Boolean(clip.loopEnabled);
                const { winStartSec, winEndSec } = resolvePlaybackWindowSec({
                    loopEnabled: isLoop,
                    reversed: Boolean(clip.reversed),
                    sourceStartSec,
                    playbackRate: pr,
                    lengthSec: clip.lengthSec,
                    sourceEndSec: Number(clip.sourceEndSec ?? mediaDur) || mediaDur,
                });
                // Loop 锚点域仍需有效终点（倒放锚点 clamp 到媒体末端）。
                const clipSourceEndSec = winEndSec;
                const effSrcEnd = Math.min(clipSourceEndSec, mediaDur);
                let clipSourceSpanSec: number;
                if (!isLoop) {
                    // 非 Loop：窗口宽度恒为 len·r（域外部分为静音，无数据可取，
                    // 取数时 clamp 到媒体内即可 —— 缺失区间自然渲染为空白）。
                    clipSourceSpanSec = Math.max(0, winEndSec - winStartSec);
                } else {
                    // Loop（循环源）：回绕发生在整个媒体文件上，音频只由锚点与
                    // 媒体时长决定 —— split 等编辑会产生 sourceStart > sourceEnd
                    // 的"环绕窗口"，可用性只取决于媒体时长本身；
                    // 若按窗口跨度判断，这类 clip 的波形会整体消失（而声音仍在播放）。
                    clipSourceSpanSec = mediaDur;
                }
                if (clipSourceSpanSec <= 1e-6) continue;

                // ── 循环分段（Loop = 循环原始音频文件）────────────────────────
                // 语义（对齐 REAPER Loop source / floor_mod 映射）：
                //   正放 src(t) = mod(sourceStart + t·pr, D)
                //   倒放 src(t) = mod(sourceEnd   − t·pr, D)
                // 其中 D = 完整媒体时长。因此波形按"进入段 + 整文件重复段"划分：
                //   段 0（头部）：源 [sourceStart, D]（正放）/ [0, sourceEnd]（倒放）
                //   段 1..n：    整个文件 [0, D] 原样重复
                // 回绕节点（倒三角）位于头部段结束处及此后每个整文件周期边界。
                //
                // 每段的 clipDuration === 该段源跨度/pr，保证
                // sourceStart + clipDuration·pr === 段窗口终点，
                // 倒放镜像数学在任意段上成立；超出 clip 长度的部分仅通过
                // 绘制矩形裁掉。
                //
                // 淡入淡出：每个分段都携带完整淡化参数，增益按 clip 局部时间
                // 求值（clipTimeOffsetSec = 分段起点；淡出锚定整条 clip 终点）
                // —— 长于一个周期的淡化横跨多段时包络保持连续；
                // 非淡化区间的分段求值自然为 1，不存在重复施加的问题。
                interface RenderSegment {
                    /** 分段在 clip 内的起始时间（秒），同时是增益求值的 clip 时间偏移 */
                    localStartSec: number;
                    /** 分段时长（秒） */
                    durationSec: number;
                    /** 该分段覆盖的源窗口起点（源域秒） */
                    srcWinStart: number;
                    /** 该分段覆盖的源窗口终点（源域秒） */
                    srcWinEnd: number;
                }
                const segmentsToRender: RenderSegment[] = [];
                if (!isLoop) {
                    segmentsToRender.push({
                        localStartSec: 0,
                        durationSec: clip.lengthSec,
                        srcWinStart: winStartSec,
                        srcWinEnd: winEndSec,
                    });
                } else if (!(mediaDur > 1e-6)) {
                    // 无有效媒体时长：退化为单片近似
                    segmentsToRender.push({
                        localStartSec: 0,
                        durationSec: clip.lengthSec,
                        srcWinStart: winStartSec,
                        srcWinEnd: winEndSec,
                    });
                } else {
                    // 锚点用 floor_mod 归一化（与引擎 mod(anchor ± t·pr, D)
                    // 一致）：负 / 超界的存储锚点正确环绕，不能用 clamp，
                    // 否则 slip 出域的锚点会让波形相位与播放错位。
                    const anchorFwd = modEuclid(sourceStartSec, mediaDur);
                    const anchorRev = modEuclid(effSrcEnd, mediaDur);
                    const headDur = (clip.reversed ? anchorRev : mediaDur - anchorFwd) / pr;
                    const bodyDur = mediaDur / pr;
                    const visLocalStart = Math.max(0, visStartSec - clipStartSec);
                    const visLocalEnd = Math.min(clip.lengthSec, visEndSec - clipStartSec);
                    if (visLocalEnd <= visLocalStart) continue;
                    // 退化保护：分段数按【可见区间】估算（而非整条 clip）——
                    // 视口内可见的周期数天然有限，长循环 clip 不会再落入
                    // "单片拉伸近似"，标记与波形内容保持一致。
                    // 首个重复段取**包含视口左缘**的那一段（floor 而非 ceil，
                    // 避免左缘出现周期级的空隙）；其左侧越界部分由裁剪矩形去掉。
                    const firstBodyIndex = Math.max(
                        0,
                        Math.floor((visLocalStart - headDur - 1e-9) / bodyDur),
                    );
                    const approxCount =
                        2 + Math.ceil((visLocalEnd - Math.max(headDur, visLocalStart)) / bodyDur);
                    if (approxCount <= 4096) {
                        if (headDur > 1e-9 && visLocalStart < headDur) {
                            segmentsToRender.push({
                                localStartSec: 0,
                                durationSec: headDur,
                                srcWinStart: clip.reversed ? 0 : anchorFwd,
                                srcWinEnd: clip.reversed ? anchorRev : mediaDur,
                            });
                        }
                        let segOffset = headDur + firstBodyIndex * bodyDur;
                        for (
                            let guard = 0;
                            segOffset < visLocalEnd - 1e-9 && guard < 4096;
                            guard += 1
                        ) {
                            segmentsToRender.push({
                                localStartSec: segOffset,
                                durationSec: bodyDur,
                                srcWinStart: 0,
                                srcWinEnd: mediaDur,
                            });
                            segOffset += bodyDur;
                        }
                    } else {
                        // 退化保护：分段数失控时回退单片近似。
                        // 用"进入段"窗口（锚点 → 媒体末端/起点）近似，
                        // 避免按 [start, start+span] 取到越界源区间。
                        segmentsToRender.push({
                            localStartSec: 0,
                            durationSec: clip.lengthSec,
                            srcWinStart: clip.reversed ? 0 : anchorFwd,
                            srcWinEnd: clip.reversed ? anchorRev : mediaDur,
                        });
                    }
                    if (segmentsToRender.length === 0) continue;
                }

                const sourcePadSec = Math.max(0.005, (2 / Math.max(1, currentPxPerSec)) * pr);

                // ── 同窗口切片缓存（单条目）─────────────────────────────────
                // 取数按"瓦片 ∩ 视口"裁剪后，相邻整文件重复段的可见子窗通常
                // 各不相同，跨瓦片复用只在退化场景成立（如视口只覆盖单个
                // 瓦片、重绘间窗口未变）。此缓存的实际作用是：同一渲染帧内
                // 相同窗口参数不重复聚合/拷贝，且 store 复用池 buffer 的持有
                // 权集中在本缓存上 —— 换窗或本 clip 结束时统一归还，
                // 保证降采样路径不会提前归还仍在使用的 store buffer。
                let fetchCacheKey: string | null = null;
                let fetchCacheResult: {
                    interleaved: Float32Array;
                    dataStartSec: number;
                    dataDurationSec: number;
                } | null = null;
                const releaseFetchCache = () => {
                    if (fetchCacheResult) {
                        waveformMipmapStore.releaseInterleaved(fetchCacheResult.interleaved);
                        fetchCacheResult = null;
                    }
                    fetchCacheKey = null;
                };

                // ── 派生降采样缓存（clip 渲染帧内）───────────────────────────
                // 整文件重复段共享同一 fetchKey（相同源窗口），其降采样结果只依赖
                // (源切片内容, 目标宽度 w)。视口内部的重复段可见宽度相等 → w 相同，
                // 可直接复用 —— 否则缩小时一帧内每个瓦片都要重扫 O(slice) 做
                // min/max（数千周期场景下是主要卡顿源）。缓存仅存活于本 clip 的
                // 本次绘制，瓦片循环结束统一归还池缓冲；内容为拷贝值，不持有
                // store buffer 引用，与 fetch 缓存的生命周期解耦。
                const derivedDownsampleCache = new Map<string, Float32Array>();
                const releaseDerivedDownsampleCache = () => {
                    for (const buf of derivedDownsampleCache.values()) {
                        releaseDownsampleBuffer(buf);
                    }
                    derivedDownsampleCache.clear();
                };

                // 渲染参数模板：clip 级恒定字段只填一次，瓦片级字段在循环内
                // 就地修补 —— 缩小时一帧可达数千瓦片，逐瓦片新建 28 字段对象
                // 是纯粹的分配 churn（对象仅被同步消费，复用安全）。
                const effectiveFadeInSec =
                    Number(clip.autoFadeInSec ?? 0) > 0
                        ? Number(clip.autoFadeInSec)
                        : Number(clip.fadeInSec ?? 0) || 0;
                const effectiveFadeOutSec =
                    Number(clip.autoFadeOutSec ?? 0) > 0
                        ? Number(clip.autoFadeOutSec)
                        : Number(clip.fadeOutSec ?? 0) || 0;
                const renderParams: WaveformRenderParams = {
                    canvasWidth: displayW,
                    canvasHeight: displayH,
                    centerY: displayH / 2,
                    zeroDbHalfHeight: displayH / 2,
                    sourceStartSec: 0,
                    clipDuration: 0,
                    playbackRate: pr,
                    reversed: Boolean(clip.reversed),
                    sourceDurationSec: mediaDur,
                    volumeGain: Number(clip.gain ?? 1) || 1,
                    // 有效 fade：自动交叉淡化（>0 时覆盖）否则手动 fade。
                    // 每个分段都携带完整淡化参数，增益按 clip 局部时间求值
                    //（clipTimeOffsetSec = 分段起点；淡出锚定整条 clip 终点）
                    // —— 长于一个周期的淡化横跨多段时包络保持连续。
                    fadeInSec: effectiveFadeInSec,
                    fadeOutSec: effectiveFadeOutSec,
                    fadeInShape: Number.isFinite(clip.fadeInShape) ? clip.fadeInShape : 0,
                    fadeInDir: clip.fadeInDir ?? 0,
                    fadeOutShape: Number.isFinite(clip.fadeOutShape) ? clip.fadeOutShape : 0,
                    fadeOutDir: clip.fadeOutDir ?? 0,
                    dataStartSec: 0,
                    dataDurationSec: 0,
                    clipTimeOffsetSec: 0,
                    clipTotalDurationSec: clip.lengthSec,
                    clipPixelOffset: 0,
                    clipTotalWidthPx: 1,
                };

                for (const seg of segmentsToRender) {
                    const tileLocalEndSec = seg.localStartSec + seg.durationSec;
                    // 该分段与可见区间、clip 长度的交集（clip 局部时间）
                    const visClipStartSec = Math.max(seg.localStartSec, visStartSec - clipStartSec);
                    const visClipEndSec = Math.min(tileLocalEndSec, visEndSec - clipStartSec);
                    if (visClipEndSec <= visClipStartSec) continue;

                    // 瓦片在主 Canvas 上的可见像素范围 —— 必须在取数/降采样
                    // 之前判定：像素区间塌陷时直接跳过，避免白做一次切片聚合，
                    // 以及降采样池缓冲取出后未归还的泄漏。
                    const tileVisLeftPx = Math.max(
                        0,
                        (clipStartSec + visClipStartSec) * currentPxPerSec - viewportStartPx,
                    );
                    const tileVisRightPx = Math.min(
                        displayW,
                        (clipStartSec + visClipEndSec) * currentPxPerSec - viewportStartPx,
                    );
                    if (tileVisRightPx <= tileVisLeftPx) {
                        continue;
                    }

                    // 该分段的源窗口（头部段 / 整文件重复段，见上方划分注释）
                    const tileSpanStartSec = seg.srcWinStart;
                    const tileSpanEndSec = seg.srcWinEnd;

                    // 仅请求当前可见部分对应的源数据，显著降低每帧处理成本。
                    // 取数范围 clamp 到媒体 [0, mediaDur]：消费窗口（尤其倒放
                    // 延伸后的 [se−len·r, se]）可越出媒体域，缺失区间无数据、
                    // 自然渲染为空白 —— 与音频的静音表达一致；像素映射由
                    // dataStartSec/dataDurationSec 回到窗口坐标系，不受影响。
                    const sourceVisStartSec = clip.reversed
                        ? tileSpanEndSec - (visClipEndSec - seg.localStartSec) * pr
                        : tileSpanStartSec + (visClipStartSec - seg.localStartSec) * pr;
                    const sourceVisEndSec = clip.reversed
                        ? tileSpanEndSec - (visClipStartSec - seg.localStartSec) * pr
                        : tileSpanStartSec + (visClipEndSec - seg.localStartSec) * pr;
                    const sourceTimeStart = Math.max(
                        0,
                        tileSpanStartSec,
                        Math.min(sourceVisStartSec, sourceVisEndSec) - sourcePadSec,
                    );
                    const sourceTimeEnd = Math.min(
                        mediaDur,
                        tileSpanEndSec,
                        Math.max(sourceVisStartSec, sourceVisEndSec) + sourcePadSec,
                    );
                    // 可见区与媒体域无交集（纯静音段）：跳过取数与绘制，
                    // 防止退化请求把媒体开头的 1ms 数据错误映射进静音区。
                    if (!(sourceTimeEnd > sourceTimeStart + 1e-9)) {
                        releaseFetchCache();
                        continue;
                    }
                    const sourceDuration = Math.max(0.001, sourceTimeEnd - sourceTimeStart);

                    // ========================================
                    // 从 mipmap 缓存获取 interleaved 数据（不 resample，与 PianoRoll 一致）。
                    // 相同源窗口的瓦片复用同一切片（见上方缓存说明）。
                    // ========================================
                    const __tSlice0 = __perfDebug ? performance.now() : 0;
                    const fetchKey = `${sourceTimeStart}|${sourceDuration}`;
                    if (!fetchCacheResult || fetchKey !== fetchCacheKey) {
                        releaseFetchCache();
                        fetchCacheKey = fetchKey;
                        fetchCacheResult = waveformMipmapStore.getInterleavedSlice(
                            clip.sourcePath,
                            stableLevel,
                            sourceTimeStart,
                            sourceDuration,
                        );
                    }
                    const result = fetchCacheResult;
                    const __tSlice1 = __perfDebug ? performance.now() : 0;

                    if (!result || result.interleaved.length < 4) {
                        if (!result) wfDiag_dataMissNull();
                        else wfDiag_dataMissShort();
                        releaseFetchCache();
                        continue;
                    }
                    wfDiag_dataHit();

                    // 该瓦片可见部分的像素宽度（用于稳定的降采样目标）
                    const tileVisibleWidthPx = Math.max(
                        1,
                        Math.ceil((visClipEndSec - visClipStartSec) * currentPxPerSec - 1e-6),
                    );

                    // ========================================
                    // 方案2：限制数据量 — 当原始数据点数远超可视像素时，快速预降采样
                    // ========================================
                    const __tDs0 = __perfDebug ? performance.now() : 0;
                    const storeInterleaved = result.interleaved;
                    let renderInterleaved: Float32Array = storeInterleaved;
                    const rawSampleCount = storeInterleaved.length / 2;
                    // 使用与可见宽度绑定的稳定采样目标，避免滚屏时分桶边界漂移导致抖动
                    const stableTargetWidthPx = Math.max(1, Math.ceil(tileVisibleWidthPx * 2));
                    const targetSamples = stableTargetWidthPx * 2;

                    if (rawSampleCount > targetSamples && targetSamples >= 2) {
                        const w = Math.ceil(targetSamples);
                        const derivedKey = `${fetchKey}|${w}`;
                        let downsampled = derivedDownsampleCache.get(derivedKey);
                        if (!downsampled) {
                            // 从局部池获取 Buffer
                            downsampled = acquireDownsampleBuffer(w * 2);

                            // 提取线性步长常数，将循环内的 4 次浮点乘除降至 1 次加法
                            const srcStep = rawSampleCount / w;

                            for (let i = 0; i < w; i++) {
                                const srcStart = i * srcStep;
                                const srcEnd = srcStart + srcStep;

                                const iStart = Math.max(0, Math.floor(srcStart));
                                const iEnd = Math.min(rawSampleCount - 1, Math.ceil(srcEnd));

                                let pMin = Infinity;
                                let pMax = -Infinity;
                                for (let j = iStart; j <= iEnd; j++) {
                                    const sMin = storeInterleaved[j * 2];
                                    const sMax = storeInterleaved[j * 2 + 1];
                                    if (sMin < pMin) pMin = sMin;
                                    if (sMax > pMax) pMax = sMax;
                                }
                                downsampled[i * 2] = pMin === Infinity ? 0 : pMin;
                                downsampled[i * 2 + 1] = pMax === -Infinity ? 0 : pMax;
                            }
                            derivedDownsampleCache.set(derivedKey, downsampled);
                        }

                        // 注意：缓存持有的派生缓冲由 releaseDerivedDownsampleCache
                        // 统一归还（本 clip 瓦片循环结束时），此处不归还，
                        // 供后续同窗口瓦片继续复用。
                        renderInterleaved = downsampled;
                    }

                    // 修补瓦片级渲染参数（模板见瓦片循环前）。偏移量相对于
                    // 主屏幕（按分段起点校正），量化到半像素粒度，
                    // 消除大浮点数相减导致的子像素漂移。
                    const tileStartPx = clipStartPx + seg.localStartSec * currentPxPerSec;
                    renderParams.sourceStartSec = tileSpanStartSec;
                    renderParams.clipDuration = seg.durationSec;
                    // 增益按 clip 局部时间求值（clipTimeOffsetSec = 分段起点）。
                    renderParams.clipTimeOffsetSec = isLoop ? seg.localStartSec : 0;
                    renderParams.dataStartSec = result.dataStartSec;
                    renderParams.dataDurationSec = result.dataDurationSec;
                    renderParams.clipPixelOffset =
                        Math.round((viewportStartPx - tileStartPx) * 2) / 2;
                    renderParams.clipTotalWidthPx = Math.max(1, seg.durationSec * currentPxPerSec);
                    const params = renderParams;

                    // 应用增益（音量 + 淡入淡出）
                    const peaksForRender = renderInterleaved;

                    const __tDs1 = __perfDebug ? performance.now() : 0;
                    const __tGain0 = __perfDebug ? performance.now() : 0;
                    const withGains = applyGainsToPeaks(peaksForRender, params);
                    const __tGain1 = __perfDebug ? performance.now() : 0;

                    // ========================================
                    // 废弃离屏 Canvas
                    // ========================================
                    const __tRender0 = __perfDebug ? performance.now() : 0;

                    const baseAlpha = (clip.muted ? 0.4 : 1.0) * (takeLane?.inactive ? 0.78 : 1);
                    // 前导重叠可视化仅作用于第一个分段（重叠区只在 clip 起始处）。
                    // 查表必须用展开前的真实 Clip id：lane 的 id 带 `::take::`
                    // 后缀，直接查会导致多 take 展示下重叠可视化静默丢失。
                    const leadingOverlapSec = Math.max(
                        0,
                        Math.min(
                            clip.lengthSec,
                            Number(
                                currentLeadingOverlapSecByClipId[clip.__originClipId ?? clip.id] ??
                                    0,
                            ) || 0,
                        ),
                    );
                    const leadingOverlapRightPx =
                        (clipStartSec + leadingOverlapSec) * currentPxPerSec - viewportStartPx;
                    const leadingOverlapVisibleRight =
                        leadingOverlapSec > 1e-9 && seg.localStartSec <= leadingOverlapSec
                            ? Math.min(
                                  tileVisRightPx,
                                  Math.max(tileVisLeftPx, leadingOverlapRightPx),
                              )
                            : tileVisLeftPx;

                    const drawSegment = (
                        segmentLeftPx: number,
                        segmentRightPx: number,
                        alpha: number,
                    ) => {
                        if (segmentRightPx - segmentLeftPx <= 1e-6) return;
                        ctx.save();
                        ctx.beginPath();
                        // 严格裁剪在片段实际可见范围内，防止越界绘制到其他片段上
                        ctx.rect(segmentLeftPx, 0, segmentRightPx - segmentLeftPx, displayH);
                        ctx.clip();
                        ctx.globalAlpha = alpha;
                        renderWaveform(
                            ctx,
                            withGains,
                            params,
                            strokeColorForClip,
                            currentStrokeWidth,
                            "line",
                        );
                        ctx.restore();
                    };

                    if (leadingOverlapVisibleRight > tileVisLeftPx + 1e-6) {
                        drawSegment(
                            tileVisLeftPx,
                            leadingOverlapVisibleRight,
                            baseAlpha * LEADING_OVERLAP_ALPHA,
                        );
                        drawSegment(leadingOverlapVisibleRight, tileVisRightPx, baseAlpha);
                    } else {
                        drawSegment(tileVisLeftPx, tileVisRightPx, baseAlpha);
                    }

                    const __tRender1 = __perfDebug ? performance.now() : 0;
                    const __tDraw0 = 0; // 已废弃 drawImage
                    const __tDraw1 = 0;

                    // 1. 归还增益 buffer
                    if (withGains !== renderInterleaved) {
                        releaseGainBuffer(withGains);
                    }

                    // 2. 派生降采样缓冲由 derivedDownsampleCache 持有，
                    //    在本 clip 瓦片循环结束后统一归还（见缓存说明）。

                    // 3. store 复用池 buffer 由 fetchCache 统一持有/归还（见缓存说明）。

                    // 收集诊断数据
                    if (__perfDebug) {
                        const fileName = clip.sourcePath?.split(/[/\\]/).pop() ?? "?";
                        __clipTimings.push({
                            name: fileName,
                            sliceMs: __tSlice1 - __tSlice0,
                            downsampleMs: __tDs1 - __tDs0,
                            gainMs: __tGain1 - __tGain0,
                            renderMs: __tRender1 - __tRender0,
                            drawImageMs: __tDraw1 - __tDraw0,
                            interleavedLen: storeInterleaved.length,
                            visibleWidthPx: tileVisibleWidthPx,
                            downsampledTo: renderInterleaved.length / 2,
                        });
                    }
                }

                // 本 clip 的所有瓦片处理完毕，归还缓存持有的 store buffer
                // 与派生降采样缓冲。
                releaseFetchCache();
                releaseDerivedDownsampleCache();

                // ── 循环节点倒三角标记（Loop 启用且存在内部回绕点时）──
                // 节点位置：头部段结束处（进入段耗尽、首次环绕）及此后
                // 每个整文件周期边界 —— 与用户示例 [0,8)=2→10、[8,18)=0→10 一致。
                if (isLoop && mediaDur > 1e-6) {
                    // 与分段构建同一 floor_mod 归一化（不能用 clamp），
                    // 保证标记位置 = 实际分段边界。
                    const anchorFwd = modEuclid(sourceStartSec, mediaDur);
                    const anchorRev = modEuclid(effSrcEnd, mediaDur);
                    const headDur = (clip.reversed ? anchorRev : mediaDur - anchorFwd) / pr;
                    const bodyDur = mediaDur / pr;
                    const markers: number[] = [];
                    // 直接跳到可视范围内的第一个回绕点（与分段构建的直接寻址
                    // 一致）：既避免从 clip 入口逐周期空转，也修复"深入长循环
                    // clip 后标记消失"（旧 guard<8192 限制）。
                    {
                        const visLocalStart =
                            viewportStartPx / Math.max(1e-9, currentPxPerSec) - clipStartSec;
                        const k0 = Math.max(
                            0,
                            Math.ceil((visLocalStart - headDur - 1e-6) / bodyDur),
                        );
                        for (
                            let markerT = headDur + k0 * bodyDur;
                            markerT < clip.lengthSec - 1e-6 && markers.length < 4096;
                            markerT += bodyDur
                        ) {
                            if (markerT <= 1e-6) continue; // 起点回绕点不绘制
                            const mx = (clipStartSec + markerT) * currentPxPerSec - viewportStartPx;
                            if (mx > displayW + 8) break;
                            if (mx < -8) continue;
                            markers.push(Math.round(mx * 2) / 2);
                        }
                    }
                    if (markers.length > 0) {
                        drawLoopMarkers(ctx, markers, displayH, strokeColorForClip);
                    }
                } else if (mediaDur > 1e-6) {
                    // ── 非 Loop：媒体边界标记（"循环节"的退化形式）──────
                    // 未循环 Clip 的循环节 = 源媒体在该 Clip 内的真实起始
                    // 位置（s=0）与真实终止位置（s=D）。它们是音频与静音的
                    // 分界线，落在 Clip 内部时绘制倒三角（前导/尾部静音、
                    // 左右延伸的视觉锚点）。
                    // 投影按**消费方向**：正放 t=(b−ss)/r；倒放 t=(se−b)/r
                    // （倒放锚定窗口终点 —— 此前用 (b−ss)/r 会把标记放到
                    // 与音频静音分界错位的位置）。
                    const markers: number[] = [];
                    for (const b of [0, mediaDur]) {
                        const tLocal = clip.reversed
                            ? (winEndSec - b) / pr
                            : (b - winStartSec) / pr;
                        if (tLocal <= 1e-6 || tLocal >= clip.lengthSec - 1e-6) continue;
                        const mx = (clipStartSec + tLocal) * currentPxPerSec - viewportStartPx;
                        if (mx < -8 || mx > displayW + 8) continue;
                        markers.push(Math.round(mx * 2) / 2);
                    }
                    if (markers.length > 0) {
                        drawLoopMarkers(ctx, markers, displayH, strokeColorForClip);
                    }
                }

                // ── SnapOffset（吸附偏移）竖线 ─────────────────────
                // SnapOffset 是 Clip 自身属性：相对 Clip 起点的偏移（秒，
                // 与倒放无关）。非 0 时以黄色竖虚线标记其位置，x 与左下角
                // ◣ 三角的左侧竖直边严格一致（含贴着 Clip 末端的情形）。
                {
                    const snapOffsetLocal = Math.max(0, Number(clip.snapOffsetSec) || 0);
                    if (snapOffsetLocal > 1e-6 && snapOffsetLocal <= clip.lengthSec + 1e-6) {
                        const mx =
                            Math.round((clipStartSec + snapOffsetLocal) * currentPxPerSec * 2) / 2 -
                            viewportStartPx;
                        if (mx >= -1 && mx <= displayW + 1) {
                            ctx.save();
                            ctx.strokeStyle = "rgba(255, 214, 102, 0.9)";
                            ctx.lineWidth = 1;
                            ctx.setLineDash([4, 3]);
                            ctx.beginPath();
                            ctx.moveTo(mx, 1);
                            ctx.lineTo(mx, displayH - 1);
                            ctx.stroke();
                            ctx.restore();
                        }
                    }
                }

                // 多 Take lane 分界线：只在 lane 顶部绘制（首条 lane 顶边
                // 即 Clip header 边界，不重复画）。颜色与轨道边框一致地保持低对比。
                // （主题色已在绘制帧开始时统一解析，见 defaultSeparatorColor。）
                if (takeLane && takeLane.index > 0) {
                    const separatorColor = takeSeparatorColorRef.current ?? defaultSeparatorColor;
                    ctx.save();
                    ctx.strokeStyle = separatorColor;
                    ctx.lineWidth = 1;
                    ctx.beginPath();
                    const boundaryY = Math.max(0.5, Math.min(displayH - 0.5, 0.5));
                    ctx.moveTo(visLeftPx, boundaryY);
                    ctx.lineTo(visRightPx, boundaryY);
                    ctx.stroke();
                    ctx.restore();
                }
            }
            displayH = baseDisplayHeight;
            ctx.setTransform(scaleX, 0, 0, scaleY, 0, 0);
            if (ctx.globalAlpha !== 1) {
                ctx.globalAlpha = 1;
            }

            // ========================================
            // 性能诊断输出
            // ========================================
            {
                const totalMs = performance.now() - __t0;
                wfDiag_frameEnd(totalMs);
            }
            if (__perfDebug) {
                const totalMs = performance.now() - __t0;
                const clipCount = __clipTimings.length;
                const sumSlice = __clipTimings.reduce((s, c) => s + c.sliceMs, 0);
                const sumDs = __clipTimings.reduce((s, c) => s + c.downsampleMs, 0);
                const sumGain = __clipTimings.reduce((s, c) => s + c.gainMs, 0);
                const sumRender = __clipTimings.reduce((s, c) => s + c.renderMs, 0);
                const sumDrawImg = __clipTimings.reduce((s, c) => s + c.drawImageMs, 0);
                console.log(
                    `%c[WaveformPerf] frame ${totalMs.toFixed(1)}ms | setup=${__tSetup.toFixed(1)}ms | clips=${clipCount} | pxPerSec=${currentPxPerSec.toFixed(0)} | canvasW=${displayW} | dpr=${dpr}`,
                    totalMs > 16 ? "color:red;font-weight:bold" : "color:green",
                );
                console.log(
                    `  ├ slice=${sumSlice.toFixed(1)}ms | downsample=${sumDs.toFixed(1)}ms | gain=${sumGain.toFixed(1)}ms | render=${sumRender.toFixed(1)}ms | drawImage=${sumDrawImg.toFixed(1)}ms`,
                );
                for (const c of __clipTimings) {
                    console.log(
                        `  └ clip "${c.name}": interleaved=${c.interleavedLen} → ds=${c.downsampledTo} | visPx=${c.visibleWidthPx} | slice=${c.sliceMs.toFixed(2)} ds=${c.downsampleMs.toFixed(2)} gain=${c.gainMs.toFixed(2)} render=${c.renderMs.toFixed(2)} draw=${c.drawImageMs.toFixed(2)}`,
                    );
                }
            }
        };

        // ========================================
        // 监听 mipmap 缓存加载完成事件，触发 invalidate
        // ========================================
        React.useEffect(() => {
            const neededPaths = new Set<string>();
            for (const clip of clips) {
                if (clip.sourcePath) neededPaths.add(clip.sourcePath);
                // 多 Take 泳道：inactive take 的源也要监听 —— 否则其 mipmap
                // 异步加载完成后不触发重绘，泳道在界面静止时无限期空白。
                for (const take of clip.takes ?? []) {
                    if (take.sourcePath) neededPaths.add(take.sourcePath);
                }
            }

            const unsub = waveformMipmapStore.addListener((sourcePath, status) => {
                if (status === "done" && neededPaths.has(sourcePath)) {
                    wfDiag_invalidateMipmap();
                    invalidate();
                }
            });

            return unsub;
        }, [clips, invalidate]);

        // ========================================
        // ★ 订阅事件总线（核心性能优化）
        // TimelinePanel.syncScrollLeft() 直接广播 → 更新 ref → invalidate
        // 完全绕过 React props 链路，与 PianoRoll 架构一致
        // ========================================
        React.useEffect(() => {
            const unsub = timelineViewportBus.subscribe((scrollLeft, pxPerSec, viewportWidth) => {
                // 直接更新 ref（不触发 React re-render）
                pxPerSecRef.current = pxPerSec;
                const vpStartSec = scrollLeft / pxPerSec;
                const vpEndSec = vpStartSec + viewportWidth / pxPerSec;
                viewportStartSecRef.current = vpStartSec;
                viewportEndSecRef.current = vpEndSec;
                viewportWidthPxRef.current = viewportWidth;
                if (canvasRef.current) {
                    canvasRef.current.style.transform = `translate3d(${scrollLeft}px,0,0)`;
                }

                wfDiag_invalidateBus();
                invalidate();
            });
            return unsub;
        }, [invalidate]);

        // ========================================
        // 低频 props 变化时 invalidate
        // 仅监听 clips / waveformHeight / strokeColor 等不频繁变化的 props
        // 高频视口参数（pxPerSec / viewportStartSec / viewportEndSec）已由事件总线处理
        // ========================================
        React.useEffect(() => {
            wfDiag_invalidateProps();
            invalidate();
        }, [
            clips,
            waveformHeight,
            strokeColor,
            strokeWidth,
            viewportWidthPx,
            showAllTakes,
            takeSeparatorColor,
            invalidate,
        ]);

        // 组件卸载时取消待执行的 rAF
        React.useEffect(() => {
            return () => {
                if (rafRef.current != null) {
                    cancelAnimationFrame(rafRef.current);
                    rafRef.current = null;
                }
            };
        }, []);

        // 移除原有的 canvasWidthPx 和 canvasLeftPx 的计算，直接替换 return
        return (
            <canvas
                ref={canvasRef}
                style={{
                    position: "absolute",
                    top: waveformTop,
                    // height 交给 style 控制比较稳定
                    height: waveformHeight,
                    pointerEvents: "none",
                    zIndex: 1,
                    left: 0,
                    // 移除 left 和 width
                    // 它们属于高频变化属性，已完全交由内部 drawRef 直接操作 DOM 更新。
                }}
            />
        );
    },
    // ★ 自定义比较函数：忽略高频 props（pxPerSec/viewportStartSec/viewportEndSec）
    // 这些高频参数由 timelineViewportBus 直接推送到 ref → invalidate，无需 React re-render
    (prev, next) => {
        return (
            prev.clips === next.clips &&
            prev.leadingOverlapSecByClipId === next.leadingOverlapSecByClipId &&
            prev.trackHeight === next.trackHeight &&
            prev.waveformTop === next.waveformTop &&
            prev.waveformHeight === next.waveformHeight &&
            prev.viewportWidthPx === next.viewportWidthPx &&
            prev.strokeColor === next.strokeColor &&
            prev.strokeWidth === next.strokeWidth &&
            prev.showAllTakes === next.showAllTakes &&
            prev.takeSeparatorColor === next.takeSeparatorColor
            // 故意不比较 pxPerSec / viewportStartSec / viewportEndSec
        );
    },
);
