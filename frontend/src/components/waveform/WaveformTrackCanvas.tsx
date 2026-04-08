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
import type { FadeCurveType } from "../layout/timeline/paths";
import { waveformMipmapStore } from "../../utils/waveformMipmapStore";
import { timelineViewportBus } from "../../utils/timelineViewportBus";
import {
    applyGainsToPeaks,
    releaseGainBuffer,
    renderWaveform,
    type WaveformRenderParams,
} from "../../utils/waveformRenderer";
// ========================================
// 局部 Buffer 复用池
// ========================================
const _downsamplePool: Float32Array[] = [];
const POOL_MAX = 8;

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
        } = props;

        // ========================================
        // refs：高频变化的参数存 ref，避免 React re-render
        // ========================================
        const canvasRef = React.useRef<HTMLCanvasElement | null>(null);
        const lastLevelByClipRef = React.useRef<Record<string, 0 | 1 | 2>>({});
        const rafRef = React.useRef<number | null>(null);

        // 高频参数用 ref 存储，避免依赖数组变化触发 useLayoutEffect
        const pxPerSecRef = React.useRef(props.pxPerSec);
        const viewportStartSecRef = React.useRef(props.viewportStartSec);
        const viewportEndSecRef = React.useRef(props.viewportEndSec);
        const clipsRef = React.useRef(clips);
        const waveformHeightRef = React.useRef(waveformHeight);
        const strokeColorRef = React.useRef(strokeColor);
        const strokeWidthRef = React.useRef(strokeWidth);
        const viewportWidthPxRef = React.useRef(viewportWidthPx);

        // 同步 ref
        pxPerSecRef.current = props.pxPerSec;
        viewportStartSecRef.current = props.viewportStartSec;
        viewportEndSecRef.current = props.viewportEndSec;
        clipsRef.current = clips;
        waveformHeightRef.current = waveformHeight;
        strokeColorRef.current = strokeColor;
        strokeWidthRef.current = strokeWidth;
        viewportWidthPxRef.current = viewportWidthPx;

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
            let __tSetup = 0,
                __clipTimings: {
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
            const currentClips = clipsRef.current;
            const currentWaveformHeight = waveformHeightRef.current;
            const currentStrokeColor = strokeColorRef.current;
            const currentStrokeWidth = strokeWidthRef.current;
            const currentViewportWidthPx = viewportWidthPxRef.current;
            const displayW = Math.max(1, Math.ceil(currentViewportWidthPx));
            const displayH = currentWaveformHeight;

            // 取消限制 dpr 为 1
            const dpr = window.devicePixelRatio || 1;
            const internalW = Math.max(1, Math.floor(displayW * dpr));
            const internalH = Math.max(1, Math.floor(displayH * dpr));

            // 仅在尺寸变化时更新 canvas 物理尺寸
            if (canvas.width !== internalW) canvas.width = internalW;
            if (canvas.height !== internalH) canvas.height = internalH;

            const ctx = canvas.getContext("2d");
            if (!ctx) return;

            const scaleX = internalW / Math.max(1, displayW);
            const scaleY = internalH / Math.max(1, displayH);
            ctx.setTransform(scaleX, 0, 0, scaleY, 0, 0);
            ctx.clearRect(0, 0, displayW, displayH);

            // left 由 timelineViewportBus 单一来源更新，避免双来源写入导致的微抖
            canvas.style.width = `${displayW}px`;
            canvas.style.height = `${displayH}px`;

            if (__perfDebug) __tSetup = performance.now() - __t0;

            for (const clip of currentClips) {
                if (!clip.sourcePath || !clip.durationSec || clip.durationSec <= 0) continue;

                const clipStartSec = clip.startSec;
                const clipEndSec = clipStartSec + clip.lengthSec;
                const clipWidthPx = clip.lengthSec * currentPxPerSec;

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
                const pr = Math.max(1e-6, clip.playbackRate);
                const sourceStartSec = Number(clip.sourceStartSec ?? 0) || 0;
                const visibleWidthPx = Math.max(1, Math.ceil(visRightPx - visLeftPx));

                // 计算源文件时间范围
                const sampleRate = clip.sourceSampleRate || 44100;
                const spp = Math.max(1, Math.round(sampleRate / currentPxPerSec));
                const levelKey = `${clip.sourcePath}::${clip.id}`;
                const previousLevel = lastLevelByClipRef.current[levelKey];
                const stableLevel = waveformMipmapStore.selectLevelStable(spp, previousLevel);
                lastLevelByClipRef.current[levelKey] = stableLevel;

                const clipSourceEndSec =
                    Number(clip.sourceEndSec ?? clip.durationSec) || clip.durationSec;
                const clipSourceSpanSec = Math.max(
                    0,
                    Math.min(clip.lengthSec * pr, clipSourceEndSec - sourceStartSec),
                );
                if (clipSourceSpanSec <= 1e-6) continue;

                // 仅请求当前可见窗口对应的源数据，显著降低每帧处理成本
                const visClipStartSec = Math.max(0, visStartSec - clipStartSec);
                const visClipEndSec = Math.min(clip.lengthSec, visEndSec - clipStartSec);
                const clipSourceWindowStartSec = sourceStartSec;
                const clipSourceWindowEndSec = sourceStartSec + clipSourceSpanSec;
                const sourceVisStartSec = clip.reversed
                    ? clipSourceWindowEndSec - visClipEndSec * pr
                    : clipSourceWindowStartSec + visClipStartSec * pr;
                const sourceVisEndSec = clip.reversed
                    ? clipSourceWindowEndSec - visClipStartSec * pr
                    : clipSourceWindowStartSec + visClipEndSec * pr;
                const sourcePadSec = Math.max(0.005, (2 / Math.max(1, currentPxPerSec)) * pr);
                const sourceTimeStart = Math.max(
                    clipSourceWindowStartSec,
                    Math.min(sourceVisStartSec, sourceVisEndSec) - sourcePadSec,
                );
                const sourceTimeEnd = Math.min(
                    clipSourceWindowEndSec,
                    Math.max(sourceVisStartSec, sourceVisEndSec) + sourcePadSec,
                );
                const sourceDuration = Math.max(0.001, sourceTimeEnd - sourceTimeStart);

                // ========================================
                // 从 mipmap 缓存获取 interleaved 数据（不 resample，与 PianoRoll 一致）
                // ========================================
                const __tSlice0 = __perfDebug ? performance.now() : 0;
                const result = waveformMipmapStore.getInterleavedSlice(
                    clip.sourcePath,
                    stableLevel,
                    sourceTimeStart,
                    sourceDuration,
                );
                const __tSlice1 = __perfDebug ? performance.now() : 0;

                if (!result || result.interleaved.length < 4) {
                    continue;
                }

                // ========================================
                // 方案2：限制数据量 — 当原始数据点数远超可视像素时，快速预降采样
                // ========================================
                const __tDs0 = __perfDebug ? performance.now() : 0;
                const storeInterleaved = result.interleaved;
                let renderInterleaved: Float32Array = storeInterleaved;
                let releasedStoreInterleaved = false;
                const rawSampleCount = storeInterleaved.length / 2;
                // 使用与 clip 自身宽度绑定的稳定采样目标，避免滚屏时分桶边界漂移导致抖动
                const stableTargetWidthPx = Math.max(1, Math.ceil(visibleWidthPx * 2));
                const targetSamples = stableTargetWidthPx * 2;

                if (rawSampleCount > targetSamples && targetSamples >= 2) {
                    const w = Math.ceil(targetSamples);

                    // 从局部池获取 Buffer
                    const downsampled = acquireDownsampleBuffer(w * 2);

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

                    waveformMipmapStore.releaseInterleaved(storeInterleaved);
                    releasedStoreInterleaved = true;
                    renderInterleaved = downsampled;
                }

                // --- 从这里开始替换 ---

                // 偏移量改为相对于主屏幕，而不是裁剪视口
                const clipPixelOffset = viewportStartPx - clipStartPx;

                // 构建渲染参数
                const params: WaveformRenderParams = {
                    canvasWidth: displayW,
                    canvasHeight: displayH,
                    centerY: displayH / 2,
                    zeroDbHalfHeight: displayH / 2,
                    sourceStartSec,
                    clipDuration: clip.lengthSec,
                    playbackRate: Number(clip.playbackRate ?? 1) || 1,
                    reversed: Boolean(clip.reversed),
                    sourceDurationSec: clip.durationSec,
                    volumeGain: Number(clip.gain ?? 1) || 1,
                    fadeInSec: Number(clip.fadeInSec ?? 0) || 0,
                    fadeOutSec: Number(clip.fadeOutSec ?? 0) || 0,
                    fadeInCurve: (clip.fadeInCurve as FadeCurveType) ?? "sine",
                    fadeOutCurve: (clip.fadeOutCurve as FadeCurveType) ?? "sine",
                    dataStartSec: result.dataStartSec,
                    dataDurationSec: result.dataDurationSec,
                    clipPixelOffset, // 相对于主 Canvas 的偏移
                    clipTotalWidthPx: Math.max(1, clipWidthPx),
                };

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

                ctx.save();
                ctx.beginPath();
                // 严格裁剪在 clip 实际可见范围内，防止越界绘制到其他片段上
                ctx.rect(visLeftPx, 0, visRightPx - visLeftPx, displayH);
                ctx.clip();

                // 处理静音片段半透明
                ctx.globalAlpha = clip.muted ? 0.4 : 1.0;

                renderWaveform(
                    ctx,
                    withGains,
                    params,
                    currentStrokeColor,
                    currentStrokeWidth,
                    "line",
                );

                ctx.restore();

                const __tRender1 = __perfDebug ? performance.now() : 0;
                const __tDraw0 = 0; // 已废弃 drawImage
                const __tDraw1 = 0;

                // 1. 归还增益 buffer
                if (withGains !== renderInterleaved) {
                    releaseGainBuffer(withGains);
                }

                // 2. 归还预降采样产生的 buffer（如果有）
                if (renderInterleaved !== storeInterleaved) {
                    releaseDownsampleBuffer(renderInterleaved);
                }

                // 3. 归还 store 复用池 buffer
                if (!releasedStoreInterleaved) {
                    waveformMipmapStore.releaseInterleaved(storeInterleaved);
                }

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
                        visibleWidthPx,
                        downsampledTo: renderInterleaved.length / 2,
                    });
                }
            }

            if (ctx.globalAlpha !== 1) {
                ctx.globalAlpha = 1;
            }

            // ========================================
            // 性能诊断输出
            // ========================================
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
            }

            const unsub = waveformMipmapStore.addListener((sourcePath, status) => {
                if (status === "done" && neededPaths.has(sourcePath)) {
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
            invalidate();
        }, [clips, waveformHeight, strokeColor, strokeWidth, viewportWidthPx, invalidate]);

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
                    willChange: "transform",
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
            prev.trackHeight === next.trackHeight &&
            prev.waveformTop === next.waveformTop &&
            prev.waveformHeight === next.waveformHeight &&
            prev.viewportWidthPx === next.viewportWidthPx &&
            prev.strokeColor === next.strokeColor &&
            prev.strokeWidth === next.strokeWidth
            // 故意不比较 pxPerSec / viewportStartSec / viewportEndSec
        );
    },
);
