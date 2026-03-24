/**
 * WaveformTrackCanvas - 轨道级波形 Canvas 组件（v2 mipmap 缓存架构）
 *
 * 核心思想：每条轨道只有一个 Canvas，负责绘制该轨道上所有可见 clip 的波形。
 * 相比之前「每 clip 一个 Canvas」的方案，大幅减少 Canvas 上下文数量（从 O(clip) 降为 O(track)）。
 *
 * 渲染流程：
 *   1. Canvas 宽度 = 视口可见宽度，通过 CSS left 偏移跟随水平滚动
 *   2. 遍历所有可见 clip，对每个 clip：
 *      a. waveformMipmapStore.getInterleavedSlice() 获取该 clip 对应源文件的稳定峰值切片
 *         （整文件级三级 mipmap 缓存 + 零拷贝切片）
 *      b. applyGainsToPeaks 应用增益/淡入淡出
 *      c. ctx.save() / ctx.clip() 限制绘制区域到 clip 的像素边界
 *      d. renderWaveform() 绘制波形
 *      e. ctx.restore()
 *
 * 性能优势：
 *   - 100+ clip 场景下只需 10-20 个 Canvas 上下文（= 轨道数）
 *   - 无 DOM 布局抖动：clip 拖拽时只需 requestAnimationFrame 重绘 Canvas
 *   - GPU 批量提交：单 Canvas 上所有 drawCall 合并为一次 GPU 提交
 *   - 整文件级缓存：无 IPC 请求，数据已在前端内存中
 *   - Float32Array 零拷贝切片：无需每帧分配新 buffer
 *
 * 数据流（v2 mipmap 架构）：
 *   waveformMipmapStore.getInterleavedSlice() → interleaved Float32Array → applyGainsToPeaks → renderWaveform
 */

import React from "react";
import type { ClipInfo } from "../../features/session/sessionTypes";
import type { FadeCurveType } from "../layout/timeline/paths";
import { waveformMipmapStore } from "../../utils/waveformMipmapStore";
import {
    applyGainsToPeaks,
    renderWaveform,
    type WaveformRenderParams,
} from "../../utils/waveformRenderer";

/** 可视区缓冲（像素），防止滚动时出现空白；固定像素数，不随缩放膨胀 */
const BUFFER_PX = 500;

export interface WaveformTrackCanvasProps {
    /** 当前轨道上所有 clip（已由 TrackLane 做过可视区过滤） */
    clips: ClipInfo[];
    /** 轨道高度（像素），包含 header 和 padding */
    trackHeight: number;
    /** 波形区域的 top 偏移（跳过 clip header 部分） */
    waveformTop: number;
    /** 波形区域高度 */
    waveformHeight: number;
    /** 每秒像素数 */
    pxPerSec: number;
    /** 视口宽度（CSS 像素），用于固定 canvas 宽度避免缩放时抖动 */
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

export const WaveformTrackCanvas = React.memo(function WaveformTrackCanvas(
    props: WaveformTrackCanvasProps,
) {
    const {
        clips,
        waveformTop,
        waveformHeight,
        pxPerSec,
        viewportWidthPx,
        viewportStartSec,
        strokeColor,
        strokeWidth = 1,
    } = props;

    const canvasRef = React.useRef<HTMLCanvasElement | null>(null);
    const backBufferRef = React.useRef<HTMLCanvasElement | null>(null);
    const lastLevelByPathRef = React.useRef<Record<string, 0 | 1 | 2>>({});

    // 强制重绘计数器（mipmap 数据加载完成时 +1 触发重绘）
    const [redrawTick, setRedrawTick] = React.useState(0);

    // 以像素为基准计算缓冲区，避免高缩放下秒↔像素往返换算造成画布原点漂移
    const viewportLeftPx = viewportStartSec * pxPerSec;
    const canvasLeftPx = Math.max(0, Math.floor(viewportLeftPx - BUFFER_PX));
    const canvasWidthPx = Math.max(
        1,
        Math.ceil(Math.max(1, viewportWidthPx)) + BUFFER_PX * 2,
    );
    const bufferedStartSec = canvasLeftPx / Math.max(1e-9, pxPerSec);
    const bufferedEndSec =
        (canvasLeftPx + canvasWidthPx) / Math.max(1e-9, pxPerSec);

    // ========================================
    // 监听 mipmap 缓存加载完成事件，触发重绘
    // ========================================
    React.useEffect(() => {
        // 收集本轮需要的 sourcePath 集合
        const neededPaths = new Set<string>();
        for (const clip of clips) {
            if (clip.sourcePath) neededPaths.add(clip.sourcePath);
        }

        const unsub = waveformMipmapStore.addListener((sourcePath, status) => {
            if (status === "done" && neededPaths.has(sourcePath)) {
                setRedrawTick((t) => t + 1);
            }
        });

        return unsub;
    }, [clips]);

    // ========================================
    // 主渲染逻辑：在单个 Canvas 上绘制所有 clip 的波形
    // ========================================
    React.useLayoutEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const dpr = Math.max(1, window.devicePixelRatio || 1);
        const displayW = canvasWidthPx;
        const displayH = waveformHeight;
        const backBuffer =
            backBufferRef.current ?? document.createElement("canvas");
        backBufferRef.current = backBuffer;

        // Canvas 内部像素 = CSS 尺寸 × dpr
        const internalW = Math.max(1, Math.floor(displayW * dpr));
        const internalH = Math.max(1, Math.floor(displayH * dpr));

        if (backBuffer.width !== internalW) {
            backBuffer.width = internalW;
        }
        if (backBuffer.height !== internalH) {
            backBuffer.height = internalH;
        }

        const backCtx = backBuffer.getContext("2d");
        if (!backCtx) return;

        const scaleX = internalW / Math.max(1, displayW);
        const scaleY = internalH / Math.max(1, displayH);
        backCtx.setTransform(scaleX, 0, 0, scaleY, 0, 0);
        backCtx.clearRect(0, 0, displayW, displayH);

        // Canvas 左边缘对应的 timeline 时间
        const canvasStartSec = bufferedStartSec;

        // 遍历所有 clip，绘制波形
        for (const clip of clips) {
            if (!clip.sourcePath || !clip.durationSec || clip.durationSec <= 0) continue;

            const clipStartSec = clip.startSec;
            const clipEndSec = clipStartSec + clip.lengthSec;
            const clipWidthPx = clip.lengthSec * pxPerSec;

            // clip 与视口交集的像素区域
            const visStartSec = Math.max(clipStartSec, bufferedStartSec);
            const visEndSec = Math.min(clipEndSec, bufferedEndSec);
            if (visEndSec <= visStartSec) continue;

            // 可见区域在 canvas 上的像素位置
            const visLeftPx = Math.max(0, (visStartSec - canvasStartSec) * pxPerSec);
            const visRightPx = Math.min(displayW, (visEndSec - canvasStartSec) * pxPerSec);
            const pr = Math.max(1e-6, clip.playbackRate);
            const clipLen = clip.lengthSec;
            const sourceStartSec = Number(clip.sourceStartSec ?? 0) || 0;
            const sampleStartSec = visStartSec;
            const sampleEndSec = visEndSec;
            if (sampleEndSec <= sampleStartSec) continue;
            const visibleWidthPx = Math.max(1, Math.ceil(visRightPx - visLeftPx));

            // ========================================
            // 计算源文件时间范围（与实际绘制窗口对齐）
            // ========================================
            const sampleRate = clip.sourceSampleRate || 44100;
            const spp = Math.max(1, Math.round(sampleRate / pxPerSec));
            const previousLevel = lastLevelByPathRef.current[clip.sourcePath];
            const stableLevel = waveformMipmapStore.selectLevelStable(
                spp,
                previousLevel,
            );
            lastLevelByPathRef.current[clip.sourcePath] = stableLevel;

            // 可见部分在 clip 内的比例
            const ratioStart = (sampleStartSec - clipStartSec) / Math.max(1e-6, clipLen);
            const ratioEnd = (sampleEndSec - clipStartSec) / Math.max(1e-6, clipLen);

            const clipSourceEndSec =
                Number(clip.sourceEndSec ?? clip.durationSec) || clip.durationSec;
            const clipSourceSpanSec = Math.max(
                0,
                Math.min(clip.lengthSec * pr, clipSourceEndSec - sourceStartSec),
            );
            const sourceTimeStart = Math.max(
                0,
                sourceStartSec + ratioStart * clipSourceSpanSec,
            );
            const sourceTimeEnd = Math.min(
                clip.durationSec,
                sourceStartSec + ratioEnd * clipSourceSpanSec,
            );
            const sourceDuration = Math.max(0.1, sourceTimeEnd - sourceTimeStart);

            // ========================================
            // 从 mipmap 缓存获取 resample 后的数据
            // ========================================
            const result = waveformMipmapStore.getInterleavedSlice(
                clip.sourcePath,
                stableLevel,
                sourceTimeStart,
                sourceDuration,
            );

            if (!result || result.interleaved.length < 4) continue;

            // 计算 clip 内的偏移
            const clipPixelOffset = (visStartSec - clipStartSec) * pxPerSec;

            // 构建渲染参数
            const params: WaveformRenderParams = {
                canvasWidth: visibleWidthPx,
                canvasHeight: displayH,
                centerY: displayH / 2,
                sourceStartSec,
                clipDuration: clip.lengthSec,
                playbackRate: Number(clip.playbackRate ?? 1) || 1,
                sourceDurationSec: clip.durationSec,
                volumeGain: Number(clip.gain ?? 1) || 1,
                fadeInSec: Number(clip.fadeInSec ?? 0) || 0,
                fadeOutSec: Number(clip.fadeOutSec ?? 0) || 0,
                fadeInCurve: (clip.fadeInCurve as FadeCurveType) ?? "sine",
                fadeOutCurve: (clip.fadeOutCurve as FadeCurveType) ?? "sine",
                dataStartSec: result.dataStartSec,
                dataDurationSec: result.dataDurationSec,
                clipPixelOffset,
                clipTotalWidthPx: Math.max(1, clipWidthPx),
            };

            // 应用增益（音量 + 淡入淡出）
            const withGains = applyGainsToPeaks(result.interleaved, params);

            // 使用 clip 裁剪区域绘制
            backCtx.save();
            backCtx.beginPath();
            backCtx.rect(visLeftPx, 0, visRightPx - visLeftPx, displayH);
            backCtx.clip();

            // 平移 ctx 使得 renderWaveform 的坐标从 0 开始
            backCtx.translate(visLeftPx, 0);

            // 静音 clip 半透明
            if (clip.muted) {
                backCtx.globalAlpha = 0.4;
            } else {
                backCtx.globalAlpha = 1.0;
            }

            renderWaveform(
                backCtx,
                withGains,
                params,
                strokeColor,
                strokeWidth,
                "line",
            );

            backCtx.restore();
        }

        if (canvas.width !== internalW) {
            canvas.width = internalW;
        }
        if (canvas.height !== internalH) {
            canvas.height = internalH;
        }
        const nextStyleWidth = `${displayW}px`;
        const nextStyleHeight = `${displayH}px`;
        if (canvas.style.width !== nextStyleWidth) {
            canvas.style.width = nextStyleWidth;
        }
        if (canvas.style.height !== nextStyleHeight) {
            canvas.style.height = nextStyleHeight;
        }

        const ctx = canvas.getContext("2d");
        if (!ctx) return;
        ctx.setTransform(scaleX, 0, 0, scaleY, 0, 0);
        ctx.clearRect(0, 0, displayW, displayH);
        ctx.drawImage(backBuffer, 0, 0, internalW, internalH, 0, 0, displayW, displayH);
        if (ctx.globalAlpha !== 1) {
            ctx.globalAlpha = 1;
        }
    }, [
        clips,
        canvasWidthPx,
        waveformHeight,
        pxPerSec,
        bufferedStartSec,
        bufferedEndSec,
        strokeColor,
        strokeWidth,
        redrawTick,
    ]);

    return (
        <canvas
            ref={canvasRef}
            style={{
                position: "absolute",
                left: canvasLeftPx,
                top: waveformTop,
                width: canvasWidthPx,
                height: waveformHeight,
                pointerEvents: "none",
                zIndex: 0,
            }}
        />
    );
});
