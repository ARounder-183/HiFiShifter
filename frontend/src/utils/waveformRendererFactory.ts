/**
 * 波形渲染器工厂
 *
 * 主要内容：根据浏览器能力检测（WebGL2 是否可用）创建对应的 WaveformRenderer 实例
 * 作用：为消费方（WaveformTrackCanvas / PianoRoll 背景波形）提供统一的渲染器入口，
 *       屏蔽底层实现差异
 * 与其他模块的关系：
 *   - 依赖 waveformRenderer.ts 的 WaveformRenderer 接口与 Canvas2DWaveformRenderer
 *   - 依赖 waveformWebGL2Renderer.ts 的 WebGL2WaveformRenderer
 *   - 被 WaveformTrackCanvas.tsx 和 PianoRollPanel.tsx 调用
 *
 * 能力检测策略：
 *   1. 读取 localStorage.hifishifter.forceCanvas2DWaveform，若为 "1" 则强制走 Canvas 2D
 *   2. 用临时 canvas 探测 WebGL2 能力（避免污染传入的 canvas，导致后续无法获取 2d context）
 *   3. 探测成功 → 在传入 canvas 上获取 webgl2 context → 创建 WebGL2WaveformRenderer
 *   4. WebGL2 构造失败 → 抛错（canvas 已被 webgl2 锁定，无法再回退到 2d）
 *   5. 探测失败或强制 Canvas 2D → 在传入 canvas 上获取 2d context → 创建 Canvas2DWaveformRenderer
 *   6. 连 Canvas 2D 都拿不到 → 抛错（极端情况）
 */

import { Canvas2DWaveformRenderer, type WaveformRenderer } from "./waveformRenderer";
import { WebGL2WaveformRenderer } from "./waveformWebGL2Renderer";

/** localStorage 强制走 Canvas 2D 的开关 key */
const FORCE_CANVAS2D_KEY = "hifishifter.forceCanvas2DWaveform";

/**
 * 读取强制 Canvas 2D 开关
 *
 * 特殊说明：用于 WebGL2 实现 bug 的应急回退、性能对比测试、兼容性问题排查
 */
function shouldForceCanvas2D(): boolean {
    if (typeof window === "undefined") return false;
    try {
        return window.localStorage?.getItem(FORCE_CANVAS2D_KEY) === "1";
    } catch {
        return false;
    }
}

/**
 * 根据浏览器能力创建波形渲染器
 *
 * 流程：
 *   1. 若 enableWebGL2 且未强制 Canvas 2D：用临时 canvas 探测 WebGL2 能力
 *   2. 探测成功 → 在传入 canvas 上获取 webgl2 context → 创建 WebGL2WaveformRenderer
 *   3. 探测失败或强制 Canvas 2D → 在传入 canvas 上获取 2d context → 创建 Canvas2DWaveformRenderer
 *   4. WebGL2 构造失败 → 抛错（canvas 已被 webgl2 锁定，无法回退）
 *
 * 特殊说明：
 *   - 先用临时 canvas 探测，避免在传入 canvas 上先获取 webgl2 再获取 2d 导致 context 锁定
 *   - WebGL2 构造失败时抛错而非静默回退，因为 canvas 已被污染
 *   - 调用方可通过 localStorage.hifishifter.forceCanvas2DWaveform=1 强制走 Canvas 2D
 *
 * @param canvas 目标 canvas 元素（必须未被分配过其他 context）
 * @param enableWebGL2 是否允许走 WebGL2 路径（默认 true）
 * @returns WaveformRenderer 实例
 * @throws {Error} WebGL2 构造失败且 canvas 已锁定时抛错；或 Canvas 2D 不可用时抛错
 */
export function createWaveformRenderer(
    canvas: HTMLCanvasElement,
    enableWebGL2: boolean = true,
): WaveformRenderer {
    const forceCanvas2D = shouldForceCanvas2D();
    const wantWebGL2 = enableWebGL2 && !forceCanvas2D;

    // 先用临时 canvas 探测 WebGL2 能力，避免污染传入的 canvas
    // 一旦在 canvas 上调用 getContext("webgl2") 成功，同一 canvas 再调用 getContext("2d") 会返回 null
    let webgl2Available = false;
    if (wantWebGL2) {
        const probe = document.createElement("canvas");
        const probeGl = probe.getContext("webgl2", {
            alpha: true,
            premultipliedAlpha: true,
            antialias: false,
            depth: false,
            stencil: false,
            preserveDrawingBuffer: false,
        });
        if (probeGl) {
            webgl2Available = true;
            // 释放探测 canvas 的 context
            const ext = probeGl.getExtension("WEBGL_lose_context");
            ext?.loseContext();
        }
    }

    if (webgl2Available) {
        const gl = canvas.getContext("webgl2", {
            alpha: true,
            premultipliedAlpha: true,
            antialias: false,
            depth: false,
            stencil: false,
            preserveDrawingBuffer: false,
        });

        if (gl) {
            try {
                return new WebGL2WaveformRenderer(canvas, gl);
            } catch (e) {
                console.warn(
                    "[WaveformRenderer] WebGL2 init failed, fallback to Canvas2D:",
                    e,
                );
                // 注意：此时 canvas 已被 webgl2 context 锁定，无法再获取 2d context
                // 需要创建新 canvas 给 Canvas 2D fallback 用
                // 但工厂不应修改 DOM，所以抛错让消费方处理
                throw new Error(
                    "WaveformRenderer: WebGL2 init failed and canvas is context-locked. " +
                    "Caller should create a new canvas and retry with enableWebGL2=false, " +
                    "or set localStorage.hifishifter.forceCanvas2DWaveform=1.",
                );
            }
        }
    }

    const ctx = canvas.getContext("2d");
    if (!ctx) {
        throw new Error("WaveformRenderer: neither WebGL2 nor Canvas2D available");
    }
    return new Canvas2DWaveformRenderer(canvas, ctx);
}
