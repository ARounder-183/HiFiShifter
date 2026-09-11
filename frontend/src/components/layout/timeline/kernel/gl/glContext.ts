/**
 * 时间轴渲染内核 · WebGL2 上下文封装
 *
 * 【主要内容】
 * 创建并持有内核唯一的 WebGL2 上下文：按 CSS 尺寸 + dpr 设置画布物理尺寸与
 * `gl.viewport`，提供透明清屏与资源释放。
 *
 * 【作用】
 * 单 WebGL2 渲染器的全部绘制共享一个上下文——避免多上下文（每个都有独立状态机与
 * 显存开销，浏览器还有上下文数量上限），也让「一次提交按层序 draw call」成为可能。
 *
 * 【与其他模块的关系】
 * - 上游：宿主视图在挂载时创建、在 `ResizeObserver` 回调里 `resize`；
 * - 下游：`sdfBoxProgram` / `glyphProgram` 等 program 共用本上下文的 gl 对象。
 * - 依赖：`glRaster.ts` 提供光栅化参数（纯逻辑，已单测）。
 *
 * 【设计约束】
 * 1. `alpha: true` + `premultipliedAlpha: true`：画布叠在 DOM 之上，必须透明；
 *    预乘 alpha 与既有 GL 渲染器一致（片元着色器输出预乘色）。
 * 2. `antialias: false`：内核所有边缘由 SDF / 设备像素吸附保证锐利，MSAA 只会
 *    增加带宽开销并让 1 物理像素的线变虚。
 * 3. `preserveDrawingBuffer: false`：允许浏览器丢弃后台缓冲，降低合成开销；
 *    内核每帧全量重绘，不依赖上一帧内容。
 */

import { resolveGlRasterTarget, type GlRasterTarget } from "./glRaster";

/** 上下文句柄。 */
export interface GlCanvasHandle {
    /** 内核唯一 WebGL2 上下文。 */
    readonly gl: WebGL2RenderingContext;
    /**
     * 按 CSS 尺寸与 dpr 设置画布物理尺寸与 viewport。
     *
     * @param cssWidthPx 宿主 CSS 宽（px）。
     * @param cssHeightPx 宿主 CSS 高（px）。
     * @param dpr 设备像素比。
     * @returns 生效的光栅化目标（含回算后的绘制坐标系尺寸）。
     */
    resize(cssWidthPx: number, cssHeightPx: number, dpr: number): GlRasterTarget;
    /** 以透明清屏（每帧绘制的第一步）。 */
    clear(): void;
    /** 释放上下文（主动 `loseContext`，让浏览器尽快回收显存）。 */
    dispose(): void;
}

/**
 * 创建 WebGL2 上下文句柄。
 *
 * 流程：`getContext("webgl2", ...)` → 失败返回 null（调用方回退 Canvas2D）→
 * 返回 resize / clear / dispose 三个操作。
 *
 * 特殊说明：创建失败**不抛错**而是返回 null——WebGL2 不可用（老驱动 / 黑名单 /
 * 远程桌面）是预期内的环境差异，调用方需要据此走降级路径。
 *
 * @param canvas 目标画布。
 * @returns 上下文句柄；WebGL2 不可用时为 null。
 */
export function createGlCanvas(canvas: HTMLCanvasElement): GlCanvasHandle | null {
    const gl = canvas.getContext("webgl2", {
        alpha: true,
        antialias: false,
        depth: false,
        stencil: false,
        premultipliedAlpha: true,
        preserveDrawingBuffer: false,
        powerPreference: "high-performance",
    });
    if (!gl) return null;

    let target = resolveGlRasterTarget(1, 1, 1);

    return {
        gl,

        resize(cssWidthPx, cssHeightPx, dpr) {
            target = resolveGlRasterTarget(cssWidthPx, cssHeightPx, dpr);
            // 仅在物理尺寸真正变化时写 canvas 尺寸：写 canvas.width 会重置
            // 整个绘制状态（缓冲被清空），无谓写入会丢掉同帧已上传的实例数据。
            if (canvas.width !== target.physicalWidthPx) canvas.width = target.physicalWidthPx;
            if (canvas.height !== target.physicalHeightPx) canvas.height = target.physicalHeightPx;
            gl.viewport(0, 0, target.physicalWidthPx, target.physicalHeightPx);
            return target;
        },

        clear() {
            gl.clearColor(0, 0, 0, 0);
            gl.clear(gl.COLOR_BUFFER_BIT);
        },

        dispose() {
            gl.getExtension("WEBGL_lose_context")?.loseContext();
        },
    };
}
