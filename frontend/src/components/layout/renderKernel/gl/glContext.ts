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

import { writeCanvasCssSize } from "../canvasRaster";
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
    /**
     * 上下文是否**当前处于丢失状态**。
     *
     * 丢失期间所有 GL 调用都是空操作，画布会保持空白；调用方据此决定"这一帧没有
     * 画成"，而不是把空白当成绘制结果。
     */
    isLost(): boolean;
    /**
     * 注册"上下文已恢复"的回调；返回注销函数。
     *
     * 恢复后 GL 资源（program / buffer / texture）全部失效，重绘是**唯一**能让内容
     * 重新出现的动作 —— 而浏览器不会替我们重新调度一帧。
     */
    onRestored(listener: () => void): () => void;
    /**
     * 释放句柄。
     *
     * 特殊说明：**不**主动丢失上下文（原因见实现处注释）——一个 canvas 只能有
     * 一个 WebGL context，丢失后无法重建，会破坏 StrictMode 双挂载 / HMR 的
     * "销毁后立即重新挂载"模式。
     */
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

    // 【为什么必须接这两个事件】标签页切换/元素尺寸归零会让浏览器回收 WebGL2 上下文
    // （显存压力下尤其如此）。恢复后全部 GL 资源失效，而**没有任何东西**会替我们
    // 重新调度一帧 —— 表现就是"画布一直空白，直到用户滚动或缩放才恢复"。
    // `preventDefault` 是恢复的前提：不阻止默认处理，浏览器不会触发 restored。
    let lost = false;
    const restoredListeners: Array<() => void> = [];
    canvas.addEventListener("webglcontextlost", (event) => {
        event.preventDefault();
        lost = true;
    });
    canvas.addEventListener("webglcontextrestored", () => {
        lost = false;
        for (const listener of restoredListeners.slice()) listener();
    });

    return {
        gl,

        isLost() {
            return lost || gl.isContextLost();
        },

        onRestored(listener) {
            restoredListeners.push(listener);
            return () => {
                const index = restoredListeners.indexOf(listener);
                if (index >= 0) restoredListeners.splice(index, 1);
            };
        },

        resize(cssWidthPx, cssHeightPx, dpr) {
            target = resolveGlRasterTarget(cssWidthPx, cssHeightPx, dpr);
            // 仅在物理尺寸真正变化时写 canvas 尺寸：写 canvas.width 会重置
            // 整个绘制状态（缓冲被清空），无谓写入会丢掉同帧已上传的实例数据。
            if (canvas.width !== target.physicalWidthPx) canvas.width = target.physicalWidthPx;
            if (canvas.height !== target.physicalHeightPx) canvas.height = target.physicalHeightPx;
            // 【必须用回算值（物理 / dpr），不能用入参的原始 CSS 尺寸】
            // 物理尺寸取整后，若把未取整的 CSS 尺寸写回样式，浏览器会把
            // `physical` 个物理像素铺到 `css * dpr` 个物理像素的布局盒上，
            // 合成器做一次非整数倍重采样（缩放比 = css*dpr/physical ≠ 1），
            // 整块画面发虚，且随窗口宽度的奇偶变化。回写 `physical / dpr`
            // 后 style × dpr 严格等于 physical，缩放比恒为 1。
            // 与 Canvas2D 路径共用同一写回策略（`writeCanvasCssSize`），
            // 两条路径因此不会出现半像素级的层间错位。
            writeCanvasCssSize(canvas, target.cssWidthPx, target.cssHeightPx);
            gl.viewport(0, 0, target.physicalWidthPx, target.physicalHeightPx);
            return target;
        },

        clear() {
            gl.clearColor(0, 0, 0, 0);
            gl.clear(gl.COLOR_BUFFER_BIT);
        },

        dispose() {
            // 刻意**不**调用 `WEBGL_lose_context.loseContext()`。
            //
            // 规范约束：一个 canvas 只能有一个 WebGL context；`loseContext()` 之后对
            // 同一 canvas 再 `getContext("webgl2")` 会返回**同一个已丢失的 context**
            // （而不是新建）。React StrictMode 的双挂载与 HMR 重新挂载恰好是
            // "dispose 后立即重建"的模式，loseContext 会让第二次挂载拿到已丢失的
            // context，表现为 `createShader` 返回 null（"Unable to create shader"）。
            //
            // 真正的 GPU 资源（program / buffer / texture）由各 program 的 dispose
            // 释放；context 本身随 canvas 被 GC 回收，无需手动丢失。
        },
    };
}
