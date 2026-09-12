/**
 * 渲染内核 · polyline program（折线描边）
 *
 * 【主要内容】
 * 封装「顶点缓冲 + 距离场」的 WebGL2 program：顶点着色器按视口原点投影，片元着色器
 * 用顶点携带的**到中心线距离**做亚像素抗锯齿、用**累积弧长**做虚线相位，并支持
 * 矩形裁剪（`gl.scissor`）与预乘 alpha 混合。
 *
 * 【作用】
 * 参数编辑器的曲线图层（检测曲线、原始 / 编辑包络、副参数、参考线、选区高亮、
 * 剪贴板预览）原先每帧对每个采样点调用一次 `lineTo`——实测单条曲线每帧 1648 次
 * JS→原生调用。本 program 把它们变成"一次顶点上传 + 一次 draw call"。
 *
 * 【与其他模块的关系】
 * - 上游：`gl/polylineGeometry` 产出顶点缓冲（每顶点 `[x, y, along, across]`）。
 * - 依赖：`glContext` 提供 gl 与光栅化目标；`instanceBuffer` 提供容量策略。
 *
 * 【设计约束】
 * 1. **`u_halfWidth` 必须由调用方显式传入**：顶点里 `across` 的范围是
 *    `±(线宽/2 + AA 余量)`，比几何边缘更宽，因此覆盖率阈值**不能**从
 *    `max|across|` 反推（见 `polylineGeometry` 文件头）。
 * 2. **抗锯齿宽度取 1 个设备像素**（`1/dpr` CSS px），与 Canvas2D 的边缘过渡
 *    尺度一致；这正是"非整数线宽（1.8 / 2.6 / 3.2 / 3.6）也要看起来一样粗"的前提。
 * 3. **虚线相位按弧长推进**，且相位从传入序列的**第一个点**起算。Canvas2D 的
 *    虚线相位从子路径起点开始，而曲线的子路径起点是**首个可见采样点**——
 *    因此调用方必须传可见点序列（见 `polylineGeometry` 特殊说明 2）。
 * 4. 预乘 alpha 输出 + `blendFunc(ONE, ONE_MINUS_SRC_ALPHA)`，与其余 program 一致；
 *    混合状态在每次 draw 前设置（各 program 自负其责，避免跨 program 状态泄漏）。
 * 5. 裁剪用 `gl.scissor`：两个需要裁剪的曲线图层（选区高亮、剪贴板预览）的裁剪区
 *    都是**轴对齐矩形**，正好落在 scissor 的能力范围内，无需模板缓冲。
 * 6. **覆盖率公式与 `gl/polylineCoverage.ts` 是两份等价实现**：GLSL 无法 import TS，
 *    而着色器数学在 node 环境（无 WebGL）无法单测。因此把公式提取为可测的 TS 参考
 *    实现，本文件的 `FRAGMENT_SHADER` 保留逐行等价副本。**改一处必须改另一处**。
 */

import type { Rgba } from "../instanceTypes";
import { resolveBufferFloats } from "./instanceBuffer";
import type { GlRasterTarget } from "./glRaster";

/** 每个顶点的 float 数（与 `polylineGeometry` 的布局一致）。 */
export const POLYLINE_VERTEX_FLOATS = 4;

const VERTEX_SHADER = `#version 300 es
in vec2 a_pos;           // 内容坐标（CSS px）
in vec2 a_meta;          // x = along（累积弧长）, y = across（到中心线的有符号距离）

uniform vec2 u_resolution;
uniform vec2 u_viewOrigin;

out float v_along;
out float v_across;

void main() {
    vec2 screen = a_pos - u_viewOrigin;
    vec2 zeroToOne = screen / u_resolution;
    gl_Position = vec4(zeroToOne.x * 2.0 - 1.0, -(zeroToOne.y * 2.0 - 1.0), 0.0, 1.0);
    v_along = a_meta.x;
    v_across = a_meta.y;
}`;

const FRAGMENT_SHADER = `#version 300 es
precision highp float;

in float v_along;
in float v_across;

uniform vec4 u_color;
/** 覆盖率阈值：线宽的一半（CSS px）。必须独立传入，不能从几何反推（见文件头约束 1）。 */
uniform float u_halfWidth;
/** 抗锯齿过渡宽度（CSS px），取 1 个设备像素（见文件头约束 2）。 */
uniform float u_aaWidth;
/** 虚线图案 [dash, gap]（CSS px）；x < 0 表示实线。 */
uniform vec2 u_dash;

out vec4 outColor;

void main() {
    // 非法 aa 必须显式回退：GLSL 的 max(NaN, x) 同样是 NaN（与 TS 参考实现的
    // normalizeAaWidth 保持同一语义，见 polylineCoverage.ts 的同步义务）。
    float aa = (u_aaWidth > 0.0) ? u_aaWidth : 1e-6;

    // ── 横向覆盖率：由到中心线的距离得到 ──
    // **必须用线性斜坡，不能用 smoothstep**：Canvas2D 的抗锯齿是**面积覆盖率**
    // （边缘落在像素内的位置决定覆盖比例，是线性的）。smoothstep 是三次 Hermite
    // 过渡，在同一位置给出 0.844 而面积覆盖率是 0.750——边缘对比更强，
    // 视觉上比 Canvas2D 更"硬"，也让非整数线宽（1.8 / 2.6 …）的粗细对不上。
    // 实测差异 0.0725（见 Phase 3 验证记录）。
    float edge = abs(v_across);
    float inner = u_halfWidth - aa * 0.5;
    float outer = u_halfWidth + aa * 0.5;
    float lateral = 1.0 - clamp((edge - inner) / max(outer - inner, 1e-6), 0.0, 1.0);

    // ── 虚线覆盖率：由累积弧长得到 ──
    float alongCoverage = 1.0;
    if (u_dash.x >= 0.0) {
        float rawPeriod = u_dash.x + u_dash.y;
        float period = (rawPeriod > 0.0) ? rawPeriod : 1e-6;
        float phase = mod(v_along, period);
        // 有符号距离：正数表示在"墨"内、负数表示在"空隙"内，0 为边界。
        // 两侧边界都做 aa 宽的过渡，因此虚线端点也有抗锯齿（与 Canvas2D 一致）。
        float dist;
        if (phase < u_dash.x) {
            dist = min(phase, u_dash.x - phase);
        } else {
            dist = -min(phase - u_dash.x, period - phase);
        }
        alongCoverage = clamp(dist / aa + 0.5, 0.0, 1.0);
    }

    float coverage = lateral * alongCoverage * u_color.a;
    if (coverage <= 0.0) discard;
    // 预乘 alpha 输出（与 sdf-box / glyph program 的混合方式一致）。
    outColor = vec4(u_color.rgb * coverage, coverage);
}`;

/** 矩形裁剪区（视口坐标 CSS px）。 */
export interface PolylineClipRect {
    readonly x: number;
    readonly y: number;
    readonly w: number;
    readonly h: number;
}

/** 一次绘制请求。 */
export interface PolylineDrawArgs {
    /** 顶点缓冲（每 4 个 float 一个顶点）。 */
    readonly vertices: Float32Array;
    /** 顶点数（用于只画缓冲的有效前缀）。 */
    readonly vertexCount: number;
    /** 光栅化目标（提供 `u_resolution`）。 */
    readonly target: GlRasterTarget;
    /** 视口原点（内容坐标 CSS px）。 */
    readonly viewOriginX: number;
    readonly viewOriginY: number;
    /** 线宽的一半（CSS px）。 */
    readonly halfWidthPx: number;
    /** 颜色。 */
    readonly color: Rgba;
    /** 抗锯齿过渡宽度（CSS px），通常传 `1 / dpr`。 */
    readonly aaWidthPx: number;
    /**
     * 虚线图案 [dash, gap]（CSS px）；缺省表示实线。
     *
     * 特殊说明：调用方必须用与 Canvas2D 路径**同一个** `getFixedDashPattern`
     * 取值（它按 dpr 量化），否则两种模式的虚线疏密会不同。
     */
    readonly dash?: readonly [number, number] | null;
    /** 矩形裁剪区（视口坐标）；缺省不裁剪。 */
    readonly clipRect?: PolylineClipRect | null;
}

/** polyline program 句柄。 */
export interface PolylineProgram {
    /** 上传顶点并绘制一次。 */
    draw(args: PolylineDrawArgs): void;
    /** 释放 program / VAO / 缓冲。 */
    dispose(): void;
}

function compile(gl: WebGL2RenderingContext, type: number, source: string): WebGLShader {
    const shader = gl.createShader(type);
    if (!shader) {
        throw new Error(
            gl.isContextLost()
                ? "WebGL 上下文已丢失（GPU 进程异常或资源耗尽），请刷新页面"
                : "无法创建着色器对象（GPU 资源不足）",
        );
    }
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        const message = gl.getShaderInfoLog(shader) ?? "Unknown polyline shader error";
        gl.deleteShader(shader);
        throw new Error(message);
    }
    return shader;
}

/**
 * 创建 polyline program。
 *
 * @param gl 内核 WebGL2 上下文。
 * @returns program 句柄；创建失败时抛错（调用方捕获后回退到 Canvas2D）。
 */
export function createPolylineProgram(gl: WebGL2RenderingContext): PolylineProgram {
    const vertexShader = compile(gl, gl.VERTEX_SHADER, VERTEX_SHADER);
    const fragmentShader = compile(gl, gl.FRAGMENT_SHADER, FRAGMENT_SHADER);
    const program = gl.createProgram();
    if (!program) throw new Error("无法创建 polyline program");
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        const message = gl.getProgramInfoLog(program) ?? "Unknown polyline link error";
        gl.deleteProgram(program);
        throw new Error(message);
    }
    gl.deleteShader(vertexShader);
    gl.deleteShader(fragmentShader);

    const resolutionLocation = gl.getUniformLocation(program, "u_resolution");
    const originLocation = gl.getUniformLocation(program, "u_viewOrigin");
    const colorLocation = gl.getUniformLocation(program, "u_color");
    const halfWidthLocation = gl.getUniformLocation(program, "u_halfWidth");
    const aaWidthLocation = gl.getUniformLocation(program, "u_aaWidth");
    const dashLocation = gl.getUniformLocation(program, "u_dash");
    const posLocation = gl.getAttribLocation(program, "a_pos");
    const metaLocation = gl.getAttribLocation(program, "a_meta");
    if (
        resolutionLocation === null ||
        originLocation === null ||
        colorLocation === null ||
        halfWidthLocation === null ||
        aaWidthLocation === null ||
        dashLocation === null
    ) {
        throw new Error("polyline uniforms are missing");
    }
    if (posLocation < 0 || metaLocation < 0) {
        throw new Error("polyline attribute is missing");
    }

    const vao = gl.createVertexArray();
    const buffer = gl.createBuffer();
    if (!vao || !buffer) throw new Error("无法创建 polyline 缓冲");

    // 容量以 **float 数** 记账，交给 `resolveBufferFloats` 统一增长策略
    // （与 glyph/sdf-box program 同一口径；字节数在写缓冲时再乘 4）。
    let capacityFloats = 0;
    const strideBytes = POLYLINE_VERTEX_FLOATS * 4;

    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.enableVertexAttribArray(posLocation);
    gl.vertexAttribPointer(posLocation, 2, gl.FLOAT, false, strideBytes, 0);
    gl.enableVertexAttribArray(metaLocation);
    gl.vertexAttribPointer(metaLocation, 2, gl.FLOAT, false, strideBytes, 8);
    gl.bindVertexArray(null);

    return {
        draw(args) {
            const count = Math.min(
                args.vertexCount,
                Math.floor(args.vertices.length / POLYLINE_VERTEX_FLOATS),
            );
            if (count <= 0) return;
            const neededFloats = count * POLYLINE_VERTEX_FLOATS;
            const capacity = resolveBufferFloats(capacityFloats, neededFloats);

            gl.useProgram(program);
            gl.bindVertexArray(vao);
            gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
            if (capacity !== capacityFloats) {
                // 只在需要更大容量时重新分配；日常帧沿用同一块缓冲（避免每帧分配）。
                gl.bufferData(gl.ARRAY_BUFFER, capacity * 4, gl.DYNAMIC_DRAW);
                capacityFloats = capacity;
            }
            gl.bufferSubData(gl.ARRAY_BUFFER, 0, args.vertices, 0, count * POLYLINE_VERTEX_FLOATS);

            gl.uniform2f(resolutionLocation, args.target.cssWidthPx, args.target.cssHeightPx);
            gl.uniform2f(originLocation, args.viewOriginX, args.viewOriginY);
            gl.uniform4f(colorLocation, args.color[0], args.color[1], args.color[2], args.color[3]);
            gl.uniform1f(halfWidthLocation, args.halfWidthPx);
            gl.uniform1f(aaWidthLocation, args.aaWidthPx);
            const dash = args.dash ?? null;
            gl.uniform2f(dashLocation, dash ? dash[0] : -1, dash ? dash[1] : 0);

            gl.enable(gl.BLEND);
            gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

            // 裁剪：轴对齐矩形直接用 scissor（见文件头约束 5）。
            const clip = args.clipRect ?? null;
            if (clip) {
                const dpr = args.target.physicalWidthPx / Math.max(1, args.target.cssWidthPx);
                // scissor 的 y 轴原点在左下角，需翻转。
                const sx = Math.round(clip.x * dpr);
                const sy = Math.round((args.target.cssHeightPx - (clip.y + clip.h)) * dpr);
                const sw = Math.max(0, Math.round(clip.w * dpr));
                const sh = Math.max(0, Math.round(clip.h * dpr));
                gl.enable(gl.SCISSOR_TEST);
                gl.scissor(sx, sy, sw, sh);
            }

            gl.drawArrays(gl.TRIANGLES, 0, count);

            if (clip) {
                gl.disable(gl.SCISSOR_TEST);
            }
            gl.bindVertexArray(null);
        },

        dispose() {
            gl.deleteBuffer(buffer);
            gl.deleteVertexArray(vao);
            gl.deleteProgram(program);
        },
    };
}
