/**
 * 时间轴渲染内核 · sdf-box program（实例化圆角盒 / 平面矩形）
 *
 * 【主要内容】
 * 封装「单位四边形 + 实例化」的 WebGL2 program：顶点着色器按实例矩形展开四边形并
 * 减去视口原点（`u_viewOrigin`）；片元着色器用圆角盒 SDF 画分区色块 / 描边 / 分隔缝，
 * 平面矩形模式（`i_mode > 0.5`）直接输出颜色。
 *
 * 【作用】
 * 网格线、轨道分界线、clip 块面共用同一个 program 与同一份实例缓冲——网格与 clip
 * 的层叠顺序由「实例在缓冲中的先后」表达，因此一次 `drawArraysInstanced` 就能完成
 * 整个底层（< 10 draw call 目标的主要贡献者）。
 *
 * 【与其他模块的关系】
 * - 上游：`scene/clipInstances` 产出 25 float/实例的 clip 实例；网格实例（`FlatInstance`）
 *   由调用方按同一布局写入。
 * - 依赖：`glContext` 提供 gl 对象与光栅化目标；`instanceBuffer` 提供容量策略。
 * - 下游：`renderLoop` 在 rAF 内调用 `render` / `repaint`。
 *
 * 【设计约束（与既有 runtime/timelineClipGlRenderer 的一致性）】
 * 1. 着色器源码与实例布局（25 float/实例，偏移见 OFF_* 常量）**刻意与既有 GL 渲染器
 *    保持逐字节一致**：Spike 阶段复制而非共享，避免改动生产代码；后续阶段 1 合并。
 *    若既有布局变更，本文件与 `scene/clipInstances.test.ts` 的布局断言会同时失败，
 *    提示同步（比静默漂移安全）。
 * 2. 滚动帧只调用 `repaint()`：实例缓冲**不重新上传**，只更新 `u_viewOrigin`
 *    uniform——这是"滚动零重绘"在 GL 层的落点。
 * 3. `u_resolution` 用光栅化目标回算的绘制坐标系尺寸（见 glRaster），保证坐标与
 *    物理像素 1:1。
 */

import { CLIP_INSTANCE_FLOATS } from "../../runtime/timelineClipGlRenderer";
import type { GlRasterTarget } from "./glRaster";
import { resolveBufferFloats } from "./instanceBuffer";

// ── 实例布局偏移（float 下标；与 runtime/timelineClipGlRenderer 的 OFF_* 一致）──────

const OFF_RECT = 0; // 4 floats: x, y, w, h
const OFF_RADIUS = 4;
const OFF_HEADER_H = 5;
const OFF_BODY_RGBA = 6; // 4 floats
const OFF_HEADER_RGBA = 10; // 4 floats
const OFF_BORDER_RGBA = 14; // 4 floats
const OFF_BORDER_WIDTH = 18;
const OFF_OVERLAP_PX = 19;
const OFF_SEAM = 20;
const OFF_SEAM_RGB = 21; // 3 floats
const OFF_MODE = 24;

/** 单实例字节步长。 */
const INSTANCE_STRIDE_BYTES = CLIP_INSTANCE_FLOATS * 4;

/** 平面矩形模式（与 `buildGuideInstance` / `buildGridInstances` 的写入端一致）。 */
export const INSTANCE_MODE_FLAT = 1;

// ── 着色器（与 runtime/timelineClipGlRenderer 逐字一致，见文件头约束 1）────────────

const VERTEX_SHADER = `#version 300 es
in vec2 a_unit;
in float i_rect[4];
in float i_radius;
in float i_headerH;
in vec4 i_bodyColor;
in vec4 i_headerColor;
in vec4 i_borderColor;
in float i_borderWidth;
in float i_overlapPx;
in float i_seamW;
in vec3 i_seamColor;
in float i_mode;

uniform vec2 u_resolution;
uniform vec2 u_viewOrigin;

out vec2 v_local;
out vec2 v_half;
out float v_radius;
out float v_headerH;
out vec4 v_bodyColor;
out vec4 v_headerColor;
out vec4 v_borderColor;
out float v_borderWidth;
out float v_overlapPx;
out float v_seamW;
out vec3 v_seamColor;
out float v_mode;

void main() {
    float pad = i_mode > 0.5 ? 0.0 : max(i_borderWidth * 0.5, 1.0);
    vec2 center = vec2(i_rect[0] + i_rect[2] * 0.5, i_rect[1] + i_rect[3] * 0.5);
    vec2 halfSize = vec2(i_rect[2] * 0.5 + pad, i_rect[3] * 0.5 + pad);
    vec2 pos = center + (a_unit - 0.5) * 2.0 * halfSize;

    vec2 screen = pos - u_viewOrigin;
    vec2 zeroToOne = screen / u_resolution;
    gl_Position = vec4(zeroToOne.x * 2.0 - 1.0, -(zeroToOne.y * 2.0 - 1.0), 0.0, 1.0);

    v_local = pos - vec2(i_rect[0], i_rect[1]);
    v_half = vec2(i_rect[2] * 0.5, i_rect[3] * 0.5);
    v_radius = i_radius;
    v_headerH = i_headerH;
    v_bodyColor = i_bodyColor;
    v_headerColor = i_headerColor;
    v_borderColor = i_borderColor;
    v_borderWidth = i_borderWidth;
    v_overlapPx = i_overlapPx;
    v_seamW = i_seamW;
    v_seamColor = i_seamColor;
    v_mode = i_mode;
}`;

const FRAGMENT_SHADER = `#version 300 es
precision highp float;

in vec2 v_local;
in vec2 v_half;
in float v_radius;
in float v_headerH;
in vec4 v_bodyColor;
in vec4 v_headerColor;
in vec4 v_borderColor;
in float v_borderWidth;
in float v_overlapPx;
in float v_seamW;
in vec3 v_seamColor;
in float v_mode;

out vec4 outColor;

float roundedBoxSdf(vec2 p, vec2 b, float r) {
    vec2 q = abs(p) - b + vec2(r);
    return min(max(q.x, q.y), 0.0) + length(max(q, 0.0)) - r;
}

void main() {
    if (v_mode > 0.5) {
        outColor = v_borderColor;
        return;
    }

    vec2 p = v_local - v_half;
    float sdf = roundedBoxSdf(p, v_half, v_radius);

    float halfBorder = v_borderWidth * 0.5;
    float inBorder = 1.0 - smoothstep(halfBorder - 0.5, halfBorder + 0.5, abs(sdf + halfBorder));
    if (sdf > halfBorder) discard;

    vec4 base = v_local.y < v_headerH ? v_headerColor : v_bodyColor;

    if (v_overlapPx > 0.0 && v_local.x < v_overlapPx) {
        base.a *= 0.55;
    }

    float sep = 1.0 - smoothstep(0.0, 1.0, abs(v_local.y - v_headerH) - 0.5);
    base.rgb = mix(base.rgb, vec3(0.0), sep * 0.14 * step(v_local.y, v_headerH + 1.0));

    if (v_seamW > 0.0) {
        float seamRight = v_half.x * 2.0 - 0.5;
        float seam = step(seamRight - v_seamW, v_local.x) * step(v_local.x, seamRight);
        base.rgb = mix(base.rgb, v_seamColor, seam);
    }

    outColor = vec4(mix(base.rgb, v_borderColor.rgb, inBorder * v_borderColor.a),
                    max(base.a, inBorder * v_borderColor.a));
}`;

/** 单位四边形（两个三角形，6 顶点）。 */
const UNIT_QUAD = new Float32Array([0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1]);

/** 编译一个着色器；失败时抛错（调用方捕获后回退）。 */
function compile(gl: WebGL2RenderingContext, type: number, source: string): WebGLShader {
    const shader = gl.createShader(type);
    if (!shader) throw new Error("Unable to create sdf-box shader");
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        const message = gl.getShaderInfoLog(shader) ?? "Unknown sdf-box shader error";
        gl.deleteShader(shader);
        throw new Error(message);
    }
    return shader;
}

/** sdf-box program 句柄。 */
export interface SdfBoxProgram {
    /**
     * 上传实例并绘制。
     *
     * @param instances 实例缓冲（有效前缀长度 = `count × CLIP_INSTANCE_FLOATS`）。
     * @param count 实例数。
     * @param target 光栅化目标（提供 `u_resolution`）。
     * @param viewOriginX 视口原点 x（内容坐标 CSS px）。
     * @param viewOriginY 视口原点 y（内容坐标 CSS px）。
     */
    render(
        instances: Float32Array,
        count: number,
        target: GlRasterTarget,
        viewOriginX: number,
        viewOriginY: number,
    ): void;
    /**
     * 复用已上传的实例缓冲，仅更新视口原点后重绘（滚动帧零上传）。
     *
     * @param target 光栅化目标。
     * @param viewOriginX 视口原点 x。
     * @param viewOriginY 视口原点 y。
     */
    repaint(target: GlRasterTarget, viewOriginX: number, viewOriginY: number): void;
    /** 释放 program / VAO / 缓冲。 */
    dispose(): void;
}

/**
 * 创建 sdf-box program。
 *
 * 流程：编译链接 → 取 uniform / 属性 location → 建立 VAO（单位四边形 + 实例属性指针）
 * → 返回 render / repaint / dispose。
 *
 * 特殊说明：属性指针在创建时一次性绑定（VAO 记录状态），绘制时只需
 * `bindVertexArray` + `bufferSubData`（实例数据）+ `drawArraysInstanced`。
 *
 * @param gl 内核 WebGL2 上下文。
 * @returns program 句柄；创建失败时抛错（调用方捕获后回退 Canvas2D）。
 */
export function createSdfBoxProgram(gl: WebGL2RenderingContext): SdfBoxProgram {
    const vertex = compile(gl, gl.VERTEX_SHADER, VERTEX_SHADER);
    const fragment = compile(gl, gl.FRAGMENT_SHADER, FRAGMENT_SHADER);
    const program = gl.createProgram();
    if (!program) throw new Error("Unable to create sdf-box program");
    gl.attachShader(program, vertex);
    gl.attachShader(program, fragment);
    gl.linkProgram(program);
    gl.deleteShader(vertex);
    gl.deleteShader(fragment);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        const message = gl.getProgramInfoLog(program) ?? "Unknown sdf-box link error";
        gl.deleteProgram(program);
        throw new Error(message);
    }

    const resolutionLocation = gl.getUniformLocation(program, "u_resolution");
    const originLocation = gl.getUniformLocation(program, "u_viewOrigin");
    if (!resolutionLocation || !originLocation) {
        gl.deleteProgram(program);
        throw new Error("sdf-box uniforms are missing");
    }

    const vao = gl.createVertexArray();
    const unitBuffer = gl.createBuffer();
    const instanceBuffer = gl.createBuffer();
    if (!vao || !unitBuffer || !instanceBuffer) {
        gl.deleteProgram(program);
        throw new Error("Unable to create sdf-box buffers");
    }

    let instanceCapacityFloats = 0;
    let uploadedCount = 0;

    /** 绑定单个 float 属性到实例缓冲的指定偏移。 */
    function bindFloatAttrib(name: string, offsetFloats: number): void {
        const location = gl.getAttribLocation(program, name);
        if (location < 0) return;
        gl.enableVertexAttribArray(location);
        gl.vertexAttribPointer(
            location,
            1,
            gl.FLOAT,
            false,
            INSTANCE_STRIDE_BYTES,
            offsetFloats * 4,
        );
        gl.vertexAttribDivisor(location, 1);
    }

    /** 绑定多分量属性（size 个连续 location）。 */
    function bindVectorAttrib(name: string, size: number, offsetFloats: number): void {
        const location = gl.getAttribLocation(program, name);
        if (location < 0) return;
        gl.enableVertexAttribArray(location);
        gl.vertexAttribPointer(
            location,
            size,
            gl.FLOAT,
            false,
            INSTANCE_STRIDE_BYTES,
            offsetFloats * 4,
        );
        gl.vertexAttribDivisor(location, 1);
    }

    gl.bindVertexArray(vao);

    gl.bindBuffer(gl.ARRAY_BUFFER, unitBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, UNIT_QUAD, gl.STATIC_DRAW);
    const unitLocation = gl.getAttribLocation(program, "a_unit");
    if (unitLocation >= 0) {
        gl.enableVertexAttribArray(unitLocation);
        gl.vertexAttribPointer(unitLocation, 2, gl.FLOAT, false, 0, 0);
        // 单位四边形是逐顶点属性（divisor 0），其余都是逐实例。
        gl.vertexAttribDivisor(unitLocation, 0);
    }

    gl.bindBuffer(gl.ARRAY_BUFFER, instanceBuffer);
    bindVectorAttrib("i_rect", 4, OFF_RECT);
    bindFloatAttrib("i_radius", OFF_RADIUS);
    bindFloatAttrib("i_headerH", OFF_HEADER_H);
    bindVectorAttrib("i_bodyColor", 4, OFF_BODY_RGBA);
    bindVectorAttrib("i_headerColor", 4, OFF_HEADER_RGBA);
    bindVectorAttrib("i_borderColor", 4, OFF_BORDER_RGBA);
    bindFloatAttrib("i_borderWidth", OFF_BORDER_WIDTH);
    bindFloatAttrib("i_overlapPx", OFF_OVERLAP_PX);
    bindFloatAttrib("i_seamW", OFF_SEAM);
    bindVectorAttrib("i_seamColor", 3, OFF_SEAM_RGB);
    bindFloatAttrib("i_mode", OFF_MODE);

    gl.bindVertexArray(null);

    /** 设置绘制状态（program / VAO / uniform）。 */
    function prepareDraw(target: GlRasterTarget, viewOriginX: number, viewOriginY: number): void {
        gl.useProgram(program);
        gl.bindVertexArray(vao);
        gl.uniform2f(resolutionLocation, target.cssWidthPx, target.cssHeightPx);
        gl.uniform2f(originLocation, viewOriginX, viewOriginY);
    }

    return {
        render(instances, count, target, viewOriginX, viewOriginY) {
            if (count <= 0) {
                uploadedCount = 0;
                return;
            }
            const needed = count * CLIP_INSTANCE_FLOATS;
            const capacity = resolveBufferFloats(instanceCapacityFloats, needed);
            gl.bindBuffer(gl.ARRAY_BUFFER, instanceBuffer);
            if (capacity !== instanceCapacityFloats) {
                // 扩容：orphan + 整块分配（避免缓冲在驱动侧反复重定位）。
                gl.bufferData(gl.ARRAY_BUFFER, capacity * 4, gl.DYNAMIC_DRAW);
                instanceCapacityFloats = capacity;
            }
            gl.bufferSubData(gl.ARRAY_BUFFER, 0, instances, 0, needed);
            uploadedCount = count;
            prepareDraw(target, viewOriginX, viewOriginY);
            gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, count);
        },

        repaint(target, viewOriginX, viewOriginY) {
            if (uploadedCount <= 0) return;
            // 纯平移帧：不碰实例缓冲，只更新 uniform 后重发 draw call。
            prepareDraw(target, viewOriginX, viewOriginY);
            gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, uploadedCount);
        },

        dispose() {
            gl.deleteBuffer(unitBuffer);
            gl.deleteBuffer(instanceBuffer);
            gl.deleteVertexArray(vao);
            gl.deleteProgram(program);
            instanceCapacityFloats = 0;
            uploadedCount = 0;
        },
    };
}
