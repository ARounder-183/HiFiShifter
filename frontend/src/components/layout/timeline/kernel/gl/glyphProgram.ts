/**
 * 时间轴渲染内核 · glyph-quad program（字形纹理四边形）
 *
 * 【主要内容】
 * 封装「单位四边形 + 实例化 + 图集纹理」的 WebGL2 program：顶点着色器按实例矩形
 * 展开并减去视口原点；片元着色器用纹理 alpha 作为覆盖度、乘以实例颜色输出**预乘**
 * 结果（与 sdf-box program 的混合方式一致）。
 *
 * 【作用】
 * 全部文字（clip 名、增益 / 速率、badge 标签、标尺刻度、轨道头文字）由本 program
 * 一次 draw call 绘制。滚动帧只需更新 `u_viewOrigin`——与块面、网格一致，
 * 文字也是"滚动零重绘"的一部分。
 *
 * 【与其他模块的关系】
 * - 上游：`gl/glyphQuads` 产出四边形；`glyph/glyphRasterizer` 提供图集像素数据。
 * - 依赖：`glContext` 提供 gl 与光栅化目标；`instanceBuffer` 提供容量策略。
 *
 * 【设计约束】
 * 1. **单页图集**（Spike 限制）：本 program 持有一张纹理；多页图集（Task 4 的
 *    `maxPages > 1`）暂不支持，超出页容量时调用方应停止新增字形（记录在 Spike 报告）。
 * 2. 纹理格式：光栅化端用白色绘制字形（见 glyphRasterizer），因此片元只需
 *    `tex.a × color` —— 颜色完全由实例决定，同一字形可复用于不同颜色文本。
 * 3. 预乘 alpha 输出 + `blendFunc(ONE, ONE_MINUS_SRC_ALPHA)`，与 sdf-box 一致；
 *    混合状态在每次 draw 前设置（program 各自负责，避免跨 program 状态泄漏）。
 */

import { resolveBufferFloats } from "./instanceBuffer";
import type { GlyphQuad } from "./glyphQuads";
import type { GlRasterTarget } from "./glRaster";

/** 单实例 float 数：rect(4) + uv(4) + color(4)。 */
export const GLYPH_INSTANCE_FLOATS = 12;

const OFF_RECT = 0;
const OFF_UV = 4;
const OFF_COLOR = 8;

const INSTANCE_STRIDE_BYTES = GLYPH_INSTANCE_FLOATS * 4;

const VERTEX_SHADER = `#version 300 es
in vec2 a_unit;
in float i_rect[4];
in float i_uv[4];
in vec4 i_color;

uniform vec2 u_resolution;
uniform vec2 u_viewOrigin;

out vec2 v_uv;
out vec4 v_color;

void main() {
    vec2 pos = vec2(i_rect[0], i_rect[1]) + a_unit * vec2(i_rect[2], i_rect[3]);
    vec2 screen = pos - u_viewOrigin;
    vec2 zeroToOne = screen / u_resolution;
    gl_Position = vec4(zeroToOne.x * 2.0 - 1.0, -(zeroToOne.y * 2.0 - 1.0), 0.0, 1.0);

    v_uv = vec2(i_uv[0], i_uv[1]) + a_unit * vec2(i_uv[2] - i_uv[0], i_uv[3] - i_uv[1]);
    v_color = i_color;
}`;

const FRAGMENT_SHADER = `#version 300 es
precision highp float;

in vec2 v_uv;
in vec4 v_color;

uniform sampler2D u_atlas;

out vec4 outColor;

void main() {
    // 字形以白色绘制，纹理 alpha 即覆盖度；颜色完全由实例决定（见文件头约束 2）。
    float coverage = texture(u_atlas, v_uv).a * v_color.a;
    // 预乘 alpha 输出（与 sdf-box program 的混合方式一致）。
    outColor = vec4(v_color.rgb * coverage, coverage);
}`;

const UNIT_QUAD = new Float32Array([0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1]);

function compile(gl: WebGL2RenderingContext, type: number, source: string): WebGLShader {
    const shader = gl.createShader(type);
    if (!shader) throw new Error("Unable to create glyph shader");
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        const message = gl.getShaderInfoLog(shader) ?? "Unknown glyph shader error";
        gl.deleteShader(shader);
        throw new Error(message);
    }
    return shader;
}

/** glyph-quad program 句柄。 */
export interface GlyphProgram {
    /**
     * 上传（或更新）图集纹理。
     *
     * @param data 图集像素数据（RGBA，长度 = sizePx × sizePx × 4）。
     * @param sizePx 图集边长（物理像素）。
     */
    uploadAtlas(data: Uint8ClampedArray, sizePx: number): void;
    /**
     * 上传字形四边形并绘制。
     *
     * @param quads 字形四边形（内容坐标 + uv）。
     * @param target 光栅化目标。
     * @param viewOriginX 视口原点 x。
     * @param viewOriginY 视口原点 y。
     */
    render(
        quads: readonly GlyphQuad[],
        target: GlRasterTarget,
        viewOriginX: number,
        viewOriginY: number,
    ): void;
    /** 复用已上传实例，仅更新视口原点后重绘。 */
    repaint(target: GlRasterTarget, viewOriginX: number, viewOriginY: number): void;
    /** 释放资源。 */
    dispose(): void;
}

/**
 * 创建 glyph-quad program。
 *
 * @param gl 内核 WebGL2 上下文。
 * @returns program 句柄；创建失败时抛错（调用方捕获后回退）。
 */
export function createGlyphProgram(gl: WebGL2RenderingContext): GlyphProgram {
    const vertex = compile(gl, gl.VERTEX_SHADER, VERTEX_SHADER);
    const fragment = compile(gl, gl.FRAGMENT_SHADER, FRAGMENT_SHADER);
    const program = gl.createProgram();
    if (!program) throw new Error("Unable to create glyph program");
    gl.attachShader(program, vertex);
    gl.attachShader(program, fragment);
    gl.linkProgram(program);
    gl.deleteShader(vertex);
    gl.deleteShader(fragment);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        const message = gl.getProgramInfoLog(program) ?? "Unknown glyph link error";
        gl.deleteProgram(program);
        throw new Error(message);
    }

    const resolutionLocation = gl.getUniformLocation(program, "u_resolution");
    const originLocation = gl.getUniformLocation(program, "u_viewOrigin");
    const atlasLocation = gl.getUniformLocation(program, "u_atlas");
    if (!resolutionLocation || !originLocation || !atlasLocation) {
        gl.deleteProgram(program);
        throw new Error("glyph uniforms are missing");
    }

    const vao = gl.createVertexArray();
    const unitBuffer = gl.createBuffer();
    const instanceBuffer = gl.createBuffer();
    const texture = gl.createTexture();
    if (!vao || !unitBuffer || !instanceBuffer || !texture) {
        gl.deleteProgram(program);
        throw new Error("Unable to create glyph buffers");
    }

    let instanceCapacityFloats = 0;
    let uploadedCount = 0;
    let quadScratch = new Float32Array(0);
    let atlasSizePx = 0;

    /** 绑定实例属性（size 个连续 location）。 */
    function bindInstanceAttrib(name: string, size: number, offsetFloats: number): void {
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
        gl.vertexAttribDivisor(unitLocation, 0);
    }

    gl.bindBuffer(gl.ARRAY_BUFFER, instanceBuffer);
    bindInstanceAttrib("i_rect", 4, OFF_RECT);
    bindInstanceAttrib("i_uv", 4, OFF_UV);
    bindInstanceAttrib("i_color", 4, OFF_COLOR);
    gl.bindVertexArray(null);

    // 纹理参数：字形图集按 1:1 采样，必须用 NEAREST（LINEAR 会在边缘混入邻居字形）。
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    /** 设置绘制状态（program / VAO / uniform / 混合）。 */
    function prepareDraw(target: GlRasterTarget, viewOriginX: number, viewOriginY: number): void {
        gl.useProgram(program);
        gl.bindVertexArray(vao);
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, texture);
        gl.uniform1i(atlasLocation, 0);
        gl.uniform2f(resolutionLocation, target.cssWidthPx, target.cssHeightPx);
        gl.uniform2f(originLocation, viewOriginX, viewOriginY);
    }

    return {
        uploadAtlas(data, sizePx) {
            if (!Number.isFinite(sizePx) || sizePx <= 0) return;
            atlasSizePx = sizePx;
            gl.bindTexture(gl.TEXTURE_2D, texture);
            gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
            gl.texImage2D(
                gl.TEXTURE_2D,
                0,
                gl.RGBA,
                sizePx,
                sizePx,
                0,
                gl.RGBA,
                gl.UNSIGNED_BYTE,
                new Uint8Array(data.buffer, data.byteOffset, data.byteLength),
            );
        },

        render(quads, target, viewOriginX, viewOriginY) {
            const count = quads.length;
            if (count <= 0 || atlasSizePx <= 0) {
                uploadedCount = 0;
                return;
            }
            const needed = count * GLYPH_INSTANCE_FLOATS;
            const capacity = resolveBufferFloats(instanceCapacityFloats, needed);
            if (quadScratch.length < capacity) quadScratch = new Float32Array(capacity);
            if (capacity !== instanceCapacityFloats) {
                gl.bindBuffer(gl.ARRAY_BUFFER, instanceBuffer);
                gl.bufferData(gl.ARRAY_BUFFER, capacity * 4, gl.DYNAMIC_DRAW);
                instanceCapacityFloats = capacity;
            }
            for (let index = 0; index < count; index += 1) {
                const quad = quads[index];
                const base = index * GLYPH_INSTANCE_FLOATS;
                quadScratch[base + OFF_RECT] = quad.x;
                quadScratch[base + OFF_RECT + 1] = quad.y;
                quadScratch[base + OFF_RECT + 2] = quad.w;
                quadScratch[base + OFF_RECT + 3] = quad.h;
                quadScratch[base + OFF_UV] = quad.u0;
                quadScratch[base + OFF_UV + 1] = quad.v0;
                quadScratch[base + OFF_UV + 2] = quad.u1;
                quadScratch[base + OFF_UV + 3] = quad.v1;
                quadScratch[base + OFF_COLOR] = quad.rgba[0];
                quadScratch[base + OFF_COLOR + 1] = quad.rgba[1];
                quadScratch[base + OFF_COLOR + 2] = quad.rgba[2];
                quadScratch[base + OFF_COLOR + 3] = quad.rgba[3];
            }
            gl.bindBuffer(gl.ARRAY_BUFFER, instanceBuffer);
            gl.bufferSubData(gl.ARRAY_BUFFER, 0, quadScratch, 0, needed);
            uploadedCount = count;
            prepareDraw(target, viewOriginX, viewOriginY);
            gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, count);
        },

        repaint(target, viewOriginX, viewOriginY) {
            if (uploadedCount <= 0 || atlasSizePx <= 0) return;
            prepareDraw(target, viewOriginX, viewOriginY);
            gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, uploadedCount);
        },

        dispose() {
            gl.deleteBuffer(unitBuffer);
            gl.deleteBuffer(instanceBuffer);
            gl.deleteVertexArray(vao);
            gl.deleteTexture(texture);
            gl.deleteProgram(program);
            instanceCapacityFloats = 0;
            uploadedCount = 0;
            atlasSizePx = 0;
        },
    };
}
