/**
 * 波形 WebGL2 渲染器
 *
 * 主要内容：将 interleaved peaks 数据上传为 RG32F 纹理，通过 instanced quad 渲染
 *         每个像素列对应一个 instance，vertex shader 内 texelFetch 取 min/max 计算 quad 顶点
 * 作用：替代 Canvas 2D per-pixel CPU 循环，将逐像素 fillRect 的开销搬到 GPU 并行执行
 * 与其他模块的关系：
 *   - 实现 waveformRenderer.ts 的 WaveformRenderer 接口
 *   - 由 waveformRendererFactory.ts 在检测到 WebGL2 支持时创建
 *   - 被 WaveformTrackCanvas.tsx 和 PianoRollPanel.tsx 消费
 *
 * 渲染流程（每次 drawClipWaveform 调用）：
 *   1. texSubImage2D 上传 peaks 到 RG32F 纹理
 *   2. 计算 uniform（pxToIdxScale 等，与现有 renderWaveform 公式一致）
 *   3. scissor rect 裁剪到 clip 可视段
 *   4. drawArraysInstanced(TRIANGLE_STRIP, 0, 4, visibleWidthPx)
 *
 * 资源管理：
 *   - 每个 renderer 实例持有一套 GL 资源（program/vao/texture）
 *   - dispose 时显式释放，避免 GPU 内存泄漏
 *   - 纹理容量动态扩展（ensureTextureCapacity）
 */

import type { DrawClipWaveformParams, WaveformRenderer } from "./waveformRenderer";

/** RG32F 纹理最大采样对数（128KB，覆盖一首 5 分钟歌 L0 级别可见窗口） */
const MAX_PEAK_SAMPLES = 65536;

/** 顶点着色器源码（GLSL ES 3.00） */
const VERTEX_SHADER_SOURCE = `#version 300 es
precision highp float;

uniform int u_visibleStartPx;
uniform int u_visibleEndPx;
uniform float u_pxToIdxScale;
uniform float u_pxToIdxBase;
uniform float u_halfPixelIdx;
uniform int u_totalSamples;
uniform float u_amplitudeScale;
uniform float u_centerY;
uniform float u_displayW;
uniform float u_displayH;
uniform float u_strokeWidth;
uniform highp sampler2D u_peaksTex;

out float v_alphaFactor;

void main() {
    int px = u_visibleStartPx + gl_InstanceID;
    float pxF = float(px);

    float centerIdx = pxF * u_pxToIdxScale + u_pxToIdxBase;
    float idxLeft = max(0.0, centerIdx - u_halfPixelIdx);
    float idxRight = min(float(u_totalSamples - 1), centerIdx + u_halfPixelIdx);

    int iStart = int(floor(idxLeft));
    int iEnd = int(ceil(idxRight));

    float pixelMin = 1e38;
    float pixelMax = -1e38;
    for (int i = 0; i < 4096; i++) {
        if (i > iEnd) break;
        if (i < iStart) continue;
        vec2 peak = texelFetch(u_peaksTex, ivec2(i, 0), 0).rg;
        if (peak.r < pixelMin) pixelMin = peak.r;
        if (peak.g > pixelMax) pixelMax = peak.g;
    }

    if (pixelMin > 1e37) {
        gl_Position = vec4(2.0, 2.0, 0.0, 1.0);
        return;
    }

    float yTop = u_centerY - pixelMax * u_amplitudeScale;
    float yBot = u_centerY - pixelMin * u_amplitudeScale;

    float halfStroke = u_strokeWidth * 0.5;
    if (yBot - yTop < 0.5) {
        float midY = (yTop + yBot) * 0.5;
        yTop = midY - 0.25;
        yBot = midY + 0.25;
    }

    float xLeft = pxF - halfStroke;
    float xRight = pxF + halfStroke;
    float x = (gl_VertexID == 0 || gl_VertexID == 2) ? xLeft : xRight;
    float y = (gl_VertexID == 0 || gl_VertexID == 1) ? yTop : yBot;

    float ndcX = (x / u_displayW) * 2.0 - 1.0;
    float ndcY = 1.0 - (y / u_displayH) * 2.0;

    gl_Position = vec4(ndcX, ndcY, 0.0, 1.0);
    v_alphaFactor = 1.0;
}
`;

/** 片元着色器源码（GLSL ES 3.00） */
const FRAGMENT_SHADER_SOURCE = `#version 300 es
precision highp float;

uniform vec4 u_color;
uniform float u_alpha;

in float v_alphaFactor;
out vec4 fragColor;

void main() {
    fragColor = u_color * u_alpha * v_alphaFactor;
}
`;

/**
 * WebGL2 波形渲染器
 *
 * 流程：见文件头注释
 * 特殊说明：
 *   - 所有坐标输入都是 CSS 像素，内部转 NDC
 *   - scissor rect 用物理像素
 *   - 颜色通过 ColorCache 解析，避免每帧创建临时 canvas
 */
export class WebGL2WaveformRenderer implements WaveformRenderer {
    readonly backend = "webgl2" as const;

    private canvas: HTMLCanvasElement;
    private gl: WebGL2RenderingContext;
    private program: WebGLProgram;
    private vao: WebGLVertexArrayObject;
    private peaksTex: WebGLTexture;
    private texCapacity = 0;

    private dpr = 1;
    private displayW = 0;
    private displayH = 0;
    private physicalW = 0;
    private physicalH = 0;

    /** uniform locations 缓存，避免每帧查询 */
    private uniforms: Record<string, WebGLUniformLocation | null> = {};

    /** 颜色解析缓存，避免重复 getImageData */
    private colorCache = new Map<string, [number, number, number, number]>();
    private colorParseCanvas: HTMLCanvasElement;
    private colorParseCtx: CanvasRenderingContext2D;

    constructor(canvas: HTMLCanvasElement, gl: WebGL2RenderingContext) {
        this.canvas = canvas;
        this.gl = gl;

        // 编译 shader 与链接 program
        this.program = this.createProgram(VERTEX_SHADER_SOURCE, FRAGMENT_SHADER_SOURCE);
        this.cacheUniformLocations();

        // 创建空 VAO（顶点全在 shader 里生成）
        const vao = gl.createVertexArray();
        if (!vao) throw new Error("WebGL2: createVertexArray failed");
        this.vao = vao;

        // 创建纹理
        const tex = gl.createTexture();
        if (!tex) throw new Error("WebGL2: createTexture failed");
        this.peaksTex = tex;

        // 颜色解析用的辅助 canvas
        this.colorParseCanvas = document.createElement("canvas");
        this.colorParseCanvas.width = 1;
        this.colorParseCanvas.height = 1;
        const colorCtx = this.colorParseCanvas.getContext("2d");
        if (!colorCtx) throw new Error("WebGL2: color parse canvas 2d context unavailable");
        this.colorParseCtx = colorCtx;

        // 初始纹理分配
        this.ensureTextureCapacity(MAX_PEAK_SAMPLES);
    }

    private createProgram(vsSource: string, fsSource: string): WebGLProgram {
        const gl = this.gl;
        const vs = this.compileShader(gl.VERTEX_SHADER, vsSource);
        const fs = this.compileShader(gl.FRAGMENT_SHADER, fsSource);
        const program = gl.createProgram();
        if (!program) throw new Error("WebGL2: createProgram failed");
        gl.attachShader(program, vs);
        gl.attachShader(program, fs);
        gl.linkProgram(program);
        if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
            const info = gl.getProgramInfoLog(program);
            gl.deleteProgram(program);
            throw new Error(`WebGL2: program link failed: ${info}`);
        }
        gl.deleteShader(vs);
        gl.deleteShader(fs);
        return program;
    }

    private compileShader(type: number, source: string): WebGLShader {
        const gl = this.gl;
        const shader = gl.createShader(type);
        if (!shader) throw new Error("WebGL2: createShader failed");
        gl.shaderSource(shader, source);
        gl.compileShader(shader);
        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
            const info = gl.getShaderInfoLog(shader);
            gl.deleteShader(shader);
            throw new Error(`WebGL2: shader compile failed: ${info}`);
        }
        return shader;
    }

    private cacheUniformLocations(): void {
        const gl = this.gl;
        const names = [
            "u_visibleStartPx",
            "u_visibleEndPx",
            "u_pxToIdxScale",
            "u_pxToIdxBase",
            "u_halfPixelIdx",
            "u_totalSamples",
            "u_amplitudeScale",
            "u_centerY",
            "u_displayW",
            "u_displayH",
            "u_strokeWidth",
            "u_peaksTex",
            "u_color",
            "u_alpha",
        ];
        for (const name of names) {
            this.uniforms[name] = gl.getUniformLocation(this.program, name);
        }
    }

    /**
     * 确保纹理容量足够容纳 sampleCount 个采样对
     *
     * 流程：
     *   - 若当前容量 >= sampleCount，跳过
     *   - 否则用 texImage2D 重新分配（罕见路径）
     */
    private ensureTextureCapacity(sampleCount: number): void {
        const gl = this.gl;
        if (this.texCapacity >= sampleCount) return;

        const newCapacity = Math.max(sampleCount, MAX_PEAK_SAMPLES);
        gl.bindTexture(gl.TEXTURE_2D, this.peaksTex);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.texImage2D(
            gl.TEXTURE_2D,
            0,
            gl.RG32F,
            newCapacity,
            1,
            0,
            gl.RG,
            gl.FLOAT,
            null,
        );
        this.texCapacity = newCapacity;
    }

    resize(displayW: number, displayH: number, dpr: number): void {
        this.displayW = displayW;
        this.displayH = displayH;
        this.dpr = dpr;
        this.physicalW = Math.max(1, Math.round(displayW * dpr));
        this.physicalH = Math.max(1, Math.round(displayH * dpr));

        this.canvas.width = this.physicalW;
        this.canvas.height = this.physicalH;
        this.canvas.style.width = `${displayW}px`;
        this.canvas.style.height = `${displayH}px`;
    }

    clear(): void {
        const gl = this.gl;
        gl.viewport(0, 0, this.physicalW, this.physicalH);
        gl.disable(gl.SCISSOR_TEST);
        gl.clearColor(0, 0, 0, 0);
        gl.clear(gl.COLOR_BUFFER_BIT);
    }

    drawClipWaveform(params: DrawClipWaveformParams): void {
        // 实现在 Task 6。
        // 以下字段在此 stub 中被引用以通过 noUnusedLocals 检查；
        // Task 6 实现 drawClipWaveform 时会移除这些 void 引用并真正使用它们。
        void this.dpr;
        void this.displayW;
        void this.displayH;
        void this.colorParseCtx;
        void params;
    }

    dispose(): void {
        const gl = this.gl;
        gl.deleteProgram(this.program);
        gl.deleteVertexArray(this.vao);
        gl.deleteTexture(this.peaksTex);
        this.colorCache.clear();

        const ext = gl.getExtension("WEBGL_lose_context");
        ext?.loseContext();
    }
}
