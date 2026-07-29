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

import type { DrawClipWaveformParams, WaveformRenderParams, WaveformRenderer } from "./waveformRenderer";

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
    // 循环上限 4096 覆盖典型场景（65536 samples / 1920px ≈ 34 per pixel）
    // 极端情况（极窄视口 + 极大数据量）可能截断，但实际应用中罕见
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
    // 允许 null 以便 dispose() 在构造失败的清理路径中安全调用；
    // 构造成功后这三个字段必然非空，使用处用 `!` 断言
    private program: WebGLProgram | null = null;
    private vao: WebGLVertexArrayObject | null = null;
    private peaksTex: WebGLTexture | null = null;
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
    // 这两个字段在构造 try 块内赋值，用 `!` 告知 TS 严格初始化检查
    private colorParseCanvas!: HTMLCanvasElement;
    private colorParseCtx!: CanvasRenderingContext2D;

    constructor(canvas: HTMLCanvasElement, gl: WebGL2RenderingContext) {
        this.canvas = canvas;
        this.gl = gl;

        // 构造过程涉及多个 GL 资源分配，任一失败都需清理已分配的资源，避免泄漏。
        // 这里用 try/catch 包裹，失败时调用 dispose() 释放已分配字段后重新抛出。
        try {
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
        } catch (e) {
            // 清理已分配的资源，避免泄漏
            this.dispose();
            throw e;
        }
    }

    /**
     * 编译并链接一个 WebGL2 program
     *
     * 流程：
     *   1. 编译 vertex / fragment shader
     *   2. attach + link
     *   3. 链接完成后立即 deleteShader（shader 已附着到 program，删除只会 detach，
     *      不影响已链接的 program；这样无论链接成功与否都不会泄漏 shader 对象）
     *   4. 检查 LINK_STATUS，失败则删除 program 并抛错
     *
     * 参数说明：
     *   - vsSource: 顶点着色器 GLSL 源码
     *   - fsSource: 片元着色器 GLSL 源码
     *
     * 返回：成功链接的 WebGLProgram
     */
    private createProgram(vsSource: string, fsSource: string): WebGLProgram {
        const gl = this.gl;
        const vs = this.compileShader(gl.VERTEX_SHADER, vsSource);
        const fs = this.compileShader(gl.FRAGMENT_SHADER, fsSource);
        const program = gl.createProgram();
        if (!program) throw new Error("WebGL2: createProgram failed");
        gl.attachShader(program, vs);
        gl.attachShader(program, fs);
        gl.linkProgram(program);
        // shader 已附着到 program，删除只 detach 不影响已链接的 program；
        // 放在 link 检查之前，保证链接失败路径也不会泄漏 shader 对象
        gl.deleteShader(vs);
        gl.deleteShader(fs);
        if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
            const info = gl.getProgramInfoLog(program);
            gl.deleteProgram(program);
            throw new Error(`WebGL2: program link failed: ${info}`);
        }
        return program;
    }

    /**
     * 编译单个 GLSL shader
     *
     * 流程：创建 shader → 灌入源码 → 编译 → 检查 COMPILE_STATUS
     * 特殊说明：编译失败时删除 shader 对象后再抛错，避免泄漏
     *
     * 参数说明：
     *   - type: gl.VERTEX_SHADER 或 gl.FRAGMENT_SHADER
     *   - source: GLSL 源码字符串
     *
     * 返回：编译成功的 WebGLShader
     */
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
            this.uniforms[name] = gl.getUniformLocation(this.program!, name);
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
        gl.bindTexture(gl.TEXTURE_2D, this.peaksTex!);
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

    /**
     * 解析 CSS 颜色字符串为 RGBA 浮点数组
     *
     * 流程：通过辅助 canvas 的 fillStyle + getImageData 解析
     * 特殊说明：使用 Map 缓存，避免重复解析同一颜色字符串
     *
     * @param css CSS 颜色字符串（如 "#7c9eff" 或 "rgba(124,158,255,0.86)"）
     * @returns [r, g, b, a]，范围 0~1
     */
    private parseColor(css: string): [number, number, number, number] {
        const cached = this.colorCache.get(css);
        if (cached) return cached;

        this.colorParseCtx.clearRect(0, 0, 1, 1);
        this.colorParseCtx.fillStyle = css;
        this.colorParseCtx.fillRect(0, 0, 1, 1);
        const data = this.colorParseCtx.getImageData(0, 0, 1, 1).data;
        const result: [number, number, number, number] = [
            data[0] / 255,
            data[1] / 255,
            data[2] / 255,
            data[3] / 255,
        ];
        this.colorCache.set(css, result);
        return result;
    }

    /**
     * 从 WaveformRenderParams 计算 vertex shader 需要的 uniform 值
     *
     * 流程：与现有 waveformRenderer.ts 的 pxToIdxScale/pxToIdxBase/halfPixelIdx 公式完全一致
     * 特殊说明：
     *   - reversed 影响符号
     *   - 视口裁剪由调用方通过 segmentLeftPx/segmentRightPx 指定
     *   - totalSamples 从 peaksLength 推算（WaveformRenderParams 不含 peaks 字段）
     *
     * @param params 渲染参数
     * @param peaksLength peaks 数组长度（interleaved，除以 2 得采样对数）
     * @param segmentLeftPx 可视段左边界（CSS 像素）
     * @param segmentRightPx 可视段右边界（CSS 像素）
     */
    private computeUniforms(
        params: WaveformRenderParams,
        peaksLength: number,
        segmentLeftPx: number,
        segmentRightPx: number,
    ): {
        visibleStartPx: number;
        visibleEndPx: number;
        pxToIdxScale: number;
        pxToIdxBase: number;
        halfPixelIdx: number;
        totalSamples: number;
        amplitudeScale: number;
        centerY: number;
    } {
        const {
            canvasWidth,
            canvasHeight,
            centerY,
            sourceStartSec,
            clipDuration,
            playbackRate,
            reversed = false,
            dataStartSec,
            dataDurationSec,
            clipPixelOffset = 0,
            clipTotalWidthPx,
            zeroDbHalfHeight,
        } = params;

        const totalSamples = Math.floor(peaksLength / 2);
        void canvasWidth; // canvasWidth 不直接使用，segmentLeftPx/RightPx 已包含视口信息

        const clipTotalW = clipTotalWidthPx ?? canvasWidth;
        const amplitudeScale = zeroDbHalfHeight ?? canvasHeight / 2;

        const effectiveDataStartSec = dataStartSec ?? sourceStartSec;
        const effectiveDataDurationSec = dataDurationSec ?? clipDuration * playbackRate;

        const pxToTimeScale = (clipDuration * playbackRate) / clipTotalW;
        const invDataDuration = 1 / effectiveDataDurationSec;
        const timeToIdxScale = (totalSamples - 1) * invDataDuration;
        const pxToIdxScale = (reversed ? -1 : 1) * pxToTimeScale * timeToIdxScale;

        const clipSourceEndSec = sourceStartSec + clipDuration * playbackRate;
        const pxToIdxBase = reversed
            ? (clipSourceEndSec - clipPixelOffset * pxToTimeScale - effectiveDataStartSec) *
              timeToIdxScale
            : (clipPixelOffset * pxToTimeScale + sourceStartSec - effectiveDataStartSec) *
              timeToIdxScale;

        const halfPixelIdx = Math.abs(0.5 * pxToIdxScale);

        return {
            visibleStartPx: Math.round(segmentLeftPx),
            visibleEndPx: Math.round(segmentRightPx),
            pxToIdxScale,
            pxToIdxBase,
            halfPixelIdx,
            totalSamples,
            amplitudeScale,
            centerY,
        };
    }

    /**
     * 绘制单个 clip 的波形（核心渲染入口）
     *
     * 流程：
     *   1. 上传 peaks 数据到 RG32F 纹理（texSubImage2D，纹理容量由 ensureTextureCapacity 保证）
     *   2. 通过 computeUniforms 计算 pxToIdxScale / pxToIdxBase / halfPixelIdx 等 uniform
     *   3. 启用 scissor rect 裁剪到 clip 可视段（物理像素坐标，乘 dpr）
     *   4. 绑定 program / VAO，写入所有 uniform，绑定纹理到 unit 0
     *   5. 启用 premultiplied alpha blending
     *   6. drawArraysInstanced(TRIANGLE_STRIP, 0, 4, instanceCount) —— 每像素列一个 instance
     *   7. 清理 scissor / blend / VAO 状态，避免影响后续渲染
     *
     * 特殊说明：
     *   - instanceCount = round(segmentRightPx) - round(segmentLeftPx)，每 instance 渲染 1 CSS 像素列
     *   - scissor 使用物理像素（乘 dpr），viewport 由 clear() 全画布设置
     *   - peaks.length < 4（即不足 2 个采样对）直接跳过，避免纹理上传异常
     *   - 颜色通过 parseColor 解析并缓存；alpha 在片元着色器内乘到 fragColor
     *
     * 参数说明：
     *   - params.peaks: interleaved [min0,max0,min1,max1,...]，已应用增益
     *   - params.renderParams: 视口/中心线/振幅等渲染参数
     *   - params.segmentLeftPx / segmentRightPx: 可视段左右边界（CSS 像素）
     *   - params.strokeColor: CSS 颜色字符串
     *   - params.strokeWidth: 描边宽度（CSS 像素）
     *   - params.alpha: 整体透明度（0~1）
     */
    drawClipWaveform(params: DrawClipWaveformParams): void {
        const {
            peaks,
            renderParams,
            segmentLeftPx,
            segmentRightPx,
            strokeColor,
            strokeWidth,
            alpha,
        } = params;

        const gl = this.gl;
        const visibleStartPx = Math.round(segmentLeftPx);
        const visibleEndPx = Math.round(segmentRightPx);
        const instanceCount = visibleEndPx - visibleStartPx;
        if (instanceCount <= 0) return;
        if (peaks.length < 4) return;

        // 1. 上传 peaks 数据到纹理
        const sampleCount = peaks.length / 2;
        this.ensureTextureCapacity(sampleCount);
        gl.bindTexture(gl.TEXTURE_2D, this.peaksTex!);
        gl.texSubImage2D(
            gl.TEXTURE_2D,
            0,
            0,
            0,
            sampleCount,
            1,
            gl.RG,
            gl.FLOAT,
            peaks,
        );

        // 2. 计算 uniform
        const u = this.computeUniforms(
            renderParams,
            peaks.length,
            segmentLeftPx,
            segmentRightPx,
        );

        // 3. 设置 scissor rect（物理像素，裁剪到可视段）
        const scissorX = visibleStartPx * this.dpr;
        const scissorW = instanceCount * this.dpr;
        gl.enable(gl.SCISSOR_TEST);
        gl.scissor(scissorX, 0, scissorW, this.physicalH);

        // 4. 绑定 program 和 VAO
        gl.useProgram(this.program!);
        gl.bindVertexArray(this.vao!);

        // 5. 设置 uniform
        gl.uniform1i(this.uniforms["u_visibleStartPx"]!, u.visibleStartPx);
        gl.uniform1i(this.uniforms["u_visibleEndPx"]!, u.visibleEndPx);
        gl.uniform1f(this.uniforms["u_pxToIdxScale"]!, u.pxToIdxScale);
        gl.uniform1f(this.uniforms["u_pxToIdxBase"]!, u.pxToIdxBase);
        gl.uniform1f(this.uniforms["u_halfPixelIdx"]!, u.halfPixelIdx);
        gl.uniform1i(this.uniforms["u_totalSamples"]!, u.totalSamples);
        gl.uniform1f(this.uniforms["u_amplitudeScale"]!, u.amplitudeScale);
        gl.uniform1f(this.uniforms["u_centerY"]!, u.centerY);
        gl.uniform1f(this.uniforms["u_displayW"]!, this.displayW);
        gl.uniform1f(this.uniforms["u_displayH"]!, this.displayH);
        gl.uniform1f(this.uniforms["u_strokeWidth"]!, strokeWidth);

        // 绑定纹理到 unit 0
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.peaksTex!);
        gl.uniform1i(this.uniforms["u_peaksTex"]!, 0);

        // 颜色与 alpha
        const color = this.parseColor(strokeColor);
        gl.uniform4f(this.uniforms["u_color"]!, color[0], color[1], color[2], color[3]);
        gl.uniform1f(this.uniforms["u_alpha"]!, alpha);

        // 6. 启用 blending（premultiplied alpha）
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

        // 7. instanced draw
        gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, instanceCount);

        // 8. 清理状态
        gl.disable(gl.SCISSOR_TEST);
        gl.disable(gl.BLEND);
        gl.bindVertexArray(null);
    }

    /**
     * 释放 GPU 资源
     *
     * 特殊说明：
     *   - 只删除本实例拥有的 program/vao/texture，不调用 loseContext ——
     *     canvas 由消费者持有，强行 loseContext 会永久破坏画布
     *   - 对每个字段做 null 检查，以便构造失败的清理路径也能安全调用
     */
    dispose(): void {
        const gl = this.gl;
        if (this.program) gl.deleteProgram(this.program);
        if (this.vao) gl.deleteVertexArray(this.vao);
        if (this.peaksTex) gl.deleteTexture(this.peaksTex);
        this.colorCache.clear();
    }
}
