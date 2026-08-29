import type { WaveformGeometry } from "./geometry.ts";
import { rasterize } from "../components/layout/timeline/runtime/canvasRaster.ts";

/**
 * 波形描边宽度（CSS 像素）。
 *
 * 两条渲染路径的**覆盖宽度契约**：Canvas2D 回退在 `setTransform(dpr)` 下
 * `lineWidth = 1` 是 1 CSS px；WebGL 端 `gl.LINES` 的 lineWidth 被所有主流
 * 实现锁死为 1 **物理**像素，在 DPR > 1 的屏幕上每列只覆盖 `1/dpr` 的宽度，
 * 亮色块从列间缝隙透出 —— 无论波形颜色多深，观感都是"浅"。因此 WebGL 端
 * 必须把线段展开成等宽四边形（见 `expandLineSegmentsToQuads`）才能与
 * Canvas2D 的覆盖严格一致。改颜色解决不了这个问题，宽度才是根因。
 */
export const WAVEFORM_STROKE_WIDTH_PX = 1;

/**
 * 把逐线段顶点（每段 2 顶点 × [x, y, r, g, b, a]）展开成逐段四边形
 * （每段 6 顶点，TRIANGLES 两次绘制），沿线段法线方向各偏移 `widthPx / 2`。
 *
 * - 竖直包络列：法线为水平 → 恰好覆盖 1 CSS px 宽的整列；
 * - 水平/斜线（take 标记）：法线为垂直/斜向 → 恒定 1 CSS px 视觉粗细；
 * - 零长度段（数字静音列）：按 1 CSS px 高的水平条带处理（Canvas2D 对
 *   零长度描边不绘制，这里选择显示一条细线，静音段在 DAW 中可见更合理）。
 *
 * 顶点颜色：A/D 沿用起点颜色，B/C 沿用终点颜色，GPU 内插值与原 LINES 一致。
 * 返回模块级复用的 scratch 缓冲（容量按需倍增），调用方不得长期持有。
 */
export function expandLineSegmentsToQuads(vertices: Float32Array): Float32Array {
    const segmentCount = Math.floor(vertices.length / 12);
    const required = segmentCount * 36;
    if (quadScratch === null || quadScratch.length < required) {
        let capacity = Math.max(4096, required);
        while (capacity < required) capacity *= 2;
        quadScratch = new Float32Array(capacity);
    }
    const out = quadScratch;
    const half = WAVEFORM_STROKE_WIDTH_PX / 2;
    for (let segment = 0; segment < segmentCount; segment += 1) {
        const base = segment * 12;
        const x1 = vertices[base];
        const y1 = vertices[base + 1];
        const x2 = vertices[base + 6];
        const y2 = vertices[base + 7];
        const dx = x2 - x1;
        const dy = y2 - y1;
        const length = Math.hypot(dx, dy);
        const nx = length > 1e-9 ? (-dy / length) * half : 0;
        const ny = length > 1e-9 ? (dx / length) * half : half;
        const outBase = segment * 36;
        // 四角：A = p1 + n, B = p2 + n, C = p2 - n, D = p1 - n
        // 三角形 1: A B C；三角形 2: A C D
        const corners = [
            [x1 + nx, y1 + ny, base + 2], // A（起点色）
            [x2 + nx, y2 + ny, base + 8], // B（终点色）
            [x2 - nx, y2 - ny, base + 8], // C（终点色）
            [x1 + nx, y1 + ny, base + 2], // A
            [x2 - nx, y2 - ny, base + 8], // C
            [x1 - nx, y1 - ny, base + 2], // D（起点色）
        ] as const;
        for (let corner = 0; corner < 6; corner += 1) {
            const [cx, cy, colorBase] = corners[corner];
            const o = outBase + corner * 6;
            out[o] = cx;
            out[o + 1] = cy;
            out[o + 2] = vertices[colorBase];
            out[o + 3] = vertices[colorBase + 1];
            out[o + 4] = vertices[colorBase + 2];
            out[o + 5] = vertices[colorBase + 3];
        }
    }
    return out.subarray(0, required);
}

let quadScratch: Float32Array | null = null;

export interface WaveformSurfaceRenderer {
    readonly kind: "webgl2" | "canvas2d";
    render(geometry: WaveformGeometry, widthPx: number, heightPx: number, dpr: number): void;
    dispose(): void;
}

function compileShader(gl: WebGL2RenderingContext, type: number, source: string): WebGLShader {
    const shader = gl.createShader(type);
    if (!shader) throw new Error("Unable to create waveform shader");
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        const message = gl.getShaderInfoLog(shader) ?? "Unknown waveform shader error";
        gl.deleteShader(shader);
        throw new Error(message);
    }
    return shader;
}

export class WebGl2WaveformRenderer implements WaveformSurfaceRenderer {
    readonly kind = "webgl2" as const;
    private readonly canvas: HTMLCanvasElement;
    private readonly gl: WebGL2RenderingContext;
    private readonly program: WebGLProgram;
    private readonly buffer: WebGLBuffer;
    private readonly resolutionLocation: WebGLUniformLocation;
    private readonly positionLocation: number;
    private readonly colorLocation: number;

    constructor(canvas: HTMLCanvasElement) {
        this.canvas = canvas;
        const gl = canvas.getContext("webgl2", {
            alpha: true,
            antialias: false,
            depth: false,
            stencil: false,
            premultipliedAlpha: true,
            preserveDrawingBuffer: false,
            powerPreference: "high-performance",
        });
        if (!gl) throw new Error("WebGL2 is unavailable");
        this.gl = gl;

        const vertex = compileShader(
            gl,
            gl.VERTEX_SHADER,
            `#version 300 es
            in vec2 a_position;
            in vec4 a_color;
            uniform vec2 u_resolution;
            out vec4 v_color;
            void main() {
                vec2 zeroToOne = a_position / u_resolution;
                vec2 clip = zeroToOne * 2.0 - 1.0;
                gl_Position = vec4(clip.x, -clip.y, 0.0, 1.0);
                v_color = a_color;
            }`,
        );
        const fragment = compileShader(
            gl,
            gl.FRAGMENT_SHADER,
            `#version 300 es
            precision mediump float;
            in vec4 v_color;
            out vec4 outColor;
            void main() { outColor = v_color; }`,
        );
        const program = gl.createProgram();
        if (!program) throw new Error("Unable to create waveform WebGL program");
        gl.attachShader(program, vertex);
        gl.attachShader(program, fragment);
        gl.linkProgram(program);
        gl.deleteShader(vertex);
        gl.deleteShader(fragment);
        if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
            const message = gl.getProgramInfoLog(program) ?? "Unknown waveform link error";
            gl.deleteProgram(program);
            throw new Error(message);
        }
        const buffer = gl.createBuffer();
        if (!buffer) throw new Error("Unable to create waveform vertex buffer");
        const resolutionLocation = gl.getUniformLocation(program, "u_resolution");
        if (!resolutionLocation) throw new Error("Waveform resolution uniform is missing");
        this.program = program;
        this.buffer = buffer;
        this.resolutionLocation = resolutionLocation;
        this.positionLocation = gl.getAttribLocation(program, "a_position");
        this.colorLocation = gl.getAttribLocation(program, "a_color");
    }

    render(geometry: WaveformGeometry, widthPx: number, heightPx: number, dpr: number): void {
        const gl = this.gl;
        const target = rasterize(this.canvas, widthPx, heightPx, dpr);
        gl.viewport(0, 0, target.physicalWidth, target.physicalHeight);
        gl.clearColor(0, 0, 0, 0);
        gl.clear(gl.COLOR_BUFFER_BIT);
        gl.useProgram(this.program);
        // u_resolution 必须传 physical/dpr：传 CSS 尺寸会让 NDC 被拉伸到
        // physical 个像素，实际缩放比变成 physical/css（≠ dpr），波形会相对
        // clip / 网格产生随窗口宽度跳动的亚像素偏移。
        gl.uniform2f(this.resolutionLocation, target.resolutionWidth, target.resolutionHeight);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);

        const stride = 6 * Float32Array.BYTES_PER_ELEMENT;
        const position = this.positionLocation;
        const color = this.colorLocation;
        gl.enableVertexAttribArray(position);
        gl.vertexAttribPointer(position, 2, gl.FLOAT, false, stride, 0);
        gl.enableVertexAttribArray(color);
        gl.vertexAttribPointer(
            color,
            4,
            gl.FLOAT,
            false,
            stride,
            2 * Float32Array.BYTES_PER_ELEMENT,
        );
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
        // LINES 的 lineWidth 恒为 1 物理像素（DPR > 1 时覆盖不足，波形发浅），
        // 改用展开后的 1 CSS px 等宽四边形，与 Canvas2D 回退路径覆盖一致。
        const quads = expandLineSegmentsToQuads(geometry.vertices);
        gl.bufferData(gl.ARRAY_BUFFER, quads, gl.DYNAMIC_DRAW);
        gl.drawArrays(gl.TRIANGLES, 0, quads.length / 6);
    }

    dispose(): void {
        this.gl.deleteBuffer(this.buffer);
        this.gl.deleteProgram(this.program);
    }
}

export class Canvas2dWaveformRenderer implements WaveformSurfaceRenderer {
    readonly kind = "canvas2d" as const;
    private readonly backCanvas = document.createElement("canvas");
    private readonly canvas: HTMLCanvasElement;

    constructor(canvas: HTMLCanvasElement) {
        this.canvas = canvas;
    }

    render(geometry: WaveformGeometry, widthPx: number, heightPx: number, dpr: number): void {
        const target = rasterize(this.canvas, widthPx, heightPx, dpr);
        rasterize(this.backCanvas, widthPx, heightPx, dpr);
        const back = this.backCanvas.getContext("2d");
        const visible = this.canvas.getContext("2d");
        if (!back || !visible) throw new Error("Canvas 2D is unavailable");

        back.setTransform(target.dpr, 0, 0, target.dpr, 0, 0);
        back.clearRect(0, 0, target.cssWidthPx, target.cssHeightPx);
        back.lineWidth = 1;
        const vertices = geometry.vertices;
        // 相邻段几乎总是同一颜色（逐像素 min/max 包络）：把同色段合并进
        // 单个 path，把每帧数千次 beginPath/stroke 降为颜色变化次数级别。
        // 一条 path 上的 moveTo 天然断开子路径，语义与逐段 stroke 一致。
        const verticesEnd = vertices.length - (vertices.length % 12);
        let batchStart = 0;
        const segmentsMatch = (a: number, b: number): boolean =>
            vertices[a + 2] === vertices[b + 2] &&
            vertices[a + 3] === vertices[b + 3] &&
            vertices[a + 4] === vertices[b + 4] &&
            vertices[a + 5] === vertices[b + 5];
        const strokeBatch = (start: number, end: number) => {
            const red = Math.round((vertices[start + 2] ?? 1) * 255);
            const green = Math.round((vertices[start + 3] ?? 1) * 255);
            const blue = Math.round((vertices[start + 4] ?? 1) * 255);
            const alpha = vertices[start + 5] ?? 1;
            back.strokeStyle = `rgba(${red},${green},${blue},${alpha})`;
            back.beginPath();
            for (let offset = start; offset < end; offset += 12) {
                back.moveTo(vertices[offset] ?? 0, vertices[offset + 1] ?? 0);
                back.lineTo(vertices[offset + 6] ?? 0, vertices[offset + 7] ?? 0);
            }
            back.stroke();
        };
        for (let offset = 12; offset < verticesEnd; offset += 12) {
            if (!segmentsMatch(batchStart, offset)) {
                strokeBatch(batchStart, offset);
                batchStart = offset;
            }
        }
        if (batchStart < verticesEnd) strokeBatch(batchStart, verticesEnd);

        visible.setTransform(1, 0, 0, 1, 0, 0);
        visible.clearRect(0, 0, target.physicalWidth, target.physicalHeight);
        visible.drawImage(this.backCanvas, 0, 0);
    }

    dispose(): void {
        this.backCanvas.width = 1;
        this.backCanvas.height = 1;
    }
}
