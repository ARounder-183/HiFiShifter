import type { WaveformGeometry } from "./geometry.ts";
import {
    clearCanvasPhysical,
    rasterize,
} from "../components/layout/timeline/runtime/canvasRaster.ts";

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
        const segDx = x2 - x1;
        const segDy = y2 - y1;
        const length = Math.hypot(segDx, segDy);
        const nx = length > 1e-9 ? (-segDy / length) * half : 0;
        const ny = length > 1e-9 ? (segDx / length) * half : half;
        const outBase = segment * 36;
        // 四角：A = p1 + n, B = p2 + n, C = p2 - n, D = p1 - n
        // 三角形 1: A B C；三角形 2: A C D
        //
        // 六个角点**全部内联展开**：此前这里是一个 `corners` 数组字面量 + 一次
        // 循环，等于每段分配 7 个数组（15,200 段 ≈ 10.6 万次/帧）。语义上它
        // 与下面 6 组直接写入完全等价，但每帧十万次分配会把 GC 拖进渲染关键
        // 路径。起点色/终点色各取一次到局部量，避免重复的 Float32Array 索引。
        const ax = x1 + nx;
        const ay = y1 + ny;
        const bx = x2 + nx;
        const by = y2 + ny;
        const cx = x2 - nx;
        const cy = y2 - ny;
        const dx = x1 - nx;
        const dy = y1 - ny;
        const r1 = vertices[base + 2];
        const g1 = vertices[base + 3];
        const b1 = vertices[base + 4];
        const a1 = vertices[base + 5];
        const r2 = vertices[base + 8];
        const g2 = vertices[base + 9];
        const b2 = vertices[base + 10];
        const a2 = vertices[base + 11];

        // A（起点色）
        out[outBase] = ax;
        out[outBase + 1] = ay;
        out[outBase + 2] = r1;
        out[outBase + 3] = g1;
        out[outBase + 4] = b1;
        out[outBase + 5] = a1;
        // B（终点色）
        out[outBase + 6] = bx;
        out[outBase + 7] = by;
        out[outBase + 8] = r2;
        out[outBase + 9] = g2;
        out[outBase + 10] = b2;
        out[outBase + 11] = a2;
        // C（终点色）
        out[outBase + 12] = cx;
        out[outBase + 13] = cy;
        out[outBase + 14] = r2;
        out[outBase + 15] = g2;
        out[outBase + 16] = b2;
        out[outBase + 17] = a2;
        // A（起点色）
        out[outBase + 18] = ax;
        out[outBase + 19] = ay;
        out[outBase + 20] = r1;
        out[outBase + 21] = g1;
        out[outBase + 22] = b1;
        out[outBase + 23] = a1;
        // C（终点色）
        out[outBase + 24] = cx;
        out[outBase + 25] = cy;
        out[outBase + 26] = r2;
        out[outBase + 27] = g2;
        out[outBase + 28] = b2;
        out[outBase + 29] = a2;
        // D（起点色）
        out[outBase + 30] = dx;
        out[outBase + 31] = dy;
        out[outBase + 32] = r1;
        out[outBase + 33] = g1;
        out[outBase + 34] = b1;
        out[outBase + 35] = a1;
    }
    return out.subarray(0, required);
}

let quadScratch: Float32Array | null = null;

/**
 * 波形面渲染器。
 *
 * 【坐标系约定】几何顶点使用**窗口局部坐标**（内容坐标减去构建窗口的左上
 * 角），视口位置由 `originXPx / originYPx` 单独给出：
 * `屏幕位置 = 局部坐标 − 视口原点`。
 *
 * 这样几何可以跨帧复用：平移只改变视口原点，几何一行都不用重算（WebGL 路径
 * 退化成一次 uniform 更新 + drawArrays）。视口原点超出已构建窗口时必须重新
 * `render()`。
 */
export interface WaveformSurfaceRenderer {
    readonly kind: "webgl2" | "canvas2d";
    /**
     * 上传几何并按给定视口原点绘制。
     *
     * @param geometry 顶点几何（窗口局部坐标）。
     * @param widthPx 视口宽（CSS 像素）。
     * @param heightPx 视口高（CSS 像素）。
     * @param dpr 设备像素比。
     * @param originXPx 视口左缘在窗口局部坐标系中的 x。
     * @param originYPx 视口上缘在窗口局部坐标系中的 y。
     */
    render(
        geometry: WaveformGeometry,
        widthPx: number,
        heightPx: number,
        dpr: number,
        originXPx: number,
        originYPx: number,
    ): void;
    /**
     * 仅平移：复用 `render()` 已上传的几何，只更新视口原点后重绘。
     *
     * 特殊说明：必须在 `render()` 之后调用，且期间不得复用几何所在的缓冲。
     * WebGL 路径几乎零成本（uniform + drawArrays）；Canvas2D 没有顶点缓冲，
     * 退化为按新原点重放 path（仍省掉 scene + geometry 两阶段）。
     */
    repaint(
        widthPx: number,
        heightPx: number,
        dpr: number,
        originXPx: number,
        originYPx: number,
    ): void;
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
    private readonly originLocation: WebGLUniformLocation;
    private readonly positionLocation: number;
    private readonly colorLocation: number;
    /** 已上传 GPU 的顶点数（`drawArrays` 的 count），供 `repaint` 复用。 */
    private uploadedVertexCount = 0;

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
            // 视口原点在几何的**窗口局部坐标系**中的位置。平移只改这一个
            // uniform，几何无需重建——这是平移帧成本降到 ~0 的关键。
            uniform vec2 u_viewOrigin;
            out vec4 v_color;
            void main() {
                vec2 screen = a_position - u_viewOrigin;
                vec2 zeroToOne = screen / u_resolution;
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
        const originLocation = gl.getUniformLocation(program, "u_viewOrigin");
        if (!originLocation) throw new Error("Waveform view-origin uniform is missing");
        this.program = program;
        this.buffer = buffer;
        this.resolutionLocation = resolutionLocation;
        this.originLocation = originLocation;
        this.positionLocation = gl.getAttribLocation(program, "a_position");
        this.colorLocation = gl.getAttribLocation(program, "a_color");
    }

    render(
        geometry: WaveformGeometry,
        widthPx: number,
        heightPx: number,
        dpr: number,
        originXPx: number,
        originYPx: number,
    ): void {
        const gl = this.gl;
        const target = rasterize(this.canvas, widthPx, heightPx, dpr);
        gl.viewport(0, 0, target.physicalWidth, target.physicalHeight);
        gl.clearColor(0, 0, 0, 0);
        gl.clear(gl.COLOR_BUFFER_BIT);
        gl.useProgram(this.program);
        gl.uniform2f(this.originLocation, originXPx, originYPx);
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
        this.uploadedVertexCount = quads.length / 6;
        gl.drawArrays(gl.TRIANGLES, 0, this.uploadedVertexCount);
    }

    /**
     * 仅平移：复用 GPU 上已有的顶点，只更新视口原点 uniform 后重绘。
     *
     * 流程：光栅化（尺寸通常未变，`rasterize` 内部按值比较后跳过写回）→
     * 清屏 → 更新 u_viewOrigin → drawArrays。**不触碰顶点缓冲**，因此平移帧
     * 的 CPU 成本与 clip 数、像素列数完全无关。
     */
    repaint(
        widthPx: number,
        heightPx: number,
        dpr: number,
        originXPx: number,
        originYPx: number,
    ): void {
        const gl = this.gl;
        const target = rasterize(this.canvas, widthPx, heightPx, dpr);
        gl.viewport(0, 0, target.physicalWidth, target.physicalHeight);
        gl.clearColor(0, 0, 0, 0);
        gl.clear(gl.COLOR_BUFFER_BIT);
        gl.useProgram(this.program);
        gl.uniform2f(this.originLocation, originXPx, originYPx);
        gl.uniform2f(this.resolutionLocation, target.resolutionWidth, target.resolutionHeight);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
        const stride = 6 * Float32Array.BYTES_PER_ELEMENT;
        gl.enableVertexAttribArray(this.positionLocation);
        gl.vertexAttribPointer(this.positionLocation, 2, gl.FLOAT, false, stride, 0);
        gl.enableVertexAttribArray(this.colorLocation);
        gl.vertexAttribPointer(
            this.colorLocation,
            4,
            gl.FLOAT,
            false,
            stride,
            2 * Float32Array.BYTES_PER_ELEMENT,
        );
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
        gl.drawArrays(gl.TRIANGLES, 0, this.uploadedVertexCount);
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
    /**
     * 最近一次 `render()` 的顶点引用，供 `repaint()` 重放。
     *
     * 生命周期约束：顶点**借用** `WaveformSurface` 的缓冲槽，只有在其未被
     * 覆写时才有效。调用方保证 `repaint()` 与上一次 `render()` 之间不会重建
     * 几何（那会覆写缓冲），因此这里持有的引用始终有效。
     */
    private lastVertices: Float32Array | null = null;

    constructor(canvas: HTMLCanvasElement) {
        this.canvas = canvas;
    }

    render(
        geometry: WaveformGeometry,
        widthPx: number,
        heightPx: number,
        dpr: number,
        originXPx: number,
        originYPx: number,
    ): void {
        this.lastVertices = geometry.vertices;
        this.paint(widthPx, heightPx, dpr, originXPx, originYPx);
    }

    /**
     * 仅平移：按新视口原点重放已缓存的顶点。
     *
     * Canvas2D 没有顶点缓冲，无法像 WebGL 那样只改一个 uniform，因此这里仍
     * 要重放全部 path——但**省掉了 scene 与 geometry 两阶段**（实测合计约
     * 0.51 ms / 帧，占满帧 0.76 ms 的 2/3）。
     */
    repaint(
        widthPx: number,
        heightPx: number,
        dpr: number,
        originXPx: number,
        originYPx: number,
    ): void {
        this.paint(widthPx, heightPx, dpr, originXPx, originYPx);
    }

    private paint(
        widthPx: number,
        heightPx: number,
        dpr: number,
        originXPx: number,
        originYPx: number,
    ): void {
        const vertices = this.lastVertices;
        if (vertices === null) return;
        const target = rasterize(this.canvas, widthPx, heightPx, dpr);
        rasterize(this.backCanvas, widthPx, heightPx, dpr);
        const back = this.backCanvas.getContext("2d");
        const visible = this.canvas.getContext("2d");
        if (!back || !visible) throw new Error("Canvas 2D is unavailable");

        // 顶点是窗口局部坐标，视口原点即平移量（物理像素 = 局部差 × dpr）。
        back.setTransform(
            target.dpr,
            0,
            0,
            target.dpr,
            -originXPx * target.dpr,
            -originYPx * target.dpr,
        );
        // 全物理清屏：否则 back 画布底部残留行会被 drawImage 带到可见画布。
        clearCanvasPhysical(back, target);
        back.lineWidth = 1;
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
