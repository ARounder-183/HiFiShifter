/**
 * clip 体的 WebGL2 实例化渲染器（P3）。
 *
 * 【主要内容】把 clip 的**主体块面**（header / body / 前导重叠半透区 /
 * 分隔线 / 相邻分隔缝 / 描边）用一个**实例化**的 SDF 圆角盒画出来：
 * 每个 clip 一条实例数据，整帧一次 `drawArraysInstanced`。
 *
 * 【作用】`drawTimelineCanvas` 的 Canvas2D 路径即便已经合批（见
 * `ac303963`），每帧仍要发几十次 path 填充与描边，且圆角、描边、分隔缝都要
 * 靠 CPU 生成几何。实例化后：
 * - CPU 侧只填一条条实例数据（纯 Float32Array 写入，无 path 构建）；
 * - GPU 侧一次 draw call 完成全部 clip 的块面；
 * - 平移可直接复用已上传的实例缓冲，只更新视口原点（与波形 P2c 同机制）。
 *
 * 【覆盖范围】**只画块面**。旋钮 / 徽标 / 文字 / 淡变曲线 / 吸附三角仍走
 * Canvas2D 细节层（它们是逐 clip 的、且数量受尺寸门控，不是瓶颈）。
 *
 * 【与其他模块的关系】
 * - 上游：`timelineCanvasRenderer.drawTimelineCanvas` 在 GL 模式开启时调用
 *   `buildClipBodyInstance()`，并把实例交给本渲染器。
 * - 横向：坐标与 `TimelineCanvasViewport` 一致——内容绝对坐标，视口位移由
 *   `originXPx / originYPx` 给出（与波形 P2c 的 `u_viewOrigin` 同约定）。
 * - 开关：dev-only，默认关闭，见 `PERF_GL_CLIP_BODIES_KEY`。
 *
 * 【为什么默认关闭】这是一次**视觉**重写（着色器重画圆角、描边、分隔缝），
 * 离线无法验证外观；默认走既有 Canvas2D 路径，由真机 A/B 确认后再切换。
 */

import { rasterize } from "./canvasRaster.js";

/**
 * 开关 key：显式置为 "0" 时关闭 GL clip 体，**其余任何值（含未设置）都开启**。
 *
 * P3 起默认走 GL；这个 key 是给真机出问题时**一键退回**用的逃生门，而不是
 * 开关。需要关闭时在控制台执行：
 *
 *     localStorage.setItem("hifishifter.glClipBodies", "0"); location.reload();
 */
export const PERF_GL_CLIP_BODIES_KEY = "hifishifter.glClipBodies";

/** 是否启用 GL clip 体（默认开启；显式写 "0" 才关闭）。 */
export function isGlClipBodiesEnabled(): boolean {
    try {
        return localStorage.getItem(PERF_GL_CLIP_BODIES_KEY) !== "0";
    } catch {
        // 读不到 localStorage（如隐私模式）时按默认行为走 GL；若 GL 本身
        // 不可用，TimelineCanvasViewport 会自行退回 Canvas2D。
        return true;
    }
}

// ── 实例数据布局 ─────────────────────────────────────────────────────
// 每实例 24 个 float。之所以存 RGBA 而不是字符串，是为了让 CPU 侧只做纯数值
// 写入——颜色解析在 `buildClipBodyInstance` 里一次性完成。

/**
 * GL 块面渲染器需要的最小接口。
 *
 * 用结构化接口而非直接引用 `GlClipBodyRenderer` 类型，是为了让调用方可以
 * 传入自己的实现（例如测试里的 mock）。
 */
export interface GlClipBodySink {
    render(
        instances: Float32Array,
        instanceCount: number,
        widthPx: number,
        heightPx: number,
        dpr: number,
        originXPx: number,
        originYPx: number,
    ): void;
}

/** 单实例的 float 个数。 */
export const CLIP_INSTANCE_FLOATS = 24;

const OFF_X = 0;
const OFF_Y = 1;
const OFF_W = 2;
const OFF_H = 3;
const OFF_RADIUS = 4;
const OFF_HEADER_H = 5;
const OFF_BODY_RGBA = 6; // 6..9
const OFF_HEADER_RGBA = 10; // 10..13
const OFF_BORDER_RGBA = 14; // 14..17
const OFF_BORDER_WIDTH = 18;
const OFF_OVERLAP_PX = 19;
const OFF_SEAM = 20; // x 分量：缝宽（0 = 无）
const OFF_SEAM_RGB = 21; // 21..23

/**
 * 解析 `rgba(r, g, b, a)` 为 0..1 的四个分量。
 *
 * 只需要覆盖 `timelineCanvasStyle` 实际产出的那一种格式（它统一用
 * `rgba(...)` 生成 header/body/描边/分隔缝色）；解析不出来时回退到不透明的
 * 洋红——故意用刺眼的颜色，让"漏解析"在真机上立刻可见而不是悄悄变透明。
 *
 * @param css 颜色字符串。
 * @returns 归一化后的 [r, g, b, a]。
 */
export function parseRgbaColor(css: string): [number, number, number, number] {
    const match =
        /rgba?\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)(?:\s*,\s*([-\d.]+))?\s*\)/.exec(css);
    if (!match) return [1, 0, 1, 1];
    const clamp01 = (value: number): number => Math.min(1, Math.max(0, value));
    return [
        clamp01(Number(match[1]) / 255),
        clamp01(Number(match[2]) / 255),
        clamp01(Number(match[3]) / 255),
        clamp01(match[4] == null ? 1 : Number(match[4])),
    ];
}

/** `buildTimelineClipVisualStyle` 的返回类型（避免引入额外依赖）。 */
export interface ClipVisualStyleLike {
    headerFill: string;
    bodyFill: string;
    borderStroke: string;
    borderLineWidth: number;
    mutedAlpha: number;
}

/** `TimelineCanvasClipModel` 的最小子集（本模块只用到这些字段）。 */
export interface ClipModelLike {
    leftPx: number;
    topPx: number;
    widthPx: number;
    heightPx: number;
    headerHeightPx: number;
    leadingOverlapPx?: number;
}

/**
 * 把一个 clip 的块面写进实例缓冲。
 *
 * 流程：解析三种颜色 → 计算收敛后的圆角与 header 高度 → 按布局写入。
 *
 * 特殊说明（**本函数与 Canvas2D 路径的等价边界**）：
 * - 圆角：`CLIP_CORNER_RADIUS_PX` 按 clip 尺寸收敛，与 Canvas2D 路径同一个
 *   公式；着色器用统一的四角半径，而 Canvas2D 路径对 header/body 分别只圆
 *   上/下两角——由于 header 下边与 body 上边都在弧线之外，两者逐像素等价。
 * - 分隔缝：Canvas2D 路径只画可见的 0.5px（右半边会被下一个 clip 覆盖）；
 *   这里同样只画 0.5px，位置一致。
 * - 编组激活（外圈描边伸出矩形 2px）**不走 GL**，由调用方排除——外圈会与
 *   邻居重叠着色，超出本渲染器的单矩形模型。
 *
 * @param out 目标 Float32Array（容量需 >= (index + 1) * CLIP_INSTANCE_FLOATS）。
 * @param index 实例序号。
 * @param clip clip 几何。
 * @param style clip 视觉样式。
 * @param seamColorCss 相邻分隔缝颜色（泳道底色）；null 表示该 clip 无分隔缝。
 */
export function buildClipBodyInstance(
    out: Float32Array,
    index: number,
    clip: ClipModelLike,
    style: ClipVisualStyleLike,
    seamColorCss: string | null,
): void {
    const base = index * CLIP_INSTANCE_FLOATS;
    const w = Math.max(1, clip.widthPx);
    const h = Math.max(1, clip.heightPx);
    const headerH = Math.max(1, Math.min(clip.heightPx, clip.headerHeightPx));
    const radius = Math.max(0, Math.min(CLIP_CORNER_RADIUS_PX_GL, w / 2, h / 2));
    const overlapPx = Math.max(0, Math.min(w - 1, clip.leadingOverlapPx ?? 0));
    const alpha = style.mutedAlpha;

    const body = parseRgbaColor(style.bodyFill);
    const header = parseRgbaColor(style.headerFill);
    const border = parseRgbaColor(style.borderStroke);

    out[base + OFF_X] = clip.leftPx;
    out[base + OFF_Y] = clip.topPx;
    out[base + OFF_W] = w;
    out[base + OFF_H] = h;
    out[base + OFF_RADIUS] = radius;
    out[base + OFF_HEADER_H] = headerH;

    out[base + OFF_BODY_RGBA] = body[0];
    out[base + OFF_BODY_RGBA + 1] = body[1];
    out[base + OFF_BODY_RGBA + 2] = body[2];
    out[base + OFF_BODY_RGBA + 3] = body[3] * alpha;

    out[base + OFF_HEADER_RGBA] = header[0];
    out[base + OFF_HEADER_RGBA + 1] = header[1];
    out[base + OFF_HEADER_RGBA + 2] = header[2];
    out[base + OFF_HEADER_RGBA + 3] = header[3] * alpha;

    out[base + OFF_BORDER_RGBA] = border[0];
    out[base + OFF_BORDER_RGBA + 1] = border[1];
    out[base + OFF_BORDER_RGBA + 2] = border[2];
    out[base + OFF_BORDER_RGBA + 3] = border[3] * alpha;

    out[base + OFF_BORDER_WIDTH] = style.borderLineWidth;
    // 前导重叠区：>0.5 时该段按 0.55 倍 alpha 绘制（与 Canvas2D 一致）。
    out[base + OFF_OVERLAP_PX] = overlapPx > 0.5 ? overlapPx : 0;

    if (seamColorCss === null) {
        out[base + OFF_SEAM] = 0;
        out[base + OFF_SEAM_RGB] = 0;
        out[base + OFF_SEAM_RGB + 1] = 0;
        out[base + OFF_SEAM_RGB + 2] = 0;
    } else {
        const seam = parseRgbaColor(seamColorCss);
        out[base + OFF_SEAM] = SEAM_WIDTH_PX;
        out[base + OFF_SEAM_RGB] = seam[0];
        out[base + OFF_SEAM_RGB + 1] = seam[1];
        out[base + OFF_SEAM_RGB + 2] = seam[2];
    }
}

/** clip 圆角半径（CSS 像素），与 `timelineCanvasStyle.CLIP_CORNER_RADIUS_PX` 同值。 */
const CLIP_CORNER_RADIUS_PX_GL = 1.5;

/** 相邻分隔缝宽度（CSS 像素）：只画可见的左半边，详见模块头注释。 */
const SEAM_WIDTH_PX = 0.5;

// ── 着色器 ───────────────────────────────────────────────────────────
// 顶点：单位四边形按实例矩形展开（含描边与分隔缝的外扩余量）。
// 片元：圆角盒 SDF + 分区着色。
//
// 为什么要留 `u_pad`：描边以路径为中心向两侧各扩 lineWidth/2，因此外边界会
// 比 clip 矩形大 lineWidth/2；不预留就会把描边裁掉。

const VERTEX_SHADER = `#version 300 es
in vec2 a_unit;          // 单位四边形 [0,1]×[0,1]
in float i_rect[4];      // x, y, w, h
in float i_radius;
in float i_headerH;
in vec4 i_bodyColor;
in vec4 i_headerColor;
in vec4 i_borderColor;
in float i_borderWidth;
in float i_overlapPx;
in float i_seamW;
in vec3 i_seamColor;

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

void main() {
    float pad = max(i_borderWidth * 0.5, 1.0);
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

out vec4 outColor;

// 圆角盒 SDF：返回到边界的有符号距离（内部为负）。
float roundedBoxSdf(vec2 p, vec2 b, float r) {
    vec2 q = abs(p) - b + vec2(r);
    return min(max(q.x, q.y), 0.0) + length(max(q, 0.0)) - r;
}

void main() {
    vec2 p = v_local - v_half;
    float sdf = roundedBoxSdf(p, v_half, v_radius);

    // 描边：距边界 borderWidth 之内的环带。
    float halfBorder = v_borderWidth * 0.5;
    float inBorder = 1.0 - smoothstep(halfBorder - 0.5, halfBorder + 0.5, abs(sdf + halfBorder));
    if (sdf > halfBorder) discard;

    // 基础色：header 区（y < headerH）用 header 色，其余用 body 色。
    vec4 base = v_local.y < v_headerH ? v_headerColor : v_bodyColor;

    // 前导重叠区：按 0.55 倍 alpha 变淡（与 Canvas2D 路径一致）。
    if (v_overlapPx > 0.0 && v_local.x < v_overlapPx) {
        base.a *= 0.55;
    }

    // header/body 分隔线：header 底边处 1px 的半透明黑。
    float sep = 1.0 - smoothstep(0.0, 1.0, abs(v_local.y - v_headerH) - 0.5);
    base.rgb = mix(base.rgb, vec3(0.0), sep * 0.14 * step(v_local.y, v_headerH + 1.0));

    // 相邻分隔缝：右缘内侧 0.5px 的泳道底色。
    if (v_seamW > 0.0) {
        float seamRight = v_half.x * 2.0 - 0.5;
        float seam = step(seamRight - v_seamW, v_local.x) * step(v_local.x, seamRight);
        base.rgb = mix(base.rgb, v_seamColor, seam);
    }

    // 描边叠加（预乘 alpha 的混合方式与 Canvas2D 的 source-over 对齐）。
    outColor = vec4(mix(base.rgb, v_borderColor.rgb, inBorder * v_borderColor.a),
                    max(base.a, inBorder * v_borderColor.a));
}`;

function compile(gl: WebGL2RenderingContext, type: number, source: string): WebGLShader {
    const shader = gl.createShader(type);
    if (!shader) throw new Error("Unable to create clip body shader");
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        const message = gl.getShaderInfoLog(shader) ?? "Unknown clip body shader error";
        gl.deleteShader(shader);
        throw new Error(message);
    }
    return shader;
}

/**
 * clip 体的 WebGL2 实例化渲染器。
 *
 * 用法：`render()` 上传实例并绘制；之后若只有视口变化，用 `repaint()` 复用
 * 已上传的实例缓冲（与波形 P2c 同机制）。
 */
export class GlClipBodyRenderer {
    private readonly canvas: HTMLCanvasElement;
    private readonly gl: WebGL2RenderingContext;
    private readonly program: WebGLProgram;
    private readonly vao: WebGLVertexArrayObject;
    private readonly unitBuffer: WebGLBuffer;
    private readonly instanceBuffer: WebGLBuffer;
    private readonly resolutionLocation: WebGLUniformLocation;
    private readonly originLocation: WebGLUniformLocation;
    private instanceCapacity = 0;
    private uploadedInstanceCount = 0;

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

        const vertex = compile(gl, gl.VERTEX_SHADER, VERTEX_SHADER);
        const fragment = compile(gl, gl.FRAGMENT_SHADER, FRAGMENT_SHADER);
        const program = gl.createProgram();
        if (!program) throw new Error("Unable to create clip body program");
        gl.attachShader(program, vertex);
        gl.attachShader(program, fragment);
        gl.linkProgram(program);
        gl.deleteShader(vertex);
        gl.deleteShader(fragment);
        if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
            const message = gl.getProgramInfoLog(program) ?? "Unknown clip body link error";
            gl.deleteProgram(program);
            throw new Error(message);
        }
        this.program = program;

        const resolutionLocation = gl.getUniformLocation(program, "u_resolution");
        const originLocation = gl.getUniformLocation(program, "u_viewOrigin");
        if (!resolutionLocation || !originLocation) {
            throw new Error("Clip body uniforms are missing");
        }
        this.resolutionLocation = resolutionLocation;
        this.originLocation = originLocation;

        const vao = gl.createVertexArray();
        const unitBuffer = gl.createBuffer();
        const instanceBuffer = gl.createBuffer();
        if (!vao || !unitBuffer || !instanceBuffer) {
            throw new Error("Unable to allocate clip body buffers");
        }
        this.vao = vao;
        this.unitBuffer = unitBuffer;
        this.instanceBuffer = instanceBuffer;

        gl.bindVertexArray(vao);

        // 单位四边形（两个三角形）。
        const unit = new Float32Array([0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1]);
        gl.bindBuffer(gl.ARRAY_BUFFER, unitBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, unit, gl.STATIC_DRAW);
        const unitLoc = gl.getAttribLocation(program, "a_unit");
        gl.enableVertexAttribArray(unitLoc);
        gl.vertexAttribPointer(unitLoc, 2, gl.FLOAT, false, 0, 0);

        // 实例属性：整个缓冲作为一个交错数组，逐属性设置 divisor。
        gl.bindBuffer(gl.ARRAY_BUFFER, instanceBuffer);
        const stride = CLIP_INSTANCE_FLOATS * Float32Array.BYTES_PER_ELEMENT;
        const setup = (name: string, size: number, offsetFloats: number): void => {
            const loc = gl.getAttribLocation(program, name);
            if (loc < 0) throw new Error(`Clip body attribute missing: ${name}`);
            gl.enableVertexAttribArray(loc);
            gl.vertexAttribPointer(
                loc,
                size,
                gl.FLOAT,
                false,
                stride,
                offsetFloats * Float32Array.BYTES_PER_ELEMENT,
            );
            gl.vertexAttribDivisor(loc, 1);
        };
        setup("i_rect", 4, OFF_X);
        setup("i_radius", 1, OFF_RADIUS);
        setup("i_headerH", 1, OFF_HEADER_H);
        setup("i_bodyColor", 4, OFF_BODY_RGBA);
        setup("i_headerColor", 4, OFF_HEADER_RGBA);
        setup("i_borderColor", 4, OFF_BORDER_RGBA);
        setup("i_borderWidth", 1, OFF_BORDER_WIDTH);
        setup("i_overlapPx", 1, OFF_OVERLAP_PX);
        setup("i_seamW", 1, OFF_SEAM);
        setup("i_seamColor", 3, OFF_SEAM_RGB);

        gl.bindVertexArray(null);
    }

    /**
     * 上传实例数据并按给定视口原点绘制。
     *
     * @param instances 交错排列的实例数据（长度 = 实例数 × CLIP_INSTANCE_FLOATS）。
     * @param instanceCount 实例数（可能小于 `instances` 的容量）。
     */
    render(
        instances: Float32Array,
        instanceCount: number,
        widthPx: number,
        heightPx: number,
        dpr: number,
        originXPx: number,
        originYPx: number,
    ): void {
        const gl = this.gl;
        const target = rasterize(this.canvas, widthPx, heightPx, dpr);

        gl.bindVertexArray(this.vao);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceBuffer);
        if (instanceCount > this.instanceCapacity) {
            this.instanceCapacity = Math.max(64, instanceCount * 2);
            gl.bufferData(
                gl.ARRAY_BUFFER,
                this.instanceCapacity * CLIP_INSTANCE_FLOATS * Float32Array.BYTES_PER_ELEMENT,
                gl.DYNAMIC_DRAW,
            );
        }
        gl.bufferSubData(gl.ARRAY_BUFFER, 0, instances, 0, instanceCount * CLIP_INSTANCE_FLOATS);
        this.uploadedInstanceCount = instanceCount;

        this.draw(target, originXPx, originYPx);
    }

    /** 仅平移：复用已上传的实例缓冲。 */
    repaint(
        widthPx: number,
        heightPx: number,
        dpr: number,
        originXPx: number,
        originYPx: number,
    ): void {
        const gl = this.gl;
        const target = rasterize(this.canvas, widthPx, heightPx, dpr);
        gl.bindVertexArray(this.vao);
        this.draw(target, originXPx, originYPx);
    }

    private draw(
        target: {
            physicalWidth: number;
            physicalHeight: number;
            resolutionWidth: number;
            resolutionHeight: number;
        },
        originXPx: number,
        originYPx: number,
    ): void {
        const gl = this.gl;
        gl.viewport(0, 0, target.physicalWidth, target.physicalHeight);
        gl.clearColor(0, 0, 0, 0);
        gl.clear(gl.COLOR_BUFFER_BIT);
        gl.useProgram(this.program);
        gl.uniform2f(this.resolutionLocation, target.resolutionWidth, target.resolutionHeight);
        gl.uniform2f(this.originLocation, originXPx, originYPx);
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
        gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, Math.max(0, this.uploadedInstanceCount));
        gl.bindVertexArray(null);
    }

    dispose(): void {
        const gl = this.gl;
        gl.deleteBuffer(this.unitBuffer);
        gl.deleteBuffer(this.instanceBuffer);
        gl.deleteVertexArray(this.vao);
        gl.deleteProgram(this.program);
    }
}
