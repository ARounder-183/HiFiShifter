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
 * 1. 实例布局（25 float/实例，偏移见 OFF_* 常量）与既有 GL 渲染器一致；着色器以既有
 *    实现为蓝本，但**已修正一处 GLSL ES 3.00 违规**：顶点输入不能声明为数组
 *    （`in float i_rect[4]`），改为 `in vec4 i_rect` 打包——macOS 的 ANGLE/Metal 后端
 *    会直接拒绝编译（`cannot declare arrays of this qualifier`），使整个 program
 *    创建失败。既有实现存在同一问题（失败后被上层 catch 静默回退 Canvas2D），
 *    阶段 1 合并为单一来源时须一并修正。
 * 2. 滚动帧只调用 `repaint()`：实例缓冲**不重新上传**，只更新 `u_viewOrigin`
 *    uniform——这是"滚动零重绘"在 GL 层的落点。
 * 3. `u_resolution` 用光栅化目标回算的绘制坐标系尺寸（见 glRaster），保证坐标与
 *    物理像素 1:1。
 * 4. 混合状态由本 program 在每次 draw 前设置（预乘 alpha + `ONE, ONE_MINUS_SRC_ALPHA`）：
 *    WebGL 的 BLEND 默认关闭，不设置会让半透明色块变成实色、层叠顺序失效。
 */

import type { GlRasterTarget } from "./glRaster";
import { resolveBufferFloats } from "./instanceBuffer";
import { CLIP_INSTANCE_FLOATS, CLIP_INSTANCE_OFFSETS } from "./instanceLayout";

/** 字段偏移：单一来源见 `instanceLayout`（该文件声明了与既有实现的一致性约束）。 */
const OFF = CLIP_INSTANCE_OFFSETS;

/** 单实例字节步长。 */
const INSTANCE_STRIDE_BYTES = CLIP_INSTANCE_FLOATS * 4;

// ── 着色器 ───────────────────────────────────────────────────────────
// 顶点：单位四边形按实例矩形展开（含描边与分隔缝的外扩余量）。
// 片元：圆角盒 SDF + 分区着色。
//
// 为什么要留 `u_pad`：描边以路径为中心向两侧各扩 lineWidth/2，因此外边界会
// 比 clip 矩形大 lineWidth/2；不预留就会把描边裁掉。
//
// 【与 runtime/timelineClipGlRenderer 逐字一致，见文件头约束 1】

const VERTEX_SHADER = `#version 300 es
in vec2 a_unit;          // 单位四边形 [0,1]×[0,1]
in vec4 i_rect;          // x, y, w, h（打包 vec4）
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
    // 平面矩形不外扩（它就是精确的矩形）；圆角盒才需要为描边预留边界。
    float pad = i_mode > 0.5 ? 0.0 : max(i_borderWidth * 0.5, 1.0);
    vec2 center = vec2(i_rect.x + i_rect.z * 0.5, i_rect.y + i_rect.w * 0.5);
    vec2 halfSize = vec2(i_rect.z * 0.5 + pad, i_rect.w * 0.5 + pad);
    vec2 pos = center + (a_unit - 0.5) * 2.0 * halfSize;

    vec2 screen = pos - u_viewOrigin;
    vec2 zeroToOne = screen / u_resolution;
    gl_Position = vec4(zeroToOne.x * 2.0 - 1.0, -(zeroToOne.y * 2.0 - 1.0), 0.0, 1.0);

    v_local = pos - vec2(i_rect.x, i_rect.y);
    v_half = vec2(i_rect.z * 0.5, i_rect.w * 0.5);
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

// 圆角盒 SDF：返回到边界的有符号距离（内部为负）。
float roundedBoxSdf(vec2 p, vec2 b, float r) {
    vec2 q = abs(p) - b + vec2(r);
    return min(max(q.x, q.y), 0.0) + length(max(q, 0.0)) - r;
}

void main() {
    // 平面矩形：直接输出颜色（四边形本身就是精确矩形，无需 SDF / 分区）。
    if (v_mode > 0.5) {
        // **预乘输出**：blend 用的是 (ONE, ONE_MINUS_SRC_ALPHA)，rgb 必须已乘 alpha。
        // 输出非预乘颜色时混合式退化为 rgb + dst*(1-a)：alpha 完全失效，
        // 0.1 的白色网格线会在深色背景上直接变成纯白（"网格线太亮"的根因）。
        outColor = vec4(v_borderColor.rgb * v_borderColor.a, v_borderColor.a);
        return;
    }

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

    // 描边叠加：先按 source-over 在**非预乘**空间求出结果色，再统一转预乘输出
    // （见 FLAT 分支的说明；不转会让半透明描边与块面在深色背景上过亮）。
    vec4 result = vec4(mix(base.rgb, v_borderColor.rgb, inBorder * v_borderColor.a),
                       max(base.a, inBorder * v_borderColor.a));
    outColor = vec4(result.rgb * result.a, result.a);
}`;

/** 单位四边形（两个三角形，6 顶点）。 */
const UNIT_QUAD = new Float32Array([0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1]);

/** 编译一个着色器；失败时抛错（调用方捕获后回退）。 */
function compile(gl: WebGL2RenderingContext, type: number, source: string): WebGLShader {
    const shader = gl.createShader(type);
    if (!shader) {
        // createShader 返回 null 几乎只发生在上下文已丢失时（规范允许的失败分支）。
        throw new Error(
            gl.isContextLost()
                ? "WebGL 上下文已丢失（GPU 进程异常或资源耗尽），请刷新页面"
                : "无法创建着色器对象（GPU 资源不足）",
        );
    }
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
 * 特殊说明：
 * - 属性指针在创建时一次性绑定（VAO 记录状态），绘制时只需 `bindVertexArray` +
 *   `bufferSubData`（实例数据）+ `drawArraysInstanced`。
 * - **属性 location 缺失直接抛错**（与既有渲染器一致）：编译器把属性优化掉时静默
 *   继续会让该属性读到常量 0（例如 `i_mode` 丢失 → 网格线被画成圆角盒），
 *   产出的画面错误在代码里搜不到根因。
 * - 创建失败路径释放已创建对象，避免重试累积显存泄漏。
 *
 * @param gl 内核 WebGL2 上下文。
 * @returns program 句柄；创建失败时抛错（调用方捕获后回退 Canvas2D）。
 */
export function createSdfBoxProgram(gl: WebGL2RenderingContext): SdfBoxProgram {
    let vertex: WebGLShader | null = null;
    let fragment: WebGLShader | null = null;
    let program: WebGLProgram | null = null;
    let vao: WebGLVertexArrayObject | null = null;
    let unitBuffer: WebGLBuffer | null = null;
    let instanceBuffer: WebGLBuffer | null = null;
    let instanceCapacityFloats = 0;
    let uploadedCount = 0;

    try {
        vertex = compile(gl, gl.VERTEX_SHADER, VERTEX_SHADER);
        fragment = compile(gl, gl.FRAGMENT_SHADER, FRAGMENT_SHADER);
        program = gl.createProgram();
        if (!program) throw new Error("Unable to create sdf-box program");
        gl.attachShader(program, vertex);
        gl.attachShader(program, fragment);
        gl.linkProgram(program);
        if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
            throw new Error(gl.getProgramInfoLog(program) ?? "Unknown sdf-box link error");
        }
        // 链接完成后着色器即可释放（program 已持有编译结果）。
        gl.deleteShader(vertex);
        gl.deleteShader(fragment);
        vertex = null;
        fragment = null;

        const resolutionLocation = gl.getUniformLocation(program, "u_resolution");
        const originLocation = gl.getUniformLocation(program, "u_viewOrigin");
        if (!resolutionLocation || !originLocation) {
            throw new Error("sdf-box uniforms are missing");
        }

        vao = gl.createVertexArray();
        unitBuffer = gl.createBuffer();
        instanceBuffer = gl.createBuffer();
        if (!vao || !unitBuffer || !instanceBuffer) {
            throw new Error("Unable to create sdf-box buffers");
        }

        /** 绑定实例属性（size 个连续 location）；缺失即抛错。 */
        function bindInstanceAttrib(name: string, size: number, offsetFloats: number): void {
            const location = gl!.getAttribLocation(program!, name);
            if (location < 0) throw new Error(`sdf-box attribute is missing: ${name}`);
            gl!.enableVertexAttribArray(location);
            gl!.vertexAttribPointer(
                location,
                size,
                gl!.FLOAT,
                false,
                INSTANCE_STRIDE_BYTES,
                offsetFloats * 4,
            );
            gl!.vertexAttribDivisor(location, 1);
        }

        gl.bindVertexArray(vao);

        gl.bindBuffer(gl.ARRAY_BUFFER, unitBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, UNIT_QUAD, gl.STATIC_DRAW);
        const unitLocation = gl.getAttribLocation(program, "a_unit");
        if (unitLocation < 0) throw new Error("sdf-box attribute is missing: a_unit");
        gl.enableVertexAttribArray(unitLocation);
        gl.vertexAttribPointer(unitLocation, 2, gl.FLOAT, false, 0, 0);
        // 单位四边形是逐顶点属性（divisor 0），其余都是逐实例。
        gl.vertexAttribDivisor(unitLocation, 0);

        gl.bindBuffer(gl.ARRAY_BUFFER, instanceBuffer);
        bindInstanceAttrib("i_rect", 4, OFF.rect);
        bindInstanceAttrib("i_radius", 1, OFF.radius);
        bindInstanceAttrib("i_headerH", 1, OFF.headerH);
        bindInstanceAttrib("i_bodyColor", 4, OFF.bodyRgba);
        bindInstanceAttrib("i_headerColor", 4, OFF.headerRgba);
        bindInstanceAttrib("i_borderColor", 4, OFF.borderRgba);
        bindInstanceAttrib("i_borderWidth", 1, OFF.borderWidth);
        bindInstanceAttrib("i_overlapPx", 1, OFF.overlapPx);
        bindInstanceAttrib("i_seamW", 1, OFF.seam);
        bindInstanceAttrib("i_seamColor", 3, OFF.seamRgb);
        bindInstanceAttrib("i_mode", 1, OFF.mode);

        gl.bindVertexArray(null);

        /**
         * 设置绘制状态（program / VAO / uniform / 混合）。
         *
         * 特殊说明：混合必须在此显式开启——片元输出预乘色，BLEND 默认关闭会让
         * 半透明块面变成实色、并破坏"实例先后即层叠顺序"的约定。
         */
        function prepareDraw(
            target: GlRasterTarget,
            viewOriginX: number,
            viewOriginY: number,
        ): void {
            gl!.useProgram(program!);
            gl!.bindVertexArray(vao);
            gl!.enable(gl!.BLEND);
            gl!.blendFunc(gl!.ONE, gl!.ONE_MINUS_SRC_ALPHA);
            gl!.uniform2f(resolutionLocation, target.cssWidthPx, target.cssHeightPx);
            gl!.uniform2f(originLocation, viewOriginX, viewOriginY);
        }

        const handle: SdfBoxProgram = {
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
        return handle;
    } catch (error) {
        // 失败路径：逆序释放已创建对象，避免重试时累积显存泄漏。
        if (vertex) gl.deleteShader(vertex);
        if (fragment) gl.deleteShader(fragment);
        if (unitBuffer) gl.deleteBuffer(unitBuffer);
        if (instanceBuffer) gl.deleteBuffer(instanceBuffer);
        if (vao) gl.deleteVertexArray(vao);
        if (program) gl.deleteProgram(program);
        throw error;
    }
}
