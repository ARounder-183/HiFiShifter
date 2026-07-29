<!--
文件说明：波形渲染 WebGL2 化重构 实施计划
主要内容：按设计文档 docs/plans/2026-07-29-waveform-webgl2-renderer.md 分阶段实施 WebGL2 波形渲染器，
         包含接口抽象、WebGL2 实现、Canvas 2D fallback、两个消费方集成、收尾。
依赖文档：docs/plans/2026-07-29-waveform-webgl2-renderer.md（设计文档）
影响模块：见设计文档附录 A
-->

# 波形渲染 WebGL2 化重构 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将波形渲染从 Canvas 2D per-pixel CPU 循环升级为 WebGL2 instanced quad GPU 渲染，保留 Canvas 2D 作为自动降级 fallback。

**Architecture:** 新增 `WaveformRenderer` 抽象接口与工厂函数，运行时检测 WebGL2 能力自动选择实现；两个消费方（`WaveformTrackCanvas`、`PianoRoll` 背景波形）统一通过 renderer 接口消费；PianoRoll 采用方案 B，新增独立背景波形 canvas，主 canvas 保持 Canvas 2D 画参数曲线。

**Tech Stack:** TypeScript / WebGL2 (GLSL ES 3.00) / Canvas 2D (fallback) / React

**Spec:** `docs/plans/2026-07-29-waveform-webgl2-renderer.md`

---

## 项目约定

- **测试框架**：项目当前未集成 vitest/jest，验证手段以 TypeScript 类型检查 + Vite 构建 + 手动浏览器验证为主。每个任务结束后运行类型检查与构建确保无回归。
- **编码规范**（来自用户全局规则）：
  - 每个代码文件最上方需要足够注释说明文件主要内容、作用、与其他模块的关系；改动时同步维护此注释。
  - 每个关键函数头部需要注释说明流程、作用、特殊说明、参数说明；改动时同步维护此注释。
  - 优先工程化、可维护方向，必要合理拆分。
- **分支**：`feature/waveform-webgl2-renderer`（已从 `develop` 创建）。
- **commit 风格**：参考现有 `git log` 风格，使用 `feat:` / `refactor:` / `perf:` / `fix:` 前缀。
- **验证命令**：
  - 类型检查：`cd frontend && npx tsc --noEmit`
  - 构建：`cd frontend && npm run build`
  - 开发服务器：`cd frontend && npm run dev`

---

## 文件结构

| 文件 | 角色 | 变更类型 | 关联任务 |
|------|------|----------|----------|
| `frontend/src/utils/waveformRenderer.ts` | 接口定义 + Canvas 2D 实现 | 修改（追加导出，保留现有 `renderWaveform`/`applyGainsToPeaks`） | Task 1, 2 |
| `frontend/src/utils/waveformWebGL2Renderer.ts` | WebGL2 实现 | 新增 | Task 4-9 |
| `frontend/src/utils/waveformRendererFactory.ts` | 工厂 + 能力检测 | 新增 | Task 3 |
| `frontend/src/components/waveform/WaveformTrackCanvas.tsx` | 上方轨道区消费方 | 修改 | Task 10-13 |
| `frontend/src/components/layout/pianoRoll/render.ts` | PianoRoll 背景波形抽取 | 修改 | Task 15 |
| `frontend/src/components/layout/PianoRollPanel.tsx` | 新增背景波形 canvas + renderer | 修改 | Task 14, 16 |
| `frontend/src/utils/waveformDebug.ts` | 诊断标识 | 修改 | Task 17 |

---

## 阶段 1：基础设施（无 UI 改动）

### Task 1: 在 `waveformRenderer.ts` 追加 `WaveformRenderer` 接口

**Files:**
- Modify: `frontend/src/utils/waveformRenderer.ts`（在文件末尾追加）

- [ ] **Step 1: 追加接口与类型定义**

在 `frontend/src/utils/waveformRenderer.ts` 文件末尾追加以下内容（保留现有所有导出不变）：

```typescript
// ============================================================================
// WaveformRenderer 抽象接口（WebGL2 / Canvas 2D / 未来 WebGPU 共用）
// ============================================================================
//
// 设计目标：
//   - 消费方（WaveformTrackCanvas / PianoRoll 背景波形）通过此接口消费，不感知底层实现
//   - 运行时由工厂函数 waveformRendererFactory 根据浏览器能力选择实现
//   - 接口兼容 WebGPU，未来可平替实现
//
// 生命周期：
//   1. createWaveformRenderer(canvas) → renderer   工厂检测能力
//   2. renderer.resize(displayW, displayH, dpr)     尺寸变化时
//   3. renderer.clear()                             每帧开始清空
//   4. renderer.drawClipWaveform(params) × N        每可见 clip 调用一次
//   5. renderer.dispose()                           canvas 卸载时
// ============================================================================

/**
 * 单次 drawClipWaveform 调用参数
 *
 * 字段对应现有 WaveformRenderParams + 颜色 + alpha + 可视段裁剪
 */
export interface DrawClipWaveformParams {
    /** peaks 数据，interleaved [min0,max0,min1,max1,...]，已应用增益 */
    peaks: Float32Array;

    /** 渲染参数（与现有 WaveformRenderParams 完全一致） */
    renderParams: WaveformRenderParams;

    /** 可视段左边界（CSS 像素，相对 canvas 左上角） */
    segmentLeftPx: number;

    /** 可视段右边界（CSS 像素） */
    segmentRightPx: number;

    /** 描边颜色（CSS 颜色字符串，内部解析为 RGBA） */
    strokeColor: string;

    /** 描边宽度（CSS 像素） */
    strokeWidth: number;

    /** 整体透明度（用于 muted / leadingOverlap 混合） */
    alpha: number;
}

/**
 * 波形渲染器抽象接口
 *
 * 实现类：
 *   - WebGL2WaveformRenderer：WebGL2 instanced quad + RG32F 纹理
 *   - Canvas2DWaveformRenderer：封装现有 renderWaveform
 *   - （未来）WebGPUWaveformRenderer：预留接口兼容性
 */
export interface WaveformRenderer {
    /** 标识实现类型，用于诊断与日志 */
    readonly backend: "webgl2" | "canvas2d" | "webgpu";

    /**
     * 调整 canvas 物理尺寸
     *
     * @param displayW CSS 像素宽度
     * @param displayH CSS 像素高度
     * @param dpr 设备像素比
     */
    resize(displayW: number, displayH: number, dpr: number): void;

    /** 清空画布（每帧开始调用） */
    clear(): void;

    /**
     * 绘制单个 clip 的一个可视段
     *
     * 流程：
     *   1. 上传 peaks 数据到 GPU（WebGL2 路径）或直接调用 renderWaveform（Canvas 2D 路径）
     *   2. 设置 scissor rect / clip 区域裁剪到 [segmentLeftPx, segmentRightPx]
     *   3. 应用 alpha 透明度（muted / leadingOverlap）
     *   4. 绘制波形
     *
     * @param params 见 DrawClipWaveformParams
     */
    drawClipWaveform(params: DrawClipWaveformParams): void;

    /** 释放资源（WebGL2 实现 destroy program/buffer/texture） */
    dispose(): void;
}
```

- [ ] **Step 2: 运行类型检查**

运行：`cd frontend && npx tsc --noEmit`
预期：无错误（仅追加类型导出，未消费）

- [ ] **Step 3: 提交**

```bash
git add frontend/src/utils/waveformRenderer.ts
git commit -m "feat(waveform): add WaveformRenderer abstract interface"
```

---

### Task 2: 实现 `Canvas2DWaveformRenderer` 类

**Files:**
- Modify: `frontend/src/utils/waveformRenderer.ts`（继续在末尾追加）

- [ ] **Step 1: 追加 Canvas2DWaveformRenderer 类**

在 `frontend/src/utils/waveformRenderer.ts` 末尾追加：

```typescript
// ============================================================================
// Canvas2DWaveformRenderer — Canvas 2D fallback 实现
// ============================================================================
//
// 封装现有 renderWaveform，行为与原 renderWaveform 完全一致。
// 用于 WebGL2 不可用时的自动降级，以及通过 localStorage 强制切换的对比测试。
// ============================================================================

/**
 * Canvas 2D 波形渲染器
 *
 * 流程：
 *   - resize：仅在物理尺寸真正变化时设置 canvas.width/height（避免强制清空）
 *   - clear：clearRect 整个画布
 *   - drawClipWaveform：save → clip 矩形 → globalAlpha → renderWaveform → restore
 *
 * 特殊说明：
 *   - 行为与原 renderWaveform 完全一致，不引入任何视觉差异
 *   - 用于 WebGL2 不可用场景的自动降级
 */
export class Canvas2DWaveformRenderer implements WaveformRenderer {
    readonly backend = "canvas2d" as const;

    private dpr = 1;
    private displayW = 0;
    private displayH = 0;
    private physicalW = 0;
    private physicalH = 0;
    private lastCanvasDims = { w: 0, h: 0 };

    constructor(
        private canvas: HTMLCanvasElement,
        private ctx: CanvasRenderingContext2D,
    ) {}

    resize(displayW: number, displayH: number, dpr: number): void {
        this.displayW = displayW;
        this.displayH = displayH;
        this.dpr = dpr;
        const internalW = Math.max(1, Math.round(displayW * dpr));
        const internalH = Math.max(1, Math.round(displayH * dpr));

        const dimsChanged =
            this.lastCanvasDims.w !== internalW || this.lastCanvasDims.h !== internalH;
        if (dimsChanged) {
            this.canvas.width = internalW;
            this.canvas.height = internalH;
            this.lastCanvasDims = { w: internalW, h: internalH };
        }
        this.physicalW = internalW;
        this.physicalH = internalH;

        // setTransform 仅在尺寸变化时重设（避免每帧重设）
        if (dimsChanged) {
            const scaleX = internalW / Math.max(1, displayW);
            const scaleY = internalH / Math.max(1, displayH);
            this.ctx.setTransform(scaleX, 0, 0, scaleY, 0, 0);
        }

        // CSS 尺寸只在变化时写入
        if (this.canvas.style.width !== `${displayW}px`) {
            this.canvas.style.width = `${displayW}px`;
        }
        if (this.canvas.style.height !== `${displayH}px`) {
            this.canvas.style.height = `${displayH}px`;
        }
    }

    clear(): void {
        this.ctx.clearRect(0, 0, this.displayW, this.displayH);
    }

    drawClipWaveform(params: DrawClipWaveformParams): void {
        const { peaks, renderParams, segmentLeftPx, segmentRightPx, strokeColor, strokeWidth, alpha } = params;
        if (segmentRightPx - segmentLeftPx <= 1e-6) return;

        this.ctx.save();
        this.ctx.beginPath();
        this.ctx.rect(segmentLeftPx, 0, segmentRightPx - segmentLeftPx, this.displayH);
        this.ctx.clip();
        this.ctx.globalAlpha = alpha;
        renderWaveform(this.ctx, peaks, renderParams, strokeColor, strokeWidth, "line");
        this.ctx.restore();
    }

    dispose(): void {
        // Canvas 2D 无需显式释放
    }
}
```

- [ ] **Step 2: 运行类型检查**

运行：`cd frontend && npx tsc --noEmit`
预期：无错误

- [ ] **Step 3: 提交**

```bash
git add frontend/src/utils/waveformRenderer.ts
git commit -m "feat(waveform): add Canvas2DWaveformRenderer as fallback implementation"
```

---

### Task 3: 实现 `waveformRendererFactory.ts` 工厂函数

**Files:**
- Create: `frontend/src/utils/waveformRendererFactory.ts`

- [ ] **Step 1: 创建工厂文件**

创建 `frontend/src/utils/waveformRendererFactory.ts`：

```typescript
/**
 * 波形渲染器工厂
 *
 * 主要内容：根据浏览器能力检测（WebGL2 是否可用）创建对应的 WaveformRenderer 实例
 * 作用：为消费方（WaveformTrackCanvas / PianoRoll 背景波形）提供统一的渲染器入口，
 *       屏蔽底层实现差异
 * 与其他模块的关系：
 *   - 依赖 waveformRenderer.ts 的 WaveformRenderer 接口与 Canvas2DWaveformRenderer
 *   - 依赖 waveformWebGL2Renderer.ts 的 WebGL2WaveformRenderer
 *   - 被 WaveformTrackCanvas.tsx 和 PianoRollPanel.tsx 调用
 *
 * 能力检测策略：
 *   1. 读取 localStorage.hifishifter.forceCanvas2DWaveform，若为 "1" 则强制走 Canvas 2D
 *   2. 尝试 canvas.getContext("webgl2", {...})，成功则创建 WebGL2WaveformRenderer
 *   3. WebGL2 初始化失败 → 回退到 Canvas2DWaveformRenderer
 *   4. 连 Canvas 2D 都拿不到 → 抛错（极端情况）
 */

import { Canvas2DWaveformRenderer, type WaveformRenderer } from "./waveformRenderer";
import { WebGL2WaveformRenderer } from "./waveformWebGL2Renderer";

/** localStorage 强制走 Canvas 2D 的开关 key */
const FORCE_CANVAS2D_KEY = "hifishifter.forceCanvas2DWaveform";

/**
 * 读取强制 Canvas 2D 开关
 *
 * 特殊说明：用于 WebGL2 实现 bug 的应急回退、性能对比测试、兼容性问题排查
 */
function shouldForceCanvas2D(): boolean {
    if (typeof window === "undefined") return false;
    try {
        return window.localStorage?.getItem(FORCE_CANVAS2D_KEY) === "1";
    } catch {
        return false;
    }
}

/**
 * 根据浏览器能力创建波形渲染器
 *
 * 流程：
 *   1. 若 enableWebGL2 且未强制 Canvas 2D：尝试获取 webgl2 context
 *   2. 拿到 gl context 后尝试创建 WebGL2WaveformRenderer（包含 shader 编译等）
 *   3. 任何步骤失败 → 回退到 Canvas2DWaveformRenderer
 *   4. 连 2d context 都拿不到 → 抛错
 *
 * @param canvas 目标 canvas 元素
 * @param enableWebGL2 是否允许走 WebGL2 路径（默认 true）
 * @returns WaveformRenderer 实例
 */
export function createWaveformRenderer(
    canvas: HTMLCanvasElement,
    enableWebGL2: boolean = true,
): WaveformRenderer {
    const forceCanvas2D = shouldForceCanvas2D();

    if (enableWebGL2 && !forceCanvas2D) {
        const gl = canvas.getContext("webgl2", {
            alpha: true,
            premultipliedAlpha: true,
            antialias: false,
            depth: false,
            stencil: false,
            preserveDrawingBuffer: false,
        });

        if (gl) {
            try {
                return new WebGL2WaveformRenderer(canvas, gl);
            } catch (e) {
                console.warn(
                    "[WaveformRenderer] WebGL2 init failed, fallback to Canvas2D:",
                    e,
                );
                // 释放可能残留的资源
                try {
                    const ext = gl.getExtension("WEBGL_lose_context");
                    ext?.loseContext();
                } catch {
                    // 忽略
                }
            }
        }
    }

    const ctx = canvas.getContext("2d");
    if (!ctx) {
        throw new Error("WaveformRenderer: neither WebGL2 nor Canvas2D available");
    }
    return new Canvas2DWaveformRenderer(canvas, ctx);
}
```

- [ ] **Step 2: 创建占位的 `waveformWebGL2Renderer.ts`**

由于 Task 3 引用了 `WebGL2WaveformRenderer`，先创建一个最小占位文件让类型检查通过，Task 4-9 会填充实现。

创建 `frontend/src/utils/waveformWebGL2Renderer.ts`：

```typescript
/**
 * 波形 WebGL2 渲染器（占位文件，Task 4-9 填充实现）
 *
 * 完整实现见后续任务：shader 编译、纹理管理、drawClipWaveform 主流程。
 */

import type { WaveformRenderer } from "./waveformRenderer";

export class WebGL2WaveformRenderer implements WaveformRenderer {
    readonly backend = "webgl2" as const;

    constructor(canvas: HTMLCanvasElement, gl: WebGL2RenderingContext) {
        // 占位：实际实现在 Task 4-9
        void canvas;
        void gl;
        throw new Error("WebGL2WaveformRenderer not implemented yet");
    }

    resize(): void {
        throw new Error("Not implemented");
    }

    clear(): void {
        throw new Error("Not implemented");
    }

    drawClipWaveform(): void {
        throw new Error("Not implemented");
    }

    dispose(): void {
        // 占位
    }
}
```

- [ ] **Step 3: 运行类型检查**

运行：`cd frontend && npx tsc --noEmit`
预期：无错误

- [ ] **Step 4: 提交**

```bash
git add frontend/src/utils/waveformRendererFactory.ts frontend/src/utils/waveformWebGL2Renderer.ts
git commit -m "feat(waveform): add renderer factory with WebGL2 capability detection"
```

---

## 阶段 2：WebGL2 Renderer 实现

### Task 4: 创建 `WebGL2WaveformRenderer` 骨架（构造、资源初始化、dispose）

**Files:**
- Modify: `frontend/src/utils/waveformWebGL2Renderer.ts`（替换占位实现）

- [ ] **Step 1: 替换为骨架实现**

用以下内容替换 `frontend/src/utils/waveformWebGL2Renderer.ts` 全文：

```typescript
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

import type { DrawClipWaveformParams, WaveformRenderer, WaveformRenderParams } from "./waveformRenderer";

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

    constructor(private canvas: HTMLCanvasElement, gl: WebGL2RenderingContext) {
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
        // 实现在 Task 8
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
```

- [ ] **Step 2: 运行类型检查**

运行：`cd frontend && npx tsc --noEmit`
预期：无错误

- [ ] **Step 3: 提交**

```bash
git add frontend/src/utils/waveformWebGL2Renderer.ts
git commit -m "feat(waveform): add WebGL2WaveformRenderer skeleton with shader and resource init"
```

---

### Task 5: 实现颜色解析与 uniform 工具方法

**Files:**
- Modify: `frontend/src/utils/waveformWebGL2Renderer.ts`（在类内追加私有方法）

- [ ] **Step 1: 追加颜色解析与 uniform 计算方法**

在 `WebGL2WaveformRenderer` 类的 `clear()` 方法之后、`drawClipWaveform` 之前追加以下方法（`computeUniforms` 直接使用 `peaksLength` 参数计算 `totalSamples`，因为 `WaveformRenderParams` 不包含 peaks 字段）：

```typescript
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
     * 流程：与现有 waveformRenderer.ts:343-352 的 pxToIdxScale/pxToIdxBase/halfPixelIdx 公式完全一致
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
```

- [ ] **Step 2: 运行类型检查**

运行：`cd frontend && npx tsc --noEmit`
预期：无错误

- [ ] **Step 3: 提交**

```bash
git add frontend/src/utils/waveformWebGL2Renderer.ts
git commit -m "feat(waveform): add color parsing and uniform computation for WebGL2 renderer"
```

---

### Task 6: 实现 `drawClipWaveform` 主流程

**Files:**
- Modify: `frontend/src/utils/waveformWebGL2Renderer.ts`（替换占位的 `drawClipWaveform`）

- [ ] **Step 1: 替换 drawClipWaveform 实现**

用以下内容替换 `drawClipWaveform` 方法：

```typescript
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
        gl.bindTexture(gl.TEXTURE_2D, this.peaksTex);
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
        gl.useProgram(this.program);
        gl.bindVertexArray(this.vao);

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
        gl.bindTexture(gl.TEXTURE_2D, this.peaksTex);
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
```

- [ ] **Step 2: 运行类型检查**

运行：`cd frontend && npx tsc --noEmit`
预期：无错误

- [ ] **Step 3: 运行构建**

运行：`cd frontend && npm run build`
预期：构建成功

- [ ] **Step 4: 提交**

```bash
git add frontend/src/utils/waveformWebGL2Renderer.ts
git commit -m "feat(waveform): implement drawClipWaveform for WebGL2 renderer"
```

---

### Task 7: 添加 context lost 处理

**Files:**
- Modify: `frontend/src/utils/waveformWebGL2Renderer.ts`

- [ ] **Step 1: 在构造函数中注册 context lost 监听**

在 `WebGL2WaveformRenderer` 构造函数末尾追加：

```typescript
        // 监听 context lost 事件（不主动恢复，由消费方重新创建 renderer）
        this.canvas.addEventListener("webglcontextlost", this.handleContextLost, false);
```

- [ ] **Step 2: 添加 handleContextLost 方法**

在类的私有方法区追加：

```typescript
    /**
     * WebGL context lost 事件处理
     *
     * 流程：preventDefault 阻止默认行为，标记 renderer 为已失活
     * 特殊说明：不主动恢复，由消费方监听同一事件后重新调用 createWaveformRenderer
     */
    private handleContextLost = (event: Event): void => {
        event.preventDefault();
        // 标记资源已失效，避免后续 drawCall 报错
        this.program = null as unknown as WebGLProgram;
        this.vao = null as unknown as WebGLVertexArrayObject;
        this.peaksTex = null as unknown as WebGLTexture;
    };
```

- [ ] **Step 3: 在 dispose 中移除监听**

修改 `dispose` 方法，在开头追加：

```typescript
    dispose(): void {
        this.canvas.removeEventListener("webglcontextlost", this.handleContextLost);
        // ...其余释放逻辑
    }
```

- [ ] **Step 4: 运行类型检查**

运行：`cd frontend && npx tsc --noEmit`
预期：无错误

- [ ] **Step 5: 提交**

```bash
git add frontend/src/utils/waveformWebGL2Renderer.ts
git commit -m "feat(waveform): handle WebGL2 context lost events"
```

---

## 阶段 3：集成 WaveformTrackCanvas

### Task 8: 在 `WaveformTrackCanvas` 中创建 renderer 实例

**Files:**
- Modify: `frontend/src/components/waveform/WaveformTrackCanvas.tsx`

- [ ] **Step 1: 引入工厂函数和接口**

在 `frontend/src/components/waveform/WaveformTrackCanvas.tsx` 顶部 import 区追加：

```typescript
import { createWaveformRenderer } from "../../utils/waveformRendererFactory";
import type { WaveformRenderer } from "../../utils/waveformRenderer";
```

- [ ] **Step 2: 创建 renderer ref 并在 canvas 挂载时初始化**

在 `WaveformTrackCanvas` 组件内，紧接 `canvasRef` 声明之后（约 `WaveformTrackCanvas.tsx:110` 附近）追加：

```typescript
        const rendererRef = React.useRef<WaveformRenderer | null>(null);

        // canvas 挂载后创建 renderer（工厂自动检测 WebGL2 能力）
        const setCanvasRef = React.useCallback((canvas: HTMLCanvasElement | null) => {
            canvasRef.current = canvas;
            if (canvas && !rendererRef.current) {
                try {
                    rendererRef.current = createWaveformRenderer(canvas);
                } catch (e) {
                    console.error("[WaveformTrackCanvas] create renderer failed:", e);
                    rendererRef.current = null;
                }
            }
        }, []);
```

- [ ] **Step 3: 在卸载时 dispose**

在组件末尾的卸载 useEffect（`WaveformTrackCanvas.tsx:568-575` 附近）修改为：

```typescript
        React.useEffect(() => {
            return () => {
                if (rafRef.current != null) {
                    cancelAnimationFrame(rafRef.current);
                    rafRef.current = null;
                }
                rendererRef.current?.dispose();
                rendererRef.current = null;
            };
        }, []);
```

- [ ] **Step 4: 替换 canvas ref 引用**

把 JSX 中 `ref={canvasRef}` 改为 `ref={setCanvasRef}`：

```tsx
            <canvas
                ref={setCanvasRef}
                style={{
                    position: "absolute",
                    top: waveformTop,
                    height: waveformHeight,
                    pointerEvents: "none",
                    zIndex: 1,
                    left: 0,
                }}
            />
```

- [ ] **Step 5: 运行类型检查**

运行：`cd frontend && npx tsc --noEmit`
预期：无错误（此时 renderer 还未被使用，仅创建和销毁）

- [ ] **Step 6: 提交**

```bash
git add frontend/src/components/waveform/WaveformTrackCanvas.tsx
git commit -m "refactor(waveform): create WaveformRenderer instance in WaveformTrackCanvas"
```

---

### Task 9: 在 `WaveformTrackCanvas` 的 draw 循环中替换为 renderer 调用

**Files:**
- Modify: `frontend/src/components/waveform/WaveformTrackCanvas.tsx`

- [ ] **Step 1: 替换尺寸管理逻辑**

在 `drawRef.current` 函数内（`WaveformTrackCanvas.tsx:191-223` 附近），把直接操作 `canvas.width/height` 和 `ctx.setTransform` 的逻辑替换为 renderer.resize。

找到这段（约 `WaveformTrackCanvas.tsx:191-223`）：

```typescript
            const displayW = Math.max(1, Math.ceil(currentViewportWidthPx));
            const displayH = currentWaveformHeight;

            const dpr = window.devicePixelRatio || 1;
            const internalW = Math.max(1, Math.round(displayW * dpr));
            const internalH = Math.max(1, Math.round(displayH * dpr));

            const lastDims = lastCanvasDimsRef.current;
            const dimsChanged = lastDims.w !== internalW || lastDims.h !== internalH;
            if (dimsChanged) {
                canvas.width = internalW;
                canvas.height = internalH;
                lastCanvasDimsRef.current = { w: internalW, h: internalH };
            }

            const ctx = canvas.getContext("2d");
            if (!ctx) return;

            if (dimsChanged) {
                const scaleX = internalW / Math.max(1, displayW);
                const scaleY = internalH / Math.max(1, displayH);
                ctx.setTransform(scaleX, 0, 0, scaleY, 0, 0);
            }

            ctx.clearRect(0, 0, displayW, displayH);

            if (canvas.style.width !== `${displayW}px`) canvas.style.width = `${displayW}px`;
            if (canvas.style.height !== `${displayH}px`) canvas.style.height = `${displayH}px`;
```

替换为：

```typescript
            const displayW = Math.max(1, Math.ceil(currentViewportWidthPx));
            const displayH = currentWaveformHeight;
            const dpr = window.devicePixelRatio || 1;

            const renderer = rendererRef.current;
            if (!renderer) return;

            renderer.resize(displayW, displayH, dpr);
            renderer.clear();
```

- [ ] **Step 2: 替换 drawSegment 函数**

找到 `drawSegment` 定义（`WaveformTrackCanvas.tsx:410-431`）：

```typescript
                const drawSegment = (
                    segmentLeftPx: number,
                    segmentRightPx: number,
                    alpha: number,
                ) => {
                    if (segmentRightPx - segmentLeftPx <= 1e-6) return;
                    ctx.save();
                    ctx.beginPath();
                    ctx.rect(segmentLeftPx, 0, segmentRightPx - segmentLeftPx, displayH);
                    ctx.clip();
                    ctx.globalAlpha = alpha;
                    renderWaveform(
                        ctx,
                        withGains,
                        params,
                        currentStrokeColor,
                        currentStrokeWidth,
                        "line",
                    );
                    ctx.restore();
                };
```

替换为：

```typescript
                const drawSegment = (
                    segmentLeftPx: number,
                    segmentRightPx: number,
                    alpha: number,
                ) => {
                    if (segmentRightPx - segmentLeftPx <= 1e-6) return;
                    renderer.drawClipWaveform({
                        peaks: withGains,
                        renderParams: params,
                        segmentLeftPx,
                        segmentRightPx,
                        strokeColor: currentStrokeColor,
                        strokeWidth: currentStrokeWidth,
                        alpha,
                    });
                };
```

- [ ] **Step 3: 移除尾部 ctx.globalAlpha 重置**

找到这段（`WaveformTrackCanvas.tsx:480-482`）并删除（不再需要）：

```typescript
            if (ctx.globalAlpha !== 1) {
                ctx.globalAlpha = 1;
            }
```

- [ ] **Step 4: 移除未使用的 import**

从 `frontend/src/components/waveform/WaveformTrackCanvas.tsx` 顶部移除不再使用的 `renderWaveform` import（`WaveformTrackCanvas.tsx:34` 附近）：

```typescript
// 移除：renderWaveform,
// 保留：applyGainsToPeaks, releaseGainBuffer, type WaveformRenderParams
```

同时移除不再使用的 `lastCanvasDimsRef` 声明（`WaveformTrackCanvas.tsx:114` 附近）。

- [ ] **Step 5: 运行类型检查**

运行：`cd frontend && npx tsc --noEmit`
预期：无错误

- [ ] **Step 6: 运行构建**

运行：`cd frontend && npm run build`
预期：构建成功

- [ ] **Step 7: 手动验证**

运行 `cd frontend && npm run dev`，打开应用：
- 加载音频文件，查看上方轨道区波形是否正常显示
- 滚动时间轴，验证波形跟随滚动
- 缩放（Ctrl+滚轮），验证波形缩放
- 多 clip 场景，验证 clip 间裁剪正确
- 静音 clip，验证 alpha 降低
- 打开 DevTools Console，确认无 WebGL 错误日志
- 在 Console 输入 `localStorage.setItem("hifishifter.forceCanvas2DWaveform", "1")` 刷新，验证 fallback 路径也正常

- [ ] **Step 8: 提交**

```bash
git add frontend/src/components/waveform/WaveformTrackCanvas.tsx
git commit -m "refactor(waveform): use WaveformRenderer interface in WaveformTrackCanvas"
```

---

## 阶段 4：集成 PianoRoll 背景波形（方案 B）

### Task 10: 在 `PianoRollPanel` 新增背景波形 canvas

**Files:**
- Modify: `frontend/src/components/layout/PianoRollPanel.tsx`

- [ ] **Step 1: 引入工厂函数与接口**

在 `PianoRollPanel.tsx` 顶部 import 区追加：

```typescript
import { createWaveformRenderer } from "../../utils/waveformRendererFactory";
import type { WaveformRenderer } from "../../utils/waveformRenderer";
```

- [ ] **Step 2: 新增背景波形 canvas ref 与 renderer ref**

在 `PianoRollPanel` 组件内，紧接 `canvasRef` 声明附近（搜索 `const canvasRef = React.useRef`）追加：

```typescript
        const bgWaveformCanvasRef = React.useRef<HTMLCanvasElement | null>(null);
        const bgWaveformRendererRef = React.useRef<WaveformRenderer | null>(null);

        const setBgWaveformCanvasRef = React.useCallback((canvas: HTMLCanvasElement | null) => {
            bgWaveformCanvasRef.current = canvas;
            if (canvas && !bgWaveformRendererRef.current) {
                try {
                    bgWaveformRendererRef.current = createWaveformRenderer(canvas);
                } catch (e) {
                    console.error("[PianoRoll] create bg waveform renderer failed:", e);
                }
            }
        }, []);
```

- [ ] **Step 3: 在 JSX 中新增背景波形 canvas**

找到主 canvas 元素（`PianoRollPanel.tsx:4109-4116`）：

```tsx
                                <canvas
                                    ref={canvasRef}
                                    className="absolute inset-0"
                                    style={{ cursor: canvasCursor }}
                                    onPointerMove={interactions.onCanvasPointerMove}
                                    onPointerLeave={interactions.onCanvasPointerLeave}
                                    onPointerDown={interactions.onCanvasPointerDown}
                                />
```

在主 canvas **之前**插入背景波形 canvas：

```tsx
                                {/* 背景波形 canvas（WebGL2，z-index 在主 canvas 之下） */}
                                <canvas
                                    ref={setBgWaveformCanvasRef}
                                    className="absolute inset-0"
                                    style={{ pointerEvents: "none", zIndex: 0 }}
                                />
                                <canvas
                                    ref={canvasRef}
                                    className="absolute inset-0"
                                    style={{ cursor: canvasCursor, zIndex: 1 }}
                                    onPointerMove={interactions.onCanvasPointerMove}
                                    onPointerLeave={interactions.onCanvasPointerLeave}
                                    onPointerDown={interactions.onCanvasPointerDown}
                                />
```

- [ ] **Step 4: 在卸载时 dispose**

在 PianoRollPanel 的某个卸载 useEffect 中追加（或在已有的 invalidate 相关 useEffect cleanup 中）：

```typescript
        React.useEffect(() => {
            return () => {
                bgWaveformRendererRef.current?.dispose();
                bgWaveformRendererRef.current = null;
            };
        }, []);
```

- [ ] **Step 5: 运行类型检查**

运行：`cd frontend && npx tsc --noEmit`
预期：无错误

- [ ] **Step 6: 提交**

```bash
git add frontend/src/components/layout/PianoRollPanel.tsx
git commit -m "feat(pianoroll): add background waveform canvas for WebGL2 rendering"
```

---

### Task 11: 抽取 PianoRoll 波形绘制为独立函数并改用 renderer

**Files:**
- Modify: `frontend/src/components/layout/pianoRoll/render.ts`

- [ ] **Step 1: 新增 `drawPianoRollBackgroundWaveform` 导出函数**

在 `frontend/src/components/layout/pianoRoll/render.ts` 文件末尾追加独立函数：

```typescript
// ============================================================================
// drawPianoRollBackgroundWaveform — PianoRoll 背景波形独立绘制函数
// ============================================================================
//
// 从 drawPianoRoll 内的背景波形循环（原 render.ts:730-837）抽取，
// 改为通过 WaveformRenderer 接口渲染（WebGL2 优先，Canvas 2D fallback）。
// 主 canvas 的参数曲线绘制仍用 Canvas 2D，不受此函数影响。
// ============================================================================

import type { WaveformRenderer } from "../../../utils/waveformRenderer";
import type { WaveformRenderParams } from "../../../utils/waveformRenderer";
import { applyGainsToPeaks, releaseGainBuffer } from "../../../utils/waveformRenderer";
import { waveformMipmapStore } from "../../../utils/waveformMipmapStore";

/** drawPianoRollBackgroundWaveform 用的 lastLevelByClip 持久化状态 */
const _bgWaveformLastLevel = new Map<string, 0 | 1 | 2>();

/**
 * 绘制 PianoRoll 背景波形（通过 WaveformRenderer 接口）
 *
 * 流程：
 *   1. resize + clear 背景 canvas
 *   2. 遍历 clipPeaks，对每个可见 clip：
 *      a. waveformMipmapStore.getInterleavedSlice 获取 peaks
 *      b. applyGainsToPeaks 应用增益
 *      c. renderer.drawClipWaveform 绘制
 *
 * 特殊说明：
 *   - 与原 drawPianoRoll 内的逻辑等价，仅切换渲染后端
 *   - clip 裁剪通过 segmentLeftPx/segmentRightPx 传入 renderer
 *   - muted clip 用 alpha = 0.3（与原版一致）
 *
 * @param renderer 背景波形 renderer 实例
 * @param canvasW  canvas CSS 宽度
 * @param canvasH  canvas CSS 高度
 * @param dpr      设备像素比
 * @param clipPeaks 可见 clip 列表
 * @param pxPerSec 每秒像素数
 * @param scrollLeft 滚动偏移（像素）
 * @param visibleStartSec 可视区起始时间（秒）
 * @param visibleDurSec 可视区时长（秒）
 * @param waveformColors 颜色配置
 */
export function drawPianoRollBackgroundWaveform(
    renderer: WaveformRenderer,
    canvasW: number,
    canvasH: number,
    dpr: number,
    clipPeaks: ClipPeaksEntry[],
    pxPerSec: number,
    scrollLeft: number,
    visibleStartSec: number,
    visibleDurSec: number,
    waveformColors: { fill: string; stroke: string },
): void {
    renderer.resize(canvasW, canvasH, dpr);
    renderer.clear();

    for (const entry of clipPeaks) {
        if (!entry.sourcePath) continue;
        if (entry.muted) continue;

        const pr = entry.playbackRate > 0 ? entry.playbackRate : 1;
        const sourceStartSec = entry.sourceStartSec ?? 0;
        const sourceDurSec = entry.sourceDurationSec;
        if (sourceDurSec <= 0) continue;

        const clipStartSec = entry.startSec;
        const clipEndSec = clipStartSec + entry.lengthSec;
        const clipWidthPx = entry.lengthSec * pxPerSec;
        if (clipWidthPx <= 0) continue;

        const visStartSec = Math.max(clipStartSec, visibleStartSec);
        const visEndSec = Math.min(clipEndSec, visibleStartSec + visibleDurSec);
        if (visEndSec <= visStartSec) continue;

        const viewportStartPx = Math.round(scrollLeft);
        const clipStartPx = Math.round(clipStartSec * pxPerSec);
        const clipEndPx = Math.round(clipEndSec * pxPerSec);
        const clipVisLeft = Math.max(0, clipStartPx - viewportStartPx);
        const clipVisRight = Math.min(canvasW, clipEndPx - viewportStartPx);
        if (clipVisRight <= clipVisLeft) continue;

        const clipSourceEndSec = Number(entry.sourceEndSec ?? sourceDurSec) || sourceDurSec;
        const clipSourceSpanSec = Math.max(
            0,
            Math.min(entry.lengthSec * pr, clipSourceEndSec - sourceStartSec),
        );
        const sourceTimeStart = sourceStartSec;
        const sourceDuration = Math.max(0.001, clipSourceSpanSec);

        const sampleRate = entry.sourceSampleRate || 44100;
        const spp = Math.max(1, Math.round(sampleRate / pxPerSec));
        const levelKey = `${entry.sourcePath}::${entry.clipId}`;
        const previousLevel = _bgWaveformLastLevel.get(levelKey);
        const stableLevel = waveformMipmapStore.selectLevelStable(spp, previousLevel);
        _bgWaveformLastLevel.set(levelKey, stableLevel);

        const result = waveformMipmapStore.getInterleavedSlice(
            entry.sourcePath,
            stableLevel,
            sourceTimeStart,
            sourceDuration,
        );
        if (!result || result.interleaved.length < 4) continue;

        const clipPixelOffset = viewportStartPx + clipVisLeft - clipStartPx;

        const params: WaveformRenderParams = {
            canvasWidth: clipVisRight - clipVisLeft,
            canvasHeight: canvasH,
            centerY: canvasH / 2,
            zeroDbHalfHeight: canvasH / 2,
            sourceStartSec,
            clipDuration: entry.lengthSec,
            playbackRate: pr,
            sourceDurationSec: sourceDurSec,
            volumeGain: Number(entry.gain ?? 1) || 1,
            fadeInSec: Number(entry.fadeInSec ?? 0) || 0,
            fadeOutSec: Number(entry.fadeOutSec ?? 0) || 0,
            fadeInCurve: entry.fadeInCurve ?? "linear",
            fadeOutCurve: entry.fadeOutCurve ?? "linear",
            dataStartSec: result.dataStartSec,
            dataDurationSec: result.dataDurationSec,
            clipPixelOffset,
            clipTotalWidthPx: Math.max(1, clipWidthPx),
        };

        const withGains = applyGainsToPeaks(result.interleaved, params);

        renderer.drawClipWaveform({
            peaks: withGains,
            renderParams: params,
            segmentLeftPx: clipVisLeft,
            segmentRightPx: clipVisRight,
            strokeColor: waveformColors.stroke,
            strokeWidth: 0.5,
            alpha: entry.muted ? 0.3 : 0.86,
        });

        if (withGains !== result.interleaved) {
            releaseGainBuffer(withGains);
        }
        waveformMipmapStore.releaseInterleaved(result.interleaved);
    }
}
```

> 注：文件顶部可能已经有 `applyGainsToPeaks`、`releaseGainBuffer`、`waveformMipmapStore` 的 import（因为原 `drawPianoRoll` 也用了它们）。如果是重复 import，合并到文件顶部现有 import 中，不要重复声明。`ClipPeaksEntry` 已在文件顶部 import（`render.ts:14`）。

- [ ] **Step 2: 从 `drawPianoRoll` 内移除原背景波形循环**

找到 `drawPianoRoll` 内的背景波形循环（`render.ts:730-837`，从 `// Background waveform: per-clip 叠加绘制` 开始到 `releaseInterleaved` 结束的整个 for 循环），整段删除。

同时删除 `lastLevelByClip` 相关的状态初始化（`render.ts:720-728`），因为已迁移到 `drawPianoRollBackgroundWaveform` 内的 `_bgWaveformLastLevel`。

- [ ] **Step 3: 运行类型检查**

运行：`cd frontend && npx tsc --noEmit`
预期：可能报 `applyGainsToPeaks` / `releaseGainBuffer` / `waveformMipmapStore` 未使用的警告（如果 `drawPianoRoll` 内不再使用它们）。若有未使用警告，保留 import（因为新函数 `drawPianoRollBackgroundWaveform` 在同一文件内使用）。

- [ ] **Step 4: 提交**

```bash
git add frontend/src/components/layout/pianoRoll/render.ts
git commit -m "refactor(pianoroll): extract background waveform rendering to use WaveformRenderer"
```

---

### Task 12: 在 `PianoRollPanel` 的 draw 循环中调用背景波形 renderer

**Files:**
- Modify: `frontend/src/components/layout/PianoRollPanel.tsx`

- [ ] **Step 1: 引入新函数**

在 `PianoRollPanel.tsx` 顶部 import 区追加：

```typescript
import { drawPianoRollBackgroundWaveform } from "./pianoRoll/render";
```

- [ ] **Step 2: 在 drawRef.current 中调用背景波形绘制**

找到 `drawRef.current = () => { drawPianoRoll({...}) }`（`PianoRollPanel.tsx:1696` 附近），在 `drawPianoRoll` 调用**之前**追加背景波形绘制：

```typescript
    drawRef.current = () => {
        // 先绘制背景波形到独立的 WebGL2 canvas
        const bgRenderer = bgWaveformRendererRef.current;
        const bgCanvas = bgWaveformCanvasRef.current;
        const viewSize = viewSizeRef.current;
        if (bgRenderer && bgCanvas && viewSize.w > 0 && viewSize.h > 0) {
            const dpr = Math.max(1, window.devicePixelRatio || 1);
            const pxPerSec = pxPerSecRef.current;
            const scrollLeft = scrollLeftRef.current;
            const visibleStartSec = scrollLeft / Math.max(1e-9, pxPerSec);
            const visibleDurSec = viewSize.w / Math.max(1e-9, pxPerSec);
            drawPianoRollBackgroundWaveform(
                bgRenderer,
                viewSize.w,
                viewSize.h,
                dpr,
                clipPeaks,
                pxPerSec,
                scrollLeft,
                visibleStartSec,
                visibleDurSec,
                waveformColors,
            );
        }

        drawPianoRoll({
            // ...现有参数保持不变
        });
    };
```

- [ ] **Step 3: 运行类型检查**

运行：`cd frontend && npx tsc --noEmit`
预期：无错误

- [ ] **Step 4: 运行构建**

运行：`cd frontend && npm run build`
预期：构建成功

- [ ] **Step 5: 手动验证**

运行 `cd frontend && npm run dev`，打开应用：
- 切换到 PianoRoll 视图，验证背景波形正常显示
- 滚动和缩放，验证波形跟随
- 验证参数曲线（音高/张力等）仍正常显示在背景波形之上
- 验证 clip 裁剪正确（clip 外不画波形）
- 在 Console 输入 `localStorage.setItem("hifishifter.forceCanvas2DWaveform", "1")` 刷新，验证 fallback 也正常

- [ ] **Step 6: 提交**

```bash
git add frontend/src/components/layout/PianoRollPanel.tsx
git commit -m "feat(pianoroll): drive background waveform rendering via WebGL2 renderer"
```

---

## 阶段 5：收尾

### Task 13: 更新 `waveformDebug.ts` 增加 renderer backend 标识

**Files:**
- Modify: `frontend/src/utils/waveformDebug.ts`

- [ ] **Step 1: 读取现有 waveformDebug.ts 结构**

运行：`cat frontend/src/utils/waveformDebug.ts | head -50` 了解现有诊断框架。

- [ ] **Step 2: 新增 backend 标识变量与 setter**

在 `waveformDebug.ts` 顶部变量区追加：

```typescript
/** 当前渲染后端标识（由 renderer 创建时设置） */
let _currentBackend: "webgl2" | "canvas2d" | "webgpu" | "unknown" = "unknown";

/**
 * 设置当前渲染后端标识（由 WaveformRenderer 实例创建时调用）
 *
 * 特殊说明：用于诊断输出中标识当前使用的渲染后端
 */
export function wfDiag_setBackend(backend: "webgl2" | "canvas2d" | "webgpu"): void {
    _currentBackend = backend;
}

/** 获取当前渲染后端标识 */
export function wfDiag_getBackend(): string {
    return _currentBackend;
}
```

- [ ] **Step 3: 在帧统计输出中包含 backend**

找到帧统计输出函数（搜索 `console.log` 或帧时间输出），在输出字符串中追加 backend 标识。若现有输出函数为 `wfDiag_frameEnd`，在其中追加：

```typescript
// 若有帧统计 console.log，追加 backend 标识
// 例如：`[WaveformPerf] backend=${_currentBackend} frame ${totalMs}ms ...`
```

- [ ] **Step 4: 在工厂函数中设置 backend**

修改 `frontend/src/utils/waveformRendererFactory.ts`，在创建 renderer 后调用 `wfDiag_setBackend`：

```typescript
import { wfDiag_setBackend } from "./waveformDebug";

// 在返回 WebGL2WaveformRenderer 之前
wfDiag_setBackend("webgl2");
return new WebGL2WaveformRenderer(canvas, gl);

// 在返回 Canvas2DWaveformRenderer 之前
wfDiag_setBackend("canvas2d");
return new Canvas2DWaveformRenderer(canvas, ctx);
```

- [ ] **Step 5: 运行类型检查**

运行：`cd frontend && npx tsc --noEmit`
预期：无错误

- [ ] **Step 6: 提交**

```bash
git add frontend/src/utils/waveformDebug.ts frontend/src/utils/waveformRendererFactory.ts
git commit -m "feat(waveform): expose renderer backend in diagnostics"
```

---

### Task 14: 更新文件头注释

**Files:**
- Modify: `frontend/src/utils/waveformRenderer.ts`
- Modify: `frontend/src/utils/waveformWebGL2Renderer.ts`
- Modify: `frontend/src/components/waveform/WaveformTrackCanvas.tsx`
- Modify: `frontend/src/components/layout/pianoRoll/render.ts`

- [ ] **Step 1: 更新 `waveformRenderer.ts` 文件头注释**

在现有文件头注释中追加说明：

```typescript
/**
 * 波形渲染工具模块（Canvas per-pixel 渲染 + WaveformRenderer 抽象接口）
 *
 * 本模块负责将已降采样的 peaks 数据绘制到 Canvas 上，是波形可视化的最后一环。
 *
 * ## 数据流
 *   waveformMipmapStore（降采样 / resample）
 *     → applyGainsToPeaks（叠加音量 + 淡入淡出增益）
 *       → renderWaveform（Canvas per-pixel 绘制，用于 Canvas2DWaveformRenderer）
 *       → WaveformRenderer.drawClipWaveform（抽象接口，WebGL2 / Canvas 2D）
 *
 * ## 渲染后端
 *   - Canvas2DWaveformRenderer：封装 renderWaveform，作为 WebGL2 不可用时的 fallback
 *   - WebGL2WaveformRenderer：见 waveformWebGL2Renderer.ts，instanced quad + RG32F 纹理
 *   - 由 waveformRendererFactory.ts 工厂函数根据浏览器能力选择
 *
 * ## 坐标系映射链
 *   canvas 本地像素 → clip 全局像素 → timeline 时间 → 源文件时间 → peaks 数据索引
 *
 * ## 导出
 * - {@link WaveformRenderParams} — 渲染参数接口
 * - {@link applyGainsToPeaks}    — 增益应用（音量 × 淡入淡出曲线）
 * - {@link renderWaveform}       — Canvas 2D 绘制（line / jitter）
 * - {@link WaveformRenderer}     — 渲染器抽象接口
 * - {@link DrawClipWaveformParams} — 单次绘制参数
 * - {@link Canvas2DWaveformRenderer} — Canvas 2D 实现
 *
 * @module waveformRenderer
 */
```

- [ ] **Step 2: 更新 `WaveformTrackCanvas.tsx` 文件头注释**

在文件头注释中追加：

```typescript
/**
 * WaveformTrackCanvas - 轨道级波形 Canvas 组件（v4 WaveformRenderer 架构）
 *
 * v4 变更（2026-07-29）：
 *   - 渲染后端切换为 WaveformRenderer 接口（WebGL2 优先，Canvas 2D fallback）
 *   - 由 createWaveformRenderer 工厂在 canvas 挂载时自动选择实现
 *   - drawSegment 改为调用 renderer.drawClipWaveform
 *   - 通过 localStorage.hifishifter.forceCanvas2DWaveform=1 可强制走 Canvas 2D
 *
 * v3 架构保留不变：
 *   - rAF + invalidate() 帧合并
 *   - 高频参数（viewportStartSec / pxPerSec）存 ref，避免 React re-render
 *   - timelineViewportBus 事件总线订阅
 *   - mipmap 三级缓存 + spp 滞后防抖
 *   - buffer 复用池
 *
 * 渲染流程（v4）：
 *   1. canvas 挂载 → createWaveformRenderer → renderer 实例
 *   2. invalidate → drawRef.current → renderer.resize + renderer.clear
 *   3. 遍历可见 clip → applyGainsToPeaks → renderer.drawClipWaveform
 *   4. canvas 卸载 → renderer.dispose
 */
```

- [ ] **Step 3: 更新 `render.ts` 文件头注释**

在文件头注释中追加：

```typescript
// 2026-07-29 变更：
//   - 背景波形绘制抽取为 drawPianoRollBackgroundWaveform 独立函数
//   - 通过 WaveformRenderer 接口渲染（WebGL2 优先，Canvas 2D fallback）
//   - 主 canvas 仍用 Canvas 2D 画参数曲线，背景波形在独立 canvas 上
//   - PianoRollPanel.tsx 负责调用 drawPianoRollBackgroundWaveform
```

- [ ] **Step 4: 运行类型检查与构建**

运行：`cd frontend && npx tsc --noEmit && npm run build`
预期：无错误，构建成功

- [ ] **Step 5: 提交**

```bash
git add frontend/src/utils/waveformRenderer.ts frontend/src/utils/waveformWebGL2Renderer.ts frontend/src/components/waveform/WaveformTrackCanvas.tsx frontend/src/components/layout/pianoRoll/render.ts
git commit -m "docs(waveform): update file header comments for WebGL2 renderer architecture"
```

---

### Task 15: 最终验收测试与 PR 准备

**Files:**
- 无文件变更

- [ ] **Step 1: 完整功能验收**

运行 `cd frontend && npm run dev`，依次验证：

1. **上方轨道区（WaveformTrackCanvas）**：
   - 加载音频，波形正常显示
   - 横向滚动，波形跟随
   - Ctrl+滚轮缩放，波形缩放
   - 多 clip 场景，clip 间裁剪正确
   - 静音 clip，alpha 降低
   - leadingOverlap 区段，半透明混合正确

2. **下方 PianoRoll 背景波形**：
   - 切换到 PianoRoll 视图，背景波形显示
   - 参数曲线显示在背景波形之上
   - 滚动和缩放，背景波形跟随
   - clip 裁剪正确

3. **Fallback 验证**：
   - Console 输入 `localStorage.setItem("hifishifter.forceCanvas2DWaveform", "1")` 刷新
   - 上述所有功能在 Canvas 2D 路径下也正常

4. **性能对比**：
   - Console 输入 `localStorage.setItem("hifishifter.debugWaveformPerf", "1")`
   - 观察帧时间，WebGL2 路径应明显低于 Canvas 2D 路径

5. **DPR 切换**：
   - 拖动窗口到外接显示器，验证尺寸自适应，无闪烁

- [ ] **Step 2: 运行 lint**

运行：`cd frontend && npm run lint`
预期：无错误

- [ ] **Step 3: 运行构建**

运行：`cd frontend && npm run build`
预期：构建成功

- [ ] **Step 4: 查看 commit 历史**

运行：`git log --oneline develop..HEAD`
预期：看到一系列 feat/refactor/docs 提交

- [ ] **Step 5: 准备合并到 develop**

询问用户是否合并到 develop，或者保留分支待 review。**不要自动合并或推送，等用户确认。**

---

## 附录：实施风险与缓解

| 风险 | 缓解 |
|------|------|
| Shader 在某些 GPU 表现异常 | localStorage 强制 Canvas 2D 开关 |
| WebGL2 context lost | 监听事件，由消费方重新创建 renderer |
| PianoRoll 双 canvas z-index 错乱 | 明确 zIndex: 0（背景）/ 1（主 canvas） |
| 视觉差异（颜色解析精度） | parseColor 用 getImageData，与 Canvas 2D 一致 |
| 性能未达预期 | 保留 Canvas 2D 路径做 A/B 对比 |
