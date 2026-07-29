<!--
文件说明：波形渲染 WebGL2 化重构设计文档
主要内容：将现有 Canvas 2D per-pixel 波形渲染（renderWaveform）抽象为 WaveformRenderer 接口，
         提供 WebGL2 实现（RG32F 纹理 + instanced quad），保留 Canvas 2D 作为自动降级 fallback，
         并为未来 WebGPU 升级预留接口空间。
影响模块：
  - frontend/src/utils/waveformRenderer.ts       （抽象接口，保留 Canvas 2D 实现作为 fallback）
  - frontend/src/utils/waveformWebGL2Renderer.ts （新增 WebGL2 实现）
  - frontend/src/utils/waveformRendererFactory.ts （新增工厂 + 能力检测）
  - frontend/src/components/waveform/WaveformTrackCanvas.tsx （改造为使用 renderer 接口）
  - frontend/src/components/layout/pianoRoll/render.ts （PianoRoll 背景波形改造为使用 renderer 接口）
关联文档：docs/plans/2026-03-20-waveform-rendering-refactor.md（前一轮 mipmap + Canvas 2D 重构）
-->

# 波形渲染 WebGL2 化重构 设计文档

**Goal:** 将波形渲染从 Canvas 2D per-pixel CPU 循环升级为 WebGL2 instanced quad GPU 渲染，消除逐像素 `fillRect` 的 CPU 开销，并为未来 WebGPU 升级预留抽象接口。

**Architecture:** 新增 `WaveformRenderer` 抽象接口，运行时通过工厂函数检测 WebGL2 支持度自动选择实现：支持则用 `WebGL2WaveformRenderer`（peaks 数据上传为 RG32F 纹理，vertex shader 通过 `gl_InstanceID` + `texelFetch` 计算每像素列的 min/max quad），不支持则回退到现有 `Canvas2DWaveformRenderer`（即当前 `renderWaveform` 的薄封装）。两个消费方（`WaveformTrackCanvas`、`PianoRoll render.ts`）统一通过 renderer 接口消费，不再直接调用 `renderWaveform`。

**Tech Stack:** TypeScript / WebGL2 (GLSL ES 3.00) / Canvas 2D (fallback) / React

---

## 1. 背景与动机

### 1.1 当前实现

波形渲染链路（见 `waveformRenderer.ts:262-437`）：

```
waveformMipmapStore.getInterleavedSlice()
  → Float32Array [min0,max0,min1,max1,...]
  → applyGainsToPeaks()           // CPU 逐采样乘增益
  → renderWaveform()              // CPU 逐像素 fillRect
  → ctx.fillRect(px, yTop, w, h)  // 每像素列一条竖线
```

两个消费方共用同一渲染函数：
- `WaveformTrackCanvas.tsx:422` — 上方轨道区波形
- `pianoRoll/render.ts:830` — 下方 PianoRoll 背景波形

### 1.2 痛点

1. **CPU 开销集中**：`renderWaveform` 是 `O(W + N)` 的 CPU 循环（`waveformRenderer.ts:376-432`），每像素列都要 `moveTo/lineTo/fillRect`，单帧 5-15ms。
2. **Canvas 2D 状态机开销**：每个 `fillRect` 都要走 Canvas 状态机，GPU 加速路径不稳定。
3. **历史闪烁问题**：git 历史显示反复出现 DPR 振荡、离屏双缓冲回退等问题（`f12761e`、`47fea22`、`38daa07`）。
4. **多轨道叠加压力**：每个可见 clip 都要循环一次，100+ clip 时 CPU 压力显著。

### 1.3 选型理由

| 方案 | 平台支持 | 性能收益 | 复杂度 | 决策 |
|------|----------|----------|--------|------|
| 继续 Canvas 2D 优化 | 全平台 | 边际收益递减 | 低 | ❌ 已到瓶颈 |
| WebGL2 | 三平台 WebView 都稳定支持 | 比 Canvas 2D 快一个数量级 | 中 | ✅ 选用 |
| WebGPU | macOS/Linux WebView 支持不确定 | 比 WebGL2 再快 20-50% | 高 | ⏳ 未来演进 |

WebGL2 在 Tauri 2 的三平台 WebView 中支持成熟（Windows WebView2 / macOS WKWebView / Linux WebKitGTK），且波形渲染场景简单（instanced quad + 纹理查找），不需要 compute shader、多 pass 等 WebGPU 优势能力，**WebGL2 是当前最佳投入产出比**。

---

## 2. 目标与非目标

### 2.1 目标

1. **性能提升**：单帧波形渲染时间从 5-15ms 降到 1-3ms（量化基准见 §9）。
2. **闪烁根治**：消除 DPR 切换、尺寸振荡导致的帧间闪烁。
3. **统一渲染路径**：`WaveformTrackCanvas` 和 `PianoRoll` 背景波形共用同一 renderer。
4. **平滑降级**：WebGL2 不可用时自动回退到 Canvas 2D，用户无感。
5. **未来演进**：`WaveformRenderer` 接口设计兼容 WebGPU，后续可平替实现。

### 2.2 非目标

1. **不重构参数曲线渲染**：PianoRoll 的参数曲线（音高/张力/气声等）仍用 Canvas 2D，因为涉及文本、虚线、渐变等 Canvas 2D 擅长特性。
2. **不重构交互层**：`usePianoRollInteractions.ts` 的 pointer/wheel/keyboard 逻辑完全不动。
3. **不重构数据层**：`waveformMipmapStore`、`applyGainsToPeaks`、mipmap 三级缓存、spp 滞后防抖逻辑全部保留。
4. **不引入图形库**：不使用 three.js / pixi.js / regl，原生 WebGL2 足够简单。
5. **不做 WebGPU 实现**：本次只预留接口，不写 WebGPU 代码。

---

## 3. 总体架构

### 3.1 分层

```
┌─────────────────────────────────────────────────────────────┐
│ 消费层（不变）                                                │
│  WaveformTrackCanvas.tsx    pianoRoll/render.ts             │
└──────────────────┬──────────────────┬───────────────────────┘
                   │                  │
                   ▼                  ▼
┌─────────────────────────────────────────────────────────────┐
│ 渲染抽象层（新增）                                            │
│  WaveformRenderer 接口                                       │
│  waveformRendererFactory.ts（能力检测 + 实例化）              │
└──────────────────┬───────────────────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
┌─────────────────┐  ┌─────────────────────────────────────────┐
│ WebGL2 实现     │  │ Canvas 2D fallback                       │
│ (新增)          │  │ Canvas2DWaveformRenderer                 │
│ RG32F 纹理 +    │  │ （封装现有 renderWaveform）              │
│ instanced quad  │  │                                          │
└─────────────────┘  └─────────────────────────────────────────┘
                   ▲
                   │
┌─────────────────────────────────────────────────────────────┐
│ 数据层（不变）                                                │
│  waveformMipmapStore  →  Float32Array (interleaved min/max)  │
│  applyGainsToPeaks   →  Float32Array (应用增益后)            │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 数据流（WebGL2 路径）

```
waveformMipmapStore.getInterleavedSlice()
  → Float32Array peaks
  → applyGainsToPeaks(peaks, params)           // CPU（保留，因为淡入淡出曲线在 CPU 实现更简单）
  → WebGL2WaveformRenderer.drawClipWaveform(peaks, params, color, alpha)
      ├─ texSubImage2D(peaks → RG32F texture)  // 上传到 GPU
      ├─ uniform 更新（pxToIdxScale、视口、颜色、alpha 等）
      ├─ scissor rect 设置（clip 可视段裁剪）
      └─ drawArraysInstanced(TRIANGLE_STRIP, 0, 4, visibleWidthPx)
          ├─ vertex shader: gl_InstanceID → px → 索引范围 → texelFetch 取 min/max → quad 顶点
          └─ fragment shader: 输出颜色 × alpha
```

---

## 4. Renderer 抽象接口

### 4.1 接口定义

```typescript
// frontend/src/utils/waveformRenderer.ts（追加导出）

/**
 * 波形渲染器抽象接口
 *
 * 生命周期：
 *   1. createWaveformRenderer(canvas) → renderer   （工厂检测能力，返回 WebGL2 或 Canvas2D 实现）
 *   2. renderer.resize(displayW, displayH, dpr)     （尺寸变化时调用）
 *   3. renderer.clear()                             （每帧开始清空）
 *   4. renderer.drawClipWaveform(params) × N        （每可见 clip 调用一次）
 *   5. renderer.dispose()                           （canvas 卸载时调用）
 *
 * 实现类：
 *   - WebGL2WaveformRenderer：WebGL2 instanced quad + RG32F 纹理
 *   - Canvas2DWaveformRenderer：封装现有 renderWaveform
 *   - （未来）WebGPUWaveformRenderer：预留接口兼容性
 */
export interface WaveformRenderer {
    /** 调整 canvas 物理尺寸（CSS 像素 + DPR），实现内部处理物理像素映射 */
    resize(displayW: number, displayH: number, dpr: number): void;

    /** 清空画布（每帧开始调用） */
    clear(): void;

    /**
     * 绘制单个 clip 的一个可视段
     *
     * @param params 见 DrawClipWaveformParams
     */
    drawClipWaveform(params: DrawClipWaveformParams): void;

    /** 释放 GPU 资源（WebGL2 实现 destroy program/buffer/texture） */
    dispose(): void;

    /** 标识实现类型，用于诊断与日志 */
    readonly backend: "webgl2" | "canvas2d" | "webgpu";
}

/**
 * 单次 drawClipWaveform 调用参数
 *
 * 字段对应现有 WaveformRenderParams + 颜色 + alpha + 裁剪段
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
```

### 4.2 工厂函数

```typescript
// frontend/src/utils/waveformRendererFactory.ts

/**
 * 根据浏览器能力创建波形渲染器
 *
 * 检测顺序：
 *   1. 尝试 canvas.getContext('webgl2', { alpha: true, premultipliedAlpha: true })
 *   2. 若失败 → 回退到 canvas.getContext('2d')
 *   3. 极端情况（连 2d 都拿不到）→ 抛错
 *
 * @param canvas 目标 canvas 元素
 * @param enableWebGL2 是否启用 WebGL2 路径（默认 true，可通过配置关闭以强制走 fallback）
 */
export function createWaveformRenderer(
    canvas: HTMLCanvasElement,
    enableWebGL2: boolean = true,
): WaveformRenderer {
    if (enableWebGL2) {
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
                console.warn("[WaveformRenderer] WebGL2 init failed, fallback to Canvas2D:", e);
            }
        }
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("WaveformRenderer: neither WebGL2 nor Canvas2D available");
    return new Canvas2DWaveformRenderer(canvas, ctx);
}
```

---

## 5. WebGL2 Renderer 实现细节

### 5.1 资源管理

每个 `WebGL2WaveformRenderer` 实例持有一套 GL 资源（对应一个 canvas）：

| 资源 | 数量 | 说明 |
|------|------|------|
| Program | 1 | 顶点 + 片元着色器链接 |
| VertexArrayObject (VAO) | 1 | 空 VAO，因为顶点全在 shader 里生成 |
| Texture | 1 | RG32F，宽度 = maxSamples，高度 = 1，动态 `texSubImage2D` 更新 |
| Uniform locations | ~10 | 缓存 location 避免 per-frame 查询 |

**纹理尺寸策略**：初始化时分配 `MAX_PEAK_SAMPLES = 65536`（对应 128KB RG32F），覆盖一首 5 分钟歌 L0 级别的数据量（~124000 对，但可见窗口通常远小于此）。若 peaks 超过容量，自动 `texImage2D` 重新分配（罕见路径）。

### 5.2 Shader 设计

#### Vertex Shader（GLSL ES 3.00）

```glsl
#version 300 es
precision highp float;

// ===== Uniforms =====
uniform int u_visibleStartPx;    // 可视段起始像素列（CSS 像素，整数）
uniform int u_visibleEndPx;      // 可视段结束像素列
uniform float u_pxToIdxScale;    // 像素 → peaks 索引的线性系数
uniform float u_pxToIdxBase;     // 像素 → peaks 索引的基准值
uniform float u_halfPixelIdx;    // 0.5 像素覆盖的索引偏移
uniform int u_totalSamples;      // peaks 总采样对数
uniform float u_amplitudeScale;  // 振幅缩放（zeroDbHalfHeight）
uniform float u_centerY;         // 波形中心 Y（CSS 像素）
uniform float u_displayW;        // canvas CSS 宽度
uniform float u_displayH;        // canvas CSS 高度
uniform float u_strokeWidth;     // 描边宽度（CSS 像素）
uniform sampler2D u_peaksTex;    // RG32F 纹理，R=min, G=max

// ===== 输出 =====
out float v_alphaFactor;  // 传给 fragment 的 alpha 修正（静音段最小高度处理）

void main() {
    // 每个 instance 对应一个像素列
    int px = u_visibleStartPx + gl_InstanceID;
    float pxF = float(px);

    // 计算该像素列覆盖的索引范围
    float centerIdx = pxF * u_pxToIdxScale + u_pxToIdxBase;
    float idxLeft = max(0.0, centerIdx - u_halfPixelIdx);
    float idxRight = min(float(u_totalSamples - 1), centerIdx + u_halfPixelIdx);

    int iStart = int(floor(idxLeft));
    int iEnd = int(ceil(idxRight));

    // 扫描范围内取 min/max
    float pixelMin = 1e38;
    float pixelMax = -1e38;
    for (int i = 0; i < 4096; i++) {  // 上限保护，实际循环到 iEnd 即 break
        if (i > iEnd) break;
        if (i < iStart) continue;
        vec2 peak = texelFetch(u_peaksTex, ivec2(i, 0), 0).rg;
        if (peak.r < pixelMin) pixelMin = peak.r;
        if (peak.g > pixelMax) pixelMax = peak.g;
    }

    // 退化情况：无数据
    if (pixelMin > 1e37) {
        gl_Position = vec4(2.0, 2.0, 0.0, 1.0);  // 裁掉
        return;
    }

    // 计算 quad 四角的 NDC
    float yTop = u_centerY - pixelMax * u_amplitudeScale;
    float yBot = u_centerY - pixelMin * u_amplitudeScale;

    // 静音段最小可见高度（0.5px，对齐现有 renderWaveform 逻辑）
    float halfStroke = u_strokeWidth * 0.5;
    if (yBot - yTop < 0.5) {
        float midY = (yTop + yBot) * 0.5;
        yTop = midY - 0.25;
        yBot = midY + 0.25;
    }

    // gl_VertexID 0~3 → quad 四角（TRIANGLE_STRIP）
    float xLeft = pxF - halfStroke;
    float xRight = pxF + halfStroke;
    float x = (gl_VertexID == 0 || gl_VertexID == 2) ? xLeft : xRight;
    float y = (gl_VertexID == 0 || gl_VertexID == 1) ? yTop : yBot;

    // CSS 像素 → NDC
    float ndcX = (x / u_displayW) * 2.0 - 1.0;
    float ndcY = 1.0 - (y / u_displayH) * 2.0;

    gl_Position = vec4(ndcX, ndcY, 0.0, 1.0);
    v_alphaFactor = 1.0;
}
```

#### Fragment Shader

```glsl
#version 300 es
precision highp float;

uniform vec4 u_color;       // 预乘 alpha 的 RGBA
uniform float u_alpha;      // 整体透明度（muted / leadingOverlap）

in float v_alphaFactor;
out vec4 fragColor;

void main() {
    fragColor = u_color * u_alpha * v_alphaFactor;
}
```

### 5.3 渲染流程（每次 drawClipWaveform 调用）

```typescript
drawClipWaveform(params: DrawClipWaveformParams): void {
    const { peaks, renderParams, segmentLeftPx, segmentRightPx, strokeColor, strokeWidth, alpha } = params;

    // 1. 上传 peaks 数据到纹理
    this.ensureTextureCapacity(peaks.length / 2);
    this.gl.bindTexture(this.gl.TEXTURE_2D, this.peaksTex);
    this.gl.texSubImage2D(
        this.gl.TEXTURE_2D, 0, 0, 0,
        peaks.length / 2, 1,
        this.gl.RG, this.gl.FLOAT, peaks,
    );

    // 2. 计算 uniform 值（与现有 renderWaveform 的 pxToIdxScale 等公式完全一致）
    const uniforms = this.computeUniforms(renderParams, segmentLeftPx, segmentRightPx);

    // 3. 设置 scissor rect（clip 可视段裁剪，替代 ctx.clip()）
    const scissorX = segmentLeftPx * this.dpr;
    const scissorW = (segmentRightPx - segmentLeftPx) * this.dpr;
    this.gl.scissor(scissorX, 0, scissorW, this.physicalH);
    this.gl.enable(this.gl.SCISSOR_TEST);

    // 4. 更新 uniform
    this.setUniforms(uniforms, strokeColor, strokeWidth, alpha);

    // 5. instanced draw
    const instanceCount = uniforms.visibleEndPx - uniforms.visibleStartPx;
    this.gl.drawArraysInstanced(this.gl.TRIANGLE_STRIP, 0, 4, instanceCount);

    this.gl.disable(this.gl.SCISSOR_TEST);
}
```

### 5.4 DPR 与坐标系

- **CSS 像素输入**：所有 `DrawClipWaveformParams` 的坐标都是 CSS 像素，与现有 `renderWaveform` 一致。
- **物理像素执行**：`resize()` 时记录 `this.dpr`，`gl.viewport` 用物理像素，`scissor` 用物理像素，shader 里 `u_displayW/displayH` 传 CSS 像素用于 NDC 转换。
- **像素对齐**：`segmentLeftPx` 在 CPU 端做 `Math.round`（与现有 `clipPixelOffset` 量化策略一致），消除子像素漂移。

### 5.5 颜色解析

`strokeColor` 是 CSS 颜色字符串（如 `"#7c9eff"` 或 `"rgba(124,158,255,0.86)"`）。WebGL2 renderer 内部维护一个 `ColorCache`：

```typescript
private colorCache = new Map<string, [number, number, number, number]>();

private parseColor(css: string): [number, number, number, number] {
    let cached = this.colorCache.get(css);
    if (!cached) {
        // 复用 canvas 2d 解析（任何 canvas 元素都可以）
        this.colorCtx.fillStyle = css;
        this.colorCtx.fillRect(0, 0, 1, 1);
        const data = this.colorCtx.getImageData(0, 0, 1, 1).data;
        cached = [data[0] / 255, data[1] / 255, data[2] / 255, data[3] / 255];
        this.colorCache.set(css, cached);
    }
    return cached;
}
```

---

## 6. Canvas 2D Fallback 实现

`Canvas2DWaveformRenderer` 是现有 `renderWaveform` 的薄封装，保持行为完全一致：

```typescript
class Canvas2DWaveformRenderer implements WaveformRenderer {
    backend = "canvas2d" as const;

    constructor(private canvas: HTMLCanvasElement, private ctx: CanvasRenderingContext2D) {}

    resize(displayW: number, displayH: number, dpr: number): void {
        // 复用 WaveformTrackCanvas.tsx:197-217 的尺寸管理逻辑
        // 仅当物理尺寸真正变化时才设置 canvas.width/height
    }

    clear(): void {
        this.ctx.clearRect(0, 0, this.canvas.width / this.dpr, this.canvas.height / this.dpr);
    }

    drawClipWaveform(params: DrawClipWaveformParams): void {
        const { peaks, renderParams, segmentLeftPx, segmentRightPx, strokeColor, strokeWidth, alpha } = params;
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

---

## 7. 集成点改造

### 7.1 `WaveformTrackCanvas.tsx` 改造

**变更范围**：`WaveformTrackCanvas.tsx:96-610`

改造要点：
1. 初始化时调用 `createWaveformRenderer(canvas)`，存入 `useRef`。
2. `drawRef.current` 内部的 `drawSegment` 函数（`WaveformTrackCanvas.tsx:410-431`）从 `ctx.save/clip/renderWaveform/restore` 改为 `renderer.drawClipWaveform`。
3. `dimsChanged` 分支（`WaveformTrackCanvas.tsx:203-217`）调用 `renderer.resize` 替代直接设置 `canvas.width/height`。
4. `ctx.clearRect` 改为 `renderer.clear()`。
5. 卸载时调用 `renderer.dispose()`。

**保留不变**：
- `invalidate` + rAF 帧合并架构（`WaveformTrackCanvas.tsx:144-150`）
- 高频参数 ref 化策略（`WaveformTrackCanvas.tsx:117-136`）
- `timelineViewportBus` 订阅（`WaveformTrackCanvas.tsx:538-555`）
- mipmap 选级、`getInterleavedSlice`、`applyGainsToPeaks`、预降采样（`WaveformTrackCanvas.tsx:227-387`）
- leadingOverlap 双段混合逻辑（`WaveformTrackCanvas.tsx:433-442`），只是把 `drawSegment` 替换为 `renderer.drawClipWaveform`
- buffer 复用池（`WaveformTrackCanvas.tsx:50-69`）
- 性能诊断（`WaveformTrackCanvas.tsx:487-511`）

**伪代码对比**：

```typescript
// 旧代码
const drawSegment = (leftPx, rightPx, alpha) => {
    ctx.save();
    ctx.beginPath();
    ctx.rect(leftPx, 0, rightPx - leftPx, displayH);
    ctx.clip();
    ctx.globalAlpha = alpha;
    renderWaveform(ctx, withGains, params, strokeColor, strokeWidth, "line");
    ctx.restore();
};

// 新代码
const drawSegment = (leftPx, rightPx, alpha) => {
    renderer.drawClipWaveform({
        peaks: withGains,
        renderParams: params,
        segmentLeftPx: leftPx,
        segmentRightPx: rightPx,
        strokeColor,
        strokeWidth,
        alpha,
    });
};
```

### 7.2 `pianoRoll/render.ts` 改造

**变更范围**：`render.ts:730-837`（背景波形循环）

改造要点：
1. `drawPianoRoll` 函数签名追加 `renderer: WaveformRenderer` 参数（由 `PianoRollPanel.tsx` 在调用时传入）。
2. `ctx.clearRect`（在循环开始前）改为 `renderer.clear()`。
3. 循环内 `ctx.save/clip/translate/renderWaveform/restore`（`render.ts:811-832`）替换为 `renderer.drawClipWaveform`，注意 `clipVisLeft` 作为 segmentLeftPx 传入。
4. 其他部分（参数曲线、网格、播放头等）保持 Canvas 2D 调用不变。

**关键约束**：PianoRoll 的主 canvas 既画波形也画参数曲线。WebGL2 renderer 只能画波形，参数曲线仍需 Canvas 2D。两种方案：
- **方案 A**：PianoRoll 主 canvas 保持 Canvas 2D，波形部分调用 `Canvas2DWaveformRenderer`（不升级）。
- **方案 B**：PianoRoll 主 canvas 保持 Canvas 2D，但在背景波形层之上叠加一个独立的 WebGL2 canvas 专画波形。
- **方案 C**：PianoRoll 主 canvas 用 WebGL2，参数曲线渲染改为在 WebGL2 canvas 上用 Canvas 2D 纹理叠加（复杂度高）。

**推荐方案 B**：在 PianoRoll 里新增一个绝对定位的 `<canvas>` 专画背景波形，z-index 在主 canvas 之下，用 `WebGL2WaveformRenderer`。主 canvas 保持 Canvas 2D 画参数曲线，每帧 clear 时波形 canvas 也同步 clear。这样波形升级到 WebGL2，参数曲线完全不动。

> 注：方案 B 会引入一个额外的 canvas 元素，但 PianoRoll 本身已有 axisCanvas 和主 canvas 两个，再加一个背景波形 canvas 是可接受的。

---

## 8. Fallback 检测与切换

### 8.1 检测时机

仅在 `createWaveformRenderer` 工厂函数中检测一次，结果缓存在 renderer 实例上。**不做运行时降级切换**（即 WebGL2 context lost 时不动态切回 Canvas 2D，因为实现复杂且 context lost 极罕见）。

### 8.2 Context Lost 处理

WebGL2 context lost（`webglcontextlost` 事件）时：
1. 监听事件，触发后调用 `renderer.dispose()`。
2. 重新调用 `createWaveformRenderer(canvas)` 创建新实例。
3. 不主动切换到 Canvas 2D，让工厂函数再次尝试 WebGL2（大多数 context lost 是暂时的，可以恢复）。

### 8.3 强制 fallback 开关

通过 `localStorage.hifishifter.forceCanvas2DWaveform = "1"` 可以强制走 Canvas 2D 路径，用于：
- WebGL2 实现出现 bug 时的应急回退
- 性能对比测试
- 兼容性问题排查

工厂函数读取该配置：

```typescript
const forceCanvas2D =
    typeof window !== "undefined" &&
    window.localStorage?.getItem("hifishifter.forceCanvas2DWaveform") === "1";

return createWaveformRenderer(canvas, !forceCanvas2D);
```

---

## 9. 性能验证方案

### 9.1 量化基准

沿用 `waveformDebug.ts` 的诊断框架，对比指标：

| 指标 | Canvas 2D 现状 | WebGL2 目标 | 测量方式 |
|------|----------------|-------------|----------|
| 单帧总耗时 | 5-15ms | 1-3ms | `performance.now()` 包裹整帧 |
| 每 clip 渲染耗时 | 1-5ms | 0.1-0.5ms | per-clip 计时 |
| 滚动流畅度 | 偶发掉帧 | 稳定 60fps | `requestAnimationFrame` 间隔统计 |
| 内存占用 | Float32Array 池 | +纹理 ~128KB | DevTools Memory 面板 |

### 9.2 测试场景

1. **单 clip 中等缩放**：1 个 3 分钟 clip，pxPerSec = 100，验证基础正确性。
2. **多 clip 高密度**：20 个 clip 同屏，pxPerSec = 200，验证多 clip 叠加。
3. **极限缩放**：pxPerSec = 5（全曲概览），验证 L2 mipmap 路径。
4. **快速滚动**：连续滚动 10 秒，验证无闪烁、无掉帧。
5. **DPR 切换**：拖动到外接显示器，验证尺寸自适应。

### 9.3 A/B 对比

通过 `localStorage.hifishifter.forceCanvas2DWaveform` 开关，可在运行时切换两个实现，配合 `localStorage.hifishifter.debugWaveformPerf = "1"` 输出帧统计，做实时对比。

---

## 10. 风险与缓解

| 风险 | 影响 | 缓解 |
|------|------|------|
| Shader 在某些 GPU 表现异常 | 渲染错误 | 保留 Canvas 2D 路径，可通过 localStorage 开关切换 |
| 纹理上传成为新瓶颈 | 性能不及预期 | RG32F 单次上传 ~1MB，实测带宽充足；必要时改用 buffer texture |
| Fragment shader alpha 混合与 Canvas 2D 不一致 | 视觉差异 | 使用 `premultipliedAlpha: true`，与 Canvas 2D 行为对齐 |
| DPR 处理不当导致模糊 | 视觉退化 | 严格区分 CSS 像素与物理像素，viewport 和 scissor 用物理像素 |
| leadingOverlap 双段混合视觉与原版不一致 | 用户感知差异 | 保留双次 draw call + 不同 alpha 的策略，逐段对比 |
| PianoRoll 双 canvas 方案导致层级错乱 | UI bug | 明确 z-index：背景波形 canvas < 主 canvas < 交互层 |
| WebGL2 context lost | 渲染中断 | 监听事件，重新创建 renderer 实例 |

---

## 11. 未来 WebGPU 演进路径

本次设计的 `WaveformRenderer` 接口已考虑 WebGPU 兼容性：

1. **接口兼容**：`resize/clear/drawClipWaveform/dispose` 四个方法都是渲染无关的高层抽象，WebGPU 实现可以直接满足。
2. **资源管理对应**：WebGL2 的 program/texture/VAO 对应 WebGPU 的 pipeline/texture/GPUBuffer，封装在实现内部。
3. **数据上传策略**：WebGPU 用 `queue.writeBuffer` 替代 `texSubImage2D`，但调用时机和数据源一致。
4. **演进时机**：当 macOS WKWebView 的 WebGPU 支持稳定，或项目放弃对老平台支持时，新增 `WebGPUWaveformRenderer` 类即可，工厂函数检测顺序变为 WebGPU → WebGL2 → Canvas 2D。

**本次不写 WebGPU 代码**，但接口设计不留技术债。

---

## 12. 实施步骤分解

> 详细到任务级别的实施计划将由 writing-plans skill 生成，这里只列阶段性里程碑。

### 阶段 1：基础设施（无 UI 改动）
1. 新增 `waveformRenderer.ts` 的 `WaveformRenderer` 接口和 `DrawClipWaveformParams` 类型
2. 实现 `Canvas2DWaveformRenderer`（纯封装，行为与现有 `renderWaveform` 一致）
3. 实现 `waveformRendererFactory.ts`（含 localStorage 强制开关）
4. 单元测试：验证 `Canvas2DWaveformRenderer` 输出与 `renderWaveform` 像素级一致

### 阶段 2：WebGL2 Renderer 实现
5. 新增 `waveformWebGL2Renderer.ts`，实现 shader 编译、纹理创建、uniform 管理
6. 实现 `drawClipWaveform`，对齐 `renderWaveform` 的坐标系与像素对齐逻辑
7. 单独测试页（开发时临时用，不合并）验证：单 clip、多 clip、muted、leadingOverlap

### 阶段 3：集成 WaveformTrackCanvas
8. 改造 `WaveformTrackCanvas.tsx`，替换 `drawSegment` 实现
9. 验证上方轨道区波形与原版视觉一致
10. 性能对比测试

### 阶段 4：集成 PianoRoll 背景波形
11. 在 `PianoRollPanel.tsx` 新增背景波形 canvas（方案 B）
12. 改造 `pianoRoll/render.ts`，把波形循环抽到独立函数，由背景 canvas 的 renderer 调用
13. 验证下方背景波形与原版一致，且参数曲线不受影响

### 阶段 5：收尾
14. 删除临时测试代码，清理 `renderWaveform` 的旧调用路径（若已无引用）
15. 更新 `waveformDebug.ts` 增加 renderer backend 标识
16. 文档更新：在 `waveformRenderer.ts` 文件头注释中补充 WebGL2 路径说明

---

## 附录 A：相关文件清单

| 文件 | 角色 | 变更类型 |
|------|------|----------|
| `frontend/src/utils/waveformRenderer.ts` | 接口定义 + Canvas 2D 实现 | 修改（追加导出） |
| `frontend/src/utils/waveformWebGL2Renderer.ts` | WebGL2 实现 | 新增 |
| `frontend/src/utils/waveformRendererFactory.ts` | 工厂 + 能力检测 | 新增 |
| `frontend/src/components/waveform/WaveformTrackCanvas.tsx` | 上方轨道区消费方 | 修改 |
| `frontend/src/components/layout/pianoRoll/render.ts` | PianoRoll 背景波形 | 修改 |
| `frontend/src/components/layout/PianoRollPanel.tsx` | 新增背景波形 canvas | 修改 |
| `frontend/src/utils/waveformMipmapStore.ts` | 数据层 | 不变 |
| `frontend/src/utils/waveformDebug.ts` | 诊断 | 修改（增加 backend 标识） |

## 附录 B：坐标系与公式对照

WebGL2 renderer 的 uniform 计算公式与 `waveformRenderer.ts:343-352` 完全一致，对照如下：

| 公式 | Canvas 2D 现有代码 | WebGL2 uniform |
|------|---------------------|----------------|
| 像素 → 索引斜率 | `pxToIdxScale = (reversed ? -1 : 1) * pxToTimeScale * timeToIdxScale` | `u_pxToIdxScale` |
| 像素 → 索引基准 | `pxToIdxBase = reversed ? (...) : (...)` | `u_pxToIdxBase` |
| 半像素索引偏移 | `halfPixelIdx = Math.abs(0.5 * pxToIdxScale)` | `u_halfPixelIdx` |
| 振幅缩放 | `amplitudeScale = params.zeroDbHalfHeight ?? canvasHeight / 2` | `u_amplitudeScale` |
| 中心 Y | `centerY` | `u_centerY` |

vertex shader 内的索引范围扫描逻辑对应 `waveformRenderer.ts:376-401` 的 CPU 循环，只是搬到 GPU 并行执行。
