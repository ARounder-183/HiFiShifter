# renderKernel — 面板共享的渲染内核模块

## 这个目录是什么

时间轴面板与参数编辑器面板**共用**的底层渲染设施：WebGL2 上下文与 program、
实例缓冲与布局、字形图集 / 布局 / 光栅化、帧循环、滚动内核、时间轴投影、
画布光栅化契约。

## 为什么单独成目录

这些模块原先散落在 `timeline/kernel/` 与 `timeline/runtime/` 下。参数编辑器在
阶段 1/2 的迁移中开始复用它们（滚动内核、GL program、字形管线、投影换算），
于是出现了"参数编辑器 import 时间轴内部目录"的耦合：目录位置暗示了错误的归属，
也让"这两个模块是否属于同一层"变得不可判断。

抽取到中性目录后：依赖方向是 `timeline` / `pianoRoll` → `renderKernel`，
两个面板之间没有直接依赖。

## 目录内容

| 路径                       | 作用                                          |
| -------------------------- | --------------------------------------------- |
| `gl/glContext.ts`          | WebGL2 上下文句柄（resize / clear / dispose） |
| `gl/glRaster.ts`           | 物理像素 vs CSS 像素的光栅化目标换算          |
| `gl/instanceBuffer.ts`     | 实例缓冲容量增长策略                          |
| `gl/instanceLayout.ts`     | 实例字段布局（`CLIP_INSTANCE_FLOATS`）与写入  |
| `gl/sdfBoxProgram.ts`      | SDF 圆角盒 / 平面矩形 program                 |
| `gl/glyphProgram.ts`       | 字形图集纹理 + 四边形 program                 |
| `gl/glyphQuads.ts`         | 字形四边形构建（内容坐标 + uv）               |
| `glyph/glyphAtlas.ts`      | 字形图集的货架式分配                          |
| `glyph/glyphLayout.ts`     | 按字符测量与截断的文本布局                    |
| `glyph/glyphRasterizer.ts` | 离屏 Canvas2D 光栅化到图集                    |
| `renderLoop.ts`            | 请求式帧调度（invalidate → rAF）              |
| `scrollKernel.ts`          | 视口真值与钳制（滚动位置的唯一事实源）        |
| `scrollbars.ts`            | 滚动条几何与拖拽 / 翻页换算                   |
| `timelineAxis.ts`          | 时间 ↔ 内容坐标 ↔ 视口坐标投影                |
| `canvasRaster.ts`          | 画布物理尺寸与清屏契约                        |
| `instanceTypes.ts`         | `FlatInstance` / `Rgba` 等共享实例类型        |

## 为什么没有留 re-export 兼容层

迁移时把 `src/` 内全部 79 处 import 一次性改到了新路径，并确认：

- `src/` 之外（脚本、配置、测试工程）没有任何引用；
- 工程未配置路径别名（tsconfig / vite 都无 `paths` / `alias`）。

因此旧路径的 re-export 文件会立刻变成**没有任何调用方的死代码**。与其留一份
会随实现演进而腐烂的转发层，不如让"旧路径不存在"这件事本身就是清晰的信号。

## 回归门

迁移后逐项验证（见 `docs/superpowers/plans/2026-09-12-pianoroll-kernel-phase2.md`）：

- `npx tsc -b --noEmit` 干净；
- `npx vitest run` 676 passed（与迁移前逐项一致，仅 2 个既有失败）；
- 时间轴内核 `KERNEL` 开关 on vs off：**0 像素差异**；
- 参数编辑器 GL on vs off：**7538 px（0.0818%）**，与迁移前**逐值相同**。
