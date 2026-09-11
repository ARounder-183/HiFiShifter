# 时间轴统一渲染内核 · 阶段 0（Spike）报告

- 日期：2026-09-11
- 分支：`feature/timeline-unified-render-kernel`
- 设计：`docs/plans/2026-09-11-timeline-unified-render-kernel-design.md`
- 实施计划：`docs/plans/2026-09-11-timeline-unified-render-kernel-implementation.md`
- 状态：**代码与自动化验证已完成；真机性能与手感待填（见 §5）**

---

## 1. 交付清单

新内核代码全部位于 `frontend/src/components/layout/timeline/kernel/`（Spike 期间与
既有 `runtime/` 目录保持隔离，仅明确列出的复用模块除外）。

| 模块 | 文件 | 职责 |
|---|---|---|
| 滚动内核 | `scrollKernel.ts` | 视口真值 + 独占钳制 + 锚点缩放 + `reclamp` |
| 输入 | `input/normalizeWheel.ts` | 滚轮增量归一化（pixel/line/page） |
| 输入 | `input/scrollbars.ts` | 自绘滚动条几何 + thumb 命中 + 拖拽换算 |
| 字形 | `glyph/glyphLayout.ts` | 文本切分 + 测量缓存 + 截断 |
| 字形 | `glyph/glyphAtlas.ts` | 多页 shelf-pack 分配器 |
| 字形 | `glyph/glyphRasterizer.ts` | 离屏 Canvas2D 按 dpr 光栅化到图集 |
| 场景 | `scene/instanceTypes.ts` | 平面矩形实例类型 |
| 场景 | `scene/gridInstances.ts` | 网格实例（内容坐标，竖直不裁剪） |
| 场景 | `scene/clipInstances.ts` | clip 实例（复用既有样式与实例写入） |
| GL | `gl/glRaster.ts` | 光栅化参数（物理 / 绘制坐标系换算） |
| GL | `gl/instanceBuffer.ts` | 实例缓冲容量策略（倍增） |
| GL | `gl/instanceLayout.ts` | 25 float 实例布局唯一来源 + FLAT 写入 |
| GL | `gl/glContext.ts` | WebGL2 上下文封装 |
| GL | `gl/sdfBoxProgram.ts` | 圆角盒 / 平面矩形实例化 program |
| GL | `gl/glyphQuads.ts` | 字形四边形构建（uv 换算） |
| GL | `gl/glyphProgram.ts` | 字形纹理四边形 program |
| 循环 | `renderLoop.ts` | 脏标记 + rAF 调度（按需，非常驻） |
| 宿主 | `host/timelineKernelHost.ts` | 装配 + 输入 + 渲染循环 + 场景重建 |
| 外壳 | `TimelineKernelSpikeView.tsx` | React 外壳（DOM 节点 + 数据镜像） |
| 开关 | `featureFlag.ts` | `hifishifter.timelineKernel`（默认关） |

接入点：`TimelinePanel.tsx` 在时间轴区域条件渲染内核视图（覆盖式，`absolute inset-0 z-30`）。

---

## 2. 关键设计落地（静态可验证部分）

| 设计目标 | 落地方式 | 验证手段 |
|---|---|---|
| 取消跨层 sync | 没有任何随原生滚动移动的 DOM 内容层；渲染完全由 rAF 调度 | 代码结构（宿主无原生 scroller） |
| 滚动零几何重建 | 几何以内容坐标常驻；滚动帧只更新 `u_viewOrigin`（`render`/`repaint` 双入口） | `gridInstances.test.ts` 的"竖直不裁剪"断言；`sdfBoxProgram.repaint` 零上传 |
| 单次提交 | 网格 + clip 共用一份实例缓冲、一次 `drawArraysInstanced`；文字一次 draw call | 宿主 `draw()` 的两次 render 调用 |
| 钳制唯一 | 全部边界钳制收敛在 `ScrollKernel`，外部边界变化走 `reclamp()` | `scrollKernel.test.ts`（14 条） |
| 缓冲零分配 | 实例缓冲倍增复用（`resolveBufferFloats`，三处共用） | `instanceBuffer.test.ts` + `clipInstances.test.ts` |
| 文字质量 | 按 dpr 光栅化到图集 + NEAREST 采样 + 预乘 alpha 混合 | 真机比对（§5） |

---

## 3. 测试与质量门

- 单元测试：**93 条**（12 个文件），覆盖滚动内核、输入、字形布局 / 图集 / 光栅化参数、
  网格与 clip 实例、GL 光栅化参数、缓冲策略、渲染循环、滚动条几何。
- 类型检查：`npx tsc -b --noEmit` 通过。
- Lint：`npx eslint`（kernel 目录 + TimelinePanel）无输出。
- 格式：`npx prettier --check` 全绿。
- 生产构建：`npm run build` 通过（2.78s）。
- 全量回归：`npx vitest run` → **412 passed / 2 failed**；2 个失败位于
  `src/features/keybindings/keybindingMatch.test.ts`，**经 develop 分支复现确认为预先存在**，
  与本次改动无关。

### 三轮独立审查记录

| 轮次 | 范围 | 发现 | 处理 |
|---|---|---|---|
| 1 | Task 1（scrollKernel） | 1 实质（测试断言无判别力）+ 1 轻微（注释缺 `@param`） | 已修 |
| 2 | Task 1（质量） | 3 Important（`reclamp` 缺失 / 引用稳定无测试 / 水平上限语义与既有实现相反）+ 6 Minor | 已修（含语义对齐既有 `resolveTimelineScrollRange`） |
| 3 | Task 2–4 | 5 处测试判别力不足（上下文写死 / 码点切分 / 边界上限 / 纵向 padding / 页复用）+ 4 其他 | 已修 |
| 4 | Task 5–8 | 1 Critical（**混合状态缺失**）+ 6 Important（着色器注释 / 错误路径泄漏 / 属性静默 / CSS 尺寸 / 页边长耦合 / 行高契约）+ 8 Minor | 已修（Critical 与全部 Important，关键 Minor） |

---

## 4. 已知简化与 A/B 预期差异

### 4.1 功能简化（Spike 范围外，不影响结论）

1. **只画网格 / clip 块面 / clip 名称文字**；细节层（旋钮 / 徽标 / fade 曲线）、波形、
   播放头、标尺、轨道头均未接入。
2. **字形图集单页**（2048²，`maxPages: 1`）；页满后新字形不再生成（跳过绘制）。
3. **主题色为固定前景色**，未接入主题变量。
4. **clip 数据直读 session**，未接选区 / 悬停 / 重叠等交互态（分隔缝判定已实现）。
5. **编组激活的 clip 仍走 GL 块面**（既有实现会退回 Canvas2D 画外圈描边）——视觉上
   少一个 2px 外圈。

### 4.2 A/B 时**不应**判为回归的差异

1. **网格竖线相位**：既有 SVG 用"居中描边 + 设备像素吸附"，强线覆盖物理列
   `[N−1, N+1]`、弱线 `[N−0.5, N+0.5]`（两列各半强度）；内核用"矩形左缘吸附"，强线覆盖
   `[N, N+2]`、弱线 `[N, N+1]`（1 列满强度）。即强线相位差 1 物理像素、弱线更锐利。
   内核实现符合设计规格（左缘吸附），属可预期的实现差异。
2. **文字基线**：光栅化用 `textBaseline = "top"`、槽位坐标即字形左上角；与既有
   Canvas2D 绘制路径（同样 top 基线）在视觉上应一致，但字体渲染管线不同（纹理采样 vs
   直接绘制），边缘可能有细微差异。

---

## 5. 真机验证（待执行）

### 5.1 步骤

> **必须在桌面应用里验证**：前端依赖 pywebview 后端（`get_ui_settings` 等），
> 纯浏览器（`cd frontend && npm run dev`）会因 "Python API not available" 报错，
> 应用本身跑不起来。

```bash
# 1. 启动桌面应用开发模式（会自动拉起 Vite dev server 并打开桌面窗口）
cd backend/src-tauri && cargo tauri dev

# 2. 在应用窗口的开发者控制台开启内核（然后刷新）
localStorage.setItem("hifishifter.timelineKernel", "1"); location.reload();

# 3. 生成性能工程（PERF 面板右下角，或控制台）
window.__hsPerf?.generate?.("400");   // 10 轨 × 40 clip
# 或 1000 clip：window.__hsPerf?.generate?.("1000")
```

### 5.2 采集项

| 场景 | 采集方式 | 记录 |
|---|---|---|
| 连续水平滚动 5s（鼠标滚轮） | DevTools Performance 录制 | p50 / p95 / p99 帧耗时、长任务数 |
| 连续水平滚动 5s（触摸板） | 同上 | 手感（惯性是否自然、有无跳变） |
| Ctrl+滚轮缩放（放大 → 缩小） | 同上 | 缩放锚点是否稳定（指针下的内容不动） |
| 滚动条拖拽 | 目视 | thumb 是否跟手、有无抖动 |
| 文字质量 | 与关闭 flag 的截图对比 | clip 名称是否清晰、有无串字 |

### 5.3 待填数据

| 指标 | 目标 | 实测 |
|---|---|---|
| 滚动帧几何重建次数 | 0（余量内） | 待填 |
| 滚动帧 p95 耗时 | < 2ms | 待填 |
| 滚动帧率 | 稳定 60fps | 待填 |
| 缩放手势帧率 | ≥ 55fps | 待填 |
| 触摸板手感 | 可接受 | 待填 |
| 文字质量 | 与既有实现可比 | 待填 |

---

## 6. 结论

**待真机数据填入后判定**：

- 若滚动帧零重建、p95 < 2ms、手感与文字质量可接受 → **推进阶段 1**（按区域接入：
  轨道区 → 标尺 → 轨道头，并迁移交互 controller）。
- 若文字质量或手感不达标 → 退回设计文档 §5 的备选形态（WebGL2 主渲染 + Canvas2D
  细节层），或先解决具体阻塞项再评估。

**当前建议**：先执行 §5.1 的验证步骤；即使暂不推进阶段 1，本分支的 11 个模块
（滚动内核、字形图集、实例构建、GL program、渲染循环）都是可独立复用的资产，
且 `hifishifter.timelineKernel` 默认关闭，对既有实现零影响。
