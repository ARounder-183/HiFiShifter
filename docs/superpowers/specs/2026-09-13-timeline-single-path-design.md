# 时间轴渲染内核收归唯一路径 · 设计

- 日期：2026-09-13
- 状态：已实施（见 docs/superpowers/plans/2026-09-13-timeline-single-path.md）
- 相关：`docs/superpowers/specs/2026-09-12-pianoroll-kernel-migration-design.md`、`docs/superpowers/plans/2026-09-12-pianoroll-kernel-phase3.md`

## 1. 背景

`11276163`（内核改为默认渲染路径）之后，时间轴上同时存在两条完整实现：

- **内核**：`TimelineKernelView` + `timelineKernelHost`，自绘滚动 + 单 WebGL2，持有视口真值。
- **旧实现**：`TimelineScrollArea` + `TimelineSurface` + `TimelineCanvasViewport` + `TrackLane`/`ClipItem`，原生滚动 + Canvas2D。
  （注：`BackgroundGrid` 由两者**共用**——旧实现经 `TimelineSurface`，参数编辑器经 barrel 直接用，因此**不在删除范围**。）

两者由 `TimelinePanel` 的一个三元表达式在运行期二选一（`:4761`），开关读 `localStorage`。

问题在于：**旧实现已被破坏，无法正常使用**（用户判断）。因此

1. 保留一条"已损坏的回退路径"没有价值——它不会被用到，却持续产生维护与认知成本；
2. 更严重的是，先前的软着陆设计（见 §3）恰恰是**回退到这条已损坏的路径**，等于把用户从"空白"送到"坏掉的界面"；
3. 两条实现并存会让每一个后续改动都要"改两遍 + 验证两遍"，而其中一条根本不工作。

**本设计的目标**：让时间轴内核成为唯一路径，彻底移除旧实现与相关开关；同时把"WebGL2 不可用"这一失败场景做成**可自助排障的明确报错**，而不是静默空白或退回坏实现。

## 2. 决策记录

### 2.1 为什么可以没有回退：CPU-only 环境实测

"内核是唯一路径"只有在 WebGL2 覆盖足够广时才成立，因此先验证纯 CPU 环境。四种渲染后端配置（真实 Chrome，非模拟）：

| 配置 | WebGL2 | 内核挂载 | 渲染器 |
| --- | --- | --- | --- |
| 正常 GPU | ✅ | ✅ | ANGLE Metal |
| 强制 SwiftShader（纯 CPU） | ✅ | ✅ | ANGLE Vulkan SwiftShader |
| `--disable-gpu`（无 GPU，允许软件回退） | ✅ | ✅ | SwiftShader |
| `--disable-gpu --disable-software-rasterizer` | ❌ | ❌ | — |

**结论 1**：纯 CPU 环境（含无 GPU 的机器）**能正常创建 WebGL2**，走 ANGLE + SwiftShader 软件光栅。只有"显式禁止软件光栅"这一种极端配置会失败。

**结论 2（正确性）**：GPU 与 CPU 两种后端的截图逐像素比对，3840×2400 中仅 420 像素不同（0.0046%），且分布在顶部文字/播放头抗锯齿区域；几何真值完全相同（`pxPerSec`、`rowHeight`、`scrollTop` 逐值相等）。即软件光栅**像素级保真**。

**结论 3（性能）**：CPU 下正确但明显更慢。程序化驱动 120 帧横向滚动（dpr 2）：

| | p50 | p95 | max | >33ms 帧 | >100ms 帧 |
| --- | --- | --- | --- | --- | --- |
| 内核（WebGL/SwiftShader） | 95.2ms | 120.7ms | 134.5ms | 114 | 28 |
| 旧实现（Canvas2D） | 135.0ms | 161.5ms | 184.2ms | 115 | 115 |

空闲时两者都是满帧（p50 16.7ms、0 慢帧），说明慢的是渲染本身。

**关键结论**：在纯 CPU 环境下**内核比旧实现更快**（p50 95ms vs 135ms），且没有灾难性超长帧（>100ms 帧：28 vs 115）。移除旧实现不会让任何用户变慢——**被删掉的是一条更慢的路径**。

诚实说明：两者在该环境下都远低于 30fps，即"都不可用，但内核没那么不可用"。CPU 软件光栅下渲染 3840×2400 本就不现实；这不是本次要解决的问题，也不构成保留旧实现的理由。

### 2.2 为什么不再提供"关闭内核"的开关

逃生门（`localStorage` 写 `"0"`）的意义是**退回一条可用的旧实现**。旧实现被移除后，开关没有可退之处，只剩两种结局：要么无效（代码里没有第二条路），要么让应用进入未定义状态。因此一并移除，包括：

- `featureFlag.ts` 的四个开关与 `KERNEL_FLAG_KEYS`
- 上一轮在「视图 → 时间轴显示设置」加的总开关（`isKernelRenderingEnabled` / `setKernelRenderingEnabled`）与 5 种语言的 3 条文案
- `dev/perfProject.ts` 的 dev 快捷切换

`kernelMount.ts` 随之退化：`enabled` 恒为 true，只剩"运行期是否失败"一个维度。**不保留一个含恒真参数的伪抽象**；将其简化为对失败状态的直接判断，但**保留单测**（用例收敛到失败维度），使"GL 失败必须报错而非静默空白"仍有测试守护。

### 2.3 为什么参数编辑器（PianoRoll）不跟着删 Canvas2D

两个面板**结构不对称**，不能一起处理：

- **时间轴内核**：网格与 clip 几何**只有 GL 一条渲染路径**（`drawTimelineCanvas` 在内核内引用数为 0）。GL 失败 ⇒ 无内容可画。
- **参数编辑器内核**：`pianoRoll/render.ts` 的 `drawPianoRoll`（Canvas2D）**在内核模式下仍是活的细节层**（曲线数值、叠加文字等仍由它绘制；只有静态层/曲线层被 GL 接管）。删掉它等于把内核本身弄坏。

因此参数编辑器的处理是：

| 项 | 处理 |
| --- | --- |
| `pianoRollKernel`（滚动/视口所有权） | **移除开关**，内核路径无条件（"唯一路径"落在代码路径的唯一性上） |
| `pianoRollKernel.gl`（静态场景层） | 移除开关 |
| `pianoRollKernel.curveGl`（曲线层） | 移除开关 |
| `drawPianoRoll`（Canvas2D 细节层） | **保留**，它是内核的一部分，不是旧路径 |

即：**"没有回退"只用在真的没有回退的地方**（时间轴的网格/clip）；参数编辑器内部 GL 与 Canvas2D 的分工是内核的**内部实现细节**，保留它是因为它承担实际绘制职责，而不是作为失败回退。

### 2.4 已知并接受的限制：参数编辑器在 GL 失败时不可用

**决定**：不为"参数编辑器 GL 运行期失败"做图层回退——GL 失败即视为该面板不可用。用户明确要求不修此项。

**背景**：当前实现里，被 GL 接管的图层通过 `skipGrid` / `skipCurves` / `skipKeyboardGeometry` / `skipAxisText` / `skipAxisCanvas` / `skipPlayhead` 让 Canvas2D **跳过绘制**，而这些参数取自**模块加载期**的常量，GL 失败却发生在**运行期**。因此 GL 失败后这些图层无人绘制。实测（dpr 2，判据为贯穿宽度的横向网格线数）：GL 正常时 **24 条**，WebGL2 被禁时 **0 条**——横向音高网格整片消失。

**为什么不修**：本次改造的目标是让时间轴收归唯一路径；参数编辑器在无 WebGL2 的环境下本就无法正常工作（网格、曲线、键盘、轴文字全归 GL），补一套 Canvas2D 回退等于把已迁走的渲染层再实现一遍，收益与成本完全不成比例。且实测确认 WebGL2 在纯 CPU 环境（含无 GPU 机器）**是可用的**，只有显式禁止软件光栅一类极端配置才会失败。

**记录此节的原因**：该行为是**已知且有意接受**的，不是未发现的缺陷。将来若要支持无 WebGL2 的环境，这里就是入口。

## 3. 范围：精确的删除清单

清单依据是**真实 import 图**（逐条 `grep` 验证），不是文件名推断。

### 3.1 删除（旧分支专用，无其他真实引用者）

| 文件 | 行数 | 依据 |
| --- | --- | --- |
| `timeline/TimelineScrollArea.tsx` | 392 | 仅被 `timeline/index.ts` 再导出 |
| `timeline/TimelineSurface.tsx` | 183 | 仅被 `timeline/index.ts` 再导出 |
| `timeline/TimelineCanvasViewport.tsx` | 188 | 仅被 `TimelineSurface` + `index.ts` |
| `timeline/ClipItem.tsx` | 945 | 仅被 `TrackLane` + `index.ts` |
| `TrackLane` **组件部分** | 见 3.2 | 仅被旧分支使用 |
| `TimelinePanel.tsx` else 分支体 | 513（`:4866`–`:5378`） | 改为无条件渲染内核分支后不可达 |

### 3.2 必须拆分而非删除：`TrackLane.tsx`

`TrackLane.tsx`（1006 行）导出一个**仍在使用**的纯函数：

```
computeLeadingOverlapSecByClipId(clips)   // TrackLane.tsx:47
```

其真实消费者**两个都在旧分支之外**：

- `TimelinePanel.tsx:223`（import）、`:4357`（调用）—— 分支外
- `TimelineWaveformSurface.tsx:11`（import）、`:92`（调用）—— **内核视图 `TimelineKernelView:37` 自己挂载的波形层**

处理：把该函数移到独立模块 `timeline/trackOverlap.ts`（职责单一：由 clips 计算各 clip 的前置重叠秒数），删除 `TrackLane` 组件体与 `ClipItem`。**这是本设计最容易做错的一步——按文件名整体删除会连带删掉内核正在调用的函数，TS 会报错但若同时改了 import 就可能被掩盖。**

### 3.3 保留（内核亦在使用，逐条验证）

| 文件/模块 | 内核侧的引用依据 |
| --- | --- |
| `TimelineWaveformSurface.tsx` | `TimelineKernelView.tsx:37` 挂载 |
| `BackgroundGrid.tsx` | **参数编辑器**（`PianoRollPanel.tsx:84` 经 barrel 导入、`:6379` 渲染、无内核守卫）。名字像旧实现，实为两用 |
| `SnapHighlightLayer.tsx` | `TimelineKernelView.tsx:33` 挂载 |
| `TimeRuler.tsx`（`timeRulerNode`） | 两个分支都用（`:4763` 与 `:4867`） |
| `renderKernel/*` | 内核的共享底座（`scrollKernel`、`glContext`、`sdfBoxProgram`、`keyboardScroll` 等） |
| `timeline/runtime/timelineClipGlRenderer.ts` | `timelineKernelHost` 引用 |
| `timeline/runtime/timelineCanvasModel.ts` | `timelineKernelHost` 引用 |
| `timeline/runtime/buildTimelineTicks.ts` | 内核标尺刻度 |
| `timeline/runtime/timelineScrollRange.ts` | `scrollKernel` 与面板共用 |
| `pianoRoll/render.ts` | 见 §2.3 |
| `ClipFormantToolWindow` | 位于分支闭合（`:5378`）之后（`:5385`），两种模式都渲染（实测确认，非旧分支专属） |

`timeline/index.ts`（barrel）同步清理：删除对已删模块的再导出行；若清空则删除该文件。

### 3.4 明确不做

- **不**删除参数编辑器的 Canvas2D 渲染器（见 §2.3）。
- **不**在本设计内做渲染重写（如把 `overlayText` 的 `fillText` 迁到 GL）——那是独立议题，混进来会让风险与收益完全不成比例。

### 3.5 连带作废的测试

`featureFlag.test.ts` 共 25 项用例，其中 24 项测的是本次要删的四个开关与总开关（`isTimelineKernelEnabled` 等）——**随实现一并删除**。

剩下 1 项是"生产默认值（源码级守护）"：断言 `featureFlag.ts` 的实现不含 `import.meta.env.DEV` / `PROD`。它的历史背景是 Phase 3 的 R8 教训——内核默认值曾跟随构建模式，导致打包后静默退回旧渲染器。

**本设计后该断言变成空洞的**（开关整个不存在，自然不可能依赖构建模式），因此同样删除。但**教训本身必须留档**，做法是：在计划文档与本次提交信息中记录"内核曾是 opt-in、默认值跟随 `import.meta.env.DEV`、导致打包后静默回退（R8）"，并在 `kernelMount` 保留的单测里以"失败必须报错而非静默空白"这一**行为**断言承接其精神。

即：删掉的是已无守卫对象的检查，保留的是可继续守护行为的测试。

## 4. GL 失败界面

失败时不再有回退，这个界面就是用户的唯一出口，必须能自助排障。替换现有的单行红字（`:941-947`）。

内容结构：

1. **标题**：时间轴无法渲染（WebGL2 不可用）
2. **一句原因**：`createGlCanvas` 返回 null ⇒「浏览器/系统未提供 WebGL2 上下文」
3. **排查清单**（按实测命中概率排序）：
   - 浏览器或系统是否禁用了硬件加速 / 软件光栅（实测 `--disable-software-rasterizer` 必然失败）
   - 是否运行在远程桌面 / 虚拟机中（GPU 常被屏蔽）
   - 显卡驱动是否过旧，或被浏览器列入黑名单
   - 是否同时打开了过多 WebGL 上下文（应用内其它 GL 面板）
4. **诊断信息 + 复制按钮**：`userAgent`、`devicePixelRatio`、`webgl2`/`webgl1` 可用性布尔值。

**已知限制（必须在界面上或代码注释中写明）**：诊断信息**拿不到具体显卡型号**。因为 renderer/vendor 需要创建 `webgl2` 上下文并读 `WEBGL_debug_renderer_info`，而在已经失败的环境里这次探测同样返回 null。因此只给布尔值，不假装能给出型号。

**不做**：不提供"重试"按钮（失败原因不会自愈，重试只会刷日志）；不静默降级；不显示英文技术栈。

## 5. 分期实施

关键约束：删除 513 行 JSX 会使大量 state/hook 变成孤儿，而 `TimelinePanel.tsx` 有 6015 行、旧分支消费的 state 与内核分支（22 个标识符）差异很大。**手工判断哪些能删不可靠**（很多是 `useCallback` 闭包与 ref，TS 挡不住全部）。

因此顺序必须是**先让旧分支不可达，再删**：

### 阶段 1：旧分支不可达（可独立验证）

把 `kernelActive ? (...) : (...)` 改为无条件渲染内核分支；旧分支 JSX 暂时保留在文件中但不可达。

验证：全量测试 + 真机确认内核渲染正常。
**这一步的价值**：若真机上发现内核尚缺能力，回滚成本是一个三元表达式；而删完 500+ 行后回滚很痛。

### 阶段 2：删除孤儿 state/hook（以工具报告为准）

以 **`npx tsc -b --noEmit` 的报告**作为权威清单删除，而非人工判断。

  说明：`noUnusedLocals` / `noUnusedParameters` **已在 `tsconfig.app.json`（与 `tsconfig.node.json`）中开启**，因此普通的 `tsc -b` 就会报出 `TS6133`（已实测验证：临时加入未使用变量后立即报错）。注意 **不能**写 `tsc -b --noUnusedLocals`——`--build` 与该命令行选项互斥（`TS5094`）。eslint 的 `no-unused-vars` 作为交叉印证。对报告未覆盖的情形（如仅被旧分支 `useCallback` 引用的 ref）人工补判并记录理由。

### 阶段 3：删除文件与拆分

- 删 §3.1 的文件与旧分支 JSX
- 拆 `timeline/trackOverlap.ts`（§3.2）
- 清理 `timeline/index.ts`

### 阶段 4：移除开关与失败界面

- 删除 `featureFlag.ts` 的开关、设置项、i18n 文案、`dev/perfProject` 切换
- 简化 `kernelMount`（保留单测）
- 实现 §4 的失败界面

每阶段独立提交，各自通过完整验证。

## 6. 验证方式

1. **全量测试**：`npx vitest run`。基线 **848 passed / 2 failed**；2 项为 `keybindingMatch.test.ts` 既有失败，已在 `develop` worktree 复现确认与本分支无关。

   **注意基线会下降**：`featureFlag.test.ts` 的 25 项用例随开关一并删除（见 §3.5），因此完成后的期望值是 **≈823 passed / 2 failed**。验收判据不是"数字不降"，而是：**除明确删除的用例外，没有任何用例从通过变为失败**。每阶段提交前逐条核对失败清单，只允许出现那 2 项既有失败。
2. **类型与静态检查**：`npx tsc -b --noEmit`、eslint（0 error）。
3. **生产构建**：`npm run build`，并确认产物中 `import.meta.env` 为 0 次（历史教训：内核默认值不得跟随构建模式）。
4. **真机渲染**：`?mock=1` 下确认轨道、clip、波形、网格、滚动条、标尺渲染正常。
5. **CPU-only 正确性与性能**：固化脚本 `frontend/scripts/cpu-render-bench.mjs`
   - 以 `--use-angle=swiftshader --enable-unsafe-swiftshader --disable-gpu-sandbox` 启动
   - 程序化驱动滚动（**不用 CDP 鼠标注入**：实测注入本身产生 80–130ms 假慢帧，会污染结论）
   - 带 idle 对照，输出 p50/p95/max/慢帧计数
   - `MODE=legacy` 在旧路径移除后应**明确报告"该模式已随旧路径移除"**，而不是静默给出错误数据
6. **GL 失败路径**：用注入脚本让 `getContext("webgl2")` 返回 null，确认失败界面出现且诊断信息可复制（而非空白或退回旧实现）。

## 7. 风险

| 风险 | 说明 | 缓解 |
| --- | --- | --- |
| 误删内核仍在用的共享模块 | 尤其 `TrackLane` 的纯函数（§3.2） | 以真实 import 图为准；每步跑 `tsc` 与全量测试 |
| 孤儿 hook 漏删或多删 | 513 行 JSX 的引用面大 | 分阶段；以工具报告为权威清单（阶段 2） |
| 内核能力缺口在删除后才暴露 | 旧实现被破坏，无参照物 | 阶段 1 先让旧分支不可达并真机验证，再删代码 |
| GL 失败的极端环境无路可走 | 仅 `--disable-software-rasterizer` 一类配置 | 失败界面给出可操作清单（§4）；实测该类环境旧实现同样不可用 |
| CPU 环境性能不足 | p50 95ms、p95 121ms | 已实测内核优于旧实现；本次不解决 CPU 性能本身 |

## 8. 验收标准

1. 时间轴只有一条渲染路径，代码中不存在第二条实现或运行期二选一。
2. 四个内核开关、设置项总开关、i18n 文案、dev 切换全部移除，无悬空引用。
3. `TrackLane.tsx` 的纯函数已拆出且两个消费者仍正常（内核波形层的重叠计算正确）。
4. WebGL2 不可用时显示可自助排障的失败界面，且诊断信息可复制。
5. 参数编辑器在移除三个开关后行为不变——即**有 WebGL2 时**网格 / 键盘 / 轴文字 / 播放头 / 曲线全部照常渲染；`skip*` 恒为 `true`，GL 失败时这些图层无人绘制，属 §2.4 **已知并接受**的限制（**不是**验收缺陷，也**不**要求补 Canvas2D 回退）。
6. 全量测试不低于基线；tsc 干净；eslint 0 error；生产构建成功且产物不含 `import.meta.env`。
7. CPU-only 基准脚本进仓库且可复现 §2.1 的结论。
