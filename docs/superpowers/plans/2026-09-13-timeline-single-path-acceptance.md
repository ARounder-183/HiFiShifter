# 时间轴唯一路径改造 · 验收记录（任务 7）

## 验收环境与基线

- 起点：`ef0bb88c`（计划提交），基线测试 `2 failed | 848 passed (850)`
- 终点：本次验收提交
- 2 项失败始终是既有问题：`keybindingMatch.test.ts` 的「默认 Shift 变体」与
  「Ctrl 微调变体」，已在 `develop` worktree 独立复现（与本分支无关）

## 11 项验收（全部通过）

| # | 项目 | 结果 |
|---|---|---|
| 1 | 5 个旧组件文件已删 | ✓ 全部不存在；`BackgroundGrid.tsx` 保留（参数编辑器依赖） |
| 2 | 开关残留 | ✓ 无（仅剩 `kernelActiveGroupIds`，是内核分组 id，非已删的 `kernelActive`） |
| 3 | `featureFlag.ts` 删除；`kernelMount.ts` 处置 | ✓ 前者已删；后者无生产消费者，保留 + 注释记录（见下） |
| 4 | `tsc -b --noEmit` | ✓ 无输出 |
| 5 | 全量测试 | ✓ `2 failed \| 838 passed (840)` |
| 6 | eslint | ✓ 0 error / 12 warning（与改动前逐项相同） |
| 7 | 生产构建 + 产物无 `import.meta.env` | ✓ 构建成功；产物中 0 次 |
| 8 | 2 项失败确为既有 | ✓ 见上 |
| 9 | 真机正常路径 | ✓ 两个内核挂载、`kernelAttr` true、旧 scroller 不存在、PageDown=132 |
| 10 | 真机失败路径（WebGL2 不可用） | ✓ 失败界面出现（标题/清单/复制按钮），旧红字不再出现 |
| 11 | CPU-only 基准可复现 | ✓ SwiftShader 软件光栅、内核挂载、idle 满帧、scroll p50 83.5ms |

## 测试数变化的完整账

| 阶段 | 通过数 | 变化原因 |
|---|---|---|
| 基线 | 848 | — |
| 任务 2 | 856 | +8（`kernelAvailability` 3 + `glDiagnostics` 5） |
| 任务 3 | 863 | +7（`trackOverlap`） |
| 任务 4 | 838 | −25（`featureFlag.test.ts` 整个删除） |

## 遗留项（有意保留，非本次范围）

1. `kernelMount.ts` + 其 8 项测试：`enabled` 维度随开关消失后无生产消费者。**不删**的原因：
   行为守护已由 `kernelAvailability.ts` 承接；且删除会使命中数变 830、与任务 4 验收判据不符。
   已在该文件头与计划文档中写明"是否删除属独立决策"。
2. 任务 3 删除 5 个旧组件后新孤立的 7 个模块（约 2469 行）：
   `ClipHeader`、`OverlapEditLayer`、`FadeHitLayer`、`ClipEdgeHandles`、`timelineHitTest`、
   `useDebouncedPersist`、`timelineViewportDispatch`。已记录在计划文档「后续清理」一节。
3. 第二个失效逃生门（key `hifishifter.glClipBodies`）：无读取者，dev 面板按钮已失效。
   已修正注释并**删除该函数**（`d3f6f516`；修正记录初稿曾误写为"函数保留"）；
   仅 dev 按钮与 key 常量保留待同上清理。

## 审查期间确认的两项既有问题（非本次引入，已排除）

### 1. `?mock=1` 下时间线 clip 体不显示波形

**不是本次改造引入。** 判据（最终审查者提供）：加载 dev 面板的「400 clip」性能工程时
两个面板的波形都正常渲染，且用"隐藏波形父层"对比截图证明波形层**确实是**绘制者
（时间线 clip 体随之变平、参数编辑器保留自己的波形）。本改动范围内 `waveform/` 目录
与 `TimelineWaveformSurface` 的挂载块**均未被修改**（该文件只改了一行 import 与注释）。
根因在 mock fixture 的峰值数据路径。

**注意**：因此上表第 9 项"渲染完整"的截图结论**不包含** mock 下的波形验证——mock 数据
本就不画波形。若要验证波形，请用 dev 面板的性能工程。

### 2. 拖动 clip 时控制台出现 `Cannot read properties of undefined (reading 'map')`

**不是本次改造引入。** 控制方实测：切到改造起点 `ef0bb88c` 的源码后执行同一拖动，
**报同样的错误**。根因是 mock 后端缺 `set_clips_state_bulk` 处理器，其 Proxy 兜底返回
`{ ok: true }`，`applyTimelineState` 随后在无 `tracks` 的 payload 上运行
（`sessionSlice.ts:353` 的 `mapTimelineTracks`）。

影响有限：该操作仍有正确的乐观本地效果。属 mock 后端问题，与本次"唯一路径"改造无关。
