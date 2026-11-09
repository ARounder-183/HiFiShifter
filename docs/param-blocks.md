# 参数块：Take 源域锚定模型

> 参数线是 Track 的连续曲线；参数块是 Take 源域的离散分段。曲线坐标在 Track，块的身份在 Take，块的显示区间是投影结果。

## 三层模型

~~~text
Take 源域                          Clip 编排                       Track 参数线
────────────────────────────      ──────────────────────         ─────────────────────────
syllable / phoneme spans           start / length                 pitch_edit / tension_edit
stretch markers                    source_start / source_end      extra_curves
game baseline                      playback_rate / reverse        参数块编辑覆盖层
（块的身份与真值）                 loop / cycle                   （跨 Clip 的用户意图）
                                    │
                                    └──────── 投影 ────────────────> ParamBlockView（绝对帧，派生）
~~~

四条不变量：

1. **区间真值在源域**：用户不持久化“块在工程上的绝对帧”；持久化的是 Take 源域区间和用户编辑意图。
2. **投影是纯函数**：`ParamBlockView = compose(base(Take analysis, Clip geometry), ParamBlockEditSet)`。
3. **曲线与块分层**：块边界跟随源内容；参数曲线的用户编辑仍按 ParamAuthority 的工程帧权威处理。
4. **单一模型**：App、ARA、VST Capture 共用内核 `param_blocks`；前端只保留交互预览端口。

## 数据模型

- `TakeKey { clipId, takeId }`：Take 在工程中的稳定出现身份。
- `TakeSpan { clipId, takeId, cycle, sourceStartSec, sourceEndSec, syllableId?, label }`：源域区间，绝不存工程秒。
- `RootSpan { startSec, endSec }`：没有 Take 来源的自由块。
- `UserParamBlock { id, geometry, deleted?, manual?, value?, transitionOutSec?, createdSeq }`：声明式用户块。
- `ParamBlockEditSet { schemaVersion: 1, revision, blocks }`：一个 root 参数线的唯一编辑权威。
- `ParamBlockView`：派生视图，**不持久化**；App 保存前清空。

## 投影规则

- 非 Loop 正放：`project = clip.start + (source - windowStart) / rate`。
- 非 Loop 倒放：以 `windowEnd` 镜像。
- Loop：按源域周期 `D` 取模，每周期一个 `ProjectedInterval`。
- 所有区间半开 `[startFrame, endFrameExclusive)`；投影后裁剪到 Clip 可见窗口。
- 宽度小于 1 帧的 base 块丢弃；用户块 resize 至少保留 1 帧。

## 编辑语义

- 合并：汇总选中块的源域成员，生成一个多成员 `UserParamBlock`；base 被抑制。
- 切分：工程切分点逆投影回源域，成员在源域切开。
- 边缘伸缩：只改最左/最右成员的源域边缘；邻居被 `place_user_block` 裁剪。
- 删除：生成 `deleted = true` 的源域墓碑；重分析后仍按源重叠匹配。
- 手动画块：工程区间逆投影为 `TakeSpan`；没有 Take 交集时才用 `RootSpan`。
- 还原 GAME：从 `ClipTake.game_block_baseline` 生成用户块，不改当前分析。

## 持久化与版本

- App 工程版本：7。`TrackParamsState.param_block_edits` 持久化；`param_block_view` 不持久化。
- 插件组件状态版本：3。`EditState.param_block_edits` 持久化；视图恢复后重新 compose。
- Capture：`CaptureEdits.param_block_edits` 与 `game_block_baselines`。
- **不做旧版兼容**：旧绝对帧块、合并意图、切分/伸缩记录全部删除；旧字段被 serde 忽略。

## 关键代码

- 内核：`backend/hifishifter-kernel/src/param_blocks/`
- App：`backend/src-tauri/src/commands/param_blocks.rs`
- 插件：`backend/hifishifter-plugin/src/editor/session.rs`
- 前端：`frontend/src/components/layout/pianoRoll/paramBlock*.ts`
- Golden fixture：`frontend/src/components/layout/pianoRoll/__fixtures__/paramBlockProjection.golden.json`

## 维护约束

- 投影公式只允许在 `param_blocks/projection.rs` 出现一份；前端 `paramBlockProjection.ts` 是预览端口，必须用 golden fixture 对拍。
- 任何新增块编辑都必须经过 `place_user_block`。
- 任何新增持久化字段都不得是绝对工程帧。
- 纯块边界编辑不得 bump 音频版本。
