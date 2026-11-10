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
- `UserParamBlock { id, geometry, deleted?, manual?, value?, transitionOutSec?, createdSeq, baseCover?, groupedFrom? }`：声明式用户块。
- `BaseCover { spans, full }`：从基块派生时记录“我替代了哪一段基块”。普通手工块没有；`full=true` 命中完整基块时隐藏整段，`full=false` 只覆盖精确投影区间。
- `ParamBlockEditSet { schemaVersion: 1, revision, blocks }`：一个 root 参数线的唯一编辑权威。
- `ProjectedMember`：除源域区间外，还携带 `frameSourceStartSec/frameSourceEndSec`（有方向的局部仿射映射）与 `visibleSourceStartSec/visibleSourceEndSec`（可见源域范围）。
- `ParamBlockView`：派生视图，**不持久化**；App 保存前清空。

## 投影规则

- 非 Loop 正放：`project = clip.start + (source - windowStart) / rate`。
- 非 Loop 倒放：以 `windowEnd` 镜像。
- Loop：按源域周期 `D` 取模，每周期一个 `ProjectedInterval`；跨周期/跨 anchor 的源域区间先用 `normalize_loop_span` 切成单一周期片段。
- 所有区间半开 `[startFrame, endFrameExclusive)`；投影后裁剪到 Clip 可见窗口。
- 编辑已有成员时不再调用全局 `inverse_project`：由成员的 `frameSource*` 局部仿射映射直接求源坐标，避免 Loop 回绕 / 周期边界 / 倒放跳到成员外。
- 宽度小于 1 帧的 base 块丢弃；用户块 resize 至少保留 1 帧。

## 编辑语义

- **基块是衬底**：base 始终参与合成，用户块/墓碑只覆盖它实际投影到的帧区间；不再“任一用户成员重叠即整段删除基块”。
- 合并：汇总选中块的源域成员，生成一个多成员 `UserParamBlock`；若所有子块携带同一 `baseCover`，编组块继承它。
- 切分：以 `block.members` 的**工程投影顺序**为准。切点在成员内部时只切该成员；切点在成员边界/缝隙时按投影顺序分组，不产生 1 帧碎片；两侧至少各 1 帧，否则拒绝。
- 边缘伸缩：重写投影顺序的成员列表；缩短时删除被跨过的成员并把落点成员裁到目标；倒放时右边缘改 `sourceStart`、左边缘改 `sourceEnd`；邻居被 `place_user_block` 按源域裁剪。
- 删除：基块派生块生成携带同一 `baseCover` 的墓碑；**纯手工块不生成墓碑**，删除后基块自然回来。旧版无 `baseCover` 的墓碑仍按成员投影区间抑制基块。
- 手动画块：工程区间经 `sourceAtProjectedFrame` 映射为 `TakeSpan`；没有 Take 交集时才用 `RootSpan`。
- 还原 GAME：只生成与目标帧范围相交的 baseline 源域片段，并裁剪相交用户块，不再“相交就删整块”。
- 新建块：调用方传入的 `blockId` 必须成为实际 `UserParamBlock.id`，预览 / 选择 / 交接共用同一个 id。

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
- 任何新增块编辑都必须经过 `place_user_block`；分割/伸缩必须走 `split_block` / `resize_block`（前端 `paramBlockGesture.ts`）这一份计划。
- 基块可见性只由“剩余帧区间”决定；禁止恢复 `suppressed_by_user` 的二元抑制。
- 任何新增持久化字段都不得是绝对工程帧。
- 纯块边界编辑不得 bump 音频版本。
