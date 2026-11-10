# 参数块：Take 源域锚定模型

> 参数线是 Track 的连续曲线；参数块是 Take 源域的离散分段。曲线坐标在 Track，块的身份在 Take，块的显示区间是投影结果。

## 三层模型

```text
Take 源域                          Clip 编排                       Track 参数线
────────────────────────────      ──────────────────────         ─────────────────────────
syllable / phoneme spans           start / length                 pitch_edit / tension_edit
stretch markers                    source_start / source_end      extra_curves
game baseline                      playback_rate / reverse        参数块编辑覆盖层
（块的身份与真值）                 loop / cycle                   （跨 Clip 的用户意图）
                                    │
                                    └──────── 投影 ────────────────> ParamBlockView（绝对帧，派生）
```

四条不变量：

1. **区间真值在源域**：用户不持久化“块在工程上的绝对帧”；持久化的是 Take 源域区间和用户编辑意图。
2. **投影是纯函数**：`ParamBlockView = compose(base(Take analysis, Clip geometry), ParamBlockEditSet)`。
3. **曲线与块分层**：块边界跟随源内容；参数曲线的用户编辑仍按 ParamAuthority 的工程帧权威处理。
4. **单一模型**：App、ARA、VST Capture 共用内核 `param_blocks`；前端只保留交互预览端口。

## 数据模型

- `TakeKey { clipId, takeId }`：Take 在工程中的稳定出现身份。
- `TakeSpan { clipId, takeId, cycle, sourceStartSec, sourceEndSec, syllableId?, label }`：源域区间，绝不存工程秒。
- `RootSpan { startSec, endSec }`：没有 Take 来源的自由块。
- `UserParamBlock { id, geometry, deleted?, manual?, value?, transitionOutSec?, createdSeq, baseCover? }`：声明式用户块。
- `BlockGeometry::Group { children }`：**一等公民组容器**。children 是组唯一的几何真值；组自身不保存合并成员，也没有历史快照。子块可以是 Take/Root 块或嵌套组，最多 8 层；同一组内不得混用 root 与 take 叶子。
- `BaseCover { spans, full }`：从基块派生时记录“我替代了哪一段基块”。普通手工块没有；`full=true` 命中完整基块时隐藏整段，`full=false` 只覆盖精确投影区间。
- `ParamBlockEditSet { schemaVersion: 2, revision, blocks }`：一个 root 参数线的唯一编辑权威。加载时把 v1 的 `groupedFrom` 快照一次性转成 `Group { children }`，转换后不再保留旧字段。
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
- 编组：选中块成为 `Group` 的 children；组只保存容器，compose 时递归收集全部叶子成员。children 不要求相邻：组投影为每个叶子自己的 run，子块之间的空隙不会被组填满。嵌套组允许最多 8 层，但同一组内不得混用 root 与 take 叶子。
- 组的结构编辑：分割 / 伸缩递归作用于子块。切点在某个子块内部时只分割那个子块，其余子块按投影顺序分到两侧；某侧只剩一个子块时该侧退化为普通块（组自然消失）。组本身不复制 `baseCover`。
- 分割：以 `block.members` 的**工程投影顺序**为准。切点在成员内部时只切该成员；切点在成员边界/缝隙时按投影顺序分组，不产生 1 帧碎片；两侧至少各 1 帧，否则拒绝。
- 边缘伸缩：重写投影顺序的成员列表；缩短时删除被跨过的成员并把落点成员裁到目标；倒放时右边缘改 `sourceStart`、左边缘改 `sourceEnd`；邻居被 `place_user_block` 按源域裁剪。
- 删除：`full=true` 的基块派生块生成携带同一 `baseCover` 的墓碑；**纯手工块不生成墓碑**，删除后基块自然回来。`full=false` 的分割半块删除后直接移除所有权，让被删半段的基块恢复。旧版无 `baseCover` 的墓碑仍按成员投影区间抑制基块。删除组时，除组本身外还要递归为组内 base 派生叶子生成 `childTombstones`，恢复时按 id 精确移除后再放回整组。
- 手动画块：工程区间经 `sourceAtProjectedFrame` 映射为 `TakeSpan`；没有 Take 交集时才用 `RootSpan`。
- 还原 GAME：只生成与目标帧范围相交的 baseline 源域片段，并裁剪相交用户块，不再“相交就删整块”。入口接受**多段**范围（`restore_game_block_ranges`），每段独立还原，段间保持原样。
- 新建块：调用方传入的 `blockId` 必须成为实际 `UserParamBlock.id`，预览 / 选择 / 交接共用同一个 id。

## 选择与编辑作用域

- 块选择没有独立状态：选区（selectionRef / selectionUi）是唯一真值，块高亮由「完整覆盖」派生。一个逻辑块的全部可见 run 都被选区覆盖，它才进入 selectedParamBlockIds。
- 逻辑块身份是 userBlockId ?? id。多 run 用户块在点击、Ctrl 切换、Shift 范围、拖动、删除、编组、取消编组、归一化、恢复与菜单计数中都是一个单位；只有绘制 / 命中 / 过渡手柄看单个 run。
- Ctrl / Cmd + 点击块：切换该逻辑块的成员资格。Shift + 点击块：从锚点到目标块按整块范围选择；Ctrl + Shift 退化为普通单选。Alt + 点击块：在点击位置分割（默认绑定，可在快捷键设置修改）；Ctrl+Alt 在块上回退为 Ctrl 选择。
- 点击与拖动以二维 3px 死区区分。Shift 点击/拖动都从锚点到目标按整块范围选择，拖动直接使用该范围作为拖动集合；Ctrl 点击移除块后不会启动拖动。Alt 点击分割在水平或纵向拖动越过死区后自动取消，进入普通块拖动。
- 块选择动作同时写入参数编辑器选区；切换工具不清空选区，块高亮只在块工具显示。块工具内的背景拖动会把碰到的逻辑块扩展为全部 run，保证高亮与 Delete / Group 的作用域一致。块工具 Ctrl/Cmd+A 只选择块实际覆盖的帧区间（多 run 全部纳入），块间空隙与轨道头尾不选；无块时清空。
- 锚点只服务 Shift：普通点击、Ctrl 点击、Shift 点击都会更新锚点；背景拖动结束落在起点块；右键空白拖动追加后按「原锚点 → 选区首块 → 播放头块」回退；清空 / 根切换清空。
- 组伸缩与删除递归保留 baseCover：组的基块所有权等于全部叶子 child 的 baseCover 并集；被缩掉 / 丢弃的 base 派生 child 生成墓碑，避免基块复活。组 resize 保持组 id；根块组按 child 生成多个 run，不填满空隙。
- 右键块体拖动 = 幅度调整：复用选择工具的 `createSelectionAmplifier` / `transformSelectionByRightDrag`，上拖放大、下拖平滑，百分比浮窗与选择工具同源；右键块体单击 = 块菜单。右键空白拖动 = 追加参数选区；右键空白单击 = 块菜单。
- 新建块：`modifier.paramBlockCreate`（默认 Alt）在空白拖动 = 普通新建并平移原曲线；`modifier.paramBlockCreateFlat`（默认 Ctrl+Alt）在空白拖动 = 新建并把当前参数线拉平。pitch 拉平保留 0 未浊哨兵，非 pitch 全部填平。普通 Alt 不再隐式拉平。创建开始后补按 / 松开 Ctrl 会实时切换普通与平直形态，提交使用最后一次预览形态。
- 分割：除播放头分割外，`modifier.paramBlockSplitAtClick`（默认 Alt）支持左键点击块内位置分割；点击与拖动以 3px 死区区分，拖动超过死区回到普通块拖动。Ctrl+Alt 在块上按 Ctrl 选择处理，不创建、不分割。分割完成后只选中右侧结果；分割点左侧的旧选中项取消，右侧既有选中项保留。
- 多选纵向拖动：按逻辑块解析每个选中块的起点中心，再扇出到它的全部 run；未选中块不进入拖动成员，提交时也只写选中逻辑块的 `value`。
- 吸附高亮：时间轴与参数编辑器共用 surface/axis 模型。参数编辑器渲染全高时间线和全宽值线，不渲染时间轴行 accent；右键幅度不发布、也不触发边缘滚动。
- 分割不发布吸附高亮：播放头分割精确落在播放头（snapMode 为 "none"）；点击分割可吸附网格 / Clip 边缘，但候选集排除播放头，且不再创建 180ms 的临时分割高亮。
- 时间轴换轨不做行吸附，也不画轨道边界横线：目标轨严格跟随指针所在行。跨轨拖动时，如果 Clip 起点回到拖拽前起点 24–32px 内，会水平吸附回原起点并显示一条竖向对齐线；水平拖出该范围后立即释放。

## 曲线与视觉预览

- 参数窗口同 scope 刷新采用 stale-while-revalidate：paramsEpoch 前进但 root / param 未变时保留上一份可见曲线，新回包到达后原子替换；只有切 scope 才清空。同窗口空回包保留旧曲线，窗口变化后的空回包则立即接受，避免旧窗口曲线留在新坐标上。
- 参数块显示值按 run 缓存：当前采样 → 手动值 → 上一次同 run 且**几何一致**的有效值；canonical 块绝不回退到值域中点。没有值的块本帧不画，而不是叠到 F#4。
- 结构命令（分割 / 删除 / 编组 / 还原 / 从选区新建）与外部历史 / 布局 / 参数变化统一走 invalidateParamBlockVisualPreviews：拖动快照、resize / create 预览、事务曲线一次性撤下，避免旧预览在结构编辑后覆盖 canonical。视觉 epoch 同时记录参数名与结构 revision；自身提交通过 ownedEditRevision 放行，第三方结构编辑立即失效。
- paramBlockSpec 是纯投影，不在 useMemo 里写 ref；清理放在 layout effect（早于 paint）。拖动提交前校验全分辨率窗口与编辑集 revision，缺帧 / 结构变化直接取消，不写 0。

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
