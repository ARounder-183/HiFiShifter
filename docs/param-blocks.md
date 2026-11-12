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
- Loop：按源域周期 `D` 取模，每周期一个 `ProjectedInterval`。
  - `TakeSpan.cycle` 是**相对当前 loop anchor** 的循环序号，不是源域属性；anchor 随 Clip 左裁一起前进。
  - 跨 anchor 的源域区间拆成 head（`[anchor, hi)`，落在**本周期开头**）与 tail（`[lo, anchor)`，落在**本周期末尾**），两段属于**同一个 cycle**。只有真正跨过一个完整周期（`width > D`）才递增 cycle。
  - `split_loop_span_parts`（Rust）/ `splitLoopTakeSpanParts`（TS）是唯一拆分点：它同时处理 anchor 缝、媒体回绕与超宽区间，并给出与时间轴方向一致的 `cut_start`/`cut_end`。
  - `normalize_loop_span` 只负责持久化/编辑层的“单周期片段”规整；`project_span` **必须自己调用拆分**，不能假设调用方已经 normalize（golden 与编辑原语会直接传原始 span）。
  - 由 anchor 截断出的片段在工程时间轴上是不完整的边，投影结果为 `cut_start = true`（或倒放时对应的另一侧），渲染层据此画平边而不是圆角。
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

- **设计原则：参数块选择是选择工具族的对象级子模式，不是第三个顶级工具。** 工具族只有选择（range / block）与绘制（draw / line / vibrato）；`block = toolMode === "select" && selectToolMode === "block"`。块层**是否绘制**是**派生**的，不落状态：`isParamBlockLayerVisible({ alwaysShowParamBlocks, toolMode, selectToolMode })` 是唯一判据，语义是「常显设置 ∨ 块选择子模式」。工具栏的「永远显示参数块」是**功能设置**（默认关），它只写自己 —— 不随工具切换改变，也不会反向改工具 / 选择子模式（旧实现「关闭图层就把子模式收回 range」已删除）。
- 块选择没有独立状态：选区（selectionRef / selectionUi）是唯一真值，块高亮由「完整覆盖」派生。一个逻辑块的全部可见 run 都被选区覆盖，它才进入 selectedParamBlockIds。
- 逻辑块身份是 userBlockId ?? id。多 run 用户块在点击、Ctrl 切换、Shift 范围、拖动、删除、编组、取消编组、归一化、恢复与菜单计数中都是一个单位；只有绘制 / 命中 / 过渡手柄看单个 run。
- Ctrl / Cmd + 点击块：切换该逻辑块的成员资格。Shift + 点击块：从锚点到目标块按整块范围选择；Ctrl + Shift 退化为普通单选。Alt + 点击块：在点击位置分割（默认绑定，可在快捷键设置修改）；Ctrl+Alt 在块上回退为 Ctrl 选择。
- 点击与拖动以二维 3px 死区区分。Shift 点击/拖动都从锚点到目标按整块范围选择，拖动直接使用该范围作为拖动集合；Ctrl 点击移除块后不会启动拖动。Alt 点击分割在水平或纵向拖动越过死区后自动取消，进入普通块拖动。
- 块选择动作同时写入参数编辑器选区；在 range / block 之间切换不清空选区，块高亮只在块子模式时显示 —— 选区本身跨工具保留（设计如此），但「选中」在画面上是一组**交互 affordance**（高亮填充、过渡手柄、可拖光标）。因此非块选择模式下即使常显开启，块层**只显示**：没有 hover 高亮、没有选中高亮、没有过渡手柄、点不动。块选择模式内的背景拖动会把碰到的逻辑块扩展为全部 run，保证高亮与 Delete / Group 的作用域一致。块选择模式 Ctrl/Cmd+A 只选择块实际覆盖的帧区间（多 run 全部纳入），块间空隙与轨道头尾不选；无块时清空。
- 锚点只服务 Shift：普通点击、Ctrl 点击、Shift 点击都会更新锚点；背景拖动结束落在起点块；右键空白拖动追加后按「原锚点 → 选区首块 → 播放头块」回退；清空 / 根切换清空。
- 组伸缩与删除递归保留 baseCover：组的基块所有权等于全部叶子 child 的 baseCover 并集；被缩掉 / 丢弃的 base 派生 child 生成墓碑，避免基块复活。组 resize 保持组 id；根块组按 child 生成多个 run，不填满空隙。
- 右键块体拖动 = 幅度调整：复用选择族的 `createSelectionAmplifier` / `transformSelectionByRightDrag`，上拖放大、下拖平滑，百分比浮窗与参数线选择同源；右键块体单击 = 块菜单。右键空白拖动 = 追加参数选区；右键空白单击 = 块菜单。块菜单顶部只有「参数块」子菜单，标准参数命令与参数线选择共用 `EditContextMenu` 的唯一描述表。
- 新建块：`modifier.paramBlockCreate`（默认 Alt）在空白拖动 = 普通新建并平移原曲线；`modifier.paramBlockCreateFlat`（默认 Ctrl+Alt）在空白拖动 = 新建并把当前参数线拉平。pitch 拉平保留 0 未浊哨兵，非 pitch 全部填平。普通 Alt 不再隐式拉平。创建开始后补按 / 松开 Ctrl 会实时切换普通与平直形态，提交使用最后一次预览形态。
- 分割：除播放头分割外，`modifier.paramBlockSplitAtClick`（默认 Alt）支持左键点击块内位置分割；点击与拖动以 3px 死区区分，拖动超过死区回到普通块拖动。Ctrl+Alt 在块上按 Ctrl 选择处理，不创建、不分割。分割完成后只选中右侧结果；分割点左侧的旧选中项取消，右侧既有选中项保留。
- 多选纵向拖动：按逻辑块解析每个选中块的起点中心，再扇出到它的全部 run；未选中块不进入拖动成员，提交时也只写选中逻辑块的 `value`。
- 吸附高亮：时间轴与参数编辑器共用 surface/axis 模型。参数编辑器只渲染全高时间线，不渲染时间轴行 accent；音高 / 参数吸附只量化，value 轴不发布 guide、不画全宽值线。右键幅度不发布高亮、也不触发边缘滚动。
- 分割不发布吸附高亮：播放头分割精确落在播放头（snapMode 为 "none"）；点击分割可吸附网格 / Clip 边缘，但候选集排除播放头，且不再创建 180ms 的临时分割高亮。
- 时间轴换轨不做行吸附，也不画轨道边界横线：目标轨严格跟随指针所在行。**只有被拖 Clip 被吸回自己的起点时**（跨轨且横向漂移落在 24–32px 阈值内），跨轨拖动的目标轨道才整行显示一圈描边（与吸附竖线同色、单一颜色，不区分 move / copy）；网格 / 相邻 Clip 边缘 / 播放头 / 选区标记的吸附命中**不画描边**，离开该状态立即消失。这个状态同时发布竖向对齐线 —— 描边与竖线是同一件事的两种投影，必须同时出现、同时消失。

## 曲线与视觉预览

- 参数窗口同 scope 刷新采用 stale-while-revalidate：paramsEpoch 前进但 root / param 未变时保留上一份可见曲线，新回包到达后原子替换；只有切 scope 才清空。同窗口空回包保留旧曲线，窗口变化后的空回包则立即接受，避免旧窗口曲线留在新坐标上。
- 参数块显示值按 run 缓存：当前采样 → 手动值 → 上一次同 run 且**几何一致**的有效值；canonical 块绝不回退到值域中点。没有值的块本帧不画，而不是叠到 F#4。
- 结构命令（分割 / 删除 / 编组 / 还原 / 从选区新建）与外部历史 / 布局 / 参数变化统一走 invalidateParamBlockVisualPreviews：拖动快照、resize / create 预览、事务曲线一次性撤下，避免旧预览在结构编辑后覆盖 canonical。视觉 epoch 同时记录参数名与**结构令牌**（`paramBlockStructureToken`：块 id / 几何 / 过渡宽度 / 删除标记 / baseCover 的指纹）。**不**记录 revision —— 它在前端是内容代次、在后端是 CAS 代次，两者不可比；旧实现拿它做「自身提交归属」（`ownedEditRevision`），于是服务端回包一落地就撤掉自己的预览（回弹的根因）。结构令牌只在形状变化时变化，所以自身提交永不撤销自己的预览，第三方结构编辑立即失效。
- paramBlockSpec 是纯投影，不在 useMemo 里写 ref；清理放在 layout effect（早于 paint）。拖动提交前校验全分辨率窗口与编辑集 revision，缺帧 / 结构变化直接取消，不写 0。
- **预览释放只有一条判据**（`paramBlockPreviewRelease`）：这次提交产生的权威已经可见。带曲线补丁的提交必须同时满足「已落地的权威曲线代次 ≥ 提交回包里的新代次 ∧ 权威水位在提交之后确认 ∧ 当前窗口包含提交片段」；纯布局提交才退回几何相等。旧实现用「本地派生几何已经等于预览几何」判断，而 `setParamBlockState` 是同步提交 —— 那个条件在 pointerup 之后一次渲染就成立，于是预览被撤、曲线回旧值、块按旧曲线重新采样，用户看到「先回弹旧参数、再生效新参数」。超时（2s）之前必须先 `flushReadback()` 对账再撤，否则撤下的第一帧仍是旧曲线。
- **权威代次只有一个单位**：后端 `param_curve_revision`（`ParamAuthoritySnapshot.curveRevision`）。旧实现同时存在「本地取数计数器」与「后端代次」两个数，过渡手柄存前者、比后者，判据几乎永远为假 —— 每次拖完过渡手柄都要等满 2s 超时才释放预览。提交回包里的新代次（`curveRevisionCommitted`）是唯一证据：没有它就不放行，本地计数器再大也没用。
- 新建 / 伸缩 / 过渡的预览事务在提交时记下「依据的曲线代次」「提交时的权威水位」与「提交的曲线片段」（`curveRevisionAtStart` / `authorityRequestSeqAtCommit` / `committedSegments`），提交成功后补上回包里的新代次（`curveRevisionCommitted`）；没有这四个字段就无法回答「权威有没有前进过」。
- **音高视图有三层语义不同的线**，不得合并：
  - 灰色虚线 `pv.orig`：根轨道组的**单值合成**原始线（`assemble_pitch_orig_from_cache`）。重叠 Clip 与音高参考块在同一帧竞争后只留一个值。
  - 白色主线 `pv.edit`：当前编辑目标。
  - 彩色细线（Clip 源音高参考线）：每个 Clip 自己的 `clipPitchCurves` 曲线，按 Clip 颜色稳定着色；重合段提高不透明度、选中 Clip 最后绘制。它只读、不参与编辑与渲染。
- Clip 源音高参考线由 `clipPitchOverlay.ts` 统一派生：数据有效性（`curveStartSec` 对齐 Clip 起点、曲线长度覆盖 Clip）、`overlapDepth`、`kind`（`sourceF0` / `midiReference`）与**有声音段切分**都在那里完成。禁止在绘制层再把 `midi <= 0` 的前后点连成一条线（那会画出贯穿静音的假斜线）。
- 该图层有独立开关（参数工具栏「Clip 音高」，默认开，仅 pitch 模式可见）。它不能替代 `pv.orig`：后者是渲染与编辑的基线；也不能把 `pv.orig` 改成多值曲线来“显示重合”。

## 参数块图层：常显设置

- 工具栏上的按钮是「**永远显示参数块**」（`alwaysShowParamBlocks`，默认关），不是「当前图层可见吗」的指示灯。它的 `aria-pressed` 读设置值 —— 块选择模式下图层**总是**可见，与这个设置无关。
- 六种组合（`isParamBlockLayerVisible`）：块选择模式 ⇒ 恒可见；选择族 range / 绘制族 ⇒ 只看设置。设置与工具**互不写入**：切工具不改设置，改设置不改工具。
- 可见 ≠ 可交互：非块选择模式下块层只画，不产生 hover、选中高亮、过渡手柄与命中。
- 代价：自动音节分析只在「块层真的在画」时跑（没有分析结果就没有块），所以常开图层意味着每次打开参数编辑器都会跑一次模型推理 —— 这也是它默认关闭的理由。

## 指针优先级

> 下面这段是**描述**，不是第二份实现。真实顺序写在 `onParamBlockPointerDown` 的 if 梯里；这里记下来，是为了让读代码的人先知道意图，而不是去照着某个「策略表」改（旧版 `paramBlockPointerPolicy.ts` 自称唯一真相，却零调用、且顺序已经和真实代码不一致，已删除）。

1. 右键块体 → 幅度调整（右键块体单击 = 块菜单）。
2. 右键空白 → 追加参数选区（单击 = 块菜单）。
3. `Ctrl+Alt` 在块上 → 回退为 Ctrl 选择（不新建、不分割）。
4. `Alt` 在块上 → 在点击处分割（越过 3px 死区后取消，回到普通块拖动）。
5. 过渡手柄 → 左右拖动改交叉淡化宽度。
6. 块边缘 → 左右拖动伸缩。
7. 块体 → 选择 / 纵向拖动。
8. 空白 + `Alt` / `Ctrl+Alt` → 新建块（普通 / 拉平）。
9. 空白 → 框选。

## 拖拽读数

- 开启「弹出展示参数」后，块手势与参数线拖拽表现一致：拖拽过程中指针旁持续显示读数，松手立即消失。
- 显示的是**块将被拖到的中心值**（`projectParamBlockDragDisplayValue(锚点起始值 + 位移)`），不是指针所在帧的曲线值 —— 指针可能抓在块的任意高度（块高 1.2 行高），用它会让读数随抓取位置偏移。多选一起拖时所有块位移相同，锚点即代表。
- 空白处 `Alt` / `Ctrl+Alt` 拖动新建：显示新块的值（已按音高吸附 / 跟随音阶解析）。
- 块边缘伸缩与过渡手柄拖动**不显示**读数：它们只改时间跨度 / 淡化宽度，值不变；显示一个静止的数字会让用户以为值被改了。
- 一次手势只有一个读数所有者：块模式下块手势负责发布；无按键的普通悬停由独立的 `onCanvasHoverPreview` 更新曲线读数，不经过块层的 hover/光标判定。`refreshParamValuePreview` 只在**块手势进行中**（拖动 / 新建 / 伸缩）退位，普通悬停与数据刷新仍会续显。

## 值轴边缘滚动

- 竖向边缘滚动的单位是**参数值**，不是 `scrollTop`：带宽 20px、满带 0.10 个视口/秒、软启动 `ratio²`、单帧上限 0.0025 个视口（只在 ≥25ms 的卡顿帧上生效，保证 60Hz 与 120Hz 手感一致）。滚动条自己的「值域中心 ↔ scrollTop」映射仍走 `verticalScrollMapping`，两者互不影响。
- 驱动只负责时钟与 rAF：帧时长按「距上一帧」计算，指针离开触发带即复位时钟，复位后的第一帧位移恒为 0。旧实现按「距上一次 step」计算，带外空闲会被补成一帧的大位移 —— 表现为「来回蹭边缘时视图一跳一跳」。
- 帧时长每帧只算一次（两个轴各算一次会让第二个轴永远拿到 0）。

## 边缘滚动的帧调度

- `rafId !== null` ⟺ **恰好一个**待执行帧。`tick()` 入口把 `rafId` 置空只完成了「已排帧」，还必须用 `ticking` 闸门表达「正在执行」：执行期间任何重入的 `track()` 都不得再排帧，唯一的排帧入口是 `schedule()`。旧实现漏了后者，而值轴的 `apply()` 会在滚动帧内部发布交互视口 ⇒ 订阅者同步合成 `pointermove` ⇒ 手势 `onMove` 重入 `track()` ⇒ **每帧净增一个无人取消的孤儿回调**；第 N 帧要跑 N 次 `tick()`、每次两回 `getBoundingClientRect()`，拖得越久越卡。
- 一帧只取一次 `getBounds()`，两轴与带内判定共用同一份矩形（`getBoundingClientRect()` 是强制同步布局）。
- 交互视口广播是**微任务合帧**（`pianoRollInteractionViewportBus.publish`）：状态立刻生效，通知推迟到下一个微任务，同一帧多次发布合并成一次，值不变不广播。这样手势的 `onMove` 永远不在 `apply()` 的调用栈里被重入。
- `stop()` 递增取消代次；`tick()` 收尾时代次已变就不再续帧 —— 在 `apply()` 内部被 `stop()` 也不会留下一帧。

## 提交事务

- 写入只有一条入口（`useParamBlockPersistence.commitPlan`，`commit` 是它的薄包装）：**前端预检**（`paramBlockValidate`，与 Rust `validate_edit_set` 逐条同构）→ 按根串行排队 → CAS → 按类别重基。
- **一个根只有一条写队列**，且「读基准 → 算补丁 → 写」整体在队列槽内（`ParamBlockWritePlan.prepare`）。旧实现有两条互不相识的队列（持久化层的 `queues` 与面板的 `paramBlockCommitQueueRef`）共用同一个后端 CAS，交错时后写者必然过期 —— 那是「各种编辑场景仍有概率报参数块操作失败」的结构性根因。
- **一次用户操作最多一条可见反馈**，且由写入层发布（`publishParamBlockWriteFailure`）。调用点只允许报**本地拒绝**（`reportParamBlockLocalRejection`，根本没有发起写入的那类）；写入失败时调用点只写结构化诊断。`paramBlockFeedback` 是单槽，重复上报必然把分类结果覆盖成一句通用红条。
- 预检失败的编辑集**永远不离开浏览器**：后端 `Invalid:` 是一类确定性失败，而 `composeParamBlockView` 会静默裁掉重叠，前端"看起来正常"不代表合法。
- 失败分四类（`paramBlockCommitError`）：`conflict`（CAS，刷新基准后可重试）、`rejected`（校验，不可重试）、`transport`（IPC，对账后最多一次）、`cancelled`（静默）。分类按 `Conflict: ` / `Invalid: ` 前缀，不按文案匹配。
- **所有 CAS 冲突都会重基后重试一次**（`canRebaseParamBlockCommit`）：重试会重新调用 `prepare()`，即「重新读一遍世界、按同一个意图重算补丁」。`transport` 永远不重试 —— 上一次写入可能已经落地，重试会把一次用户操作变成两步历史。
- 多块全分辨率读在窗口之间出现 revision 不一致时**整轮重读**（最多 3 次），而不是把 `Conflict: parameter curve changed during read` 抛给用户 —— 曲线 revision 是进程全局计数，任何无关写入都会让它前进。
- 回读失败与写入失败**不共用回调**：后台回读失败只写诊断，不会再弹一条看起来像"你的编辑丢了"的 danger 提示。

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
  - 菜单描述：`paramBlockCommandMenu.ts`
  - **命令执行表（唯一）**：`panel/paramBlockEditOps.ts` —— 键盘通道与右键菜单共用；新增块命令只改这里。
  - 键盘通道：`panel/useParamBlockEditOpChannel.ts`（只把 stroke 折成 `snapToggleHeld`，不 switch op）。
  - 渲染规格的唯一派生点：`panel/useParamBlockRenderSpec.ts`。
  - 指针 hover / 光标：`panel/paramBlockHover.ts`。
- Golden fixture：`frontend/src/components/layout/pianoRoll/__fixtures__/paramBlockProjection.golden.json`

## 交叉 Clip 的块层级

- 参数块层级与时间轴视觉顺序一致：Clip 按 `startSec` 升序，同起点按 id 的 UTF-16 码元序；后开始的 Clip 在上层。前端与 Rust 合成必须复用同一个比较器，禁止在合成器里另写数组下标排序。
- 上层 base 块先占用时间区间，下层 base 被 `occupied + subtract` 切成仍可见的片段，不整体消失；最终 `ParamBlockView` 仍满足时间不重叠。
- 用户块 / 编组块的优先级（3/4）高于 base，不受 Clip 层级影响。
- 仅交换 clips 数组顺序不得改变 blocks 或 `layoutKey`。

## 块标签的渲染上下文

- 参数块标签只属于主画布：主画布与左侧轴画布各自持有独立的 `GlyphSurface`（program / rasterizer / atlas 上传页）。`GlyphProgram` 与创建它的 WebGL 上下文绑定，`GlRasterTarget` 只描述尺寸，不切换上下文。
- 禁止把块标签的 glyph program 指向轴画布；那会把主画布坐标压缩进 56px 轴列，表现为轴列白色碎片、块内无标签。

## 音符块来源

- 参数块与静音检测共享 `note_blocks::source_note_spans` 的优先级：`syllable_analysis` → `game_block_baseline` → `midi_note_data`；空来源不伪造整段一块。
- 静音检测的 `note_blocks` 方式只消费这份来源与 `projected_note_spans`，缺失分析时返回可操作错误，不重新启动 GAME 推理。

## 维护约束

- 投影公式只允许在 `param_blocks/projection.rs` 出现一份；前端 `paramBlockProjection.ts` 是预览端口，必须用 golden fixture 对拍。
- 任何新增块编辑都必须经过 `place_user_block`；分割/伸缩必须走 `split_block` / `resize_block`（前端 `paramBlockGesture.ts`）这一份计划。
- 基块可见性只由“剩余帧区间”决定；禁止恢复 `suppressed_by_user` 的二元抑制。
- 任何新增持久化字段都不得是绝对工程帧。
- Loop 源域片段**不得**在 `normalize_loop_span` 里把跨 anchor 的 head 写成 `cycle + 1`；投影层必须对未 normalize 的 span 也正确拆分，否则左裁越过音符起点后首个参数块会在投影阶段被丢弃。
- 纯块边界编辑不得 bump 音频版本。

- 值轴手感常量只有一份：`kernel/valueAxisEdgeScroller.ts`（带宽 / 满带速度 / 软启动指数 / 单帧上限）。文档与代码冲突时以该文件为准。
