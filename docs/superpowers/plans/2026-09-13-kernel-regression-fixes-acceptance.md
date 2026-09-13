# 内核回归修复与钢琴背景 · 验收记录

## 验收环境与基线

- 分支：`feature/timeline-unified-render-kernel`
- 起点：`f41f5bfe`（上一轮"内核唯一路径"改造的终点）
- 真机：`?mock=1`（6 轨 / 24 clip 示例工程），1920×1200，dpr=2，Vite dev server `127.0.0.1:5174`
- 基线测试：`2 failed | 838 passed (840)`（上一轮改造的终点；本会话各任务的增量见「测试数变化的完整账」）
- 2 项失败始终是既有问题：`keybindingMatch.test.ts` 的「默认 Shift 变体」与
  「Ctrl 微调变体」，已在 `develop` worktree 独立复现（与本分支无关）。**全程未"修"它们。**

## 缺陷清单与验收结果

### 缺陷 1：修饰键 + 拖拽复制无法落到新轨道 / 其他轨道

| 项 | 结果 |
| --- | --- |
| 根因 | 提交分支把 `dropToNewTrack` / `trackOffset` 写死为 `false` / `0`，丢弃了预览分支已算出的落点；且注入的 `resolveTrackIdByOffset` **忽略偏移量**（返回原轨），使修复静默无效 |
| 修复 | 抽出纯函数 `resolveKernelDropTarget`（Task 1），提交分支改用它与真实建轨 creator，并修 `resolveTrackIdByOffset` |
| 真机 | ⌘ 拖到全部轨道之下 → `add_track` 1 次 + `duplicate_clips_bulk` `trackMode=explicit_mapping {track-1→track-new-1}`；⌘ 拖到轨 2 → `explicit_mapping {track-1→track-2}`（**无** `add_track`）；向下跨轨 → `{track-1→track-4}` |
| 附带修复 | 虚线新轨行**不残留**（Esc 取消同样清理）；越界时改为**夹取到末轨**（与预览分支一致，消除"幽灵到末轨、落库留原轨"的分叉） |

### 缺陷 2：点标尺后网格播放头不动

| 项 | 结果 |
| --- | --- |
| 根因 | 内核渲染循环**纯脏标记驱动**（`start()` 不绘制、无常驻 rAF），而播放头变化从不标脏；原 `if (playheadMoved) loop.invalidate()` 写在 `draw()` **内部**，永远无法调度第一次帧 |
| 修复 | 新增 `invalidatePlayhead()`（**只标脏、不置 `sceneDirty`**，避免每帧重建 GPU 几何）；播放头改走实时 getter（镜像滞后一次提交）；桥接每帧请求重绘；删除孤儿 `playheadRef` |
| 真机（两次独立测量） | 点标尺 12.5 → 4.293 时内核 `transform` 1875px → **644px**（= 4.293×150）；再点 → 6.96 / **1044px**；再点 → 1.627 / **244px**。修复前恒为 1875px |
| 负对照 | 运行时把 `invalidatePlayhead` 换为空函数 → 症状**原样复现**（store 变、`transform` 不变），证明新入口是承重的 |
| 性能 | 3 次 seek + 60fps 播放脉冲期间 `invalidateScene` 调用 **0** 次；空闲 2s `invalidatePlayhead` **0** 次；帧 p50/p95 = 16.7ms |

### 缺陷 3：点空白不取消选中 / 播放头不变（**两个独立缺陷**）

| 项 | 结果 |
| --- | --- |
| 根因 (a) | 内核 seek 只派发 `seekPlayhead`，缺乐观写；而 `seekPlayhead.fulfilled` 只在"后端返回值 ≠ 请求值"时采纳，后端对非负请求**原样回显** → store 值压根不变 |
| 根因 (b) | 取消选中是**纯本地**的，后端仍记着 `selected_clip_id`；`selectTrackRemote` 用**纯字符串**派发使闸门判为"要恢复"，把刚清掉的选中**异步复活**。这是 `019e93ed` 修过、`464a78bb` 静默解除的**回归** |
| 修复 (a) | `handleKernelSeek` 两条分支 + `handleKernelActivateTake` 补 `setplayheadSec` |
| 修复 (b) | 六处 `selectTrackRemote` 全部改为 `{applySelectedClip: false}`（经核实后端 `selected_clip_id` 是**全工程唯一**字段，不存在"每轨记住的 clip"这一语义） |
| 真机（用户原场景） | 轨 1 选中 → 点轨道头把焦点漂到轨 2（`{sel:track-1-clip-1, trk:track-2}`）→ 点轨 1 空白 → `{sel:null, multi:0, trk:track-1}`，**2.5s 后仍为 null**（异步复活消失） |
| 变异测试 | 把闸门改回恒真 → 新增的守卫用例立刻失败（`expected null, received "clip-a"`） |
| mock 修正 | `select_clip` / `select_track` 补 `selected_clip_id`（旧桩缺该字段，**掩盖**了本缺陷；且旧桩会让 fulfilled 抛 `Cannot read properties of undefined (reading 'map')`） |

### 缺陷 4：参数编辑器选区框残留（**系统性缓存键缺陷**）

| 项 | 结果 |
| --- | --- |
| 根因 | 主画布内容签名用 `[...].join("|")` 构造，`join` 把每个对象/数组元素串成字面量 `"[object Object]"` → **两个不同选区产生同一签名**，缓存命中，`drawPianoRoll` 在清屏前就 return。只有 `null ↔ object` 能改变签名，精确解释"第一次可见、再次选择不更新、拖到边缘才更新" |
| 影响面 | `selectionRef` / `paramMorphOverlay`（morph 手柄拖拽同样静默失效，3 点 → 9 点签名相同）/ `liveEditOverrideRef` / `paramViewsRef` / `detectedPitchCurves` / `referencePitchOverlays` / `secondaryParamViews` |
| 修复 | 新增纯模块 `mainCanvasSignature.ts`（`Object.is` 逐项引用比较），缓存改为比较数组；补 `overlayText` 签名项（提取为单一局部变量，消除"签名写 A、绘制喂 B"）；新建选区分支补 `invalidate()` |
| 真机 | 第二次按下后主画布 `toDataURL()` **必须变化** → 实测 `true`（修复前**逐字节相同**）。像素差定位：旧框区间 `[400,1000]` 被清除、新零宽边界出现在 `[1400,1401]` |
| 变异测试 | 比较实现退回 `join("|")` → 两条对象塌缩守卫失败（`2 failed / 6 passed`） |
| 逐图层回归 | 选区框 / morph 手柄 / morph 拖拽跟随 / 剪贴板预览 / 曲线（滚动 + 竖向缩放）/ 中央提示文字 —— 全部仍会更新 |
| 缓存未失效 | 空闲 1.5s 主画布绘制 **0** 次；选区拖拽期间 18 次（引用比较未被每帧新对象击穿） |

### 缺陷 5：MIDI / 音高参考块内部为空（**改造回归**）

| 项 | 结果 |
| --- | --- |
| 根因 | `MidiPitchTrackCanvas.tsx` 仍存在但**零消费者**——唯一挂载点 `TrackLane.tsx` 被 `98b4956c` 删除。波形层无法替代（`clipToSceneClip` 对 `!sourcePath` 返回 null） |
| 数据 | 管线完好，无需重接（`midiNoteCount` / `midiNoteData` / Redux `clipPitchCurves`） |
| 修复 | 折线数学**逐字**搬进纯模块 `midiPitchCurve.ts`（458 行），由细节层绘制（含环回 ▽ 标记）；孤儿组件**删除**（避免两份数学漂移）；mock 补 `midi_note_data` 与 `pitch_range`（只给 count 会让该 clip 在 `develop` 上同样空白，这是当初无法视觉复现的原因） |
| 真机（模型层，独立复验） | 24 个 clip 中**恰好 1 个**产出折线（`track-3-clip-5`），**其余 23 个音频 clip 均无**（无多余折线）；921 个抽稀点；**7 个不同 y 值**且序正确（note 72 → y 29.02 最高，note 60 → y 33.93），末段 1 个 `null` 对应音符之后的静音 |
| 环回标记 | 注入 loop clip（40s 长 / 4s 媒体）后可见两个 ▽，位于 1× 与 2× 周期处 |
| 抽稀有界 | 1200s clip：4px/s → 38 点，150px/s → 921 点（原始 24 万帧），步长 0.75/1.5/3.0px |

### 缺陷 6：音阶高亮按钮失效（**改造回归**）

| 项 | 结果 |
| --- | --- |
| 根因 | 该图层只画在 Canvas2D 的 `skipGrid` 分支里，而 `skipGrid` 恒为 `true` → 按钮能按、Redux 会变、画面不动 |
| 修复 | 迁到 GL 网格层；新输入（背景色 / 音阶音级 / 强调色）纳入 `gridGeometrySignature` |
| **第二处接线缺陷（实现者发现）** | 仅改 GL 层**不够**：拥有 `buildGridSpec()` 的 `useLayoutEffect` 依赖数组缺 `themeMode` / `s.scaleHighlightMode` / `effectiveProjectScale`，spec 永不重建、签名永不变 → 按钮仍无效。补齐依赖后**切主题**也一并修好（实测 GL 亮度 p10 225.72 → 29） |
| 真机（元素截图，非 `toDataURL`） | 关闭 → 开启：截图 55899 → 57483 字节（画面确实变化）；**再关回来逐字节相同**（无残留）。目视确认琥珀色强调线出现在音阶音级上，**叠在黑键带之上** |
| ⚠️ **方法学纠正** | 本工程 GL 画布用的是 `preserveDrawingBuffer: false`（`glContext.ts:72`），因此 `toDataURL()` 对 GL 画布**恒返回全透明图**、哈希对任何变化都**不变**。排查阶段用 `toDataURL` 得到的"逐字节相同"是**假信号**（即便修好了也会相同）。结论改由**元素截图 + 像素分析**得出；代码层面的诊断（kernel 目录 0 次 `scaleHighlight`）仍然成立 |

### 缺陷 6b：新功能 · 钢琴背景（黑键行）

| 项 | 结果 |
| --- | --- |
| 实现 | 黑键行半透明暗带，**先于所有网格线发射**（FLAT 按缓冲顺序合成，否则会盖住线）；几何取**键体范围** `[valueToY(midi+1), valueToY(midi)]` 而非线的中心位置；复用 `buildRectInstances` 的设备像素覆盖率修正；`Math.max(1, …)` 键高下限 |
| 真机（元素截图） | 深浅交替带清晰可见（浅色主题：白键行亮度 239.72 vs 黑键带 225.72，**Δ14**）；目视见 `/tmp/v6-off.png` |
| **网格线仍可见（硬性验收点）** | 检出 45 条线，带内 10 条、白键行 35 条；带内最小对比 **Δ13.07** vs 白键行 Δ14.0 —— 基本一致，背景带**没有盖掉**网格线 |
| ⚠️ 已知不足 | 深色主题下带宽只有 **Δ2**（8% 黑叠在 `#1f1f1f` 上已接近下限），可见但偏弱。常量已隔离在 `colors.ts` 便于调强（唯一手段是改用**偏亮**的带，因为黑色在深底上没有余量） |

## 测试数变化的完整账

| 阶段 | 累计通过数 | 本步增量 |
| --- | --- | --- |
| 上轮改造终点 `f41f5bfe` | 838 | — |
| Task 1 落点解析 | 844 | +6 |
| Task 3 播放头标脏 | 849 | +5 |
| Task 3 提取取值判定 | 853 | +4 |
| Task 4 签名引用比较 | 861 | +8 |
| Task 6b 缺陷 7 | 861 | 0（纯逻辑改动，靠真机验证） |
| Task 5 折线数学 | 872 | +11 |
| Task 6 钢琴背景 + 音阶 | **880** | +8 |
| **实测终值** | **880 passed / 2 failed (882)** | **+42** |

> 两个并行任务各自的增量已按其提交核对：Task 5 = +11、Task 6 = +8。

## 门禁

| 门禁 | 结果 |
| --- | --- |
| `npx vitest run` | **2 failed（既有）/ 880 passed (882)**，139 个测试文件（基线 840 → **+42**） |
| `npx tsc -b --noEmit` | **无输出** |
| `npx eslint src` | **0 error / 12 warning**（warning 数与改动前基线相同） |
| `npm run build` | **成功**（2.51s） |
| 产物 `import.meta.env` | **0 次** |

## 未决 / 已知残留

1. **缺陷 7 的一帧原生 scroller 滞后**（瞬时方向反转）：`applyHorizontalScrollPosition` 先写内核、原生 scroller 靠内核镜像回写，因此慢一帧。稳态与常速拖动差值为 0。
2. **音阶高亮的 Tempo Map 分段路径未搬迁**：GL 网格几何是视口级、不含时间维度，无法表达"不同时间段用不同音阶"。含 Tempo Map 音阶变化的工程会高亮**工程音阶**而非各段音阶。已在四处注释中记录。
3. **深色主题钢琴背景偏弱（Δ2）**：见上，常量可调。
4. **`clipPitchCurves` 回退路径已实现但当前不可达**：`timelineKernelHost` 未透传 `clipPitchCurves` / `clipPitchRanges`，因此线上只走 `midiNoteData` 主路径。回退逻辑与降级行为已实现并如实注释；若要真正启用，宿主补两行透传即可。
5. **`pitchAnalysisPending` 是死参数**：全历史无任何调用方传入，`render.ts` 的该参数从未生效。本次修正了"要求包含它"的错误注释，未新增签名项（新增会让缓存因恒 `undefined` 的值而失效；真正接上会改变行为——提前 return 会让 morph 手柄在分析期消失）。


### 缺陷 7：同步时间轴视图对不上、拖动抽搐（用户后续报告）

| 项 | 结果 |
| --- | --- |
| 根因 | 两面板**可滚域时长来源不同**：时间轴 `session.projectSec`（120s）vs 参数编辑器 `getDynamicProjectSec(clips)`（59.5s）→ 内容宽 18000 vs 8925，可滚上限 18000 vs 9125。同步写入超出部分被**浏览器钳制**，两边永久错位；快速来回拖动时共享值在两端跳变（实测差值 **7650px**）＝抽搐 |
| 修复 | 新增唯一来源 `resolveScrollableProjectSec(projectSec, clips)`（取大）；三处消费者（参数编辑器 / 内核 / 标尺与内容宽）共用 |
| 真机 | 参数编辑器可滚上限 9125 → **18200**；拖到时间轴最右时两侧 `scrollLeft` 均 **18000**（差 0）；缩放三轮差 0–1px；标尺刻度数两侧一致（18/18） |
| 同步开关往返 | off 时参数编辑器独立滚动（2000，不动时间轴）；on 后重新对齐（0） |
| 残留（已记录） | 一帧内把 scrollbar 从最左甩到最右时，参数编辑器**原生 scroller** 滞后一帧（`applyHorizontalScrollPosition` 先写内核、原生靠镜像回写）。稳态与常速拖动差值为 0 |

## 测试数变化的完整账

| 阶段 | 通过数 | 变化原因 |
| --- | --- | --- |
| 起点 `f41f5bfe` | 861 | — |
| Task 1 落点解析 | 867 | +6 |
| Task 3 播放头标脏 | 872 | +5 |
| Task 3 提取取值判定 | 876 | +4 |
| Task 4 签名引用比较 | 884 | +8 |
| Task 6b 缺陷 7 | 884 | 无新增（纯逻辑改动，靠真机验证） |
| Task 5 / Task 6 | 待填 | 折线数学 +11、钢琴背景与音阶 +8 |

## 门禁

| 门禁 | 结果 |
| --- | --- |
| `npx vitest run` | 2 failed（既有）/ 其余全通过 |
| `npx tsc -b --noEmit` | 无输出 |
| `npx eslint` | 0 error |
| `npm run build` | 待复验 |
| 产物 `import.meta.env` | 待复验 |

## 未决 / 已知残留

1. **缺陷 7 的一帧原生 scroller 滞后**（瞬时方向反转）——见上，已记录。
2. **音阶高亮的 Tempo Map 分段路径**：GL 网格几何是视口级、不含时间维度，因此只实现
   了单音阶（工程音阶）高亮；旧 Canvas2D 的"按时间段用不同音阶"未搬迁。
3. `pitchAnalysisPending` 从未被任何调用方传入 `drawPianoRoll`（全历史搜索为空），
   其 `render.ts` 参数是**死参数**。本次修正了要求包含它的注释，未新增签名项——
   新增会让缓存因一个恒为 `undefined` 的值而失效，且把它真正接上会改变行为
   （提前 return 会让 morph 手柄在分析期消失），超出本轮范围。
