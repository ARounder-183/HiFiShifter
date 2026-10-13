# 宿主扩展边界：哪些功能只能靠原生扩展

> 面向插件维护者与集成方。本文回答一个问题：**同一个功能，为什么在 REAPER 里能用、
> 在 Studio One / Cubase 里不能用？**
>
> 结论先行：不是"还没实现"，而是 **ARA 2.0 契约不允许插件做**。把这些能力标成"缺失"
> 会误导产品与用户；正确的说法是"宿主扩展专属"。

---

## 1. 两种后端

插件在启动时按"宿主有没有给出原生扩展接口"二选一（`host/adapter.rs` 的
`HostBackend`，`host/ara_only.rs` 的 `AraOnlyHost`）：

| 后端 | 触发条件 | 能力来源 |
|---|---|---|
| `HostBackend::Native` | 宿主加载了插件导出的原生扩展入口（当前实现是 REAPER 的 `reaper_vst3_interfaces`） | 原生扩展 API + ARA + VST3 |
| `HostBackend::AraOnly` | 没有原生扩展 | ARA 2.0 + VST3 |

身份与能力是**两套**词汇，都在 `kernel::editor::host`：

- `HostKind`（`Reaper` / `AraOnly`）—— "我是哪个后端"；
- `HostCapabilities` —— "我能做什么"，每个字段对应一条用户可见操作族。

`HostKind::clip_write_allowed()` 把"ARA-only ⇒ 只读"写成**身份蕴含**的结论：ARA 2.0
不允许插件改写 region 几何，所以这不是"某台宿主碰巧缺了某个 API"，而是契约。

**绝不按宿主名分支。** `EditorHost::app_name()` 只用于诊断（"你是哪个宿主"），任何能力
判断都走 `capabilities()`。宿主名是任意字符串，不可枚举。

---

## 2. 只能靠宿主扩展的部分

以下能力 ARA 2.0 **没有**对应接口，只有原生扩展能给。ARA-only 后端上它们是**诚实的
"没有"**，前端据此禁用对应控件，而不是让用户点了之后撞一个字符串错误。

| 能力 | 为什么 ARA 2.0 给不了 | 原生扩展的入口（REAPER） |
|---|---|---|
| 写回片段几何（位置 / 长度 / 源窗 / 倍率） | ARA 的 region 几何是宿主的**输入**，插件只能读 | `GetSetMediaItemTakeInfo_Value` 的 `D_POSITION` / `D_LENGTH` / `D_STARTOFFS` / `D_PLAYRATE` |
| 分割片段 | ARA 没有"在播放时间轴某点切开 region"的原语 | `SplitMediaItem` |
| 建轨 / 导入音频 | ARA 不建 item、不建轨 | `InsertTrackAtIndex` / `InsertMedia` |
| 含宿主 item 状态的剪贴板 | ARA 不暴露 item 的 RPP chunk | `GetItemStateChunk` / `SetItemStateChunk` |
| 宿主撤销栈集成 | ARA 有自己的编辑会话撤销，但不是宿主工程栈 | `Undo_BeginBlock2` / `Undo_EndBlock2` |
| take / lane 枚举 | ARA 只认 region，不认"一个 item 有多个 take" | `GetMediaItemNumTakes` / `GetMediaItemTake` |
| 轨道层级（folder / Track Stack） | ARA 没有 folder 概念 | `I_FOLDERDEPTH` + `GetTrack` 遍历 |
| 倒放 / 循环源 / 拉伸标记 / 源文件指纹 / item 增益 | 这些是 **item / take 级**属性，ARA 表达不到 | `PCM_Source_GetSectionInfo`（revOut）/ `B_LOOPSRC` / `I_CHANMODE` / `D_VOL` |

> `ara::mapping::LOST_FIELDS` 是上表在代码里的登记处。**注意**：那里的注释曾写成
> "都补回来了"——对 ARA-only 是**假的**。正确措辞是"**在存在原生扩展时**补回"。

---

## 3. ARA-only 上仍然完整的功能

这些不依赖原生扩展，在任意 ARA 2.0 宿主上都工作：

| 功能 | 来源 |
|---|---|
| 时间线枚举与片段几何（只读） | `ARAPlaybackRegion` 的播放时间轴字段 |
| 音频读取 / 渲染 | `read_source_pcm` + VST3 `ProcessBlock` |
| 播放位置与状态 | VST3 `ProcessContext` |
| 音高 / 参数编辑 | 插件自己的参数模型（kernel） |
| 内容淡化 / 时间拉伸标志 | `ARAPlaybackRegion::hasContentBasedFade*` / `isTimestretchEnabled` |
| 参数随编辑保留（移动 / 裁剪 / 分割 / 拉伸 / 取消裁剪） | 插件自己的参数权威表（源域曲线 + 稳定宿主身份） |
| 宿主保存工程时携带插件状态（含**部分**存档） | `storeObjectsToArchive` / `restoreObjectsFromArchive`（含筛选与 persistentID 重映射） |
| 告诉宿主"这些对象的持久状态变了" | `ARAModelUpdateControllerInterface`（经 `notifyModelUpdates` 投递） |
| 导出处理结果到音频文件 | 授权 PCM（`ARAAudioAccessControllerInterface`）+ 内核编码器；不需要任何宿主写接口 |
| 淡变形状轴（有 / 无、legacy / continuous） | 原生扩展**报告**版本；ARA-only 恒为"无" |

**参数随宿主编辑保留**（第五轮加入）。参数权威按**稳定宿主身份**索引（有原生扩展时用
item GUID，否则用 modification persistentID），曲线以**源域**存储，因此移动 / 裁剪 /
分割 / 拉伸都不再让曲线消失 —— 这是此前"在 Cubase 里编辑一次，参数就被重置"的根因。
分割时父段参数通过 `cloneAudioModification` 回执或 `end_editing` 的结构 diff 整份播给
两个半段；取消裁剪能完整恢复曲线（裁剪不缩短权威里的源曲线）。
「锁定参数线」决定参数跟着**素材**走（源域锚定，默认）还是留在**工程位置**（工程域锚定）。
设 `HIFISHIFTER_ARA_LEGACY_ATLAS=1` 可退回旧的几何认领路径做真机对照（保留一个发布周期）。

**渲染失效通路必须对两种后端都成立。** `RenderFacts`（`host/geometry.rs`）把"会改变
渲染结果"的宿主事实收成一张逐 item 的表，任何一项变化都推进 `DocumentSession::render_epoch`。
其中 `reversed` / `loop_enabled` / `channel_mode` 只可能来自原生扩展，而
`timestretch` / `content_fade_head` / `content_fade_tail` 来自 ARA region —— 因此
**ARA-only 后端也必须采到后三项并推进 epoch**（`refresh_ara_render_facts`），否则宿主在
ARA 侧改了时间拉伸 / 内容淡化，插件会继续用旧配置渲染。

---

## 4. 可选的跨宿主增强

**剪贴板**可以换成 ARA 原生的实现（源 id + modification + region 属性 + 私有参数块），
比 RPP chunk 更可移植，能在 REAPER ↔ Cubase 之间工作。列为可选增强，未实现。

---

## 5. 各宿主可达性预期（验收用，不是承诺）

| 宿主 | ARA2 | 可达 |
|---|---|---|
| Studio One / Cubase / Nuendo / Samplitude / Sequoia / Pyramix / Sonar | 是 | 读 + 渲染 + 调音；**不可写**、无 take、无 folder、无宿主撤销栈 |
| Logic Pro（10.8+，ARA 面较薄） | 是 | 读 + 渲染 |
| WaveLab / Sound Forge / Acoustica | 是 | 读 + 渲染；transport 可能缺席 |

**没有任何一家暴露 `GetMediaItemInfo_Value`。** 所以产品必须在原生扩展调用全部为 0 时
依然完整。

---

## 6. 代码索引

| 面 | 文件 |
|---|---|
| 后端选择器 | `backend/hifishifter-plugin/src/host/adapter.rs` |
| ARA-only 后端 | `backend/hifishifter-plugin/src/host/ara_only.rs` |
| 原生适配器（当前 = REAPER） | `backend/hifishifter-plugin/src/host/reaper/` |
| 身份 / 能力词汇表 | `backend/hifishifter-kernel/src/editor/host.rs` |
| 丢失字段登记 | `backend/hifishifter-plugin/src/ara/mapping.rs`（`LOST_FIELDS`） |
| 渲染事实 | `backend/hifishifter-plugin/src/host/geometry.rs` |
