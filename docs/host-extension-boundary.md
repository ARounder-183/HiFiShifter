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
| 快速导出选中片段 / 合成结果另存 / 导出布局·主题·颤音预设 | 同上（音频）或纯前端数据写文件；不需要任何宿主 API |
| 宿主编辑器视图的选择同步（宿主里选中 ⇒ 插件跟着选） | `ARAEditorViewInterface::notifySelection`（需 `kARAEditorViewRole`） |
| 轨道名与**轨道颜色**取宿主真值 | `ARARegionSequenceProperties::name` / `color` |
| 音高参考 / 链接参数 / 参数剪贴板 | 插件自己的参数模型，只改曲线不碰宿主几何 |
| 淡变形状轴（有 / 无、legacy / continuous） | 原生扩展**报告**版本；ARA-only 恒为"无" |

**参数随宿主编辑保留**（第五轮加入主轴，第六轮修完）。参数权威按**稳定宿主身份**索引
（有原生扩展时用 item GUID，否则用 modification persistentID），曲线以**源域**存储，
因此移动 / 裁剪 / 分割 / 拉伸都不再让曲线消失。

> **第六轮修掉的那个洞**：权威键是 modification 级（**多条 region 共享**），而曲线当时
> 按"捕获时那条 region 的**工程帧**"索引（每条**独占**）。Cubase 一次分割产生的所有半段
> 共享同一个 modification ⇒ 它们写同一个键、后写的覆盖先写的 ⇒ 前几条半段投影时落到界外、
> 退回基线（用户看到"参数全部归零"）。现在曲线改按**源帧**索引：每条 region 只是同一条
> 共享曲线上的一个**投影窗口**，各写自己的源区间，互不覆盖。
> 权威里的曲线与既有部分**取并集**，不再是整份替换。

分割时父段参数通过 `cloneAudioModification` 回执、`end_editing` 的结构 diff
（判据已放宽为"任意 N 段、源窗首尾相接"）或原生 `SplitMediaItem` 回执播种。
**共享一个 modification 的半段（Cubase 的常态）不需要任何播种** —— 它们本来就读同一条
曲线的不同子段（ARA 的别名语义，`ARAInterface.h:950-952`）。
取消裁剪能完整恢复曲线（裁剪不缩短权威里的源曲线）。
「锁定参数线」决定参数跟着**素材**走（源域锚定，默认）还是留在**工程位置**（工程域锚定）。
它只在**几何事务**（`param_geometry`）一处生效，判据是宿主给的三个事实：

| 宿主做了什么 | 判据 | 锁开 | 锁关 |
|---|---|---|---|
| 移动 item | 源窗口不变、工程起点变 | 该 item 范围内的曲线整段平移 | 不动 |
| 拉伸 item | **播放倍率**变、工程时长也变 | 按新旧时长比缩放（值不变） | 不动 |
| 裁剪 / **分割** / 滑源 / 倒放 / Loop 开关 | 以上都不成立 | **不动** | 不动 |

> **倍率必须是宿主读数，不能由窗口除出来**（2026-11-09 修）。`源窗口跨度 / 工程时长`
> 这个商只在"源窗口 == 消费窗口"时等于倍率；**循环源**（REAPER 新建 item 的默认
> `B_LOOPSRC=1`）的源窗口是 `[锚点, 媒体末)`，比消费窗口长得多。于是分割一个
> 10 s 的循环 item，左半段长度减半而窗口不变 ⇒ 商从 1.0 变 2.0 ⇒ 被分类成一次
> **拉伸** ⇒ 整条参数线被压进左半段。现场日志的指纹正是
> `[geom] clips=N ... deltas=[item:{GUID}:stretch]` 紧跟
> `[param] mutate cause=geometry ...`。分割只改变"露出素材的哪一段"，素材本身一个
> 采样都没有被拉长压短，所以它必须落在"以上都不成立"那一行。

> **它什么时候生效**：放置策略（`Placement`）决定"用哪条映射"，而**几何代次**
> （`ui_geometry_revision`）决定"映射到哪"。两者都必须在**投影**时读同一份几何，
> 开关才是一个真正的开关。此前的实现只在模型收口（`remap_and_log`）那一刻重投影，
> 而"在宿主里拖动一个 item"既不改 edit 也不改 model —— 于是曲线停住，只有分割
> （它推进了 edit/scope、走整份重载）才让曲线跟上。现在清单刷新
> （`refresh_ui_geometry`）也会把重投影结果落到 GUI 读的那份数组上
> （`adopt_view_into_timeline`），并且**只换曲线**：`pitch_orig_key` /
> `dyn_orig_key` / `pending_pitch_offset` 是会话自己的记账，换掉它们会让每次拖动
> 都触发一轮重分析。

**persistentID 的生命周期必须跟着搬权威**：宿主改名（`updateAudioModificationProperties`）、
整份 / 部分存档恢复时的 ID 重映射，都要同步把权威表搬过去，否则曲线被孤儿化。
宿主动销毁 modification 时权威**不立即删**（撤销会把同一个 ID 带回来），
按"连续 64 轮不在文档图里"回收（避免存档无界增长到 64MiB 上限而写不出去）。

**宿主编辑器视图的选择会被采纳**（`notifySelection` → 插件选区）。采纳是**有条件的**：
只有与上次采纳的不同才覆盖，空选择与已销毁片段都不清空插件自己的选择 ——
该回调是"通知"不是"设定"（`ARAInterface.h:3646-3655`），宿主重开一次窗口就会重发。

设 `HIFISHIFTER_ARA_LEGACY_ATLAS=1` 可退回旧的几何认领路径做真机对照（保留一个发布周期）。

**Cubase（ARA 2.0 Draft）不会调用部分存档。** `storeObjectsToArchive` 属于 2.0 Final
的接口尾部（`ARAInterface.h:2992`），Draft 宿主拿到的是被按代次截短的 vtable ——
因此它在 Cubase 上的存档保证来自**整份存档**（`storeDocumentToArchive`）。
（第六轮复核了第五轮"启用 2.0 Final 专属能力会让 Cubase 的控制器创建失败"这一结论：
它**不成立**。调用点 `ara2-bridge-plugin/src/controller.rs` 会按协商到的代次把能力降级为
全 false，`ffi/vtable.rs` 那道硬闸门因此永远不会被 Draft 宿主触发。）

**渲染失效通路必须对两种后端都成立。** `RenderFacts`（`host/geometry.rs`）把"会改变
渲染结果"的宿主事实收成一张逐 item 的表，任何一项变化都推进 `DocumentSession::render_epoch`。
其中 `reversed` / `loop_enabled` / `channel_mode` 只可能来自原生扩展，而
`timestretch` / `content_fade_head` / `content_fade_tail` 来自 ARA region —— 因此
**ARA-only 后端也必须采到后三项并推进 epoch**（`refresh_ara_render_facts`），否则宿主在
ARA 侧改了时间拉伸 / 内容淡化，插件会继续用旧配置渲染。

---

## 3.1 CLAP（ARA + Capture 双形态）

> **现状勘误（2026-11-08，第二次修订）**：本节先写作「Capture-only 外壳」，随后一度
> 改为「REAPER 7.85 已跑通 CLAP ARA」。**后一结论是错的**：那次验证只覆盖了空工程，
> 一旦时间线上有音频 item，REAPER 7.85 就会崩在自己的代码里
> （`reaper.exe+0x5a677`，`0xC0000005`，读地址 `0x8`）。取证见
> [报告](2026-11-08-reaper-clap-ara-crash-analysis.md)。
>
> 根因是 **REAPER 的 2022 私有草案实现不完整**，不是插件侧可修补的：该崩溃函数在
> `test rbx,rbx` 之后仍解引用 `rbx`（空的是它自己的 playback 渲染器槽位）；强制本
> 实例同时承担 `playback|editor|view`（日志确认三个接口指针全非空）、补齐 ARA 1 遗留
> 字段、关闭 `samples_access` 的同步 PCM 读取，三种改法都照崩。
> 关键在于**实例数量**：`ARACLAP.h` 规定 `bind_to_document_controller` 在一个
> `clap_plugin` 实例上只能调一次、只承担一组角色，多角色必须多实例；实测 REAPER 全程
> 只调用**一次** `create_plugin`，从不补建 playback 实例，也从不向 entry 级
> `get_factory` 索要 ARA 工厂（其二进制里只有那一个私有的实例级 ID）。
> 同一台机器同一份音频下 **VST3 形态稳定**（REAPER 为 VST3 建两个实例并分配
> `playback`；CLAP 只建一个，`assigned=editor|view`）。
>
> 因此本仓只应答 ARACLAP.h 定稿的两个 ID，**明确拒绝**
> `com.celemony.ara.plugin_extension.draft/1`：REAPER 里的 CLAP 插件以 Capture
> 形态稳定运行，需要 ARA 时用 **VST3 形态**。下表「ARA 工厂 / 实例扩展」两行已按此
> 更新；「三个格式共用 `onnx + soundtouch` 内核」仍成立。

当前 `hifishifter-clap` 是可被 CLAP 1.1.9 宿主扫描与加载的插入效果器；宿主把实例
绑到 ARA 文档后进入 ARA 形态，未绑定时走 Capture 形态：

| 能力 | 状态 | 契约原因 |
| --- | --- | --- |
| 实例化 / 激活 / process / 保存 state / 销毁 | 有 | `clap.plugin-factory` + `clap.state` + `clap.audio-ports`；生命周期按 CLAP 状态机严格校验 |
| 延迟 / 尾音 | 如实为 0 | `clap.latency` + `clap.tail`；输出是整段渲染结果的**逐帧拷贝**，没有额外缓冲，0 是唯一正确答案 |
| 宿主日志 | 有 | `clap.log`：`plugin.init` 后查询 `clap_host_log` 并装上 logger，诊断不再静默丢弃 |
| transport 读数 | 按事件如实给 | 只从 `clap_event_transport_t` 读播放/录音/循环/拍号；`steady_time` 不推断播放。编辑器轮询的 `get_playback_state`（光标与捕获带右缘的唯一来源）由实例把位置抄进原子量后回答 |
| 离线 bounce | 按扩展如实给 | 只从 `clap.render` 的 `CLAP_RENDER_OFFLINE` 置位 |
| ARA 工厂 / 实例扩展 | **有，仅限 ARACLAP.h 定稿 ID** | entry 发布 `ClapAraEntry`（stable + `draft/2` compat）；实例返回 `clap_ara_plugin_extension_t`。REAPER 的 `com.celemony.ara.plugin_extension.draft/1` 被**显式拒绝**（应答会让 REAPER 崩在 `+0x5a677`）。绑定走与 VST3 同一份 `AraBinder` |
| `ara:supported` / `ara:required` | 声明 supported，不声明 required | 定稿宿主据此发现 ARA。REAPER 的私有草案既不读它也不查 entry 级工厂，那条路径在本插件里不被支持。没有 ARA 的宿主里插件仍可作普通效果器 |
| 内嵌 WebView | Windows 有 | `clap.gui` + 独立 WebView2 宿主，加载同一份 `frontend/dist`；非 Windows 如实降级 |
| 编辑器命令 | Capture 回执后端 | `plugin_health`、时间线、参数曲线（`get/set_param_frames`、静态参数）、波形峰值、传输读数都由本模块在进程内回答；未接线的命令立刻回结构化错误，不复用 VST3 的 ARA 会话 |
| 音频输出 | **有（渲染结果）** | 捕获的素材 + 用户曲线由 `mixdown::render_mixdown_with_pcm`（与 App 导出同一份）在后台线程渲染成 PCM，`process` 只覆盖"属于素材"的帧；素材之外按 `outside_passthrough` 直通或静音。VST3 / CLAP / App 现共用同一套 `onnx + soundtouch` 推理内核 |
| 捕获（录音） | **有** | `capture_status` / `start` / `stop` / `cancel` / `clear` / `delete_take` / `set_options` 与 VST3 同名同载荷；音频从 `process` 的输入总线按工程位置写进插件自己的时间线，实现与 VST3 是**同一个 crate**（`hifishifter-capture`） |
| 写回 / 分割 / 建轨 / 宿主撤销 / take / folder | 不可用 | 与 ARA-only 相同：REAPER 原生扩展是 VST3 COM 专属，CLAP 没有对应接口 |

**这张表是当前实现的契约，不是缺陷清单**：共享 ARA 运行时、`ClapAraEntry` /
`clap_ara_plugin_extension_t` 两级入口与 `AraBinder` 绑定均已接线，按 ARACLAP.h
定稿协议发现的宿主可完整使用。**REAPER 的 CLAP ARA 不可用**，原因在宿主侧（见上）；
REAPER 用户应使用 VST3 形态获得完整 ARA。其余宿主（Bitwig / Studio One /
MultitrackStudio）的 CLAP ARA 仍需逐个真机验收。

### 产物侧的工程证据（非真机宿主结论）

以下各条由产物自检与 cargo 用例覆盖，**不等价于**"已在 Bitwig / REAPER 里加载过"：

- **ABI 布局**：`backend/hifishifter-clap/tests/clap_layout.c` 用 `plugin-sdks.json`
  钉住的 CLAP 1.1.9 头文件编译原生 oracle，`clap_layout.rs` 逐项比对全部
  `#[repr(C)]` 结构的 size / align / 字段偏移（含 `clap_process_t` 的
  `transport` 指针、`clap_event_transport_t`、GUI/state/ports/render/timer 扩展表）。
  手写声明与黑盒探针不再互相印证同一份可能出错的假设。
- **宿主流**：`clap/stream.h` 允许宿主限制单次读/写字节数；`state.rs` 循环推进，
  短读不当成 EOF、短写不丢尾部、零进度与负数判失败，并有 64 MiB 上限防止挂死。
  `scripts/clap/probe.c` 用每次 7 字节的假宿主实际复现这条路径。
- **每实例状态**：采样率、`clap.render` 模式与 transport 观察值从进程级 static 移到
  `Instance::audio`。同一宿主进程里两个实例（不同采样率 / 一个实时一个离线）不再
  互相覆盖；`process` 在未 `activate` / 未 `start_processing` / 并发进入时明确拒绝。
- **窗口与资源**：`clap.gui` 真实创建子 HWND 并挂到宿主的 win32 父窗口；
  Windows 产物把 `frontend/` 与 `models.json` 装到模块同级的
  `HiFiShifter-resources`（`ResourceLayout::Sibling`），安装器与自检都断言
  `plugin.html` 落地。此前只装 `.clap` 单文件，宿主扫描得到插件却只会给空白窗口。
- **日志落盘只有一套**：VST3 与 CLAP 共用 `hifishifter-plugin::diagnostics`
  （目录、行格式、会话头、8 MiB 轮转、级别、panic hook 全部来自
  `kernel::logfile`）。文件名是 `plugin-<format>-<host>.log`，例如
  `plugin-vst3-reaper.log` / `plugin-clap-reaper.log`：格式来自唯一的
  `set_log_tag` 登记点，新增格式自动适配。外壳只把日志额外转发给宿主的
  `clap.log`，不再自己装 logger、开文件或定级别。
- **editorRoute 不能说 `unbound``：前端的 `captureStore` 在路由为 `unbound` 时**完全跳过**
  `capture_status` 轮询（那本是"视图连不上处理器"的降频判据）。CLAP 视图由本模块在
  进程内回答命令，路由因此报 `bound`；真正缺的 ARA/编辑能力由 `hostCapabilities`
  全 false 表达。这正是用户报的"当前不可用"的第二层原因。
- **输出接在渲染结果上**：`capture_stop`、清空/删 take、写入曲线/静态参数都会请求一次重渲染
  （防抖 600 ms），结果以 `ArcSwapOption<RenderTape>` 无锁发布；音频线程在直通之后
  覆盖素材区间，因此"没处理的地方"听起来与没插插件完全一致。渲染失败会写
  `[clap][render] render failed`，不会变成静默的"没声音"。
  端到端判据是**频率**而不是"有样本"：把捕获的 220 Hz 正弦整体抬高 12 个半音，渲染
  结果的主频必须变成 440 Hz（`a_pitch_edit_changes_the_rendered_audio`）。
- **WORLD 的 F0 估计器**：本仓库静态链接的 WORLD 里 `Harvest` 对任何输入都返回全 0，
  而"全 0 的 f0"会让声码器判定"没有有声音高"并**原样返回输入** —— 表现就是"曲线画满、
  声音没变"，且没有任何报错。因此默认改用 DIO + StoneMask，并在首选估计器产不出任何
  有声音高时自动换另一个再试一次（`vocode_one`）。`the_vocoder_actually_shifts_the_pitch`
  用 220→440 Hz 钉住这条。
- **SoundTouch 是可选 feature**：它是**链接期**依赖，一打开产物就多一个必须与宿主同目录
  的 `SoundTouchDLL.dll`，而 CLAP 外壳没有 loader 层。内核因此把 `soundtouch` 拆成
  feature（默认开，VST3/App 行为不变）；CLAP 关掉它，走静态链接的 Signalsmith。
  判据：`link /dump /dependents dist\HiFiShifter.clap` 里不得出现 `SoundTouchDLL`。
- **参数编辑器有一条真实的音高线**：`get_processor_params` 按算法给出参数描述符；
  `get_param_frames` / `set_param_frames` 读写时间线上的曲线（与 VST3 同一份
  `hifishifter_kernel::editor::params`）；音高**原线**由静态链接的 WORLD
  （`world_vocoder::estimate_f0_hz`）在后台线程上从捕获 PCM 估计后铺到工程帧网格 ——
  CLAP 外壳不链接 ONNX 推理栈，这是不引入 40 MB 依赖又让原线真实存在的那条路。
- **宿主调整窗口大小**：`clap.gui` 的 `adjust_size` 只把请求夹到合法区间后**原样放行**。
  此前它无条件写回 960×640，于是窗口永远拖不动、WebView 也收不到尺寸变化。
- **捕获的界面读数必须由外壳回答**：编辑器不推送传输事件，光标与捕获带右缘都读
  `get_playback_state`；时间线的重取由 `plugin_get_apply_state.host_version` 的**前进**
  触发（CLAP 里它等于捕获存储的修订号）。缺了前一条，页面上的光标永远停在 0、带子
  右缘跟着冻住（"捕获在计时，界面什么都不动"）；缺了后一条，take 已经落盘、时间线却
  永远不重取（"捕获结束没有片段"）。
- **波形峰值来自内存**：CLAP 的源是 `capture://blake3:<hash>`，磁盘上没有文件。
  `get_waveform_mipmap_binary` / `batch_get_waveform_mipmap` 由外壳按 `WFPK` v2
  协议直接从捕获 PCM 编码（分块 min/max，三级除数 16/512/4096），结果按源缓存 ——
  否则片段只是一块没有内容的色块。
- **捕获运行时是共享的**：`recorder` / `sidecar` / `store` / `timeline` / `transport`
  从 VST3 crate 搬进 `backend/hifishifter-capture/`（纯机械搬迁 + 中性类型转换），VST3 与
  CLAP 两个外壳都只是它的适配层：VST3 把 `ProcessContext` 变成 `InputBlock`，CLAP 把
  `clap_event_transport_t` 变成同一个形状。因此 "CLAP 能录" 与 "VST3 能录" 是同一条
  代码路径 —— 不是一份为 CLAP 重写的简化版。VST3 侧的 779 条库用例与捕获 crate 的 125 条
  用例逐条通过，判定零改动。
- **WebView2 用户数据目录**：profile 固定在
  `%LOCALAPPDATA%\HiFiShifter\webview\clap-<pid>`（`hifishifter_plugin_core::local_data`，
  与内核 `app_paths` 由跨 crate 测试逐字比对）。它**绝不能**落在模块旁边：模块装在
  `Program Files`，浏览器进程在那里无权建 `EBWebView`，宿主会弹出
  「Microsoft Edge 无法读取和写入其数据目录」而窗口一片空白（用户实测）。建目录后
  还会写一次探针文件，写不进去就在窗口里给出原因、且完全不拉起 WebView2；
  资源目录一旦混进 `webview/`，构建与自检脚本都会硬失败（运行期状态不进安装包）。
- **能力诚实性**：黑盒探针把 `clap.state` / `clap.render` / `clap.audio-ports` /
  `clap.timer-support` / `clap.gui` 必须存在、descriptor 声明 `ara:supported` 且
  不含 `ara:required`、entry ARA 工厂表与实例 ARA 扩展必须返回同一张可用接口都写成
  硬失败；REAPER 的私有草案 ID 则断成**必须缺席**（应答即崩宿主）；
  `verify.sh` 逐项断言探针输出。
  探针自身实现 `clap.log` 假宿主并断言收到日志、`clap.latency` / `clap.tail`
  必须返回 0。打印 "absent" 后照常返回成功的老判据不再成立。

REAPER 7.85 / Windows x64 的 CLAP 扫描、实例化与 Capture 形态已由隔离真机复测覆盖
（含音频 item 场景，退出码 0）；**CLAP ARA 在该宿主不可用**，取证与结论见
[崩溃分析报告](2026-11-08-reaper-clap-ara-crash-analysis.md)。其余宿主
（Bitwig / Studio One / MultitrackStudio）与保存重开、WebView 编辑仍按
`docs/plugin-host-survey.md` 保持「未测」。

### ARA 发现表的工程证据

`backend/hifishifter-clap/tests/ara_discovery.rs` 用 `ara2-bridge-companion` 的
**真实** CLAP 类型钉住四件改错就要等到宿主里才暴露的事：

- `ARACLAP.h` 的四个扩展 ID（entry 工厂稳定/兼容、实例扩展稳定/兼容）与
  `ara:supported` 常量逐字一致，另单独钉死 REAPER 的私有草案 ID —— 且断言它
  **永远拿不到扩展表**（`the_reaper_legacy_id_is_never_answered`）；
- `ClapAraEntry::new` 能由 descriptor ID 构造出恰好一条工厂关联，
  `get_factory_count` / `get_ara_factory` / `get_plugin_id` 三个回调可用；
- 宿主侧 `ClapAraHostFactory::discover` 读回的 `plugin_id` 与 descriptor
  `com.hifishifter.clap` 逐字相同（ARA-over-CLAP 的配对键）；
- 未接绑定器的实例对全部 ARA ID 如实返回 null。

companion 的 `clap` feature 只挂在 `[dev-dependencies]`：它是纯 checked-in 声明，
不需要 ARA/VST3/AudioUnit SDK，也不进产品依赖图。共享 ARA 运行时与
`AraBinder` 绑定已在产品代码中，REAPER 7.85 真机握手与授权 PCM 已通过；
其余宿主的读 region / 渲染 / 存读编辑仍待逐个验收。


### 契约层的依赖边界

`hifishifter-plugin-core` 只持有纯数据契约（时间块、音频平面、组件 state、资源根、
FFI panic 守卫），产品依赖只有 `log`。两条容易回退的归属规则：

- `fade`（校验 `hifishifter_kernel::fade_curves` 的形状接受域）与
  `host_identity`（Windows ToolHelp 进程探测 / VST3 宿主名证据）是 **VST3 专属**，
  留在 `hifishifter-plugin`；core 曾把它们连同 WebView2 / Windows API / ONNX /
  SoundTouch 一起机械复制进来，结果一个只做字节转发的 CLAP 外壳也要链接整个推理内核。
- 格式 crate 不得因为「core 有了」就继承宿主专属依赖。新增格式时先问"这个模块是否
  被至少一个非 VST3 格式消费"，答不出就留在 VST3。`no_tauri_in_dependency_tree.rs`
  与 `cargo tree -p <格式>` 是这条边界的判据。

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
| CLAP 外壳 | `backend/hifishifter-clap/` |
