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

## 3.1 CLAP（Capture-only 外壳，阶段 1）

当前 `hifishifter-clap` 是可被 CLAP 1.1.9 宿主扫描与加载的普通插入效果器：

| 能力 | 状态 | 契约原因 |
| --- | --- | --- |
| 实例化 / 激活 / process / 保存 state / 销毁 | 有 | `clap.plugin-factory` + `clap.state` + `clap.audio-ports`；生命周期按 CLAP 状态机严格校验 |
| 延迟 / 尾音 | 如实为 0 | `clap.latency` + `clap.tail`；逐帧直通、不缓存样本，0 是唯一正确答案 |
| 宿主日志 | 有 | `clap.log`：`plugin.init` 后查询 `clap_host_log` 并装上 logger，诊断不再静默丢弃 |
| transport 读数 | 按事件如实给 | 只从 `clap_event_transport_t` 读播放/录音/循环/拍号；`steady_time` 不推断播放。编辑器轮询的 `get_playback_state`（光标与捕获带右缘的唯一来源）由实例把位置抄进原子量后回答 |
| 离线 bounce | 按扩展如实给 | 只从 `clap.render` 的 `CLAP_RENDER_OFFLINE` 置位 |
| ARA 工厂 / 实例扩展 | **不暴露** | 共享 ARA 文档模型仍在 VST3 外壳里；暴露不能创建控制器的工厂会让宿主写穿空指针 |
| `ara:supported` / `ara:required` | 都不声明 | descriptor 只描述已履行的能力 |
| 内嵌 WebView | Windows 有 | `clap.gui` + 独立 WebView2 宿主，加载同一份 `frontend/dist`；非 Windows 如实降级 |
| 编辑器命令 | Capture 回执后端 | `plugin_health`、时间线、参数曲线（`get/set_param_frames`、静态参数）、波形峰值、传输读数都由本模块在进程内回答；未接线的命令立刻回结构化错误，不复用 VST3 的 ARA 会话 |
| 捕获（录音） | **有** | `capture_status` / `start` / `stop` / `cancel` / `clear` / `delete_take` / `set_options` 与 VST3 同名同载荷；音频从 `process` 的输入总线按工程位置写进插件自己的时间线，实现与 VST3 是**同一个 crate**（`hifishifter-capture`） |
| 写回 / 分割 / 建轨 / 宿主撤销 / take / folder | 不可用 | 与 ARA-only 相同：REAPER 原生扩展是 VST3 COM 专属，CLAP 没有对应接口 |

**这张表是阶段 1 的契约，不是缺陷清单**：阶段 2 只有把共享 ARA 运行时抽进 core、接好
`ClapAraPluginAdapter` 的六个生命周期观察点并跑完真机验收，才允许把 ARA 能力写回 descriptor。

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
- **CLAP 的日志落盘**：外壳除了把日志交给宿主的 `clap.log`，还写
  `%LOCALAPPDATA%\HiFiShifter\logs\plugin-clap.log`（8 MiB 软上限）。"录制键一直是灰的"
  这类问题只能靠"处理过几块 / 见过哪些 transport 标志 / 是否推出过工程位置"三个读数
  定位 —— 它们现在每次会话第一条 `capture_status` 就会写进日志。
- **editorRoute 不能说 `unbound``：前端的 `captureStore` 在路由为 `unbound` 时**完全跳过**
  `capture_status` 轮询（那本是"视图连不上处理器"的降频判据）。CLAP 视图由本模块在
  进程内回答命令，路由因此报 `bound`；真正缺的 ARA/编辑能力由 `hostCapabilities`
  全 false 表达。这正是用户报的"当前不可用"的第二层原因。
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
  `clap.timer-support` / `clap.gui` 必须存在、descriptor 不得含任何
  `ara:` feature、ARA 工厂必须缺席都写成硬失败；`verify.sh` 逐项断言探针输出。
  探针自身实现 `clap.log` 假宿主并断言收到日志、`clap.latency` / `clap.tail`
  必须返回 0。打印 "absent" 后照常返回成功的老判据不再成立。

真机宿主的扫描、录音、WebView、保存重开与 ARA 配对仍按 `docs/plugin-host-survey.md`
保持「未测」，阶段 2 / 3 的 No-Go 结论不变。

### 阶段 2 的可执行前置（已就绪，但不是 ARA 能力）

`backend/hifishifter-clap/tests/ara_discovery.rs` 用 `ara2-bridge-companion` 的
**真实** CLAP 类型钉住四件以后接线时必须正确、且改错就要等到宿主里才暴露的事：

- `ARACLAP.h` 的四个扩展 ID（entry 工厂稳定/兼容、实例扩展稳定/兼容）与
  `ara:supported` 常量逐字一致；
- `ClapAraEntry::new` 能由 descriptor ID 构造出恰好一条工厂关联，
  `get_factory_count` / `get_ara_factory` / `get_plugin_id` 三个回调可用；
- 宿主侧 `ClapAraHostFactory::discover` 读回的 `plugin_id` 与 descriptor
  `com.hifishifter.clap` 逐字相同（ARA-over-CLAP 的配对键）；
- 阶段 1 对这四个 ID 的查询**仍然全部返回 null**。

companion 的 `clap` feature 只挂在 `[dev-dependencies]`：它是纯 checked-in 声明，
不需要 ARA/VST3/AudioUnit SDK，也不进 `cargo tree -p hifishifter-clap --edges normal`
（仍是 34 个 crate）。因此这**不是**"ARA 已暴露"，只是把阶段 2 的 ID 与发现表
契约提前验完。真正放行仍缺：共享 ARA 运行时从 VST3 外壳解耦、六个 `observe_*`
生命周期接线、以及支持 ARA-over-CLAP 的宿主上读 region / 渲染 / 存读编辑的真机验收。


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
