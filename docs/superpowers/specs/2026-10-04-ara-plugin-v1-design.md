# HiFiShifter ARA 插件 v1 · 实现设计（方案）

> 状态：待评审
> 日期：2026-10-04
> 分支：`codex/ara-plugin`
> 上位文档：[ARA 桥接产品设计](2026-10-04-ara-bridge-design.md)（**冲突时以它为准**）
> 实测边界：[内核边界](2026-10-04-ara-kernel-boundary.md)
> 探针结论：[HANDOFF](../../../probe/ara/HANDOFF.md) · [ledger](../../../probe/ara/EXECUTION-LEDGER.md)

---

## 0. 这份文档解决什么

产品意图（"让 HiFiShifter 以 ARA2 插件挂在 DAW 里、时间线随 DAW 走"）已在
[产品设计](2026-10-04-ara-bridge-design.md) 定稿；可行性（R1 / R2）已由探针实测坐实。
缺的是**把它做出来的完整工程方案**：内核怎么切、插件怎么写、参数怎么回本体、
每条假设怎么验、按什么顺序交付。

本文是那一份方案。它是**实现设计**，不重复产品设计的取舍论证。

---

## 1. 目标与验收口径（v1）

产品设计的成功判据是：

> 在 REAPER 里挂上插件、把一段人声素材切片，HiFiShifter 能呈现与 DAW 一致的 clip
> 布局，调参后的结果在 DAW 里播放正确，且工程重新打开后不需要重新合成。

把它拆成可判定的四条。**每条都必须能在 REAPER 里被观测，不能只靠单测**：

| # | 判据 | 观测方式 | 归属阶段 |
| --- | --- | --- | --- |
| A1 | 插件被 REAPER 以 ARA 方式加载，且它看到的 source / region 数与工程一致 | 插件日志 + 轨道截图 | Phase 2 |
| A2 | DAW 侧的切片 / 移动 / 走带变化，插件侧的 `TimelineState` 跟着变 | 插件日志打印时间线摘要 | Phase 2 |
| A3 | 播放时 DAW 听到的是**修音后**的音频，不是原声 | 盲听 + 波形比对 | Phase 3 |
| A4 | 关掉本体、重开工程，插件仍能渲染出同一结果（曲线随工程走） | 重开工程后比对渲染产物哈希 | Phase 4 |

另有一条**贯穿性约束**，它不是功能而是架构不变量：

> **A5：插件进程里不出现 Tauri / WebView2。**
> 判据是构建产物审计（`dumpbin /dependents` 里无 `WebView2Loader.dll`、无 Tauri 相关
> 导入），而不是"代码里没写"。

---

## 2. 已确立的事实（探针实测，勿重查）

| # | 事实 | 证据 |
| --- | --- | --- |
| F1 | Rust 能以可接受成本接触 ARA：`ara2-bridge` 0.3.0 + 自写 VST3 外壳即可被 REAPER 加载并绑定 ARA | [rust-path FINDINGS](../../../probe/ara/rust-path/FINDINGS.md) |
| F2 | ARA 模型 → `TimelineState` 在渲染所需字段上无损；丢失项是 7 条**显式降级** | [roundtrip FINDINGS](../../../probe/ara/captures/roundtrip/FINDINGS.md) |
| F3 | 映射后的时间线在逐样本口径上与手工构造的参考一致（阈值 1e-6，已通过） | 同上 §7 的补做（`hifishifter-plugin` 测试） |
| F4 | 基线：app `762 passed / 4 failed / 1 ignored`，内核 `10 passed`；4 个失败是既有 `/tmp` 路径环境性失败 | [plan](../../plans/2026-10-04-ara-bridge-probe.md) 环境前提 |
| F5 | `sampleAccessEnabled: false` 是**撤销**语义，不是拒绝（`enable=true` 在绑定后、`false` 在停用时） | ledger "Task 2 Step 3" |
| F6 | 一个 ARA 实例 = 一条轨道；REAPER 为一条轨道建 3 个 `IAudioProcessor` 实例，**每实例必须有自己的 companion 绑定** | 同上 |
| F7 | 内核抽取的真实依赖闭包是 **43 模块 / 2.57 MB**，其中只有 5 个模块碰 `tauri::`（`state` 10 处、`pitch_clip` 4、`pitch_analysis` 2、`recording` 2、`audio_engine` 10） | 本次实测（见 §4.2） |

### 2.1 仍未验证的四项（**不是已解决**）

| # | 未决项 | 为什么重要 | 收口阶段 |
| --- | --- | --- | --- |
| U1 | **ARA 没有反向位**。宿主如何表达倒放（改源内容 / region 属性）未在真实宿主里观测过 | 若是后者，倒放区域会渲染成正放 | Phase 2 实验 |
| U2 | **宿主级拉伸观测缺一个**。awkward 样本实际不含拉伸（5 个 region 全部 `durMod == durPlay`） | 拉伸分支目前只在合成样本上验证过 | Phase 2 实验 |
| U3 | **R4：渲染窗口内能否稳定供音**。探针的 `process()` 是空实现 | 失败表现为可听见的空洞 | Phase 3 |
| U4 | **源内容版本与缓存键**：宿主在插件未运行时改源内容、且几何字段不变时，是否会给缓存造成过期命中 | 静默复用过期渲染 | Phase 3 实验 |

### 2.2 一个必须记住的探针更正

Task 1 的 ledger 里"拉伸 = 时长差，awkward 样本已证明"**证据不成立**：那份样本里
5 个 region 全部 `durationInModificationTime == durationInPlaybackTime`，被读成拉伸的
那一条只是**区间被裁短**。公式本身成立（合成样本已验证），但**宿主级观测仍缺**（U2）。

---

## 3. 架构总览

### 3.1 一份内核、两种形态

用户在会话里问过"ARA 是独立的还是原本的 app 版可以同时支持 plugin"。
答案是**一份内核、两种形态**（产品设计 §5.1）：

```
        ┌────────────────────────────┐
        │  hifishifter-kernel        │  离线内核：不依赖 Tauri，不依赖音频设备
        │  state(model) / mixdown /  │
        │  renderer / vocoder /      │
        │  render_cache / render_key │
        └───────┬────────────┬───────┘
                │            │
   ┌────────────▼───┐   ┌────▼─────────────────────┐
   │ Tauri app      │   │ hifishifter-plugin        │
   │ （HiFiShifter  │   │ （DAW 内的 ARA 插件）      │
   │  独立产品）     │   │                           │
   │ + audio_engine │   │ + VST3 外壳               │
   │   （cpal 设备） │   │ + ARA 宿主适配            │
   │ + commands IPC │   │ + 宿主回调替代 cpal        │
   └────────────────┘   └───────────────────────────┘
```

- app 仍是**完整独立产品**：不插 DAW 也能用全部功能。
- 插件是**额外构建目标**：两者可同时安装、互不排斥（不同进程、不同文件）。
- v1 插件**不带编辑器 UI**（产品设计 §4.4）：修音界面仍在本体。

### 3.2 进程与线程视图

```
DAW 进程
 ├─ 宿主线程 ── ARA 回调 ──→ PluginModel（我们实现）
 │                            └─ TimelineState（内核类型）
 ├─ 渲染线程 ── process() ──→ render_callback_f32（内核）+ 快照
 └─ 编辑线程 ── 状态通道服务端（命名管道）
                                  ▲
HiFiShifter.exe（独立进程）        │ JSON 行协议
 └─ 状态通道客户端 ────────────────┘
```

### 3.3 crate 布局（目标态）

```
backend/
  Cargo.toml                 ← workspace 根（members = src-tauri / kernel / plugin）
  Cargo.lock                 ← 单一锁文件
  src-tauri/                 ← app（Tauri）
  hifishifter-kernel/        ← 离线内核（新）
  hifishifter-plugin/        ← ARA 插件（新，cdylib）
```

现状是三个各自带 `[workspace]` 的独立 crate、两份 `Cargo.lock`。合并的理由与代价见 §4.1。

---

## 4. 关键决策

### 4.1 D1：把 workspace 根提到 `backend/`

**决定**：新建 `backend/Cargo.toml` 作为 workspace 根，members = `["src-tauri",
"hifishifter-kernel", "hifishifter-plugin"]`；删掉两个新 crate 的 `[workspace]`；
`backend/src-tauri/Cargo.lock` 移到 `backend/Cargo.lock`，删掉两个 crate 自带的 lock 副本。

**理由**：现在插件 crate 自带一份从 app 复制的 `Cargo.lock`，这是**已实测的踩坑点** ——
不复制就会解析出不同版本的 `windows-core`，`backend_lib` 作为依赖直接编译失败
（`webview2_accelerators.rs` 的 `cast()` 找不到 trait）。靠"记得复制"维持的东西迟早会漂。
合并成一份 lock 之后，这个问题在结构上消失，且 `cargo test --workspace` 成为可能。

**代价（须明确接受）**：cargo 的 `target/` 从 `backend/src-tauri/target` 变成
`backend/target`。影响：`.gitignore`（已忽略 `target/`，需确认）、本地重建一次依赖
（约 10 分钟）、文档里的路径引用。

**错了的代价**：一次失败的 cargo 解析，回退是 `git mv` 回来。可逆。

### 4.2 D2：先拆 `state.rs`，再搬模块

**实测**：从 `mixdown` 出发的依赖闭包是 **43 模块 / 2.57 MB**，其中包含
`project` / `notebook_assets` / `hfspeaks_v2` / `temp_manager` / `media` / `recording` ——
这些**都不是内核**。它们被卷进来，是因为 `state.rs`（11826 行 / 513 KB）把
"数据模型"与"运行时容器 `AppState`"放在同一个模块里：

```
state.rs
 ├─ 纯模型：TrackParamsState / ClipTake / Clip / Track / TimelineState /
 │          HistoryOp / TimelineHistory / ProjectState / SplitTransition* / RippleMode
 │          + impl TimelineState（两段，共约 7000 行）
 └─ 非模型：AppState（持有 tauri::AppHandle / 波形缓存 / 渲染缓存句柄 / recording 状态）
            + impl AppState（10 处 tauri::，含 Emitter 与 Manager）
```

**结论：依赖闭包 ≠ 内核边界。** 在拆开 `state.rs` 之前算出来的闭包必然被 `AppState` 污染。

**决定**：分两步，**先拆后搬**。

1. **拆**（行为零变化、路径零变化）：`state.rs` → `state/mod.rs`（纯再导出）+
   `state/model.rs`（纯模型）+ `state/app.rs`（`AppState` 与其 impl）。
   `state/mod.rs` 写 `pub use model::*; pub use app::*;`，于是 `crate::state::X` 全部照旧。
   验收：**app 单测数一个不变**。
2. **搬**：把 `state/model.rs` 移进内核；app 侧 `state/mod.rs` 保留
   `pub use hifishifter_kernel::state::*;` 接回路径。验收：app 减少 / 内核增加 / 合计不变。

**拆完之后必须重算闭包**，因为只有那一次的模块集才是真内核。§4.2 的 43 是**拆之前**的数。

**实测（2026-10-04，拆分已落地后重算）**：

| | 模块数 | 字节 | 碰 `tauri::` 的模块 |
| --- | --- | --- | --- |
| 拆分之前 | 43 | 2 568 908 | `state`、`pitch_clip`、`pitch_analysis`、`recording`、`audio_engine` |
| 拆分之后（`state` 只剩 `model.rs`） | **40** | **2 149 796** | **`pitch_clip`、`pitch_analysis`** |

**预测被部分证伪，据实记录**：当初推断 `hfspeaks_v2` / `notebook_assets` / `temp_manager` /
`recording` 会离开闭包。实测 **`recording` 离开了，另外三个没有** —— 说明它们不只被
`AppState` 引用，模型侧也引用了它们（`project` / `models` 这条链）。这条推断当时已标注
为推断、并写明"以重算为准"，所以没有造成误施工，但结论必须按实测改写。

**另一条实测（比上述更正更要紧）**：`state/model` **不是一个叶模块**，它引用
`project`（`CustomScale`）、`models`、`midi_import`、`audio_utils`、`time_stretch`。
所以"把 model 搬进内核"必须等这些依赖先搬 —— **机械搬迁（原计划的 Task 6）要排在
搬 `state/model` 之前**，不能反过来。见 §4.9。

**第三条实测（2026-10-04 补，测量脚本本身有 bug，修好后推出的结论）**：
测量脚本的正则漏掉了 `pub(crate) mod` 声明，导致闭包被低估。修正后，从 `mixdown` 出发的
闭包含 **46 模块**（仅生产代码 44），多出来的 6 个（`commands` / `recording` / `search` /
`system_clipboard` / `linux_clipboard` 及 `commands` 子模块）**全部由一条边拉进来**：

```
pitch_analysis/schedule.rs -> crate::commands::playback::request_background_render
                           -> crate::commands::playback::AUTO_BG_RENDER_ENABLED / BG_RENDER_PITCH_PENDING
```

**这条边就是 §4.3 表里的第 2 类"向音频引擎投递命令"** —— 也就是 `HostServices` 的职责。
把这**一条边**改掉（外加把 `project.rs` 里那几处 `#[cfg(test)]` 的
`commands::channel_scan` 测试挪回 app），闭包就回到 **39 模块 + `state/model`**，
与上面那张表的数字一致。

**所以执行顺序是硬的：先 `EngineCommand` + `HostServices`，再大搬迁。** 反过来做，
`pitch_analysis` 会把整条 `commands` 链（196 处 `tauri::`）带进内核，
"内核不认识 Tauri"这条不变量当场破产。

**错了的代价**：若拆开后闭包仍然很大（例如 `TimelineState` 的方法真的依赖 `project`
的复杂逻辑），则内核会比预期大，插件二进制更大、编译更慢；但架构方向不变，只是收益变小。

### 4.3 D3：内核新增宿主回调出口 `HostServices`

**问题**（实测清单，来自内核边界文档 §2）：内核 worker 回头找宿主只有三类，
第 1 类（发事件）已有出口，第 2、3 类没有：

| # | 用途 | 现状 |
| --- | --- | --- |
| 1 | 向 UI 发进度事件 | 出口已建（`events::{EventSink, SharedEventSink}`），**但调用点还没迁** |
| 2 | 向音频引擎投递命令 | `mpsc::Sender<EngineCommand>`（`pitch_clip.rs` 3 处） |
| 3 | 调度音高分析 | `app.state::<AppState>()`（`engine.rs` 4 处） |

**决定**：新增 `hifishifter_kernel::host::{HostCallbacks, SharedHostCallbacks, HostServices}`，
形状与 `events` 一致 —— **trait 定义契约、具体类型持有 `OnceLock`、固有方法转发**，
这样调用点的改写量最小（把 `state.app_handle.get()` 换成 `state.host.request_pitch_analysis(..)`）。

```rust
/// 内核 worker 回头找宿主的出口。宿主（Tauri app / ARA 插件）各自注入实现。
pub trait HostCallbacks: Send + Sync + 'static {
    /// 把一条引擎命令投递给设备层。插件侧可以丢弃，或转成 ARA renderer 的请求。
    fn send_engine_command(&self, command: EngineCommand);
    /// 请求为某个根轨调度音高分析。app 侧读 `AppState`；插件侧读自己的文档状态。
    fn request_pitch_analysis(&self, root_track_id: &str);
}

/// 未安装实现时的出口：不发命令、不调度。
///
/// 【为什么要能"未安装"】内核会被单测与插件直接使用，那时没有宿主。
/// 语义必须是**静默降级**而不是 panic —— 内核方法在无宿主时应当仍然完成
/// 自己的那份工作，只是不对外发通知。
pub struct HostServices { inner: OnceLock<SharedHostCallbacks> }
```

**错了的代价**：若插件侧其实需要第 2、3 类的更多能力（例如读宿主 tempo map），
trait 需要再加方法 —— 那是加方法，不是改架构。

### 4.4 D4：`EngineCommand` 成为纯内核类型，`AppHandle` 不再走命令通道

**实测**：`EngineCommand` 里唯一非数据型的变体是 `SetAppHandle { handle: tauri::AppHandle }`，
它是 `audio_engine/types.rs` 碰 `tauri::` 的原因，也是"命令枚举无法进内核"的唯一障碍。

**决定**：

1. `EngineCommand` 及其载荷（`StretchKey` / `AudioKey` / `MetronomeConfig` / `MetronomeClick`）
   移进内核，**删掉 `SetAppHandle`**。
2. engine worker 需要的 `AppHandle` 改从 `app_events::app_handle()` 取（它已经是进程级出口），
   不再经命令通道传递。

**注意**：`audio_engine` **整体留在 app 层**（它是 cpal 设备边界，产品设计 §5.1 要替换的
正是它）。所以 `engine.rs` / `snapshot.rs` 里那 10 处 `tauri::` 与约 35 处 `emit` **不动** ——
只有 `EngineCommand` 的类型定义搬走。这条边界必须写死在评审里，否则搬迁会失控。

**错了的代价**：worker 拿不到 handle 会导致启动早期的事件丢失；`app_events::app_handle()`
在 setup 里注册，早于引擎启动，所以时序上安全（需要在计划里用一个测试钉住）。

### 4.5 D5：宿主的内容版本折进**既有**的渲染键输入

**问题**（产品设计 §4.2）：宿主的音频源是权威源；在 DAW 里改源内容而不改路径是常规操作。
现有渲染键已经包含 `source_path` / `source_file_mtime` / `source_file_size` /
`source_file_fingerprint` 四个源侧输入，并有测试 `every_render_input_changes_the_hash`
逐项钉住它们必须影响哈希。

**决定**：**不改 `render_key.rs`**。插件在把 ARA 源映射成 `Clip` 时：

- `source_path` ← `audioSource.persistentID`（REAPER 实测就是素材绝对路径）；
- `source_file_fingerprint` ← `hash(persistentID ‖ host_content_version ‖ sampleCount ‖
  sampleRate ‖ channelCount ‖ duration)`，其中 `host_content_version` 在
  `PluginModel::update_audio_source_content` 回调里逐源自增；
- `source_file_mtime` / `source_file_size` ← 能读到真实文件时填真实值（REAPER 的
  persistentID 就是路径，常见情形可读），否则留 `None`。

于是"源内容变更 ⇒ 渲染键变更"这条产品设计要求，落在**已经存在且已被测试覆盖**的
输入上，改动面从"给管线加一个新类型"缩小为"映射时多算一个指纹"。

**残余风险（U4，不得当作已解决）**：若宿主在插件未运行期间改了源内容、且既没有递增
版本号、几何字段也没变、路径也读不到元数据，就可能命中过期缓存。Phase 3 有一个
专门的实验（§6.3）来判定这个窗口是否存在。

**错了的代价**：过期缓存被复用 → DAW 里听到旧音频。这是**静默出错**类型，
所以宁可多失效（多渲染一次）也不能少失效。若实验证明窗口存在，退回方案是
"源 PCM 读入后按首个 64 KB 内容算指纹"。

### 4.6 D6：参数编辑通道（收敛产品设计 §5.4）

产品设计把 §5.4 标为"本设计里最未收敛的部分"。这里给出 v1 的具体形态。

**权威归属：插件实例持有曲线的权威副本。**

理由不是偏好，而是被 §1 的 A4 判据逼出来的：工程重开后曲线必须还在，而工程文件由宿主保存。
曲线的落点必须是与工程一起走的东西。选 **VST3 组件 state**（`IComponent::getState` /
`setState`）而不是 ARA 文档归档：归档在 ARA 语义上属于"模型对象图的归档"，
把插件私有的参数曲线塞进去会混淆语义；而 VST3 state 块本来就是"插件任意状态"的天然落点，
探测外壳里这两个函数已经是空实现，填进去即可。

**通道形态：本体（Tauri app）是客户端，插件实例是服务端。**

| 方面 | v1 决定 | 理由 |
| --- | --- | --- |
| 传输 | 本机命名管道（Windows），长度前缀 + JSON 行协议 | 编辑是人手操作、低频，延迟要求宽松；管道自带就绪与背压语义，不需要自己发明 |
| 依赖 | `interprocess` crate（同时给 Windows 命名管道与 Unix domain socket） | 为将来的 macOS 铺路，且不引入 async 运行时 |
| 发现 | 插件把 `{pid, pipe_name, document_name, track_name, heartbeat_ms}` 写进 `%LOCALAPPDATA%\HiFiShifter\ara-instances\<instance>.json` | 目录扫描比端口探测简单，且天然支持多实例 |
| 并发 | **单写者 + 乐观并发**：`commit(curves, base_revision)`；版本不匹配返回 `Conflict{current_revision}` | "最后写者胜"会**静默丢编辑**，那是这类工具的致命缺陷。宁可让用户重做一次 |
| 撤销 | 插件侧在 `Persistence::restore_document` 时使 `revision` 前进 | DAW 的 undo 会把 state 块回滚，本体必须察觉 |
| 实例消亡 | 曲线从 VST3 state 重建，不依赖本体在线 | A4 判据；也是"本体没开也能正确播放"的前提 |
| 本体不在线 | 插件用自己持有的副本渲染，结果正确；只是不能编辑 | **这条要写进用户可见说明**：DAW 里能听能播，改参数要开本体 |

**v1 的编辑对象**：五条逐采样参数曲线（pitch / breath / tension / formant / volume）
与每轨静态参数。**播放头位置与选中对象不进 v1** —— 它们不影响渲染正确性，
而每多一条通道就多一份并发语义要收敛。

**错了的代价**：若"一次只有一个本体在编辑"这个前提不成立（用户开了两个本体窗口），
乐观并发会频繁冲突。代价是体验变差（要重做），不是数据损坏。

### 4.7 D7：插件侧用 `ara2-bridge-plugin` 的 `PluginModel`，不手写回调委托

探针用的是 `ara2-bridge-companion` 的**低层**适配器 + 自己收集模型
（`probe/ara/rust-path/src/model.rs`，236 行手工收集）。

**实测发现**（本次读锁定的 crate 源码得出）：同仓库的 `ara2-bridge` **默认 feature `plugin`**
已经提供了一层**高层插件框架**，且就在本机 registry 里：

```
ara2-bridge-plugin-0.3.0/src/
  traits/{model,content,persistence}.rs   ← 要实现的语义 trait
  processing.rs                           ← Plugin / PluginBuilder / SemanticCapabilities
  realtime.rs                             ← RealtimeHeadTailAdapter（对应 R4）
  analysis.rs                             ← AnalysisCoordinator / AnalysisProgress
```

要实现的 trait 与 ARA 的模型回调一一对应且**已经是安全的 Rust 语义**：
`DocumentLifecycle` / `MusicalContexts` / `RegionSequences` / `AudioSources` /
`AudioModifications` / `PlaybackRegions`，外加 `ContentProvider` / `AnalysisProvider` /
`Persistence`。构造方式是 `PluginBuilder::new(model).build()?`。

**决定**：产品插件实现这些 trait，**不再手写回调委托**。探针那份手工收集保留为
对照物（它的价值是"用最小代码证明路径可通"），但产品代码不复用它的委托层，
只复用它的 **VST3 模块外壳**（`GetPluginFactory` / `InitDll` / `ExitDll` + 最小组件，
以及三条 ABI 硬事实：GUID 布局的 IID、`IPluginFactory` 直接继承 `FUnknown`、
REAPER 要求编辑控制器）。

**错了的代价**：若高层框架在某处与 REAPER 不兼容，回退路径是探针已验证的低层适配器 ——
那是一条**已知可行**的路，所以这个决定是低风险的。

### 4.8 D8：沿用产品设计的三条边界

- v1 单实例（一个实例 = 一条人声编辑轨）；
- v1 插件内不做参数编辑器 UI（§4.4）；
- `vslib` 的分发不承诺、Linux 出局、FL Studio 不支持（§3、§7）。

### 4.9 D9：原生依赖的构建归属（**执行时发现，待评审确认**）

计划里把 `time_stretch` / `metronome` / `state/model` 当成"叶模块，`git mv` 即可"。
执行到这一步时发现两件事都不成立，其中一件是**结构性的**：

**（1）`state/model` 不是叶模块。** 它引用 `project`（`CustomScale`）、`models`、
`midi_import`、`audio_utils`、`time_stretch`。所以顺序必须是**先搬依赖，再搬 model** ——
原计划的 Task 3 / Task 6 顺序是反的。

**（2）`time_stretch` 拖着一批"由 build.rs 编译的原生代码"。** 它的两个后端：

| 后端 | 形态 | 现状 |
| --- | --- | --- |
| `sstretch.rs` | **静态链接**的 Signalsmith Stretch + 一层 C 包装（`sstretch-c.cpp`） | 由 `backend/src-tauri/build.rs` 编译并用 `cargo:rustc-link-lib=static=signalsmith_stretch` 链接 |
| `soundtouch.rs` | 运行时加载的 `SoundTouchDLL.dll`（LGPL 规避） | 由同一个 `build.rs` 用 CMake 构建 |

这意味着：**把这两个模块 `git mv` 进内核，在 app 里"看起来"仍然能编译**（因为链接参数
来自 app 的 build 脚本，最终二进制的链接不受影响）—— 但**插件单独构建内核时会链接失败**。
这是"看起来成功、到 DAW 里才炸"的一类陷阱，必须在设计里封死。

**决定（建议，待确认）**：**把这两个原生依赖的构建搬进 `hifishifter-kernel` 自己的
`build.rs`**，并从 app 的 `build.rs` 里删掉对应段落。理由：

1. 内核是"离线音频内核"，时间拉伸是它的一部分；原生依赖跟着它走才自洽。
2. 插件与 app 都需要这两个后端 —— 放在内核里，两边都自动获得。
3. 若改成"宿主注入后端函数"（像 `HostServices` 那样），插件**仍然**需要自己编译这两个
   原生库，问题只是被挪了个位置，没有消失。

**代价**：`backend/src-tauri/build.rs` 里 soundtouch（CMake）与 sstretch（C++ 包装）
两段约 250 行要搬到内核的 build.rs，且两边的构建产物路径要重新对齐。这是一次
**中等规模**的构建系统改动，失败的表现是链接错误，不会静默出错。

**替代方案（若不想现在动构建系统）**：保留 `sstretch` / `soundtouch` / `time_stretch`
在 app 层，只把 `UserStretchAlgorithm` / `StretchAlgorithm` 两个 **enum 与设置结构**
搬进内核（`EngineCommand` 只需要它们）。代价是内核的 `time_stretch` 无法真正拉伸，
渲染管线必须在 Task 6 再拆一次 —— 相当于把同一件事做两遍。

---

## 5. 组件设计

### 5.1 内核 crate：模块集与搬迁顺序

搬迁顺序按"依赖深度从叶到根"，每一步都保持 app 侧路径零改写、测试合计不变：

1. `fade_curves`、`byte_budget_cache`、`util`（**已迁**）
2. `events`、`host`（**出口层**，先于搬模块，因为后面的模块要用）
3. `EngineCommand` + 载荷类型（`audio_engine/types.rs` 的数据部分）
4. `state/model.rs`（D2 拆分之后）
5. `mixdown`、`render_key`、`render_cache`、`synth_clip_cache`、`renderer/*`、`vocoder/*`
6. `pitch` 系、`pitch_editing`、`formant_cache`、`formant_morph`
7. `import/*` 中属于内核的部分（`reaper_import`、`midi_import`）与 `time_stretch`、`encode`

**每步的验收口径**（内核边界文档 §6 已定，此处重申）：app 单测减少、内核增加、
**两边合计不变**。当前基准：app `762 / 4 / 1`，内核 `10`，合计 `772 / 4 / 1`。

### 5.2 插件 crate：结构

```
hifishifter-plugin/src/
  lib.rs          ← cdylib 入口：GetPluginFactory / InitDll / ExitDll
  vst3/           ← VST3 模块外壳（从探针生产化）
    factory.rs    ← IPluginFactory(+2)：类清单 = 音频效果 + ARA 主工厂 + 编辑控制器
    component.rs  ← IComponent + IAudioProcessor
    controller.rs ← 最小编辑控制器（无参数、无 GUI、createView 返回 null）
  ara/            ← ARA 宿主适配（← 探针 src/model.rs 的产品化替身）
    mod.rs
    model.rs      ← 实现 PluginModel 的各 trait；累积出 AraDocument
    mapping.rs    ← AraDocument → TimelineState（← 现有 src/ara.rs）
  render/         ← 渲染闭环
    source.rs     ← AudioSourceReader：经 ARA 读源 PCM
    engine.rs     ← render_callback_f32 的宿主侧驱动
  state_channel/  ← §4.6 的状态通道（服务端）
```

### 5.3 数据流

```
宿主 beginEditing → createAudioSource / createAudioModification / createPlaybackRegion
      ↓（PluginModel 回调，逐对象）
  AraDocument（内存模型）
      ↓ mapping
  TimelineState（内核类型）—— 与本体加载工程后的类型**是同一个**
      ↓
  渲染：render_mixdown_interleaved(timeline, opts) / render_callback_f32(..)
      ↓
  产物：VST3 音频输出 + 缓存（render_cache，键由 §4.5 保证不过期）
```

### 5.4 生命周期

| 事件 | 插件侧动作 |
| --- | --- |
| VST3 `initialize` | 建工厂信息；**不**建 ARA 运行时 |
| 宿主查询 `IPlugInEntryPoint2` | 建 ARA 运行时与 `PluginModel` 实例 |
| `bindToDocumentControllerWithRoles` | **每个处理器实例一份** companion 绑定（F6） |
| `beginEditing` … `endEditing` | 累积模型变更；`endEditing` 时映射成 `TimelineState` 并置脏 |
| `enableAudioSourceSamplesAccess(true)` | 登记该源可读；准备 `AudioSourceReader` |
| `updateAudioSourceContent` | 该源 `content_version += 1`（D5） |
| `process()` | 从快照取音频；未就绪时用 `RealtimeHeadTailAdapter` 的提前窗口 |
| `getState/setState` | 序列化 / 反序列化参数曲线（D6） |
| `enableAudioSourceSamplesAccess(false)` | 释放该源的 reader（F5：这是撤销语义） |
| `terminate` | 释放 companion 绑定与 ARA 运行时（探针把适配器泄漏到进程结束，产品不可） |

---

## 6. 验证计划

### 6.1 R3 / R4 怎么判定

| 假设 | 判定实验 | 失败意味着 |
| --- | --- | --- |
| R3：设备边界可替换（cpal 出、宿主回调进） | Phase 3 在 REAPER 里播放，音频从 `process()` 出而不是本地声卡 | 播放模型要重做 |
| R4：渲染窗口内总能给出音频 | Phase 3 播放 30 秒含静音间隙的素材，逐帧检查输出的非零样本比例；人为让首块 miss 以触发等渲染分支 | 出现空洞 → 需要提前渲染调度或同步阻塞策略 |

### 6.2 未决项 U1 / U2 的收口实验（Phase 2）

- **U2（拉伸）**：把插件的 `SemanticCapabilities` 声明为
  `Timestretch | ReflectTempo | ContentFades`，在 REAPER 里对一个 region 做真实拉伸
  （拖动 item 边缘改变长度），重采 ARA 模型，断言 `durationInModificationTime !=
  durationInPlaybackTime`。
- **U1（倒放）**：在 REAPER 里对 item 执行 Reverse，重采模型，检查
  (a) region 属性是否变化、(b) `audioSource.persistentID` / 内容版本是否变化。
  判据：若 (a)(b) 都不变，则"宿主用改源内容表达倒放"**不成立**，需要在设计里
  另找表达（这是会导致"倒放渲染成正放"的缺陷）。

### 6.3 未决项 U3 / U4 的收口实验（Phase 3）

- **U3**：见 §6.1 的 R4。
- **U4（缓存过期窗口）**：① 插件未运行时替换源文件内容（保持路径与时长）；
  ② 打开工程播放；③ 断言插件重新渲染（日志里出现 cache miss）而不是命中旧条目。

### 6.4 每阶段的验收口径

每个阶段都必须给出**实测证据**（日志、截图、哈希），不接受"应该没问题"。

---

## 7. 分阶段交付

> 每阶段**独立可交付、独立可验收**。下一阶段的计划在上一阶段验收通过后再写 ——
> 这是探针计划里"不通过就不往下"的同一纪律。

| 阶段 | 交付物 | 验收 | 杀死判据 |
| --- | --- | --- | --- |
| **Phase 1 内核边界落地** | 插件 crate 只依赖 `hifishifter-kernel`；workspace 合并 | A5 的静态部分（依赖树里无 `tauri`）；app+内核测试合计 `772 / 4 / 1` 不变 | 若拆完 `state` 后闭包仍含 `commands` / `audio_engine` 等非内核模块且无法在 2 天内切开 → 停下来重估"内核"的定义 |
| **Phase 2 插件骨架产品化** | 插件被 REAPER 加载、绑定 ARA、打印真实 source/region 数；U1/U2 收口 | A1、A2 | 若高层 `PluginModel` 框架在 REAPER 下不可用，回退探针低层路径；两条都失败则停 |
| **Phase 3 渲染闭环** | 播放听到修音结果；U3/U4 收口 | A3 | 若 R4 不可解（窗口内拿不到音频）→ 整个进程内方案重估，报告用户 |
| **Phase 4 参数通道与持久化** | 本体改参数 → DAW 渲染变化；重开工程曲线还在 | A4 | 若乐观并发在真实使用中频繁冲突到不可用 → 降级为"某时刻只允许一个本体窗口" |
| **Phase 5 打包与分发** | 可安装的插件包 + 用户说明（含"本体不在线时能听不能改"） | 干净机器上装 → 判据 A1–A4 | — |

**本次交付的计划文档只覆盖 Phase 1 + Phase 2**（见
[plan](../plans/2026-10-04-ara-plugin-v1-phase1-2.md)）。Phase 3–5 的逐任务计划在
Phase 2 验收后按本节表格展开 —— 这符合"每个计划独立产出可工作软件"的纪律。

---

## 8. 风险与残余不确定性

| # | 风险 | 影响 | 缓解 |
| --- | --- | --- | --- |
| R1 | 内核在拆完 `state` 后仍然很大（预测会变小，**未实测**） | 插件二进制大、编译慢 | Phase 1 的重算步骤给出真实数字；若过大，把"内核"进一步按 feature 切分 |
| R2 | `ara2-bridge` 是 0.3.0 的第三方 crate，API 可能变动 | 升级成本 | 把 ARA 适配集中在 `src/ara/`，不让它渗透到映射层 |
| R3 | REAPER 在插件被拒绝时的卸载时序会导致崩溃（探针实测过 `0xc0000005`） | 加载失败路径不安全 | Phase 2 必须处理 `initialize` 失败路径，不能只做成功路径 |
| R4 | U4 的缓存过期窗口可能存在 | 静默复用旧音频 | §6.3 实验；退路是按内容指纹 |
| R5 | 状态通道的实例发现/心跳在异常退出时会留垃圾条目 | 本体连到死实例 | 心跳超时 + 连接失败即清理 |

---

## 9. 不承诺

1. `vslib` 的分发（闭源、仅 Windows、文件 IO 型）。
2. FL Studio（宿主不支持 ARA）。
3. Linux（`vslib` 仅 Windows；ARA 宿主生态也不在 Linux）。
4. macOS 纳入 v1 与否仍待定（ARA 宿主存在，但 `vslib` 缺失且打包链路不同）。
5. 插件内嵌修音 UI（v1 不做，修音仍在本体）。

---

## 10. 与既有文档的关系

| 文档 | 关系 |
| --- | --- |
| [产品设计](2026-10-04-ara-bridge-design.md) | 上位。本文不推翻它的任何决定；§4.6 是对它 §5.4 的收敛 |
| [内核边界](2026-10-04-ara-kernel-boundary.md) | 本文 §4.2 修正了它的一处判断：**先拆 `state` 再算闭包**，否则闭包被 `AppState` 污染 |
| [探针计划](../../plans/2026-10-04-ara-bridge-probe.md) | 探针已完结（Task 1–3 全部有结论）；本文接手它的四个未决项 |
| [EXECUTION-LEDGER](../../../probe/ara/EXECUTION-LEDGER.md) | 历史 Ruling 的唯一记录处；本文产生的新 Ruling 追加在那里 |
