# 本地补丁说明（HiFiShifter）

这份源码是 `ara2-bridge-plugin` 0.3.0 的**逐字拷贝**，外加一处**把模型图的边找回
委托层**的补丁，以及 renderer 分配/控制器身份通知补丁。

## 补了什么

三个 trait 方法各加了宿主本来就有、却**没有往委托层传**的父子边：

| 方法 | 新增参数 | 为什么 |
| --- | --- | --- |
| `AudioModifications::create_audio_modification` | `source: &Self::AudioSource` | `Take → Clip.source_path` 的映射输入 |
| `AudioModifications::clone_audio_modification` | `audio_source: &Self::AudioSource` | 克隆出的 take 也要知道自己挂在哪个源上 |
| `PlaybackRegions::create_playback_region` | `modification`、`sequence` | clip ← take 与 clip ← 轨道，两条边都是建时间线的必需输入 |

为了让这些参数可命名，`AudioModifications` / `PlaybackRegions` 各自多声明了一份关联类型，
并在 `PluginModel` 上用等式约束钉回"正主" trait 的同名类型（实现方仍只需给一套类型）。

`runtime.rs` 里三处派发点改成把边一并传下去（那些值本来就在运行时的节点上）。

## renderer 区域分配通知（Phase 3a）

`ExtensionBinding::new_with_assignment_observer` 在 ARA 2 模型线程 add/remove playback
region 后提供 `(role, sorted_model_ref_keys)`。观察器在所有内部锁释放后运行，允许
重入只读查询；原 `new` 不安装观察器，行为保持。接口拒绝 ARA 1，旧 constructor
仍保留原 ARA 1 行为。新增 `new_with_renderer_observer` 同时提供显式 playback region
与 editor sequence keys；editor.rs 在 add/remove sequence 后解除锁并通知。产品模型
维护真实 sequence -> region 成员表，editor 与显式分配展开后去重；序列更新必须迁移成员。

观察器不是音频线程接口；不能在 process 中复制、锁分配表或读宿主 PCM。
身份来源为 `CreateContext::realtime_key()`（runtime 内 model-ref 地址），不是 RawHandle。
集成回归在产品 crate `tests/renderer_assignments.rs`，经真实扩展 FFI 检查两个 renderer
隔离、remove 以及 controller/companion 两种释放顺序。

## controller 身份通知

`PluginBuilder::controller_identity(Fn(usize))` 是可选模型线程通知。controller.rs 分配
成功后、把实例交给宿主前通知真实 `documentControllerRef`（不透明键）。原 builder 不
安装通知时行为不变。产品工厂将键登记到各自 DocumentSession；VST3 bind 的参数
**也是 controllerRef，绝不是 ARADocumentControllerInstance**。不可解引用。

产品 entry 用原 companion 的 audited C++ shim + CompanionProcessorBinding，但自持
上下文以获知真实 controller 身份；不改 registry 源码或 SDK。控制器侧 session 独立
持 ExtensionControllerLease，实际 ModelHandle::destroy_document / Drop 撤销许可；
entry 的 owning 引用保留 companion storage 至最终 release，支持两个销毁顺序。

## 为什么必须补

上游的委托层**丢掉了整个模型图**：它内部持有 `source` / `modification` / `sequence`
三条父子边（`runtime.rs` 的节点上），但 `PluginModel` 的 trait 签名里一个都没有。
于是实现方拿到的是一堆**没有连线的对象** —— 能计数（探针干的就这个），
不能建时间线。

## 影响面

- 纯增量：不改任何既有行为，只多传参数。
- 上游修好后，删掉三个签名上的新参数与运行时派发点的新传参、
  删掉 `PluginModel` 上的等式约束、删掉本目录与 `backend/Cargo.toml` 的对应
  `[patch.crates-io]` 行即可。

## 标准宿主播放请求租约

新增PlaybackRequestHandle和PluginBuilder::host_playback：从原HostClients已验证的可选
PlaybackAccess交付可存储租约，HostClients销毁后撤销，原模型线程之外拒绝调用。
HiFiShifter只在WebMessageReceived主线程执行Start/Stop/SetPosition，不在actor/process调用，
也不保留未经验证的ARA host裸指针。缺可选接口时GUI禁用播放控制。
