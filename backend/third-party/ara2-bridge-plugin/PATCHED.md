# 本地补丁说明（HiFiShifter）

这份源码是 `ara2-bridge-plugin` 0.3.0 的**逐字拷贝**，外加一处**把模型图的边找回
委托层**的补丁。

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
