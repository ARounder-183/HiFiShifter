# 本地补丁说明（HiFiShifter）

这份源码是 `ara2-bridge-core` 0.3.0 的**逐字拷贝**，外加 `src/properties/model.rs` 里
一处**纯增量**的补丁。

## 补了什么

`PlaybackRegionProperties` 的四个读访问器 + `name()`：

- `start_in_modification_time()`
- `duration_in_modification_time()`
- `start_in_playback_time()`
- `duration_in_playback_time()`
- `name()`

## 为什么必须补

上游 0.3.0 的这个类型**只有一个公开读访问器**（`transformation_flags()`）。
而那四个数是 `ARA → TimelineState` 映射的**全部输入**（placement 与拉伸都靠它们）。
高层 `PluginModel` 的 `PlaybackRegions::create_playback_region` 只交出这个不透明结构 ——
也就是说，不补就**拿不到**，不是"拿得别扭"。

## 影响面

- 只加读访问器，不改任何既有行为；`[patch.crates-io]` 之后其余 ara2-bridge crate
  的行为与本机未打补丁时一致。
- 上游若补了同名 getter，删掉 `src/properties/model.rs` 里那段标注为"本地补丁"的代码、
  删掉 `backend/Cargo.toml` 的 `[patch.crates-io]` 段、删掉本目录即可。

## 为什么不改成"自己写一层 ARA 文档控制器"

那等于重写上游的整个回调委托层（`generated_callbacks` 那些表），规模远大于五行 getter。
只有当上游在**别的**地方也不可用时，才值得走那条路。
