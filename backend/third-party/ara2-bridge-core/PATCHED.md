# 本地补丁说明（HiFiShifter）

这份源码是 `ara2-bridge-core` 0.3.0 的拷贝 + 下表所列补丁。

## 补丁清单（升级 0.4 时逐项核对；"行为"= 不只是新增代码）

| # | 位置 | 增量/行为 | 内容 | 上游 0.4 状态 |
|---|---|---|---|---|
| 1 | `src/properties/model.rs` | 增量 | `PlaybackRegionProperties` 补 `start/duration_in_{modification,playback}_time()` 与 `name()` 读访问器（上游 0.3.0 只有 `transformation_flags()`） | 待查 |
| 2 | FFI ASCII 校验（`src/properties/model.rs` 的 persistent-ID 输入路径） | **行为** | 有界非空有效 UTF-8 直通（REAPER 对中文素材路径传 UTF-8 对象 ID，严格 ASCII 拒绝会让 REAPER 空指针崩溃）——见下方「REAPER中文宿主对象ID兼容」 | 待查 |
| 3 | `src/properties/mod.rs` | **删除** | 删掉被 #2 孤立的上游辅助函数 `copy_required_id`（宿主 ID 的严格 ASCII 复制器，调用点已全部换成 `copy_host_object_id`）。保留它只会让整个 workspace 每次编译都报一条 `dead_code` 警告 | 待查 |

## 1 的细节：`PlaybackRegionProperties` 的四个读访问器 + `name()`

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

## REAPER中文宿主对象ID兼容（2026-10-06）

SDK规定ARAPersistentID为七位ASCII，但REAPER实际对含中文素材路径传入UTF-8音频对象ID。
严格拒绝会使createAudioSource返回null，REAPER随后在源列表流程空指针崩溃。
仅AudioSourceProperties/AudioModificationProperties的FFI输入改用有界、非空、有效UTF-8
复制，原始字节不归一化、不转换、不制造新身份。插件自己生成的ID及其它ASCII校验不变。
定向边界回归见hifishifter-plugin/tests/host_persistent_ids.rs；宿主复现记录在probe/ara。

## 原方案取舍（续）

那等于重写上游的整个回调委托层（`generated_callbacks` 那些表），规模远大于五行 getter。
只有当上游在**别的**地方也不可用时，才值得走那条路。
