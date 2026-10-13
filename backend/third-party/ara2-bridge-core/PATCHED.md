# 本地补丁说明（HiFiShifter）

这份源码是 `ara2-bridge-core` 0.3.0 的拷贝 + 下表所列补丁。

## 补丁清单（升级 0.4 时逐项核对；"行为"= 不只是新增代码）

| # | 位置 | 增量/行为 | 内容 | 上游 0.4 状态 |
|---|---|---|---|---|
| 1 | `src/properties/model.rs` | 增量 | `PlaybackRegionProperties` 补 `start/duration_in_{modification,playback}_time()` 与 `name()` 读访问器（上游 0.3.0 只有 `transformation_flags()`） | 待查 |
| 2 | FFI ASCII 校验（`src/properties/model.rs` 的 persistent-ID 输入路径） | **行为** | 有界非空有效 UTF-8 直通（REAPER 对中文素材路径传 UTF-8 对象 ID，严格 ASCII 拒绝会让 REAPER 空指针崩溃）——见下方「REAPER中文宿主对象ID兼容」 | 待查 |
| 3 | `src/properties/mod.rs` | **删除** | 删掉被 #2 孤立的上游辅助函数 `copy_required_id`（宿主 ID 的严格 ASCII 复制器，调用点已全部换成 `copy_host_object_id`）。保留它只会让整个 workspace 每次编译都报一条 `dead_code` 警告 | 待查 |
| 4 | `validate_playback`（`src/properties/model.rs`） | **行为** | 未知 `transformationFlags` 位不再在**解析层**被拒。此前它让插件侧"降级接受并留痕"的两段日志**永远不可达**：解析层已先把整条 region 判成 null，宿主设任何未来位就丢整条片段。位仍原样保留在结构里，由插件侧比对后降级 | 新增 |
| 5 | ARA2 `regionSequenceRef` 为 null 的解析（`src/properties/model.rs`） | **行为** | 不再 `Err`（实测宿主会给 null），改记 `None` 并放行；归属由 trait 参数（补丁 1）与 `mapping.rs` 的顺序归属兜住 | 新增 |
| 6 | `PlaybackRegionProperties::color()` | 增量 | 补读访问器。`color` 早已被正确解析，但此前没有任何 getter ⇒ 宿主给的颜色解析后即丢弃（`MusicalContextProperties` / `RegionSequenceProperties` 都有，唯独它没有） | 新增 |

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

## RawHandle 身份访问器（2026-10-13）

## 补了什么

- `RawHandle::identity()`

## 为什么必须补

上游 0.3.0 的 `RawHandle` 字段全部私有、且**没有任何访问器**。而 ARA 2 的部分存档
（`storeObjectsToArchive` / `restoreObjectsFromArchive`）交给插件的筛选对象里，
对象身份正是这个类型（`StoreFilter::audio_modifications()`）。

插件侧的状态按 `persistentID` 索引、句柄按 `RawHandle` 索引，两边必须能对上 ——
否则"宿主只保存它选中的那几条 modification"这条路径无法实现，插件只能返回
`kARAFalse`，宿主做增量保存时就会丢数据。

`Handle::index()` / `Handle::session()` 已经是公开的，本补丁只是把同一对值从**擦除后**
的类型上取回来，不引入任何新的身份语义。

## 影响面

- 只加一个只读访问器，不改任何既有行为。
- 上游若补了同名访问器，删掉 `src/handles.rs` 里那段标注为"本地补丁"的代码即可。
