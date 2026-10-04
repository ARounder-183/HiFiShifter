# 内核边界：`hifishifter-kernel` 与 app 层怎么切

> 写于 2026-10-04，分支 `codex/ara-plugin`。
> 触发原因：内核抽取在"搬 `audio_engine`"这一步撞墙。这份文档记录**实测到的边界形状**
> 与建议的切法，供继续抽取时照做。冲突时以 [ARA 设计文档](2026-10-04-ara-bridge-design.md) 为准。

## 1. 撞到的问题

最初按"依赖闭包"划界：从 `{mixdown, state}` 出发，得到 **43 个模块 / 约 2.58 MB**。
但把 `AppState.app_handle` 换成事件出口后，编译器给出 68 处错误，其中暴露了真正的阻碍：

**`audio_engine` 会反过来调用 app 层。** 它拿 `AppHandle` 不只是发事件：

```
engine.rs:1594   app.state::<crate::state::AppState>()   // 调度音高分析
engine.rs:1621   app.state::<crate::state::AppState>()
engine.rs:1650   app.state::<crate::state::AppState>()
engine.rs:1704   app.state::<crate::state::AppState>()
```

app 层也拿同一个 handle 做窗口（`get_webview_window` 5 处）、路径（`path`）、state 访问。

结论：**依赖闭包不等于内核边界**。闭包把设备层与 app 回调也圈进来了。

## 2. 内核 worker 与 app 层的三种回调（实测清单）

内核侧代码需要"回头找宿主"的地方只有三类：

| # | 用途 | 现状 | 证据 |
| --- | --- | --- | --- |
| 1 | 向 UI 发进度事件 | `tauri::Emitter::emit` | `state.rs` 14 处、`pitch_clip` 11 处、`recording` 4 处、`pitch_analysis` 3 处、`audio_engine` ~35 处 |
| 2 | 向音频引擎投递命令 | `mpsc::Sender<EngineCommand>` | `pitch_clip.rs:656/891/1014` |
| 3 | 调度音高分析（要读 app 状态） | `app.state::<AppState>()` + `maybe_schedule_pitch_orig` | 上面那 4 处 |

第 1 类已经有出口了：`hifishifter_kernel::events::{EventSink, Events}`（提交 `e180360b`）。
第 2、3 类还没有。

## 3. `audio_engine` 内部也是两类东西

| 文件 | 性质 | 证据 |
| --- | --- | --- |
| `byte_budget_cache.rs`、`util.rs` | **内核**（已迁走，`992e1fa3`） | 零 `crate::`、零 `tauri::` |
| `mix.rs`（58KB） | 内核 | 零 `tauri::`，但 24 处 `crate::`（要 `state`/`render_cache` 等一起过去） |
| `types.rs`、`snapshot.rs`、`engine.rs` | **app / 设备层** | 含 `tauri::` 与 `app.state::<AppState>()`；`engine.rs` 就是 cpal 设备边界 |

设计文档 §5.1 说的"设备边界替换"要替换的正是 `engine.rs`：插件里宿主回调取代 cpal，
`AppState` 根本不存在。所以它**不该**进内核。

## 4. 建议的切法

1. **内核新增宿主回调 trait**（暂名 `HostServices`），把第 2、3 类也变成注入：

   ```rust
   pub trait HostServices: Send + Sync + 'static {
       /// 把一条引擎命令投递给设备层（插件侧可能直接丢弃或转成 ARA renderer 的请求）。
       fn send_engine_command(&self, command: EngineCommand);
       /// 请求为某个根轨调度音高分析（app 侧读 AppState；插件侧读自己的文档状态）。
       fn request_pitch_analysis(&self, root_track_id: &str);
   }
   ```

   与 `Events` 一样做成具体类型 + 固有方法，调用点改写量最小。

2. **`EngineCommand` 一分为二**：数据型变体（`ClipPitchReady` 等）进内核；
   `SetAppHandle` 这类只属于 app 的留在 app 层（或用 `#[cfg]` 挂）。

3. **重算闭包**：把 `engine.rs` / `snapshot.rs` 排除后，再跑
   `.build-tmp/kernel-closure2.ps1` 的同类脚本，得到真正的内核模块集。

4. **然后才是机械搬迁**：搬文件 + app 侧 `pub use` 接回路径。因为搬的是闭包，
   模块内 `crate::X` 在内核里仍指向内核自己的 `X`，**路径零改写**。

## 5. 搬迁进度

| 模块 | 状态 | 提交 |
| --- | --- | --- |
| `fade_curves` | 已迁入 | `a5568ef5` |
| 事件出口（`events`） | 已建立 | `e180360b` |
| `byte_budget_cache`、`util` | 已迁入 | `992e1fa3` |
| `mix`、`metronome`、`io` | 待迁（需先有宿主回调，因为 `mix` 依赖 `state` 等） | — |
| `state`、`mixdown`、`renderer`、`render_cache`、`vocoder`、`pitch*` | 待迁 | — |
| `engine.rs`、`snapshot.rs` | **不迁**（设备层，插件侧由宿主回调取代） | — |

## 6. 每步的验收口径

搬一个模块就要核对：**app 侧单测减少、内核侧增加、两边合计不变**。
当前基准：app `762 passed / 4 failed / 1 ignored`，内核 `10 passed`，合计 `772 / 4 / 1`。
那 4 个失败是既有的 `/tmp` 路径环境性失败，不要修。
