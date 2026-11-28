# 线程与守卫模型（总纲）

三形态（独立 App / DAW 内插件 / App 音频引擎）共用 `hifishifter-kernel`，但
"什么代码跑什么线程、panic 与锁的纪律是什么"此前散在 20+ 处行内注释里。本页是
总纲：新入口/新线程先对照这里，再写代码。

## 1. 三条铁律

1. **panic 绝不越过 FFI / Win32 边界。** Rust 1.71+ 起 panic 越过
   `extern "system"` = abort：
   - 插件侧 = DAW 进程崩溃；App 侧 = 整个应用退出。
   - 插件每个 `unsafe extern "system"` 入口必须自带守卫
     （`lib.rs::guard_ffi` / `view.rs::boundary` / `vst3.rs` 音频路径的
     `catch_unwind`），把 panic 拦成"该入口的诚实失败值"并记日志。
   - App 的同步 Tauri 命令在 WebView IPC 线程内联执行，panic 会穿过 Win32
     消息循环 —— 每条同步命令必须包 `commands/common.rs::guard_command`
     （async 命令由 tokio 任务边界结构性遏制，不需要）。
   - **CI 门禁**：`scripts/checks/check-ffi-guards.py`（vst3.yml 第一步）强制以上两条，
     新的无守卫入口直接红。
2. **锁毒化必须恢复而不是 panic。** 生产代码禁止 `.lock().unwrap()` —— 用
   `MutexExt::lock_or_recover()`（毒化后取回内部数据继续）或
   `unwrap_or_else(|e| e.into_inner())`。`scripts/checks/check-ffi-guards.py` 的基线门禁
   禁止新增（`scripts/checks/baseline-lock-unwrap.txt`，现存 0 处）。门禁会剥掉
   `#[cfg(test)] mod` 的函数体，测试里的 `.lock().unwrap()` 不计入。
3. **错误语义不靠子串匹配传播。** 禁止新增 `error.contains("...")` 式的分类
   （基线 `scripts/checks/baseline-error-contains.txt`，现存 5 处：`editor::errors`
   的 `classify_error` / `error_code_for` 与 `editor::gui` 客户端各 2 处、
   `render::extension` 的绑定失败判定 1 处）。协议语义用结构化错误码
   （`hifishifter-host-link` 的 `Response.code`；notebook 家族的
   `notebook_*` 稳定码），字符串只做展示。

## 2. 线程清单与各线程的允许/禁止

### 插件形态（跑在 DAW 进程里）

| 线程 | 谁在跑 | 允许 | 禁止 | 强制点 |
|---|---|---|---|---|
| DAW 音频回调线程 | `vst3.rs::audio_process` | 计算、读原子快照 | 锁等待、文件 IO、日志、堆分配（分配由测试断言兜底） | `catch_unwind` → 静音输出；`test_allocator` 进程级分配断言；快照原子读区 + 非实时回收 |
| DAW 主/UI 线程 | 其余全部 vtbl 入口（工厂/组件/控制器/连接点） | 拿锁（必须 `lock_or_recover`）、初始化、IPC | panic 越界 | `guard_ffi` 包裹全部入口 |
| 插件编辑器窗口线程 | `editor/webview.rs`（WndProc + 20ms 泵） | COM/WebView2 调用、命令排队 | 阻塞等待 worker | `window_proc` 自带 `catch_unwind`；UI 回调只排队（`UiRequest`） |
| 后台渲染/分析线程 | kernel 渲染器、音高分析 | 一切 | 无界并发（渲染有 budget） | 完成门重试预算（`HOST_REPLY_RETRY_BUDGET`）、`log_limiter` 限流日志 |

### 独立 App 形态

| 线程 | 谁在跑 | 允许 | 禁止 | 强制点 |
|---|---|---|---|---|
| WebView IPC 线程 | `#[tauri::command]` 同步命令 | 短平快的读操作 + guard 包裹 | panic 越界、长阻塞 | `guard_command` 包裹（`Value`/`TimelineStatePayload`/`Result<_,String>` 三类全覆盖；typed getter 长尾未包，见 §4） |
| tokio 阻塞池 | async 命令的 `spawn_blocking` | 一切 | —— | tokio 任务边界（panic 变 JoinError） |
| 音频引擎线程 | `audio_engine/engine.rs`（cpal 流） | 命令队列消费、meter 发射 | 锁等待、分配之外的阻塞 | 命令通道 + `drain_commands_without_output` 优雅降级 |
| 后台渲染调度线程 | `commands/playback.rs` 的去抖调度器 | 文件 IO、全量渲染 | 并发多发（单飞 + 合流） | 原子请求时间戳 + 去抖窗口 |

### 跨进程

- **配置文件**：App 与插件是两个进程、共用一份 `app_config.json`。
  读-改-写必须走 `config.rs::update_config`（进程内 `CONFIG_WRITE` +
  `app_config.json.lock` 跨进程文件锁，`File::lock`）。
- **IPC**：`hifishifter-host-link` 的 **workspace 文件交接**（前身是命名管道；
  为什么改成文件见该 crate `lib.rs` 的文档）；错误带结构化 `code`。

## 3. 新增入口的 checklist

- [ ] 是 `extern "system"` 入口？→ 包 `guard_ffi`（选对失败值：QI → `K_NO_INTERFACE`，
      refcount → 计数、视图 → null……）。
- [ ] 是 App 同步命令？→ 包 `guard_command`（返回类型有三类现成 fallback；
      新返回类型要先定义它的失败形态）。
- [ ] 拿锁？→ `lock_or_recover`，不得 `.unwrap()`。
- [ ] 在音频回调里？→ 先读 `audio_abi` 与快照协议，禁清单见 §2 表格。
- [ ] 新增错误分类？→ 结构化 code，不是子串。

## 4. 已知边界（诚实清单）

- App 侧 25 个返回 typed payload 的同步 getter（`get_ui_settings`、
  `get_gpu_devices` 等纯读命令）尚未包 panic 守卫 —— 它们不做 IO/编辑，
  panic 面小；包它们需要先给各 payload 类型定义失败形态（Default/错误构造）。
  新命令**必须**从三类已覆盖的返回类型里选。
- 插件 `vst3.rs` 的音频路径（`component_set_state` / `component_get_state` /
  `audio_process`）有自己的专项 `catch_unwind`（音频语义的静音降级），
  不走通用 `guard_ffi`。

## 5. Windows VST3 native loader 锁契约

Windows 插件 bundle 里的 HiFiShifter.vst3 是一个 C++ shim，它按绝对路径加载
HiFiShifterEngine.dll。shim 的 gate 是**唯一**的生命周期锁，保护
engine / factory / initialize / engine_exit / load_failed。

- load_engine_locked() 要求调用者已经持有 gate，自己**绝不再取锁**。
- GetPluginFactory() / InitDll() / ExitDll() 在入口各取一次 Gate。
- GetPluginFactory() 在 Gate 内调用 engine 的工厂函数；InitDll() 在 Gate 内调用
  engine 的初始化函数，保证 ExitDll() 不会在“读取函数指针”和“调用函数指针”之间
  清空/卸载 engine。
- ExitDll() 在 Gate 内清空状态，FreeLibrary() 放到 Gate 外执行（避免在持锁时
  运行 engine 的 DllMain/静态析构）。
- 任何新增的导出入口都必须遵守同一契约；不要为了“保护一个指针”在已经持锁的调用链
  上再取一次 SRWLOCK——它不是递归锁。

回归守卫：scripts/vst3/windows/load_probe.cpp + verify.ps1，由
.github/workflows/vst3.yml 的 Windows VST3 load gate 执行。vst3_exports.rs 只覆盖
Rust engine DLL，不能替代这个黑盒门禁。

## 6. 几何与时钟：两条不被授权门控的读通道

本节补在 2026-10-27 的编辑完整性改造之后，回答两个**曾经被门住**的通道。

### 6.1 几何只有一个投影面

- 编辑器看到的片段位置 / 长度来自 **REAPER 清单**（`D_POSITION` / `D_LENGTH`）；
  渲染输入读的是 `document.timeline`（ARA 映射产物），而后者只在 `remap_and_log`
  收口时重建。
- **移动 item 不保证产生收口**。于是两份几何会分叉：GUI 动了、渲染没动、声音也没动。
  修复不是"再同步一次"，而是让渲染输入在读之前**接受同一份清单投影**
  （`DocumentSession::project_host_item_geometry_locked`）。
- 推论：任何新增的"渲染前几何来源"都必须**接到这条投影上**，不得另起一份读法。
  ARA 侧仍然由 `remap_and_log` 供值（那一侧本来就在收口时整体重算）。

### 6.2 时钟不受几何授权门控

- `host_query_stamp()` 回答的是"这份清单 / 几何此刻能不能查"（需要 document、分配、
  未关闭）。它**不是**时钟的授权：时钟只需要"宿主在哪儿、在不在播"。
- 旧实现把两者绑在一起，于是没有 document / 没有分配的实例一次也不发布宿主位置，
  150 ms 后 `host_is_fresh()` 为假，光标退回音频块读数 —— 而编辑器实例不处理音频，
  那两个量恒为 `(0, false)`。用户看到的就是"跳到位之后冻住一会儿"。
- 现在：先无条件采一次并发布，再用 stamp（可选）决定要不要继续做清单那一段。
- 播放位在 `read()` 里取**并集**（任一来源说在播就是真），位置仍取最新鲜的来源。
  "该走的时候不走"是看得见的卡顿，"多走几毫秒"下一拍就被采样纠回来 —— 两者代价不对称。

### 6.3 纯几何变化不能让渲染当场重排

- 清单刷新可能跑在**会话 actor 自己**的调用栈上（`ensure_loaded` → 清单刷新），
  而 `ExtensionOwner::prepare()` 要投递回同一个 actor。在那种栈上直接调
  `invalidate_render_input()` 就是"投递并等自己"——实测会让整份测试套件挂死。
- 只有"ARA 表达不到的渲染事实"（方向 / 循环源 / 声道模式）才走
  `invalidate_render_input()`；纯位置 / 长度变化只推进 `render_epoch`
  （`note_render_geometry_changed`），由 20 ms 定时器驱动的下一次 prepare 接手。


## 7. 传输权威：谁给的读数、什么时候算数

本节补在 2026-10-27 的第二轮传输改造之后。`6.2 只解决了"时钟不被门住"，
这一节解决"时钟**有没有写者**，以及写进来的东西算不算数"。

### 7.1 一个读数必须能说出自己是谁给的

`TransportClock::read_view()` 返回 `TransportView { position_sec, playing, provenance, age_ms, seek_pending }`，
`provenance` 只有四档：

| 档 | 谁写的 | 位置可信吗 |
|---|---|---|
| `HostSample` | 宿主主线程采样（`GetPlayPositionEx` / `GetCursorPositionEx`） | 是 |
| `RealtimeBlock` | 某个实例的 **realtime** 处理块 | 是（一块之内） |
| `SeekIntent` | 我们自己请求、宿主还没确认的一次定位 | 是"意图"，不是观测 |
| `Stale` | 以上都没有：上一次观测的残值 | **不是** |

三条纪律：

1. **prefetch 块只能投"播不播"，不能投"在哪儿"。** 它的 `kPlaying` 与 realtime 块同源，
   而它的 `project_time_samples` 是**渲染进度**（领先播放头 1~1.5 s）。
2. **播放位取并集**（host / block / prefetch / seek 任一为真即真），位置取最新鲜的来源。
   理由与 `6.2 相同：代价不对称。
3. **残值不得改光标。** `Stale` 的语义是"上一次观测"，不是"现在在哪"；
   前端与 `payload.playhead_sec` 都据此放行/拒写。

### 7.2 定位有一个"未落地窗口"

ARA 明说播放请求 "typically scheduled and executed with some delay"
（`ARAInterface.h:2157-2161`），而 `on_ara_model` 最多等 250 ms。于是"命令回执"
**不等于**"宿主落地"。把这两件事混为一谈，就是"先跳过去、再被拽回来"。

`TransportClock::request_seek(target)` 在**投递之前**记下意图：

- 窗口内 `read_view()` 一律返回 `target`（`provenance = SeekIntent`）；
- 宿主报回的位置落进 `±0.25 s` ⇒ 窗口关闭，读数回到 `HostSample`；
- 超过 `1 s` ⇒ 窗口自己关闭，`seek_unsettled` 计数 +1，并留一行 WARN。

推论：**前端不得再有"我压住采样多久"的判断**。它没有观测宿主的能力，
任何这样的判断都只能靠猜。

### 7.3 推送优先，轮询兜底

- 视图定时器（20 ms）每拍读一次原子时钟，**变了就发**一条 `transport` 视图事件
  （最多 50 Hz，另有 200 ms 心跳）。这条通道没有往返时延可丢。
- 轮询退化为看门狗（插件形态恒 400 ms）。33 ms 档只保留给独立 App —— 那里没有推送。
- 两条通道共用**同一个** `applyTransportSample`，否则"同一个宿主状态画出两条光标"。

### 7.4 宿主节拍的自愈

`plugin_register("timer")` 失败时只剩一行 WARN，而"节拍没挂上"的后果是
**每次宿主调用白等 250 ms 再熔断**（`host::marshal`）。现在视图定时器每 5 秒
问一次 `NativeHost::ensure_host_timer()`：没挂上就再 `acquire` 一次（幂等、
带退避），并且 `acquire` 只在自己真的加了一个引用时才让调用方在析构里减计数
—— 否则"宿主拒绝"会被记成"挂上了"，节拍照样没挂却再也不会被摘。

### 7.5 喂入流的工程锚点：一个读数什么时候算数

捕获形态下有两个域，它们不许互相赋值：

* **喂入域**（磁带位置）= 这条喂入流喂进来多少帧，永远从 0 起；
* **工程域** = 这一帧在工程里落在哪一秒。

把前者翻译成后者需要**一个数**：喂入域第 0 帧在工程第几帧。第六轮之前这个数藏在
映射公式里（隐含假设 = 片段起点），于是"它到底是什么"没有人能审阅、断言，也没有任何
一行日志说出来。`FeedAnchor` 把它提成一个有来源、有日志、有测试的值。

| 宿主在做什么 | 首块 `projectTimeSamples` 是什么 | 锚点 |
|---|---|---|
| 播放预览（VEGAS 的预卷：`kPlaying=0`、位置钉住、内容在流动） | **播放头** —— 它从当前光标开始喂，片段起点只是这段内容在工程里的落点 | 宿主读数 |
| Render As | **上一次的播放头** —— 既不是片段起点，也不是渲染范围起点 | 片段起点 |
| 从片段起点起播 | 两者相同 | 两者同值（逐帧相同） |

判据只有一个可观测的问题：**它落在片段的工程窗口里吗**。落在里面 ⇒ 它是个合法的
播放头；落在外面 ⇒ 它显然不是"这一段内容现在的位置"。

现场读数（VEGAS，片段在 15.000 s，播放头 33.625 s，预卷 9 040 帧）：

```text
（第六轮之前：读的是片段起点）
[capture] feed begin kind=pinned raw=1614000 tape=23488 frames=960 project=743488 mapping=clip anchor=clip first_raw=1614000
（第六轮：读的是播放头）
[capture] feed begin kind=pinned raw=1614000 tape=23488 frames=960 project=1637488 mapping=clip anchor=playhead first_raw=1614000
```

那 743 488（15.489 s）对应的工程位置，而宿主正在播 33.625 s —— 差正好是
`playhead − clip_start = 894 000` 帧，与 `[click] … jump=894000` 逐帧吻合。
听感就是"起播 1 s 内放的是片段起点的处理结果"，直到 `kPlaying` 升起、喂入流切回
传输域才恢复。

`HFS_CAPTURE_ANCHOR=auto|playhead|clip` 让现场一键对照三种答案；`feed begin` 的
`anchor=` 是"这一块用了哪条证据"的读数。

**这一节和 §9 是同一条纪律的两面**：位置和锁一样，都只能有一个权威，而且那个权威
必须能被单独指认。

## 8. 让路：目标、保真度与焦点租借

### 8.1 三级目标，每一级都有证据

| 级 | 什么时候 | 依据 |
|---|---|---|
| `Focus` | 宿主线程的焦点窗口**不在**我们子树里 | "用户此刻按键会去哪" |
| `PreviousFocus` | 焦点在我们子树里，且那个窗口仍属于宿主树 | `WM_KILLFOCUS` 的 wParam：插件抢走焦点之前用户正在用的窗口 |
| `RootOwner` | 上面都不成立 | owner 链顶端（旧行为） |

`Parent` 这一级已被删除：编辑器是 `WS_CHILD`，父窗口与它同属一个顶层窗口，
于是那条分支**恒成立**，所有按键都被投给宿主为装插件视图而建的容器窗口 ——
一个没有宿主键位表的窗口。

### 8.2 两种保真度

| `Fidelity` | 通道 | 忠实转交修饰键？ | 动焦点？ |
|---|---|---|---|
| `Synthetic` | `PostMessage` | **否**（不更新线程键状态） | 否 |
| `Injected` | `SendInput` + 焦点租借 | 是 | 是（RAII 归还） |

规则：**先用便宜的**。合成投出去、观察窗到期、宿主毫无反应 ⇒ 按码记账
（`ESCALATION_TTL = 120 s`），下一次这个键直接走注入。带修饰的组合键**只有**注入
一条路（合成会送出一个错键）。

### 8.3 焦点租借必须是 RAII

`FocusLoan` 用 `AttachThreadInput` + `SetFocus` 把焦点临时交给目标窗口，
`Drop` 里无条件还回去。失败路径（`SendInput` 返回的条数不符）也必须经过 `Drop`
—— 否则插件会"忽然不理人"，而界面上看不出任何异常。

## 9. 宿主线程不取 GUI 锁

本节补在 2026-11-26 的第五轮改造之后。它把"插件为什么会在宿主里失去响应"
从一条现场猜测变成一条**可执行的不变量**。

### 9.1 不变量

> **宿主 UI 线程不得取任何可能被会话 actor 或工作线程长期持有的锁。**

"宿主 UI 线程"有四个入口，缺一不可：

| 入口 | 位置 | 谁在跑 |
|---|---|---|
| WebView 消息回调 | `editor/gui/client.rs::on_message` | WebView2 / WebKit 的消息泵线程 = 宿主 UI 线程 |
| 视图节拍 | `editor/gui/client.rs::tick`（20 ms） | `plugin_register("timer")` / GTK 定时器 / CLAP timer |
| ARA 模型回调 | `ara/model.rs`（`beginEditing` / `endEditing` / `notifyModelUpdates`） | 宿主 model 线程（与 UI 线程同一泵） |
| 显式宿主调用 | `host::HostCallGuard` | 调用方线程 |

四条都由 `host::thread_mark::RoleGuard` 打标，于是"我在不在宿主线程上"
在运行期可判定，而不是靠读代码推断。

### 9.2 为什么这把锁特别危险

宿主 UI 线程一旦停住，它就不再泵消息；而 WebView2 的 `PostWebMessageAsJson`
是**同步**调用，页面侧的投递推进依赖同一条线程泵消息。于是：

```text
actor:      持 timeline 写锁 → ... → post_message（等宿主泵）
host thread: 在 on_message 里等 timeline 写锁
```

两边互等。现场的读数是"进程级静默"：最后一行之后连每 60 s 一次的
`[gui] heap` 都没有了，而 `notifyModelUpdates` 在挂起前 356 次全部正常到达
—— 宿主的泵没停，是我们把它占住了。

### 9.3 三条落地规则

1. **读快照，不读活对象。** GUI 侧需要的时间线事实由 `EditorSession::ui_clips()`
   提供：`id / track / start / length / active take`。它在每次发布时重建，读方只克隆
   一个 `Arc`，写方持有那把锁的时间也只有一次 `Arc` 交换 —— 两个方向都不跨
   `post_message`。`close_track_gaps` 与 `rename_clip_take` 走这条路。
2. **算法只留一份。** `close_track_gaps_moves_from(&[(id, start, length)], from)` 是
   `TimelineState::close_track_gaps_moves` 的实现体。两个调用点（actor 与 GUI）用同一份，
   于是"两条路算出不同的位移"不可能发生。
3. **出站先分类再投递。** 回执永不丢、状态只留最新、其余按 256 KB/拍预算。
   一条 1.9 MB 的消息会把宿主线程按住一次，而 100 条"去重取一次"的提示会让前端
   取 100 次 —— 两者都是同一条链子。

### 9.4 强制点

| 机制 | 位置 | 管什么 |
|---|---|---|
| 角色标记 | `host::thread_mark::RoleGuard` | 让"本线程是宿主线程"可判定 |
| 长锁闸 | `host::thread_mark::host_thread_may_not_lock` | `HFS_HOST_LOCK=probe`（默认，只记读数）/ `refuse`（拒绝）/ `allow`（静默放行） |
| 静态门禁 | `scripts/checks/check-host-thread-locks.py` | `on_message` 里出现任何互斥量、ARA 回调里出现 `self.timeline.lock` 即失败；正反两向自检 |
| 停滞看门狗 | `host::stall_watch` | 采样型：一次**永不返回**的宿主调用也会被点名（`Drop` 型守卫看不见它）；点名之后立刻落盘 |
| 重入探针 | `plugin_core::sync::lock_tracked` | 调试构建下同一线程二次取同一把锁**当场点名**（报两处 file:line）；`ui_tracks` 的所有访问都走它 |
| 心跳预算 | `HFS_NOTIFY_BUDGET_MS` | `notifyModelUpdates` 单次调用的时间上界 |

### 9.5 推论：不要用 Drop 型守卫证明"没有挂起"

`HostCallGuard` / `AraCallbackGuard` 都在 `Drop` 里量耗时。一次永不返回的调用
永远不 `Drop` —— 于是"日志里一条告警都没有"不是"没问题"，是"看不见"。
判据只能是**另一条线程**定期读表（`stall_watch`），以及"所有线程的日志一起停"
这个形状本身。

### 9.6 出站桥为什么**没有**发送线程

第五轮方案里写过"独立发送线程 + 合并 + 预算"。落地时只做了后两件，这是有意的：

| 事实 | 后果 |
|---|---|
| `EditorPort` 是 `Rc<dyn EditorPort>`（`editor/gui/port.rs` 明说三个平台的视图句柄**只属于宿主 UI 线程**） | 给它加 `Send` 只能逼 macOS / Linux 的实现去藏 `Rc`，换不来任何保证 |
| WKWebView 与 WebKitGTK 都要求投递发生在主线程 | "在另一条线程上 `post_message`"在这两个平台上不是优化，是崩溃 |

于是这一层解决的是**量**，不是线程：同一批字节不再重复发、状态消息只留最新、
每拍有 256 KB 硬预算、回执永不丢。宿主线程每次 `post_message` 的阻塞时间因此有了
上界 —— 而"宿主线程被 1 MB/s 按住"这个现象正是方案要消除的那一条。

**所以方案 §8 里那条 `HFS_GUI_SENDER=inline` 回滚开关没有发布**：没有线程可关。
需要退回旧行为时，把 `coalesce_events` 的预算调成"不限制"即可（那是一行常数，
不是环境变量，因为它不是运行期策略）。

同理，`[gui] outbound big message` 只补了 `kind=` 与 `view=`，没有 `part=i/n`：
本端口只能投递**整条 JSON 字符串**，切片会让页面收到半个对象。要减小单条消息，
唯一的办法是让上游少发 / 发小（本轮的做法），而不是在投递层切开。

### 9.7 推论：会取锁的函数不许被持锁的作用域调用

第五轮把"宿主线程不取锁"立成了不变量，但第六轮又栽在同一类形状上，只是换了一条
线程：`present_host_inventory` 从 `ui_tracks.lock_or_recover()` 那一行起整段持有
guard，并在函数末尾调**待定导入对账**；对账为了回答"这条 item 在不在清单里"**又取了
一次同一把表**。`std::sync::Mutex` 不可重入 ⇒ 同一线程自锁 ⇒ 导入成功之后宿主整机
未响应（Windows 记 `AppHangB1`），状态栏那条 `plugin_get_apply_state` 只是宿主停止
泵消息 **30 秒之后**才浮现的读数。

这一条不是"小心一点"能防的：自锁的形状与"什么都没发生"完全同形（进程安静地冻住、
日志里一个字都没有）。所以三层都要有：

| 层 | 做法 | 保证 |
|---|---|---|
| 结构 | 对账器的输入是 `ImportClaims`（**数据**：清单里的 item、region→clip、region→item），它手里没有表 | 结构上不可能再取锁 |
| 防御 | 对账内部每一次取锁都是 `try_lock_or_recover`；拿不到就跳过这一轮并落一行 `[import] reconcile skipped` | 万一又出现嵌套，结果是一行字而不是挂死 |
| 诊断 | `lock_tracked` 在调试构建里让重入点名；静态门禁禁止"对账器取阻塞锁"与"清单函数在 guard 里对账" | 下一次改动不会静默复发 |

对账**跳过**是安全的：它在每一次 ARA 发布、每一次清单刷新时都会再跑一遍，
"这一轮不动"没有任何语义代价 —— 这正是它可以用 `try_lock` 的前提。
`begin_pending_import` 是唯一的例外（阻塞取锁）：它是**写**，丢掉登记就等于丢掉了
这条导入，而账本是叶子锁（持有者只做算术，不取别的锁、不做 I/O、不投递）。

## 10. 三条"看起来成功了，实际只做了一半"的不变量

第七轮修的三件事都不在并发原语上，而在**一个动作的完整性**上。它们放这一份文档里，
是因为失效时的形状与锁问题完全同形：进程安静、日志里一个字都没有、用户只看到"没反应"。

### 10.1 一轮渲染没有内容 ⇒ **什么都不发布**

`RenderOutcome::{Snapshots, Nothing}`：

| 输入 | 结论 | 为什么 |
|---|---|---|
| `regions` 为空 | `Nothing` | 这一轮没有活，不是失败 |
| `timeline.clips` 为空 | `Nothing` | 宿主正处在一次空模型的瞬间（拖动 clip 时每几百毫秒一次） |
| 还剩 clip，但某条源没发布 | `Err(host PCM unavailable: <id>)` | 可重试的暂时失败，且措辞是短句 |

**发布一份空快照是一个真实的动作**：它替换上一份好快照，`is_ready()` 变真，
`copy_block_at` 逐样本返回 0 —— 整条轨静音，直到下一次真正的分配。所以 `Nothing`
的两条发布路径（ARA 的准备作业、自动应用）都**不发布**；自动应用也**不写 `applied`**，
下一轮 150 ms 后还会再来一次。连续为空 10 s（40 轮 @ 4 Hz）才说一次，且只说一次。

### 10.2 状态栏红字只有一个写入口

`render_error` 的读者是 4 Hz 的 `plugin_get_apply_state`，而那个轮询失败才写日志。
三处直接赋值 ⇒ 红字在屏幕上存在 250 ms ~ 1 s、用户看不完、日志里查不到、下一轮被清空。

`EditorSession::note_render_error` 是**唯一**写入口，它同时做四件事：
短句进 `error`（≤ 120 字符）、协议码进 `errorCode`、全量诊断进 `errorDetail`、
按"短句变了"去重写一行 WARN。清空同样收口（`clear_render_error`）。

### 10.3 淡变只有一个权威：插件；宿主是镜子

`D_FADEINLEN_AUTO` / `D_FADEOUTLEN_AUTO` 是**同一个落点**：插件写的、和 REAPER 按
工程级 auto-crossfade 在 `D_POSITION` 落下那一刻写的，落在同一个字段上。把回读无条件
镜像进模型，就是"两个权威同时存在" —— 关掉自动交叉淡化后拖动 clip，包络先闪成
交叉淡化再变回去。

| 规则 | 做法 |
|---|---|
| 写入 | 改几何的补丁**连带**写两侧 auto，且写在 `D_POSITION` 之后（同一次调用里） |
| 回读 | 宿主那一对进 `auto_fade_*_sec_host`（镜像，只做诊断）；模型值来自插件自己的记录 |
| 冷加载 | 本会话还没有记录 ⇒ 宿主值**播种**进模型（工程里已有的交叉淡化不丢） |
| 自愈 | 模型值与宿主值不一致 ⇒ 一行 Debug 读数，下一次写入把两侧一起写回 |
| 手动淡化 | `D_FADEINLEN` / `D_FADEOUTLEN` 仍然镜像（用户在 REAPER 里拖手柄是真意图） |

判据不是我发明的：`D_FADE*LEN_AUTO` 的定义就是"REAPER 为自动交叉淡化预留的长度"，
独立 App 的 `.rpp` 解析器早已按同一条语义把 auto 与 manual 分开。

### 10.4 一次发布必须把它推导不出来的东西一起带走

第八轮的两条现象各自都不是"某个判断写错了"，而是**一个发布动作只做了一半**。

#### 淡变：装配不是淡变的权威

装配的输入（`ItemRow`）只有几何与媒体身份 —— `guid / track_id / name / start_sec /
length_sec / takes / active_take / loop_source / muted / gain`。淡变与吸附偏移**不在
其中**。而装配在每一次清单刷新后重跑（拖动 item 期间约 4.5 次/秒），于是"发布一条
clip"这个动作每一轮都把它们的值写成 serde 默认值 0。

修法有两条，缺一不可：

| 规则 | 做法 | 为什么 |
|---|---|---|
| A | `assembly::inherit_presentation` 把上一轮**已发布**的同 id clip 的呈现字段借过来 | 装配产出的那十一个字段是"没有观测"，不是"观测到 0" |
| B | `FadeProjection` 是**稀疏补丁**：只有宿主几何这一轮说得出来的 clip 在里面 | "没有答案"与"答案是 0"必须在类型上分开 |
| C | `apply_fade_projection` 只改补丁里出现过的 clip | 覆盖不到的保持清单刚写下的宿主真值 |

`ui_fade_projection` 因此不再克隆整份工作区时间线（含全部 clip / take / 参数）只为拷
十二个浮点数 —— 而那份克隆的淡变值在装配重跑之后正是 0。

#### 锚点：一个数的位置，不能回答它是什么

`AnchorPolicy::Auto` 原先只问一件事：宿主首块 `projectTimeSamples` 落在片段的工程
窗口里吗。预览里它是播放头，判据成立；`Render As` 里 VEGAS 把它钉在**上一次的光标**
上，而光标通常正落在片段里 —— 于是判据给出 `project_zero = cursor`，整条渲染平移
`cursor − 渲染起点`，覆盖不到的部分直通宿主干声。

捕获通路能**量**出位置：录制下来的干声就在我们手上，而 Render As 喂进来的还是同一份。

| 规则 | 做法 | 为什么 |
|---|---|---|
| 索引 | `TapeEnvelope`：每 64 帧一格 RMS，`hop < 最小宿主块` | 第七轮把 hop 写成 1024，一块折不出两格，探测一次都不跑 |
| 同代 | `TapeProbe { envelope, samples }` 一次 `Arc` 换指针 | 包络与样本不同代 ⇒ 一个自洽但错误的偏移 |
| 判决 | 格级**提名**，样本级**判决** | 15 格 RMS 的相关系数标准差约 0.26；960 个样本是 0.03 |
| 定稿 | 每条喂入流只赋值一次；静音块不消耗机会 | 中途重锚等于在产物中间切一刀 |

`AnchorSource::Content` 是第三个来源，与 `Playhead` / `Clip` 并列：前两个回答"宿主说的
这个数落在窗口里吗"，它回答"这段内容在素材的哪一处"。

### 10.5 读数必须能回答"这一次到底发生了什么"

两条现象在现场都只能靠用户复述（"闪了一下"、"偏移跟着光标跑"），因为日志里没有
任何一行说出"我们发了什么"。第八轮补的三处读数都遵循同一条纪律：**一行日志必须能
否证一个假设**。

| 读数 | 位置 | 它否证什么 |
|---|---|---|
| `[ara] fade projection: N of M` | `refresh_ui_geometry`（部分覆盖 WARN，全零 DEBUG） | "这一轮到底有没有几何" |
| `[ara] fades: clips=… min_in=…`（High 档，限流） | 同上 | "发出去的那一帧是不是 0" |
| `[capture] anchor probe: … verdict=… offset=… score=… second=…@… hint=… policy=…` | `note_stream_probe` | "为什么这次没按内容判" |

`feed begin` 只**增加**一个字段（`probe=score/second@offset`），既有字段一个不动 ——
现场笔记里已有的判读方法（`project=` / `anchor=`）继续成立。

### 10.6 读数与事件是两个通道，级别不是一个装饰

第九轮的现场日志：30 秒的渲染里有 182 条 `[click] envelope`、30 条 `[click] step`、
110 行 `[capture] clock`、91 行 `[capture] ctx`、8 行 `[capture] pass`，
全部是 WARN。用户的原话是「这次我全靠过滤才看清有用的行」。

它们不是「日志太多」，而是**两个不同的问题被写进了同一个渠道**：

| | 问的问题 | 什么时候该出现 | 归宿 |
|---|---|---|---|
| **读数** | 「这一刻的数值是多少」 | 只要数值变了 | DEBUG（High 档），可以合并 |
| **事件** | 「发生了什么、要不要行动」 | 一次一条 | WARN / ERROR，必须限流但不得合并 |

判据只有一条：**这一行能不能否证一个假设**。能，它是读数，留在 High；它自己就是
「有事发生」的证据，才是事件。

#### 每条读数的级别由它自己的归因能力决定

`[click]` 量的是**输出波形**，它手里没有内容的「标准答案」。一次起伏既可能是音乐换了
口气，也可能是我们换错了源 —— 只看波形永远分不开。唯一分得开的事实是**覆盖**：
`uncovered > 0`（这一块里有帧不是我们写的）与 `edge`（覆盖边落在块内）。
于是 `click::blames_us` 只回答一件事：**这一块里有不属于我们的帧吗**。

| 读数 | 归因 | 级别 | 合并窗口 |
|---|---|---|---|
| `switch` / `step` / `envelope` / `hole`（覆盖完整） | 不能 | DEBUG | 5 s |
| 同上，但 `uncovered > 0` 或 `edge` | 能 | WARN | 1 s（证据的价值随位置精度衰减） |

合并**不损失定位**：一个桶只留最强的那一条完整读数（含 `at=` 与 `jump=`），
并在前面加 `xN in T`。计数回答「有多少」，正文回答「在哪一下」。

#### 「没有位置」必须能说出它是哪一种

`capture.no-project-mapping` 在每次渲染开头出现 1~4 条，而那几块正是 −102 dBFS 的
数字静音与一个 128 帧的过短块 —— 探测按设计不为它们定稿（静音块在任何锚点下输出
都是静音），直通输出的是**同一份静音**，一个样本都没变。

它与真正的缺陷（映射表还没到）在输出上完全同形。所以原因必须**带出求解函数**：
`AnchorAbsence` 从 `feed_anchor` 一路走到 `BlockEvidence`，由输出仲裁按
`is_expected()`（只有 `silent` / `short`）决定级别。
把它留在几何里让仲裁去读，就会读到**另一条流**或**另一块**的结论。

#### 同一个数，「宿主给了」与「我们用了」是两件事

`[capture] ctx` 里的 `music=16677.1`（≈2.3 小时 @120 BPM）被反复问过：这个值会不会
把我们带偏？答案原本要靠人读代码。现在这一行多了 `src=` —— 本块**实际采用**的
位置来源。一个荒谬的 `music` 配 `src=project`，意思是「宿主给了，我们没信它」。

#### 倒退是事件，不是读数

`[capture] clock` 的判读方式是「稳态下 `rewinds == 0`」。但整行每 100 ms 就可能变一次
（`jumps` 在正常会话里一直在动），于是它每拍打一行 WARN，而真正要报警的只有
**位置倒退**。现在读数降到 DEBUG，倒退增长时另发一条 WARN **事件**，带前后读数与所在流。
