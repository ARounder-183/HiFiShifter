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
