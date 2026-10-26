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
