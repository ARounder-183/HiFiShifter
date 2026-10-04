# HiFiShifter ARA 插件 v1 · Phase 1–2 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把内核从 Tauri app 里切出来（插件 crate 只依赖内核），并把 ARA 插件从探针原型做成产品代码 —— REAPER 能加载它、它能看到真实的 ARA 文档、并把时间线映射成内核的 `TimelineState`。

**Architecture:** 一份内核两种形态。先把 `backend/` 合成一个 workspace，再把 `state.rs` 拆成"纯模型 / 运行时容器"，把模型与叶模块搬进 `hifishifter-kernel`；然后用 `ara2-bridge` 的 `PluginModel` 框架实现插件侧，复用探针已验证的 VST3 外壳。

**Tech Stack:** Rust 1.97 · `ara2-bridge` 0.3.0（含 `plugin` / `companion` / `vst3`）· 锁定的 VST3 v3.8.0_build_66 与 ARA SDK · REAPER 7.81 · Windows / MSVC 14.44

**Spec:** [`docs/superpowers/specs/2026-10-04-ara-plugin-v1-design.md`](../specs/2026-10-04-ara-plugin-v1-design.md)

## Global Constraints

- **绝不 push**。只做本地提交。分支 `codex/ara-plugin`。
- **绝不用 `git add -A`**。只 add 本任务明确列出的路径。上一个 agent 因此把 REAPER 生成的 2356 个配置/主题文件提交进了仓库。
- **文件顶部中文头注释、关键函数中文 doc 注释**（本仓库硬约定）。
- **不碰 `frontend/`**。`frontend/dist` 是 gitignored 构建产物，已存在，别删。
- **构建前置**：每一条 `cargo` 命令前都必须先做这两件事，否则 `cl.exe` 会报 `D8050: cannot execute 'c1xx.dll'`（两种不同成因，症状相同）：

  ```powershell
  cd E:\code\HiFiShifter\.worktrees\ara-plugin
  . .\tools\msvc-env.ps1                                    # 必须点源
  $t = "$PWD\.build-tmp\cl"; New-Item -ItemType Directory -Force $t | Out-Null
  $env:TEMP = $t; $env:TMP = $t                             # 必须在 vcvars 之后
  ```

- 所有 cargo 命令带 `--offline --jobs 1`。
- **4 个既有失败不要修**：`audio_engine::snapshot::tests` 里硬编码 POSIX `/tmp/…`，Windows 上解析成 `E:\tmp\…` 不存在。它们是环境性失败，改动前后一致。
- **REAPER 采集铁律**：先杀掉所有 REAPER 进程，再用 `-cfgfile` + 独立 `vstpath64` 启动。**绝不**用 `-nonewinst` 往运行中的实例送脚本 —— 那是单实例应用，脚本会进用户的工程。
- **每搬一个模块的验收口径**：app 单测减少、内核增加、**两边合计不变**。

### 基线（开工前先自己核一遍，数字必须一致）

```powershell
# app
cd backend\src-tauri; cargo test --lib --jobs 1 --offline 2>&1 | Select-String 'test result:'
# → 期望 762 passed / 4 failed / 1 ignored

# 内核
cd ..\hifishifter-kernel; cargo test --offline 2>&1 | Select-String 'test result:'
# → 期望 10 passed

# 插件
cd ..\hifishifter-plugin; cargo test --jobs 1 --offline 2>&1 | Select-String 'test result:'
# → 期望 11 passed
```

合计口径：**772 passed / 4 failed / 1 ignored**（app + 内核）。集成测试 17 个另计。

---

## 文件结构（本计划会创建 / 改动的文件）

| 路径 | 责任 |
| --- | --- |
| `backend/Cargo.toml` | **新建**。workspace 根，三个 crate 共享一份 lock |
| `backend/Cargo.lock` | 从 `backend/src-tauri/Cargo.lock` 移来 |
| `backend/hifishifter-kernel/src/host.rs` | **新建**。内核找宿主的出口（`HostCallbacks` / `HostServices`） |
| `backend/hifishifter-kernel/src/engine_command.rs` | **新建**。`EngineCommand` 及载荷（从 `audio_engine/types.rs` 搬） |
| `backend/hifishifter-kernel/src/state/{mod,model}.rs` | **新建**。`state.rs` 拆出的纯模型 |
| `backend/src-tauri/src/state/{mod,app}.rs` | 拆分后 app 侧的运行时容器与再导出 |
| `backend/hifishifter-plugin/src/vst3/*.rs` | **新建**。VST3 模块外壳（从探针生产化） |
| `backend/hifishifter-plugin/src/ara/model.rs` | **新建**。`PluginModel` 实现 |
| `.build-tmp/*.ps1` | 计数脚本与闭包重算脚本（gitignored） |

---

## Phase 1 — 内核边界落地

### Task 1: 把 workspace 合并到 `backend/`

**为什么先做**：插件 crate 现在自带一份从 app 复制来的 `Cargo.lock`，不复制就会解析出不同版本的 `windows-core`，`backend_lib` 作为依赖直接编译失败。合并之后这个问题在结构上消失，并且 `cargo test -p <crate>` 可用 —— 后面每个任务都要用它。

**Files:**
- Create: `backend/Cargo.toml`
- Modify: `backend/hifishifter-kernel/Cargo.toml`（删 `[workspace]`）
- Modify: `backend/hifishifter-plugin/Cargo.toml`（删 `[workspace]`）
- Move: `backend/src-tauri/Cargo.lock` → `backend/Cargo.lock`
- Delete: `backend/hifishifter-kernel/Cargo.lock`、`backend/hifishifter-plugin/Cargo.lock`

**Interfaces:**
- Consumes: 无
- Produces: 一个 workspace，后续任务用 `cargo test -p hifishifter-kernel` / `-p hifishifter-plugin` / `-p HiFiShifter` 从 `backend/` 运行

- [ ] **Step 1: 查清三个 lock 的差异来源（不是"必须相同"）**

```powershell
cd E:\code\HiFiShifter\.worktrees\ara-plugin\backend
(Get-FileHash src-tauri\Cargo.lock).Hash
(Get-FileHash hifishifter-kernel\Cargo.lock).Hash
(Get-FileHash hifishifter-plugin\Cargo.lock).Hash
git diff --no-index --stat src-tauri\Cargo.lock hifishifter-plugin\Cargo.lock
```

**实测（2026-10-04）**：三者**互不相同**，差异只有两类，都是预期的：

| 文件 | 行数 | 与 app 的 lock 的差异 |
| --- | --- | --- |
| `src-tauri/Cargo.lock` | 6935 | 基准 |
| `hifishifter-plugin/Cargo.lock` | 6943 | 多一个 `hifishifter-plugin` 包条目（9 行） |
| `hifishifter-kernel/Cargo.lock` | 127 | 内核自己的最小闭包（`lru` / `serde_json` 及其依赖） |

**判据不是"哈希相同"，而是"差异可解释"**：除上述两类之外，若出现**任何 `version` 字段的变化**，
说明已经有依赖漂移，先查清再合并 —— 合并会把漂移固化。用这条命令只看版本差异：

```powershell
git diff --no-index src-tauri\Cargo.lock hifishifter-plugin\Cargo.lock | Select-String '^[+-]version'
```

Expected: **无输出**。

- [ ] **Step 2: 写 workspace 根**

创建 `backend/Cargo.toml`：

```toml
# HiFiShifter 后端 workspace。
#
# 【为什么需要它】插件 crate（`hifishifter-plugin`）依赖 app crate（`backend_lib`）里的内核。
# 三个 crate 各自带一份 `Cargo.lock` 时，插件会自行解析出**不同版本**的 `windows-core`，
# 于是 `backend_lib` 作为依赖编译失败（`webview2_accelerators.rs` 的 `cast()` 找不到 trait）——
# 这是实测踩过的坑。合并成一份 lock 后，这个问题在结构上消失。
#
# 设计依据：docs/superpowers/specs/2026-10-04-ara-plugin-v1-design.md §4.1
[workspace]
resolver = "2"
members = ["src-tauri", "hifishifter-kernel", "hifishifter-plugin"]
```

**然后必须把 `backend/src-tauri/Cargo.toml` 末尾的三段 `[profile.*]` 搬过来**
（`[profile.release]` / `[profile.dist]` / `[profile.dev-opt]` 及其注释），
并在 workspace 根上保留它们。

> **这是执行时实测到的一个静默陷阱**：cargo 对**非根包**的 `[profile.*]` 只发一条
> warning（`profiles for the non root package will be ignored`）就忽略。不搬的话，
> release 构建会丢掉 `strip` / `opt-level = 3` / `dist` 的 fat LTO —— 而症状只是
> "产物变大变慢"，不报错、不进测试。放在根上会让三种 profile 同时作用于
> kernel 与 plugin，这是期望行为（插件的 release 产物同样该 strip）。

- [ ] **Step 3: 摘掉两个新 crate 的独立 workspace 声明**

在 `backend/hifishifter-kernel/Cargo.toml` 与 `backend/hifishifter-plugin/Cargo.toml` 中删除这一段（含它上面的注释）：

```toml
[workspace]
```

并把 `hifishifter-plugin/Cargo.toml` 里那段关于"自带 Cargo.lock / 必须复制"的注释替换为：

```toml
# lock 由 `backend/` workspace 统一持有（见 backend/Cargo.toml 的说明）。
```

- [ ] **Step 4: 移动 lock 文件**

```powershell
cd E:\code\HiFiShifter\.worktrees\ara-plugin
git mv backend/src-tauri/Cargo.lock backend/Cargo.lock
git rm --cached backend/hifishifter-kernel/Cargo.lock backend/hifishifter-plugin/Cargo.lock
Remove-Item backend/hifishifter-kernel/Cargo.lock, backend/hifishifter-plugin/Cargo.lock
```

> `git rm --cached` 只对**已跟踪**的文件有效。若某个 lock 从未提交，用 `Remove-Item` 即可，报错可忽略。

- [ ] **Step 5: 验证 workspace 解析稳定（lock 不被改写）**

```powershell
cd E:\code\HiFiShifter\.worktrees\ara-plugin\backend
. ..\tools\msvc-env.ps1
$t = "$PWD\..\.build-tmp\cl"; New-Item -ItemType Directory -Force $t | Out-Null
$env:TEMP = $t; $env:TMP = $t
cargo metadata --offline --format-version 1 > $null
cd ..; git diff --stat -- backend/Cargo.lock
```

Expected: `git diff --stat` 只显示 `backend/Cargo.lock` **增加了约 9 行**（把 `hifishifter-plugin`
这个 workspace member 的包条目写进去）。这是唯一允许的差异。

**若出现任何 `version` 行的增删 → 停**：那就是 Cargo 重新解析了版本，而重新解析正是这次
合并要消除的风险。逐条查清后再决定接受或回退。

验证一下只加了这一条：

```powershell
git diff -- backend/Cargo.lock | Select-String '^[+-]name =|^[+-]version'
```

Expected: 只有 `+name = "hifishifter-plugin"` 一行。

- [ ] **Step 6: 三个 crate 全部跑一遍测试**

```powershell
cd backend
cargo test -p hifishifter-kernel --offline 2>&1 | Select-String 'test result:'
cargo test -p hifishifter-plugin --jobs 1 --offline 2>&1 | Select-String 'test result:'
cargo test -p HiFiShifter --lib --jobs 1 --offline 2>&1 | Select-String 'test result:'
```

Expected: `10 passed` / `11 passed` / `762 passed, 4 failed, 1 ignored` —— 与基线逐字一致。

- [ ] **Step 7: 提交**

```powershell
cd E:\code\HiFiShifter\.worktrees\ara-plugin
git add backend/Cargo.toml backend/Cargo.lock backend/hifishifter-kernel/Cargo.toml backend/hifishifter-plugin/Cargo.toml
git commit -m "chore(ara): merge the backend crates into one workspace"
```

---

### Task 2: 拆 `state.rs` 成 `model` / `app`

**为什么**：`state.rs`（11826 行）把纯数据模型与运行时容器 `AppState` 放在一起，而 `AppState` 持有 `tauri::AppHandle` 并引用波形缓存、渲染缓存句柄、`recording` 状态。这导致**任何**从 `state` 出发的依赖闭包都被非内核模块污染（实测 43 模块 / 2.57 MB）。必须先拆，才能算出真正的内核边界。

**这一步是纯搬运：行为零变化、路径零变化。** 验收标准是"app 单测数一个不变"。

**Files:**
- Move: `backend/src-tauri/src/state.rs` → `backend/src-tauri/src/state/original.rs`
- Create: `backend/src-tauri/src/state/mod.rs`

**Interfaces:**
- Consumes: 无
- Produces: `crate::state::*` 的全部路径不变（`state/mod.rs` 做 `pub use`）；下一步的切分有了落点

- [ ] **Step 1: 目录化（零风险的第一步）**

```powershell
cd E:\code\HiFiShifter\.worktrees\ara-plugin\backend\src-tauri\src
New-Item -ItemType Directory -Force state | Out-Null
git mv state.rs state/original.rs
```

创建 `state/mod.rs`：

```rust
//! 时间线状态：数据模型与运行时容器。
//!
//! 【本文件的存在意义】原先 `state.rs` 是一个 11826 行的单文件，把"纯数据模型"
//! 与"运行时容器 `AppState`"混在一起。`AppState` 持有 Tauri 句柄与设备层句柄，
//! 于是任何从 `state` 出发的依赖闭包都会被非内核模块污染 —— 这是内核抽取撞墙的根因。
//!
//! 拆成两个文件、用一个 `mod.rs` 再导出，是为了让 `crate::state::X` 的**全部既有路径
//! 保持不变**（内核搬迁的通用做法：搬内容，接路径）。

mod original;

pub use original::*;
```

- [ ] **Step 2: 跑测试，确认目录化没改变任何事**

```powershell
cd E:\code\HiFiShifter\.worktrees\ara-plugin\backend
cargo test -p HiFiShifter --lib --jobs 1 --offline 2>&1 | Select-String 'test result:'
```

Expected: `762 passed, 4 failed, 1 ignored` —— 不变。

- [ ] **Step 3: 把 `original.rs` 切成 `model.rs` / `app.rs`**

创建 `state/app.rs`，把下面这些**顶层项**从 `original.rs` 整体移进去（含各自的 `impl` 块与 doc 注释）：

| 项 | 说明 |
| --- | --- |
| `struct AppState` | 运行时容器 |
| `impl Default for AppState` | |
| `impl AppState`（**两处**） | 含所有 `use tauri::Emitter;` / `use tauri::Manager;` 的方法 |
| `struct RuntimeState` | 只被 `AppState` 持有 |
| `struct WaveformInflightGuard<'a>` + 它的 `impl` + `impl Drop` | 只服务于 `AppState` 的波形缓存 |
| `struct SourceFileCheckItem` | 只被 `impl AppState` 使用 |

**不要按行号裁剪** —— 行号会漂。以 `^\s*(pub )?(struct|enum|impl|fn|const|static) ` 的顶层项边界为准，逐项剪切。

**测试也要跟着走**：`original.rs` 里的 `mod tests` 中，凡是调用 `AppState::default()` 的测试（当前在 `SYNC_EDITS_TEST_LOCK` 附近，至少 3 处）必须移入 `app.rs` 的 `mod tests`；其余测试留在 `model.rs`。

Then rename and wire up:

```powershell
git mv state/original.rs state/model.rs
```

把 `state/mod.rs` 改成：

```rust
//! 时间线状态：数据模型（`model`）与运行时容器（`app`）。
//!
//! 拆分的理由与"路径零改写"的做法见 `model.rs` 与 `app.rs` 的文件头注释。

mod app;
mod model;

pub use app::*;
pub use model::*;
```

在 `model.rs` 顶部删除 `use crate::audio_engine::AudioEngine;`（它只服务 `AppState`），在 `app.rs` 顶部补上它以及 `model` 里被用到的项（编译器会逐条指出来）。

- [ ] **Step 4: 跑测试，确认拆分仍是纯搬运**

```powershell
cd E:\code\HiFiShifter\.worktrees\ara-plugin\backend
cargo test -p HiFiShifter --lib --jobs 1 --offline 2>&1 | Select-String 'test result:'
```

Expected: **仍是** `762 passed, 4 failed, 1 ignored`。少一个或多一个都说明搬运时改变了行为，必须查清再往下。

- [ ] **Step 5: 钉住"model 里不出现 tauri"这条边界**

```powershell
cd E:\code\HiFiShifter\.worktrees\ara-plugin\backend\src-tauri\src
Select-String -Path state\model.rs -Pattern 'tauri::'
```

Expected: **无输出**。若有，说明该项属于 `app.rs`，移过去。

- [ ] **Step 6: 提交**

```powershell
cd E:\code\HiFiShifter\.worktrees\ara-plugin
git add backend/src-tauri/src/state
git commit -m "refactor(ara): split state into a pure model and the runtime container"
```

---

### Task 3: 把 `state/model`、`time_stretch`、`metronome` 搬进内核

**为什么是这三个**：它们是离 Tauri 最远的三块，且是 `EngineCommand` 的前置（命令里的 `UpdateTimeline(TimelineState)` 与 `UserStretchAlgorithm` / `MetronomeConfig` 来自这里）。

**Files:**
- Move: `backend/src-tauri/src/state/model.rs` → `backend/hifishifter-kernel/src/state/model.rs`
- Move: `backend/src-tauri/src/audio/time_stretch.rs` → `backend/hifishifter-kernel/src/time_stretch.rs`
- Move: `backend/src-tauri/src/audio_engine/metronome.rs` → `backend/hifishifter-kernel/src/metronome.rs`
- Modify: `backend/hifishifter-kernel/src/lib.rs`、`backend/src-tauri/src/state/mod.rs`、`backend/src-tauri/src/lib.rs`、`backend/src-tauri/src/audio_engine/mod.rs`

**Interfaces:**
- Consumes: Task 2 的 `state/model.rs`
- Produces: `hifishifter_kernel::{state, time_stretch, metronome}`

- [ ] **Step 1: 搬 `time_stretch`（最叶的一块，先验证搬运配方）**

```powershell
cd E:\code\HiFiShifter\.worktrees\ara-plugin
git mv backend/src-tauri/src/audio/time_stretch.rs backend/hifishifter-kernel/src/time_stretch.rs
```

在 `backend/hifishifter-kernel/src/lib.rs` 追加 `pub mod time_stretch;`。

在 `backend/src-tauri/src/lib.rs` 里把 `#[path = "audio/time_stretch.rs"] mod time_stretch;` 替换为：

```rust
// `time_stretch` 已迁到 `hifishifter-kernel`。这里再导出，app 侧
// `crate::time_stretch::…` 的路径保持不变。
pub use hifishifter_kernel::time_stretch;
```

`time_stretch.rs` 内部若有 `crate::X` 引用，把 `X` 指向内核里的同名模块（内核还没有的，本任务不动它 —— 编译器会报出来，那时先只搬能搬的）。

- [ ] **Step 2: 跑测试，核对计数**

```powershell
cd E:\code\HiFiShifter\.worktrees\ara-plugin\backend
cargo test -p hifishifter-kernel --offline 2>&1 | Select-String 'test result:'
cargo test -p HiFiShifter --lib --jobs 1 --offline 2>&1 | Select-String 'test result:'
```

Expected: 内核从 `10` 变成 `10 + N`，app 从 `762` 变成 `762 − N`，**N 相同**。`time_stretch.rs` 当前没有单测时 N = 0 —— 那也是合法结果。

- [ ] **Step 3: 提交叶模块的搬运**

```powershell
git add backend/hifishifter-kernel/src/time_stretch.rs backend/hifishifter-kernel/src/lib.rs backend/src-tauri/src/lib.rs
git commit -m "refactor(kernel): move time_stretch into the kernel crate"
```

- [ ] **Step 4: 搬 `metronome`**

```powershell
git mv backend/src-tauri/src/audio_engine/metronome.rs backend/hifishifter-kernel/src/metronome.rs
```

在 kernel 的 `lib.rs` 加 `pub mod metronome;`；在 `backend/src-tauri/src/audio_engine/mod.rs` 里把 `pub(crate) mod metronome;` 替换为：

```rust
// 已迁到 `hifishifter-kernel`。再导出，`crate::audio_engine::metronome::…`
// 与 `super::metronome::…` 的路径都保持不变。
pub(crate) use hifishifter_kernel::metronome;
```

`metronome.rs` 里的 `crate::state::TempoPointData` 在内核里指向内核自己的 `state`（`crate::state` 语义相同），无需改写。

- [ ] **Step 5: 跑测试并提交**

```powershell
cd ..\..\backend
cargo test -p hifishifter-kernel --offline 2>&1 | Select-String 'test result:'
cargo test -p HiFiShifter --lib --jobs 1 --offline 2>&1 | Select-String 'test result:'
cd E:\code\HiFiShifter\.worktrees\ara-plugin
git add backend/hifishifter-kernel/src/metronome.rs backend/hifishifter-kernel/src/lib.rs backend/src-tauri/src/audio_engine/mod.rs
git commit -m "refactor(kernel): move metronome into the kernel crate"
```

Expected: 合计不变（app 与内核此消彼长，总和恒定）。

- [ ] **Step 6: 搬 `state/model.rs`**

```powershell
git mv backend/src-tauri/src/state/model.rs backend/hifishifter-kernel/src/state/model.rs
```

（`backend/hifishifter-kernel/src/state/` 目录需要先建。）

在 `backend/hifishifter-kernel/src/lib.rs` 加 `pub mod state;`。

创建 `backend/hifishifter-kernel/src/state/mod.rs`：

```rust
//! 时间线数据模型。
//!
//! 从 `backend/src-tauri` 的 `state.rs` 拆出来的纯模型部分 —— 不含 `AppState`
//! （那是持有 Tauri 句柄的运行时容器，留在 app 层）。

pub mod model;

pub use model::*;
```

把 `backend/src-tauri/src/state/mod.rs` 改成：

```rust
//! 时间线状态。纯模型已迁到内核，这里再导出 + 挂住 app 层的运行时容器。

mod app;

pub use app::*;
// 模型已迁到 `hifishifter-kernel`。再导出，`crate::state::Clip` 这类路径保持不变。
pub use hifishifter_kernel::state::*;
```

- [ ] **Step 7: 跑测试，解决 `use` 与可见性**

```powershell
cd backend
cargo test -p hifishifter-kernel --offline 2>&1 | Select-String 'test result:'
cargo test -p HiFiShifter --lib --jobs 1 --offline 2>&1 | Select-String 'test result:'
```

Expected: 内核 `10 + N_model`，app `762 − N_model`，合计不变。

**注意：`state/model.rs` 里的项用的是 `pub`，内核 crate 内不需要改可见性。** 但 app 侧 `state/app.rs` 里对模型项的引用走 `hifishifter_kernel::state::*` 再导出，`crate::state::X` 依然可解析。

- [ ] **Step 8: 提交**

```powershell
cd E:\code\HiFiShifter\.worktrees\ara-plugin
git add backend/hifishifter-kernel/src/state backend/hifishifter-kernel/src/lib.rs backend/src-tauri/src/state/mod.rs
git commit -m "refactor(kernel): move the timeline data model into the kernel crate"
```

---

### Task 4: `EngineCommand` 进内核，删掉 `SetAppHandle`

**为什么**：`EngineCommand` 是内核 worker 与设备层之间唯一的命令词汇表。它现在住在 `audio_engine/types.rs`，而那个文件碰 `tauri::` 的唯一原因就是 `SetAppHandle { handle: tauri::AppHandle }` 这一个变体。

**Files:**
- Create: `backend/hifishifter-kernel/src/engine_command.rs`
- Modify: `backend/src-tauri/src/audio_engine/types.rs`
- Modify: `backend/src-tauri/src/audio_engine/engine.rs`

**Interfaces:**
- Consumes: Task 3 的 `hifishifter_kernel::state::TimelineState`、`hifishifter_kernel::time_stretch::UserStretchAlgorithm`、`hifishifter_kernel::metronome::{MetronomeConfig, MetronomeClick}`
- Produces: `hifishifter_kernel::engine_command::{EngineCommand, StretchKey, AudioKey}` —— 后续 `HostCallbacks::send_engine_command` 用它

- [ ] **Step 1: 写内核侧的编译期测试（先失败）**

创建 `backend/hifishifter-kernel/src/engine_command.rs`，**先只放测试**：

```rust
//! 内核与设备层之间的命令词汇表。
//!
//! 【为什么在内核里】内核 worker（音高分析、渲染调度）需要向设备层投递命令，
//! 而设备层（cpal 流）是宿主相关的、不进内核。命令本身是**纯数据**，所以它属于内核：
//! 这样内核既不需要认识 cpal，也不需要认识 Tauri。
//!
//! 【为什么没有 `SetAppHandle`】那个变体曾把 `tauri::AppHandle` 塞进命令通道，
//! 使整个枚举无法离开 app 层。现在 engine worker 需要的句柄改从
//! `crate::app_events::app_handle()` 取（它已经是进程级出口），命令通道回到纯数据。

#[cfg(test)]
mod tests {
    use super::*;

    /// 命令必须是 `Send`：它跨线程投递给 device worker。
    #[test]
    fn commands_are_send() {
        fn assert_send<T: Send>() {}
        assert_send::<EngineCommand>();
    }
}
```

在 `backend/hifishifter-kernel/src/lib.rs` 加 `pub mod engine_command;`。

- [ ] **Step 2: 运行，确认失败**

```powershell
cd backend\hifishifter-kernel
cargo test --offline 2>&1 | Select-String 'error|cannot find'
```

Expected: `cannot find type EngineCommand` —— 测试引用了还不存在的类型。

- [ ] **Step 3: 把枚举与载荷搬进来**

从 `backend/src-tauri/src/audio_engine/types.rs` 剪出 `AudioKey`、`StretchKey`、`EngineCommand`（保留 `StretchJob`、`EngineClip`、`EngineSnapshot`、`ResampledStereo`、`TrackMeterValue`、`AudioEngineStateSnapshot` 在原地 —— 它们是设备层），粘到 `engine_command.rs`，并把 `crate::` 引用改成内核路径：

```rust
use crate::metronome::{MetronomeClick, MetronomeConfig};
use crate::state::TimelineState;
use crate::time_stretch::UserStretchAlgorithm;
use std::path::PathBuf;
use std::sync::Arc;

/// 解码缓存键：源路径 + 目标采样率。
pub type AudioKey = (PathBuf, u32);

/// 拉伸任务键。
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StretchKey {
    pub path: PathBuf,
    pub out_rate: u32,
    pub algorithm: UserStretchAlgorithm,
    /// 保留字段以兼容 `Hash`，固定为 0。
    pub bpm_q: u32,
    pub trim_start_q: i64,
    pub trim_end_q: i64,
    pub playback_rate_q: u32,
}

/// 内核与设备层之间的命令。**纯数据**，不含任何宿主句柄。
pub enum EngineCommand {
    UpdateTimeline(TimelineState),
    SeekSec { sec: f64 },
    SetPlaying { playing: bool, target: Option<String> },
    PlayFile { path: PathBuf, offset_sec: f64, target: String },
    StretchReady { key: StretchKey },
    #[allow(dead_code)]
    AudioReady { key: AudioKey },
    /// clip pitch MIDI 异步预计算完成，触发 snapshot rebuild。
    ClipPitchReady { clip_id: String },
    /// 请求 worker 侧为「动态（DYN）」提交后台分析任务。
    ScheduleDynLevelAnalysis,
    /// 使指定源路径的解码缓存和拉伸缓存失效（源文件被替换时调用）。
    EvictSourcePath { path: String },
    /// 更新节拍器配置（开关 / 音量 / 细分模式 / 重音 / 音色）。
    SetMetronome { config: MetronomeConfig },
    /// 换入节拍器响点表（命令层按工程 Tempo Map + 网格预展开）。
    SetMetronomeSchedule { clicks: Arc<Vec<MetronomeClick>> },
    /// 渲染结果已变更（**发送方是渲染线程**）。
    RenderedClipsChanged,
    Stop,
    Shutdown,
}
```

> **把原文件里那些解释性 doc 注释一并搬过来** —— 它们记录了"为什么 `RenderedClipsChanged` 是解除原地等待的唯一机制"这类不能丢的信息。这里为了篇幅只留了骨架。

- [ ] **Step 4: 删掉 `SetAppHandle`，让 worker 从进程级出口取句柄**

在 `backend/src-tauri/src/audio_engine/types.rs` 顶部加：

```rust
// 命令词汇表已迁到 `hifishifter-kernel`。再导出，`crate::audio_engine::types::EngineCommand`
// 与 `super::types::EngineCommand` 的路径都保持不变。
pub(crate) use hifishifter_kernel::engine_command::{AudioKey, EngineCommand, StretchKey};
```

删掉 `types.rs` 里的原定义。`StretchJob.app_handle: Option<Arc<tauri::AppHandle>>` 保留（它是设备层，且 worker 需要它 emit 进度）。

在 `audio_engine/engine.rs` 里删掉 `set_app_handle`（约 137 行）与 `EngineCommand::SetAppHandle { handle } => { ... }` 分支（约 817 行），把 `with_app_handle(app_handle: Option<tauri::AppHandle>)` 换成：

```rust
/// 设备 worker 需要的宿主句柄从进程级出口取。
///
/// 【为什么不再经命令通道传】`EngineCommand` 已迁进内核，而它必须是纯数据
/// （内核不认识 Tauri）。句柄由 `app_events::install` 在 Tauri setup 里注册，
/// 时序上早于引擎启动（见 `lib.rs` 的 setup），所以这里取得到。
fn host_app_handle() -> Option<tauri::AppHandle> {
    crate::app_events::app_handle().cloned()
}
```

把 `EngineWorkerState.app_handle` 字段改成在构造时用 `host_app_handle()` 初始化。

- [ ] **Step 5: 跑测试**

```powershell
cd backend
cargo test -p hifishifter-kernel --offline 2>&1 | Select-String 'test result:'
cargo test -p HiFiShifter --lib --jobs 1 --offline 2>&1 | Select-String 'test result:'
```

Expected: 内核 `+1`（`commands_are_send`），app 数字不变 —— 合并枚举没有搬走任何测试。

**另外手工启动一次 app**（`cargo run -p HiFiShifter --jobs 1 --offline`）确认能起来、
播放一次音频 —— 句柄改道是这一条里唯一有时序风险的地方。

- [ ] **Step 6: 提交**

```powershell
cd E:\code\HiFiShifter\.worktrees\ara-plugin
git add backend/hifishifter-kernel/src/engine_command.rs backend/hifishifter-kernel/src/lib.rs backend/src-tauri/src/audio_engine/types.rs backend/src-tauri/src/audio_engine/engine.rs
git commit -m "refactor(kernel): move EngineCommand into the kernel and drop the app-handle variant"
```

---

### Task 5: 内核新增宿主回调出口 `host`

**为什么**：内核 worker 回头找宿主有三类，第 1 类（发事件）已有出口，第 2 类（投引擎命令）与第 3 类（调度音高分析）还没有。没有它，`pitch_clip` / `pitch_analysis` 就无法离开 app 层。

**Files:**
- Create: `backend/hifishifter-kernel/src/host.rs`
- Modify: `backend/hifishifter-kernel/src/lib.rs`

**Interfaces:**
- Consumes: Task 4 的 `hifishifter_kernel::engine_command::EngineCommand`
- Produces: `hifishifter_kernel::host::{HostCallbacks, SharedHostCallbacks, HostServices}`，含 `install` / `is_installed` / `send_engine_command` / `request_pitch_analysis`

- [ ] **Step 1: 写失败测试**

创建 `backend/hifishifter-kernel/src/host.rs`：

```rust
//! 内核 worker 回头找宿主的出口。
//!
//! 【为什么需要独立出口】内核不认识 Tauri，也不认识 ARA 宿主。它需要"往外说三件事"：
//! ① 发 UI 事件（已有 [`crate::events`]）；② 向设备层投引擎命令；
//! ③ 请求为某个根轨调度音高分析。后两件由本模块承载。
//!
//! 【为什么是"具体类型 + 固有方法"而不是裸 trait 对象】调用点的改写量最小：
//! `state.host.request_pitch_analysis(root)` 与原来的 `app.state::<AppState>()`
//! 后面接一句调用，形状几乎一样。

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine_command::EngineCommand;
    use std::sync::{Arc, Mutex};

    /// 记录型的假宿主：把内核发来的调用原样存下来供断言。
    #[derive(Default)]
    struct RecordingHost {
        commands: Mutex<usize>,
        pitch_requests: Mutex<Vec<String>>,
    }

    impl HostCallbacks for RecordingHost {
        fn send_engine_command(&self, _command: EngineCommand) {
            *self.commands.lock().unwrap() += 1;
        }
        fn request_pitch_analysis(&self, root_track_id: &str) {
            self.pitch_requests.lock().unwrap().push(root_track_id.to_string());
        }
    }

    /// 没有宿主时必须是**静默降级**，不是 panic：内核会被单测与插件直接使用，
    /// 那时根本没有宿主。
    #[test]
    fn an_uninstalled_outlet_is_a_silent_no_op() {
        let host = HostServices::default();
        assert!(!host.is_installed());
        host.request_pitch_analysis("root-1");
        host.send_engine_command(EngineCommand::Stop);
    }

    /// 装了宿主就必须原样转发。
    #[test]
    fn an_installed_outlet_forwards_every_call() {
        let host = HostServices::default();
        let recorder = Arc::new(RecordingHost::default());
        assert!(host.install(recorder.clone()));

        host.request_pitch_analysis("root-1");
        host.send_engine_command(EngineCommand::Stop);

        assert_eq!(recorder.pitch_requests.lock().unwrap().as_slice(), ["root-1"]);
        assert_eq!(*recorder.commands.lock().unwrap(), 1);
    }

    /// 二次安装必须失败且**不替换**已有宿主：宿主在一个进程里只装配一次，
    /// 静默替换会让先装的那份实现变成幽灵。
    #[test]
    fn installing_twice_keeps_the_first_host() {
        let host = HostServices::default();
        let first = Arc::new(RecordingHost::default());
        let second = Arc::new(RecordingHost::default());

        assert!(host.install(first.clone()));
        assert!(!host.install(second.clone()));

        host.request_pitch_analysis("root-1");
        assert_eq!(first.pitch_requests.lock().unwrap().len(), 1);
        assert!(second.pitch_requests.lock().unwrap().is_empty());
    }
}
```

在 kernel `lib.rs` 加 `pub mod host;`。

- [ ] **Step 2: 运行，确认失败**

```powershell
cd backend\hifishifter-kernel
cargo test --offline 2>&1 | Select-String 'error|cannot find'
```

Expected: `cannot find type HostServices`。

- [ ] **Step 3: 实现**

在 `host.rs` 的测试模块**之前**插入：

```rust
use crate::engine_command::EngineCommand;
use std::sync::{Arc, OnceLock};

/// 宿主必须能提供的两类回调。
pub trait HostCallbacks: Send + Sync + 'static {
    /// 把一条引擎命令投递给设备层。插件侧可以丢弃它，或转成 ARA renderer 的请求。
    fn send_engine_command(&self, command: EngineCommand);
    /// 请求为某个根轨调度音高分析。app 侧读 `AppState`；插件侧读自己的文档状态。
    fn request_pitch_analysis(&self, root_track_id: &str);
}

/// 可跨线程持有的宿主回调。
pub type SharedHostCallbacks = Arc<dyn HostCallbacks>;

/// 内核侧的宿主出口。由宿主在装配时 `install` 一次；未安装时所有调用都是静默降级。
#[derive(Default)]
pub struct HostServices {
    inner: OnceLock<SharedHostCallbacks>,
}

impl HostServices {
    /// 安装宿主实现。返回 `false` 表示已经装过（**不替换**）。
    pub fn install(&self, callbacks: SharedHostCallbacks) -> bool {
        self.inner.set(callbacks).is_ok()
    }

    /// 宿主是否已装配。
    pub fn is_installed(&self) -> bool {
        self.inner.get().is_some()
    }

    /// 投递一条引擎命令；无宿主时什么都不做。
    pub fn send_engine_command(&self, command: EngineCommand) {
        if let Some(host) = self.inner.get() {
            host.send_engine_command(command);
        }
    }

    /// 请求调度某个根轨的音高分析；无宿主时什么都不做。
    pub fn request_pitch_analysis(&self, root_track_id: &str) {
        if let Some(host) = self.inner.get() {
            host.request_pitch_analysis(root_track_id);
        }
    }
}
```

- [ ] **Step 4: 跑测试，确认通过**

```powershell
cd backend\hifishifter-kernel
cargo test --offline 2>&1 | Select-String 'test result:'
```

Expected: 新增 3 条通过。

- [ ] **Step 5: 提交**

```powershell
cd E:\code\HiFiShifter\.worktrees\ara-plugin
git add backend/hifishifter-kernel/src/host.rs backend/hifishifter-kernel/src/lib.rs
git commit -m "feat(kernel): add the host-callback outlet for engine commands and pitch scheduling"
```

---

### Task 6: 重算闭包，按实测清单搬迁剩余内核模块

**为什么**：Task 2 拆掉 `state` 之后，"真正的内核模块集"才第一次可测。**设计文档 §4.2 的 43 模块是拆之前的数**，不能拿它当施工清单。

**Files:**
- Create: `.build-tmp/kernel-closure3.ps1`（gitignored 的施工工具）
- Create: `probe/ara/kernel-closure-measured.md`（**进版本控制**的实测记录）
- Move: 按脚本输出确定的模块（下面给固定配方）

**Interfaces:**
- Consumes: Task 2 拆分后的 `state/{model,app}.rs`
- Produces: 内核模块集与真实字节数的**实测数字**（写进 ledger）

- [ ] **Step 1: 写重算脚本**

创建 `.build-tmp/kernel-closure3.ps1`：

```powershell
# 重算内核闭包：从 mixdown 出发，整体排除设备层（audio_engine），
# 目录型模块扫描其**全部**子文件（第一版脚本漏了这一点）。
$src = 'E:\code\HiFiShifter\.worktrees\ara-plugin\backend\src-tauri\src'
$files = @{}
$lines = Get-Content (Join-Path $src 'lib.rs')
for ($i = 0; $i -lt $lines.Count; $i++) {
    if ($lines[$i] -match '^\s*(?:pub )?mod\s+([A-Za-z_][A-Za-z0-9_]*)\s*;') {
        $name = $matches[1]; $list = @()
        if ($i -ge 1 -and $lines[$i-1] -match '#\[path\s*=\s*"([^"]+)"\]') {
            $rel = $matches[1] -replace '/', '\'; $p = Join-Path $src $rel
            if ($rel -like '*\mod.rs') {
                $list = Get-ChildItem -Recurse (Split-Path $p -Parent) -Filter '*.rs' | ForEach-Object { $_.FullName }
            } else { $list = @($p) }
        } else {
            $a = Join-Path $src "$name.rs"; $b = Join-Path $src "$name\mod.rs"
            if (Test-Path $a) { $list = @($a) }
            elseif (Test-Path $b) { $list = Get-ChildItem -Recurse (Join-Path $src $name) -Filter '*.rs' | ForEach-Object { $_.FullName } }
        }
        if ($list.Count -gt 0) { $files[$name] = $list }
    }
}
function Get-Refs($paths) {
    $refs = New-Object System.Collections.Generic.HashSet[string]
    foreach ($p in $paths) {
        if (-not (Test-Path $p)) { continue }
        foreach ($m in [regex]::Matches((Get-Content $p -Raw), 'crate::([A-Za-z_][A-Za-z0-9_]*)')) {
            $null = $refs.Add($m.Groups[1].Value)
        }
    }
    return $refs
}
# 设备层整体排除：audio_engine 是 cpal 边界，插件侧由宿主回调取代（设计 §4.4）。
$excluded = @('audio_engine')
$seen = New-Object System.Collections.Generic.HashSet[string]
$queue = New-Object System.Collections.Queue
foreach ($s in @('mixdown')) { if (-not $seen.Contains($s)) { $seen.Add($s) | Out-Null; $queue.Enqueue($s) } }
while ($queue.Count -gt 0) {
    $name = $queue.Dequeue()
    if (-not $files.ContainsKey($name)) { continue }
    foreach ($ref in Get-Refs $files[$name]) {
        if ($files.ContainsKey($ref) -and $excluded -notcontains $ref -and -not $seen.Contains($ref)) {
            $seen.Add($ref) | Out-Null; $queue.Enqueue($ref)
        }
    }
}
$total = 0; $tauri = @()
foreach ($name in ($seen | Sort-Object)) {
    $bytes = 0; $hasTauri = $false
    foreach ($p in $files[$name]) {
        $bytes += (Get-Item $p).Length
        if (Select-String -Path $p -Pattern 'tauri::' -Quiet) { $hasTauri = $true }
    }
    $total += $bytes
    if ($hasTauri) { $tauri += $name }
    Write-Output ("{0,-24} {1,9} {2,3} files {3}" -f $name, $bytes, $files[$name].Count, $(if ($hasTauri) { 'TAURI' } else { '' }))
}
Write-Output ''
Write-Output ("模块数 = {0}" -f $seen.Count)
Write-Output ("总字节 = {0}" -f $total)
Write-Output ("碰 tauri 的模块 = [{0}]" -f ($tauri -join ', '))
```

- [ ] **Step 2: 运行并记录实测数字**

```powershell
cd E:\code\HiFiShifter\.worktrees\ara-plugin
pwsh -NoProfile -File .build-tmp\kernel-closure3.ps1 | Tee-Object -FilePath .build-tmp\kernel-closure3.out.txt
```

把输出**原样**写成 `probe/ara/kernel-closure-measured.md`，并注明"这是拆分 `state` 之后的实测；设计文档 §4.2 的 43 模块是拆分之前的数，两者不可混用"。

- [ ] **Step 3: 把实测数字与预测做对比，差异写进 ledger**

设计文档 §4.2 给了一条**明确标注为推断**的预测：`hfspeaks_v2` / `notebook_assets` / `temp_manager` / `recording` 会离开闭包。比一比，若不符，把原因查清再往下 —— 不符本身是重要信息（说明那些模块不只被 `AppState` 引用）。

在 `probe/ara/EXECUTION-LEDGER.md` 追加：

```markdown
### Task 6: Ruling: <实测的模块数 / 字节数 / 碰 tauri 的模块> — <拆分 state 之后的真内核集是它> — <错了的代价：若把拆分前的 43 模块当清单施工，会白搬一批非内核模块并让插件二进制白白变大>
```

- [ ] **Step 4: 按固定配方逐个搬模块**

**固定配方**（对清单里每个模块执行一次；以 `render_key` 为例）：

```powershell
cd E:\code\HiFiShifter\.worktrees\ara-plugin

# 1) 搬文件（目录型模块整个目录搬）
git mv backend/src-tauri/src/render_key.rs backend/hifishifter-kernel/src/render_key.rs

# 2) 内核声明：在 backend/hifishifter-kernel/src/lib.rs 追加 pub mod render_key;

# 3) app 侧接回路径：在 backend/src-tauri/src/lib.rs 把 `mod render_key;`
#    换成 `pub use hifishifter_kernel::render_key;`（原路径上的注释一并更新）

# 4) 计数核对
cd backend
cargo test -p hifishifter-kernel --offline 2>&1 | Select-String 'test result:'
cargo test -p HiFiShifter --lib --jobs 1 --offline 2>&1 | Select-String 'test result:'

# 5) 提交（只 add 这一个模块涉及的文件）
cd E:\code\HiFiShifter\.worktrees\ara-plugin
git add backend/hifishifter-kernel/src/render_key.rs backend/hifishifter-kernel/src/lib.rs backend/src-tauri/src/lib.rs
git commit -m "refactor(kernel): move render_key into the kernel crate"
```

**两个必须遵守的约束**：

1. **一次只搬一个模块**。合起来搬会让"合计不变"这个验收失去分辨力 —— 两个模块一增一减恰好抵消时，你什么都看不出来。
2. 模块内部的 `crate::X` 引用若指向还没搬的模块，**先搬被依赖的那个**（叶先动）。

- [ ] **Step 5: 每搬完一批，核对总账**

```powershell
cd E:\code\HiFiShifter\.worktrees\ara-plugin\backend
cargo test -p hifishifter-kernel --offline 2>&1 | Select-String 'test result:'
cargo test -p HiFiShifter --lib --jobs 1 --offline 2>&1 | Select-String 'test result:'
```

Expected: **app 减少的数 == 内核增加的数**，每一项都对得上。

---

### Task 7: 插件 crate 只依赖内核（A5 的静态部分）

**为什么**：这是"插件不把 WebView2 / Tauri 带进 DAW 进程"从**约定**变成**事实**的那一步。在此之前，`hifishifter-plugin` 依赖的是整个 app crate。

**Files:**
- Modify: `backend/hifishifter-plugin/Cargo.toml`
- Modify: `backend/hifishifter-plugin/src/ara.rs`、`src/render.rs`、`src/lib.rs`
- Create: `backend/hifishifter-plugin/tests/no_tauri_in_dependency_tree.rs`

**Interfaces:**
- Consumes: `hifishifter_kernel::{state, mixdown, render_key, time_stretch, ...}`
- Produces: 一个**不依赖 `backend_lib`** 的插件 crate

- [ ] **Step 1: 写失败测试 —— 依赖树里不许有 Tauri**

创建 `backend/hifishifter-plugin/tests/no_tauri_in_dependency_tree.rs`：

```rust
//! 架构不变量的守卫：**插件进程里不出现 Tauri / WebView2**（设计 §1 判据 A5）。
//!
//! 【为什么用测试而不是人看】这条约束会随着后续搬迁被无意破坏
//! （某个模块为了图方便去 `use backend_lib::…`），而症状要到 DAW 里才显形。
//! 放在测试里，破坏的第一时间就报。
//!
//! 手段是读 `cargo metadata` 的解析结果：只要依赖图里没有 `tauri` / `wry` /
//! `webview2-com`，就说明插件 crate 的依赖闭包是干净的内核。

use std::process::Command;

#[test]
fn the_plugin_dependency_tree_contains_no_tauri_stack() {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let output = Command::new(env!("CARGO"))
        .args(["metadata", "--offline", "--format-version", "1", "--manifest-path"])
        .arg(format!("{manifest}/Cargo.toml"))
        .output()
        .expect("cargo metadata must run");
    assert!(output.status.success(), "cargo metadata failed: {output:?}");

    let text = String::from_utf8_lossy(&output.stdout);
    for banned in ["\"tauri\"", "\"wry\"", "\"webview2-com\"", "\"HiFiShifter\""] {
        assert!(
            !text.contains(banned),
            "插件依赖树里出现了 `{banned}` —— 插件会把 DAW 进程污染成 WebView 宿主（判据 A5）"
        );
    }
}
```

> 禁用项写 `"HiFiShifter"`（app 的包名）而不是 `backend_lib`（库名）：`cargo metadata`
> 输出的是包名。这一点在第一次运行时就会显形。

- [ ] **Step 2: 运行，确认失败**

```powershell
cd backend\hifishifter-plugin
cargo test --jobs 1 --offline --test no_tauri_in_dependency_tree 2>&1 | Select-Object -Last 20
```

Expected: FAIL，且报出 `"tauri"` —— 当前 `Cargo.toml` 依赖 `backend_lib`。

- [ ] **Step 3: 换依赖，改引用**

`backend/hifishifter-plugin/Cargo.toml`：

```toml
[dependencies]
hifishifter-kernel = { path = "../hifishifter-kernel" }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
```

删掉 `backend_lib = { package = "HiFiShifter", path = "../src-tauri" }`。

把 `src/ara.rs` / `src/render.rs` 里的 `use backend_lib::kernel::…` 换成 `use hifishifter_kernel::…`（`render.rs` 需要的 `render_mixdown_interleaved` / `MixdownOptions` / `OutputSpec` / `QualityPreset` / `StretchAlgorithm` 都得先在内核里存在 —— 缺哪个就把它加进 Task 6 的搬迁清单）。

把 `src/lib.rs` 的文档注释里"内核入口经 `backend_lib::kernel` 暴露"改成"插件**只**依赖 `hifishifter-kernel`：这是插件不把 WebView2 带进 DAW 进程成立的时刻"。

- [ ] **Step 4: 跑测试，确认新守卫通过、原有 11 条映射测试仍通过**

```powershell
cd backend\hifishifter-plugin
cargo test --jobs 1 --offline 2>&1 | Select-String 'test result:'
```

Expected: `11 passed`（映射）+ `1 passed`（新守卫）。

- [ ] **Step 5: 提交**

```powershell
cd E:\code\HiFiShifter\.worktrees\ara-plugin
git add backend/hifishifter-plugin/Cargo.toml backend/hifishifter-plugin/src backend/hifishifter-plugin/tests
git commit -m "refactor(ara): make the plugin crate depend on the kernel only"
```

**Phase 1 完成判据**：`the_plugin_dependency_tree_contains_no_tauri_stack` 通过；app + 内核测试合计仍为 `772 / 4 / 1`；`hifishifter-plugin` 的 `Cargo.toml` 里没有 `backend_lib`。

---

## Phase 2 — 插件骨架产品化

### Task 8: 把 VST3 外壳从探针搬成产品代码

**为什么**：探针已经证明这个外壳能被 REAPER 加载（`GetPluginFactory` / `InitDll` / `ExitDll` + 最小组件 + 最小编辑控制器）。产品代码不需要重新发明它，但需要它**不再是可丢弃的探针**。

**Files:**
- Create: `backend/hifishifter-plugin/src/vst3/mod.rs`、`factory.rs`、`component.rs`、`controller.rs`
- Modify: `backend/hifishifter-plugin/src/lib.rs`、`Cargo.toml`
- Test: `backend/hifishifter-plugin/tests/vst3_exports.rs`

**Interfaces:**
- Consumes: `probe/ara/rust-path/src/vst3.rs`（1169 行，已实测可加载）
- Produces: 一个导出 `GetPluginFactory` / `InitDll` / `ExitDll` 的 cdylib

- [ ] **Step 1: 写失败测试 —— 导出符号必须存在**

创建 `tests/vst3_exports.rs`：

```rust
//! VST3 模块入口的守卫：三个导出符号必须存在且拼写正确。
//!
//! 【为什么能测】VST3 在 Windows 上就是一个导出 `GetPluginFactory` / `InitDll` /
//! `ExitDll` 的 DLL。符号名拼错时 REAPER 的表现是"这个文件 0 个类"而不报错，
//! 属于最难定位的一类失败 —— 所以用测试把它钉住。

#[test]
fn the_vst3_module_entry_points_are_exported() {
    // 构建产物的导出表里必须同时出现这三个名字。
    // 实现方式二选一：
    //   (a) 用 `libloading` 在本进程的模块符号表里查（需要 cdylib 已被加载）；
    //   (b) 构建后对 target/debug/hifishifter_plugin.dll 跑 `dumpbin /exports` 并断言。
    // 优先 (b)：它直接检查**产物**，不依赖加载时序。
    let dll = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target/debug/hifishifter_plugin.dll");
    let out = std::process::Command::new("dumpbin")
        .args(["/exports", dll.to_str().unwrap()])
        .output();
    let Ok(out) = out else {
        eprintln!("dumpbin 不可用，跳过（需要 MSVC 环境）");
        return;
    };
    let text = String::from_utf8_lossy(&out.stdout);
    for name in ["GetPluginFactory", "InitDll", "ExitDll"] {
        assert!(text.contains(name), "缺少 VST3 导出符号 `{name}`：\n{text}");
    }
}
```

在 `Cargo.toml` 的 `[lib]` 段加：

```toml
[lib]
name = "hifishifter_plugin"
crate-type = ["cdylib", "rlib"]
```

（`rlib` 是为了让测试能链接它；`cdylib` 才是 REAPER 加载的产物。）

- [ ] **Step 2: 运行，确认失败**

```powershell
cd backend\hifishifter-plugin
cargo test --jobs 1 --offline --test vst3_exports 2>&1 | Select-Object -Last 20
```

Expected: FAIL —— 还没有导出。

- [ ] **Step 3: 搬外壳**

把 `probe/ara/rust-path/src/vst3.rs` 的内容按探针文件里已有的分段拆成 `src/vst3/{factory,component,controller}.rs`：

| 探针里的段 | 落点 |
| --- | --- |
| `PluginFactoryVtbl` / `PluginFactoryObj` / `factory_*` / `get_plugin_factory` | `factory.rs` |
| `ComponentVtbl` / `AudioProcessorVtbl` / `Processor` / `component_*` / `audio_*` | `component.rs` |
| `EditControllerVtbl` / `EditController` / `edit_controller_*` | `controller.rs` |

`mod.rs` 汇总并把入口函数（`GetPluginFactory` / `InitDll` / `ExitDll`）暴露给 `lib.rs`。

**三条 ABI 硬事实必须连同代码一起搬进注释**（它们是用崩溃换来的）：

1. VST3 的 IID 在 Windows 上是 **GUID 布局**（`l1` 小端 + `l2` 拆两个 u16 各自小端 + `l3`/`l4` 大端）。按"每字小端"实现会让 REAPER 认为该文件 **0 个类**，不报错。
2. `IPluginFactory` **直接继承 `FUnknown`**，不是 `IPluginBase`（多插两个 vtable 槽位会让 `countClasses` 落到 `getFactoryInfo` 上）。
3. REAPER **要求 ARA 插件提供编辑器控制器**。没有 `kVstComponentControllerClass` 时，REAPER 会 bind 完再放弃、卸载模块，然后对失效控制器指针回调 → 崩溃（实测 `0xc0000005`，偏移落在 `ara2_bridge_plugin::ffi::generated_callbacks::begin_editing`）。

把探针里的 `eprintln!` 调试输出换成 `log::info!`，并去掉写死了工作树路径的日志文件回退链。

- [ ] **Step 4: 跑测试，确认导出存在**

```powershell
cd backend\hifishifter-plugin
cargo test --jobs 1 --offline 2>&1 | Select-String 'test result:'
```

Expected: `vst3_exports` 通过，其余测试不回归。

- [ ] **Step 5: 提交**

```powershell
cd E:\code\HiFiShifter\.worktrees\ara-plugin
git add backend/hifishifter-plugin/src/vst3 backend/hifishifter-plugin/src/lib.rs backend/hifishifter-plugin/Cargo.toml backend/hifishifter-plugin/tests/vst3_exports.rs
git commit -m "feat(ara): productise the VST3 module shell from the probe"
```

---

### Task 9: 实现 `PluginModel` —— 把 ARA 回调累积成文档模型

**为什么用 `PluginModel` 而不是手写委托**：探针用的是低层适配器 + 手工收集（236 行）。同仓库的 `ara2-bridge` 默认 feature `plugin` 已经提供了高层语义 trait（`DocumentLifecycle` / `MusicalContexts` / `RegionSequences` / `AudioSources` / `AudioModifications` / `PlaybackRegions`），实现它们比手写回调委托更安全；而探针的低层路径已被证明可行，所以这是低风险选择（设计 §4.7）。

**Files:**
- Create: `backend/hifishifter-plugin/src/ara/model.rs`、`src/ara/mod.rs`
- Modify: `backend/hifishifter-plugin/src/lib.rs`

**Interfaces:**
- Consumes: `ara2_bridge::plugin::{PluginBuilder, Plugin}`、`ara2_bridge::plugin::traits::*`
- Produces: `AraDocumentAccumulator` —— 累积出 `hifishifter_plugin::ara::AraDocument`（已存在于 `src/ara.rs`）

- [ ] **Step 1: 写失败测试 —— 累积器必须记下三个层级**

在 `src/ara/model.rs` 末尾写测试：

```rust
#[cfg(test)]
mod tests {
    use super::*;

    /// 一次最小的文档装配：一个 musical context → 一个 region sequence →
    /// 一个 audio source → 一个 audio modification → 一个 playback region。
    /// 累积器必须把每一层都记下来，且父子关系正确。
    #[test]
    fn the_accumulator_records_every_model_level() {
        let mut acc = AraDocumentAccumulator::default();

        acc.begin_document("probe-project");
        acc.note_region_sequence("seq-1", "track-1");
        acc.note_audio_source("src-1", "E:\\fixtures\\tone44100.wav", 44_100, 88_200, 1, 2.0);
        acc.note_audio_modification("mod-1", "src-1");
        acc.note_playback_region("region-1", "mod-1", "seq-1", 0.0, 0.0, 2.0, 2.0);

        let doc = acc.into_document();
        assert_eq!(doc.audio_sources.len(), 1);
        assert_eq!(doc.audio_modifications.len(), 1);
        assert_eq!(doc.playback_regions.len(), 1);
        assert_eq!(doc.musical_contexts[0].region_sequences[0].name, "track-1");
        assert_eq!(doc.playback_regions[0].region_sequence_index, Some(0));
    }
}
```

> **为什么先写这个测试**：它是"插件看到的世界"与"映射器要求的世界"之间的接口。映射器（`src/ara.rs`）已经 11 条测试全绿，所以这一层只要形状对，Phase 2 的端到端就有把握；形状不对会在这里立刻炸出来，而不是到 REAPER 里。

- [ ] **Step 2: 运行，确认失败**

```powershell
cd backend\hifishifter-plugin
cargo test --jobs 1 --offline 2>&1 | Select-String 'cannot find|error'
```

Expected: `cannot find type AraDocumentAccumulator`。

- [ ] **Step 3: 实现累积器与六个语义 trait**

在 `src/ara/model.rs` 里：

1. 定义 `AraDocumentAccumulator`：与 `src/ara.rs` 的 `AraDocument` 同形，外加逐对象的 `HashMap<RawHandle, usize>` 索引（ARA 回调给的是**句柄**，句柄与我们的 id 之间必须有稳定映射）。
2. 为它实现 `DocumentLifecycle`（`Document = AraDocumentAccumulator`）以及 `MusicalContexts` / `RegionSequences` / `AudioSources` / `AudioModifications` / `PlaybackRegions` —— 每个回调把参数写进自己的那张表。
3. 特别处理两个：

```rust
/// 宿主授予/撤销样本访问。
///
/// 【为什么单独记】探针实测：绑定后宿主调 `enable=true`，停用时 `enable=false`。
/// 所以 `false` 是**撤销**语义，不是拒绝 —— 采集到 `false` 时不能判定"宿主不给样本"。
fn enable_audio_source_samples_access(
    &mut self,
    _context: &CreateContext,
    source: RawHandle,
    enable: bool,
    _host: &HostContentScope<'_, '_>,
) -> Result<(), AraError> {
    if let Some(entry) = self.sources.get_mut(&source) {
        entry.sample_access_enabled = enable;
    }
    Ok(())
}

/// 宿主报告某个源的内容变了。
///
/// 【为什么必须处理】渲染缓存不能复用旧内容（设计 §4.5）：这里把该源的
/// `content_version` 自增，映射时折进 `Clip.source_file_fingerprint`。
fn update_audio_source_content(
    &mut self,
    _context: &CreateContext,
    source: RawHandle,
    _range: Option<ContentTimeRange>,
    _host: &HostContentScope<'_, '_>,
) -> Result<(), AraError> {
    if let Some(entry) = self.sources.get_mut(&source) {
        entry.content_version += 1;
    }
    Ok(())
}
```

**具体的关联类型与参数形状以编译器为准**（`ara2_bridge::plugin::traits` 的签名就是契约）。不要凭记忆写 —— 打开 `%USERPROFILE%\.cargo\registry\src\index.crates.io-1949cf8c6b5b557f\ara2-bridge-plugin-0.3.0\src\traits\model.rs` 对着抄。

- [ ] **Step 4: 跑测试，确认通过**

```powershell
cd backend\hifishifter-plugin
cargo test --jobs 1 --offline 2>&1 | Select-String 'test result:'
```

Expected: 新测试通过；映射的 11 条与依赖守卫不回归。

- [ ] **Step 5: 提交**

```powershell
cd E:\code\HiFiShifter\.worktrees\ara-plugin
git add backend/hifishifter-plugin/src/ara backend/hifishifter-plugin/src/lib.rs
git commit -m "feat(ara): accumulate the ARA document model through the plugin traits"
```

---

### Task 10: 接上映射与日志 —— A1 / A2 端到端

**判据（设计 §1）**：**A1** 插件被 REAPER 以 ARA 方式加载、看到的 source / region 数与工程一致；**A2** DAW 侧的切片 / 移动能反映到插件的 `TimelineState`。

**Files:**
- Modify: `backend/hifishifter-plugin/src/ara/mod.rs`
- Create: `probe/ara/build_task10_probe.lua`
- Create: `probe/ara/captures/task10-plugin.log`

**Interfaces:**
- Consumes: Task 9 的累积器、`hifishifter_plugin::ara::ara_document_to_timeline`
- Produces: 一条可核对的行 `ara: sources=N modifications=M regionSequences=K playbackRegions=P clips=C`

- [ ] **Step 1: 在 `end_editing` 里做映射并打日志**

```rust
/// 宿主一次编辑结束：把累积的 ARA 模型映射成 `TimelineState` 并记录摘要。
///
/// 【为什么在 `end_editing` 而不是每个回调里】ARA 的一次用户操作会拆成多次回调
/// （例如"复制 item" = createPlaybackRegion × N），逐次映射会得到中间态。
/// `end_editing` 是一次编辑的收口点，这是 ARA 的约定，不是我们的偏好。
fn end_editing(
    &mut self,
    document: &mut Self::Document,
    _host: &HostContentScope<'_, '_>,
) -> Result<(), AraError> {
    let doc = document.clone_document();
    match ara_document_to_timeline(&doc) {
        Ok(timeline) => {
            log::info!("{}", summary_line(&doc, &timeline));
            *document.timeline_mut() = Some(timeline);
        }
        Err(err) => {
            // 映射失败必须显式记录：它是"DAW 里看到的时间线不对"这类问题的唯一线索。
            log::error!("ara: mapping failed: {err:?}");
        }
    }
    Ok(())
}
```

判据行的形状固定为：

```
ara: sources=1 modifications=1 regionSequences=1 playbackRegions=2 clips=2
```

- [ ] **Step 2: 写单元测试钉住摘要行格式**

```rust
/// 摘要行的字段与顺序是运维契约（采集日志要能被 diff），不许随手改。
#[test]
fn the_summary_line_has_a_stable_shape() {
    let doc = load_fixture("../../probe/ara/captures/ara-model.reaper.json");
    let timeline = ara_document_to_timeline(&doc).unwrap();
    assert_eq!(
        summary_line(&doc, &timeline),
        "ara: sources=1 modifications=1 regionSequences=1 playbackRegions=1 clips=1"
    );
}
```

夹具是干净的 REAPER 单 region 采集；先跑一次取实际值，确认它符合该样本的语义再写死。

- [ ] **Step 3: 构建并部署到隔离 VST 目录**

```powershell
cd E:\code\HiFiShifter\.worktrees\ara-plugin
. .\tools\msvc-env.ps1
$t = "$PWD\.build-tmp\cl"; New-Item -ItemType Directory -Force $t | Out-Null
$env:TEMP = $t; $env:TMP = $t
$env:ARA_VST3_SDK_DIR = "$PWD\probe\ara\rust-path\.third-party\vst3sdk"
$env:ARA_SDK_DIR      = "$PWD\probe\ara\rust-path\.third-party\ARA_SDK"
cd backend\hifishifter-plugin
cargo build --jobs 1 --offline

cd E:\code\HiFiShifter\.worktrees\ara-plugin
Copy-Item backend\hifishifter-plugin\target\debug\hifishifter_plugin.dll probe\ara\vst3\HiFiShifter.vst3 -Force
```

> `ARA_VST3_SDK_DIR` / `ARA_SDK_DIR` 指向的两个检出做过仓库 + commit + tree hash 三重校验，**tree 一脏就拒绝编译**。不要改动它们的内容，也不要重新克隆。若 `probe/ara/rust-path/.third-party/` 不存在（新 worktree），按 `probe/ara/README.md` 重建，**不要**换版本。

- [ ] **Step 4: 在隔离实例里采集（铁律：先杀进程）**

```powershell
$win = 'E:\code\HiFiShifter\.worktrees\ara-plugin'
$cfg = "$win\probe\ara\reaper-profile"

# 1) 关掉所有 REAPER —— 单实例应用，不关会把脚本送进用户的工程
Get-Process -Name reaper -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep 4

# 2) 隔离配置：只扫工作区 VST 目录
@"
[REAPER]
vstpath64=$win\probe\ara\vst3
"@ | Set-Content "$cfg\REAPER.ini" -Encoding ascii

# 3) 启动（工作目录决定日志落点）
Start-Process -FilePath 'D:\Softwares\REAPER (x64)\reaper.exe' `
  -ArgumentList @('-cfgfile', "$cfg\REAPER.ini", '-new', "$win\probe\ara\build_task10_probe.lua") `
  -WorkingDirectory "$win\probe\ara"
Start-Sleep 40
```

采集脚本按 `probe/ara/build_task2_probe.lua` 改：建一条轨、放**同一素材两次**、插入插件。之后手动在 REAPER 里把第二个 item 拖到别的位置，观察插件日志是否出现新的一行摘要（A2）。

- [ ] **Step 5: 核对判据并把实测写进 FINDINGS**

- **A1**：日志里 `playbackRegions=2` 且 `clips=2`；
- **A2**：拖动 item 后出现新摘要行，且截取到的 `start_sec` 随之变化。

写入 `probe/ara/rust-path/FINDINGS.md` 的新一节，**原样保留日志行**，并明确标注哪些是实测、哪些是推断。

- [ ] **Step 6: 提交**

```powershell
cd E:\code\HiFiShifter\.worktrees\ara-plugin
git add backend/hifishifter-plugin/src/ara/mod.rs probe/ara/build_task10_probe.lua probe/ara/captures/task10-plugin.log probe/ara/rust-path/FINDINGS.md
git commit -m "feat(ara): map the live ARA document and log the timeline summary"
```

---

### Task 11: 收口未决项 U1（倒放）与 U2（拉伸）

**为什么这一条不能省**：倒放若没有表达，倒放区域会**渲染成正放**（静默错误）；拉伸分支目前只被合成样本验证过（探针的 awkward 样本实际不含拉伸）。两者都是"看起来能用、结果错"的类型。

**Files:**
- Modify: `backend/hifishifter-plugin/src/ara/model.rs`（声明能力）
- Create: `probe/ara/captures/task11-stretch-reverse.json`、`task11-FINDINGS.md`
- Modify: `probe/ara/EXECUTION-LEDGER.md`

**Interfaces:**
- Consumes: Task 10 的采集流程
- Produces: U1 / U2 的**实测**结论（成立 / 不成立 / 不成立时的退路）

- [ ] **Step 1: 声明支持拉伸与基于内容的淡化**

```rust
// 不声明这三项时，REAPER 不会把拉伸/淡化写进 ARA 模型（探针实测：awkward 样本里
// 5 个 region 全部 durMod == durPlay）。这是 U2 存在的直接原因。
let mut caps = plugin.capabilities_mut();
caps.set_supported_playback_transformation_flags(
    PlaybackTransformationFlags::Timestretch
        | PlaybackTransformationFlags::ReflectTempo
        | PlaybackTransformationFlags::ContentFades,
);
```

具体 API 名以 `ara2-bridge-plugin-0.3.0/src/processing.rs` 的 `SemanticCapabilities` 为准 —— 对着源码抄，不要猜。

- [ ] **Step 2: 采集拉伸样本（U2）**

在隔离实例里：放一个 2 秒素材，把 item 右边拖长到 4 秒（真实拉伸），采一份模型。判据：

```
durationInModificationTime != durationInPlaybackTime
```

把 JSON 存成 `probe/ara/captures/task11-stretch-reverse.json`。

**若仍相等**：说明这条路径不通过 ARA 表达拉伸，需要重新评估"拉伸 = 时长差"公式的宿主级适用性 —— 记进 ledger，**不要**当作"已验证"。

- [ ] **Step 3: 采集倒放样本（U1）**

对同一个 item 执行 Reverse，采一份模型，检查两件事：

1. region 的属性（`startInModificationTime` / `durationIn*`）是否变化；
2. `audioSource.persistentID` 或它指向的内容是否变化。

判据：

- 若 (2) 变化 → 宿主用**改源内容**表达倒放，插件读到的是已反向的源，**没问题**；
- 若两者都不变 → ARA 层没有倒放的表达，**这是会导致渲染方向错误的缺陷**，必须写进 FINDINGS 并回报（退路是让本体侧用 `Clip.reversed` 表达，但那要求倒放信息能进到插件 —— 目前没有通道，属于设计缺口）。

- [ ] **Step 4: 写结论并提交**

```powershell
cd E:\code\HiFiShifter\.worktrees\ara-plugin
git add backend/hifishifter-plugin/src/ara/model.rs probe/ara/captures/task11-stretch-reverse.json probe/ara/captures/task11-FINDINGS.md probe/ara/EXECUTION-LEDGER.md
git commit -m "probe(ara): close out the host-level stretch and reverse observations"
```

**Phase 2 完成判据**：A1 / A2 有实测证据；U1 / U2 各有"成立 / 不成立 / 不成立时的退路"三种明确结论之一，没有一项写成"待定"。

---

## 收尾

- [ ] 把 Phase 1 / Phase 2 的实际结论回写[设计文档](../specs/2026-10-04-ara-plugin-v1-design.md)的 §2.1（未决项表）与 §7（阶段表）
- [ ] 把每条 Ruling 追加进 [`EXECUTION-LEDGER.md`](../../../probe/ara/EXECUTION-LEDGER.md)，格式：`Task N: Ruling: <发现> — <决定与理由> — <错了的代价>`
- [ ] 决定 `probe/ara/rust-path/` 的处置（保留为对照物，还是删除）—— 产品代码已不再依赖它，但它的 FINDINGS 仍是三条 ABI 硬事实的原始记录

## Review Focus

本计划里最容易咬到人的地方，各由对应任务的验收钉住：

| 风险 | 归属 |
| --- | --- |
| 搬迁时两个模块的测试数一增一减抵消掉，掩盖行为变化 | Task 6 Step 4 的"一次只搬一个模块" |
| 拆 `state` 时把 `AppState` 依赖的测试留在了 `model.rs` | Task 2 Step 4 的计数核对 |
| `EngineCommand` 去 Tauri 后，启动早期的事件丢失 | Task 4 Step 5 的手工启动 app |
| 插件**看起来**不依赖 Tauri，实际通过传递依赖带进来 | Task 7 的 `cargo metadata` 守卫 |
| 拉伸/倒放"看起来对"但其实没被 ARA 表达 | Task 11 的宿主级采集 |
| 采集时把脚本送进用户正在用的 REAPER 工程 | 每个采集步骤的"先杀进程"铁律 |
