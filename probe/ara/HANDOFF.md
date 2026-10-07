# 交接文档 —— HiFiShifter ARA2 探针

> 写于 2026-10-04。给接手本分支的下一个 agent。
> **先读这份，再读 spec 与 plan。** 这份记录的是"踩过的坑与已确立的规矩"，
> 那些不写下来就会重踩一遍。

---

## 0. 一句话现状

**Task 1（取得宿主侧 ARA 模型的真实样本）已完成。Task 2（确定 Rust 接触 ARA 的路径）
完成一半 —— 已证明 `ara2-bridge` 在本机可编译可运行、且带本平台 VST3 原生探测证据，
但"能加载进 REAPER"的判据尚未满足。Task 3 未开始，且被基线阻塞。**

---

## 1. 工作区与分支

```
E:/code/HiFiShifter                             52211e61 [develop]     ← 主工作区，未动
E:/code/HiFiShifter/.worktrees/ara-bridge-probe f04e72e8 [feature/ara-bridge-probe]
```

**所有工作都在 worktree 里做。** 主工作区只有过一个提交（`52211e61`，把 `.worktrees/`
加进 `.gitignore`），那是 git worktree 的硬要求。

提交序列（新→旧）：

| commit | 内容 |
| --- | --- |
| `f04e72e8` | Task 2：`ara2-bridge` 在锁定 SDK 下编译并运行 |
| `373ce2ad` | Task 1 FINDINGS（丢失字段清单） |
| `860ad26c` | awkward 素材采集（拉伸/淡化/非 44.1k） |
| `78bf8221` | 第一份真实 REAPER ARA 模型 |
| `06f9283c` | 插桩插件（dump ARA 模型图） |
| `6b34dcc6` | ARA SDK 就位 + 测试插件构建 |
| `2751f0d3` | 忽略 `.superpowers/` |
| `8b93e2ce` | 探针计划 |
| `60d29f8a` | 设计 spec |

**绝不 push。** 只做本地提交。

`develop` 目前 ahead of origin 1（就是那个 `.gitignore` 提交），没有 push。

---

## 2. 权威文档

| 文档 | 作用 |
| --- | --- |
| [spec](docs/superpowers/specs/2026-10-04-ara-bridge-design.md) | 设计。**冲突时以它为准** |
| [plan](docs/superpowers/plans/2026-10-04-ara-bridge-probe.md) | 三步探针，含杀死判据 |
| [EXECUTION-LEDGER.md](probe/ara/EXECUTION-LEDGER.md) | **执行记录与全部 15 条 Ruling**。比本文件细 |
| [probe/ara/README.md](probe/ara/README.md) | SDK 构建配方 + 插桩补丁步骤 |
| [probe/ara/captures/FINDINGS.md](probe/ara/captures/FINDINGS.md) | Task 1 产出：字段清单 + 丢失字段 |

> **ledger 有两份**：`.superpowers/sdd/.../progress.md` 是 SDD 工作区里的**活文件**
> （已 gitignore，不随 git 走）；`probe/ara/EXECUTION-LEDGER.md` 是进版本控制的
> **冻结快照**。接手时读后者；若你继续用 SDD 流程，把新条目写进前者并同步。

---

## 3. 环境前提（不做这些，所有失败都无法归因）

### 3.1 MSVC 环境 —— 必须加载

`cc` / `cmake` 两个 crate **直接调用 `cl.exe`，从不初始化 MSVC 环境**。缺变量时的症状
极具误导性：`cl.exe` 明明在磁盘上，却报

```
cl : Command line error D8050 : cannot execute '...\c1xx.dll'
```

```powershell
cd E:\code\HiFiShifter\.worktrees\ara-bridge-probe
. .\tools\msvc-env.ps1        # 若报"未数字签名"，先 Set-ExecutionPolicy -Scope Process Bypass
```

### 3.2 陈旧临时文件会让 cl.exe 报同一个 D8050

**即使 MSVC 环境正确，`%TEMP%` 里有残留也会导致 D8050。** 解法是把 `TEMP`/`TMP`
指到一个干净目录，并且**在 vcvars 之后再设一次**（vcvars 会重置它们）：

```powershell
$tmp = "<worktree>\probe\ara\rust-path\.build-tmp\cl"
New-Item -ItemType Directory -Force $tmp | Out-Null
# ... 加载 vcvars ...
$env:TEMP = $tmp; $env:TMP = $tmp   # vcvars 之后再设
cargo build --jobs 1
```

**这条是本次会话最耗时的坑。** 报错信息把人引向"缺工具链"，实际是临时文件。

### 3.3 前端产物

`backend/src-tauri/build.rs` 在缺 `frontend/dist` 时会跑 `npm run build` 并可能 panic。
本 worktree 已构建好 `frontend/dist`，**别删**。

### 3.4 npm 安装

```powershell
cd frontend
$env:npm_config_cache = "<worktree>\.npm-cache"   # 默认缓存在工作区外，会被沙箱拒
npm install --ignore-scripts                       # postinstall 的 spawn 会被沙箱拦
$env:ESBUILD_TMPDIR = "<worktree>\frontend\.esbuild-tmp"
npm run build
```

### 3.5 基线：已取得（2026-10-04）✅

**结论先给**：`backend/src-tauri` 的 `cargo test --no-fail-fast` 跑通，合计
**789 passed / 4 failed / 1 ignored**；4 个失败全部是 `audio_engine::snapshot::tests` 里
硬编码 POSIX 路径 `/tmp/…`（Windows 上解析成 `E:\tmp\…`，该目录不存在）导致的**既有环境性失败，
不要修**。完整数字与原因见 `EXECUTION-LEDGER.md` 的 "Baseline — OBTAINED" 一节，
并已回填 `docs/superpowers/plans/2026-10-04-ara-bridge-probe.md` 的基线表。

以下是**历史上的**受阻说明，保留以便理解为什么曾经拿不到 —— 当时是沙箱对 cargo 孙进程
编译器的干扰，沙箱放开后同一配方一次跑通。

`cargo test` 的基线**拿不到**。`fdk-aac-sys` / `opusic-sys` 经 `cmake` crate 调 MSBuild，
撞上临时文件拒绝，而那条路径**没有** `TrackFileAccess` 之类的钩子可关。

**需要人在普通终端跑**：

```powershell
cd E:\code\HiFiShifter\.worktrees\ara-bridge-probe
. .\tools\msvc-env.ps1
cd backend\src-tauri
cargo test 2>&1 | Select-String 'test result:'
```

**Task 3 依赖它。** 在基线为空时改代码，后续失败无法区分"本来就坏"和"改坏了"。

---

## 4. REAPER 采集流程（已跑通，照抄）

### 4.1 三条铁律

1. **`-cfgfile` 不会在 REAPER 已运行时开出新实例。** REAPER 是单实例应用，脚本会被送进
   运行中的那个。**这是本会话真实造成的一次破坏**：往用户在用的 `test.rpp` 里写了轨道。
2. **因此：先杀掉所有 REAPER 进程，再用 `-cfgfile` + 独立 `vstpath64` 启动。**
   **永远不要**用 `-nonewinst` 往运行中的实例送脚本。
3. **采集实例要配独立 VST 目录**，否则插件文件被别的实例锁住（`Copy-Item` 会失败，
   而失败原因是"文件正由另一进程使用"，与插件本身无关）。

### 4.2 完整步骤

```powershell
$win = 'E:\code\HiFiShifter\.worktrees\ara-bridge-probe'
$cfg = "$win\probe\ara\reaper-profile"     # 隔离配置（已 gitignore）
$cap = "$win\probe\ara\captures"

# 1) 关掉所有 REAPER
Get-Process -Name reaper -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep 4

# 2) 部署插件（此时无人锁定）
Copy-Item "$win\probe\ara\ARA_SDK\ARA_Examples\build-vs2022\bin\Release\ARATestPlugIn.vst3" `
          "$win\probe\ara\vst3\ARATestPlugIn.vst3" -Force

# 3) 隔离配置：只扫工作区 VST 目录
@"
[REAPER]
vstpath64=$win\probe\ara\vst3
"@ | Set-Content "$cfg\REAPER.ini" -Encoding ascii

# 4) 清旧产物，带环境变量启动（工作目录决定 dump 回退路径）
$env:ARA_PROBE_OUT = "$cap\ara-model.reaper.json"
Start-Process -FilePath 'D:\Softwares\REAPER (x64)\reaper.exe' `
  -ArgumentList @('-cfgfile', "$cfg\REAPER.ini", '-new', "$win\probe\ara\build_probe_project_capture.lua") `
  -WorkingDirectory "$win\probe\ara"
Start-Sleep 40
```

### 4.3 脚本清单（`probe/ara/`）

| 脚本 | 用途 |
| --- | --- |
| `build_probe_project_capture.lua` | 单 region 采集；分阶段延迟并写 `captures/capture.log` |
| `build_awkward_fixture.lua` | 拉伸/倒放/淡化/同源多放/非 44.1k |
| `query_tracks.lua` | **只读**：列出轨道与 FX（用于确认实例里到底有什么） |
| `query_undo.lua` | **只读**：undo 状态；路径用十六进制输出以规避免码页问题 |

**关键点**：`ARA_PROBE_OUT` **不保证被继承**（job 启动时实测为 `nil`）。插桩里有回退链，
最终落到 `.\captures\ara-model.auto.json`（相对启动工作目录）+ 一条指向本工作树的绝对路径。
所以**启动时必须设 `-WorkingDirectory "$win\probe\ara"`**。

---

## 5. 关键结论（已实测，勿当推断重查）

### 5.1 ARA 实际给什么（来自真实 REAPER dump）

- **`audioSource.persistentID` 就是素材绝对路径** → 直接对应 `Clip.source_path`
- `regionSequence.name` = REAPER 轨道名
- `audioModification` 是独立对象，persistentID 与 source 相同（本插件 1:1）
- `startInPlaybackTime` / `durationInPlaybackTime` 是**秒（double）**，无需采样换算
- `sampleRate` / `sampleCount` **逐源保真**（44.1k 与 48k 各带自己的值）
- **拉伸 = 时长差**：`playback_rate = durationInModificationTime / durationInPlaybackTime`
  —— **不是** `isTimestretchEnabled` 标志位
- **淡化形状/曲率 ARA 不提供**：只有两个布尔，且插件不声明支持时恒为 false

### 5.2 关于 ara2-bridge

- 0.3.0 **在本机编译并运行通过**（官方 `minimal-plugin` 输出 `model: document model`）
- 那些看似不同的 crate（`ara-bridge` / `ara2-bridge` / `…-companion` / `…-sys`）
  **同属一个仓库** `github.com/entrepeneur4lyf/ara2-bridge`
- 它要求 **VST3 v3.8.0_build_66**；Celemony 安装器给的 v3.7.11_build_10 **不可互换**
- `build.rs` 做**仓库+commit+tree hash 三重校验**，tree 脏了拒绝编译。
  **Windows 上必须 `core.autocrlf=false`**，否则 CRLF 转换让 tree hash 对不上
- 已提供：`Vst3MainFactoryAdapter`、`Vst3PluginEntryAdapter`、以及
  `ara2_vst3_plugin_entry_create` / `query_interface` / `add_ref` / `release` 等 FFI
- 本平台原生探测证据：`probes/vst3-windows-x86_64.json`（IID、vtable 尺寸/对齐，实测）
- **边界**：它给的是 ARA↔VST3 **桥**，**不是**成品可加载模块

---

## 6. 待办

### 6.1 Task 2 Step 3 —— 解锁 T2 完成判据（可立即做）

目标：一个 **cdylib 形式的 VST3+ARA 插件**，能被 REAPER 识别为 ARA 并打印源/region 计数。

要做：导出 `GetPluginFactory` / `InitDll` / `ExitDll`，接到 `ara2_vst3_plugin_entry_create`
上。SDK 路径已就位：

```
ARA_VST3_SDK_DIR = <worktree>\probe\ara\rust-path\.third-party\vst3sdk
ARA_SDK_DIR      = <worktree>\probe\ara\rust-path\.third-party\ARA_SDK
```

编译配方见 §3.1 + §3.2（干净 TEMP + `--jobs 1`）。**已下载的 crate 源码在**
`%USERPROFILE%\.cargo\registry\src\index.crates.io-1949cf8c6b5b557f\ara2-bridge-companion-0.3.0\`
—— 读它的 `src/vst3/ffi.rs` 和 `probes/`，别猜 API。

**Computer Use 现在可用**（`list_apps` / `get_app_state` / `click` / `press_key` 等），
这是加载验证的手段 —— 上一个 agent 没有这个能力。

### 6.2 未决问题（必须先澄清）

**`sampleAccessEnabled` 在两次采集里不一致**：最小采集 `true`，awkward 采集 `false`。
插件**读不到样本就无法做音高分析**。若是持续为假，则"宿主供源 + 本地合成"整条链路不成立。
**不得当作已解释。**

### 6.3 Task 3 —— 被基线阻塞

需要 §3.5 的基线数字。之后按 plan 实现 `ara_document_to_timeline`。

---

## 7. 这个分支上确立的规矩（别破坏）

1. **绝不 push。**
2. **绝不用 `-nonewinst` 往运行中的 REAPER 送脚本**（§4.1）。
3. **绝不用 `git add -A`**：本会话曾因此把 REAPER 生成的 2356 个配置/主题文件
   （5 万行）提交进仓库。**只 add 明确列出的路径。**
4. **探针产物与产品代码隔离**：全部在 `probe/ara/` 下，且标注为一次性。
5. **插桩插桩文件的权威副本在 `probe/ara/instrumentation-reference/`**（可评审），
   实际编译的那份在 git-ignored 的 SDK 克隆里。改完要按 README 同步。
6. **文件顶部中文头注释、关键函数中文 doc 注释**（仓库硬约定）。
7. **REAPER 工程是用户资产**：采集只在隔离实例里做，且用一次性工程。

---

## 8. 已 gitignore 的内容（重建而非提交）

```
probe/ara/ARA_SDK/            克隆的 ARA SDK + VST3 SDK + 构建产物（~93MB）
probe/ara/vst3/               暂存的插件二进制
probe/ara/reaper-profile/     REAPER 首次启动生成的配置目录（~2356 文件）
probe/ara/fixtures/peaks/     REAPER 生成的峰值缓存
probe/ara/rust-path/target/   Rust 构建产物
probe/ara/rust-path/.third-party/  锁定的 SDK 检出
.superpowers/                 SDD 工作区（ledger 在此）
```

重建命令都在 `probe/ara/README.md`。

---

## 9. 关于本探针与 HiFiShifter 的关系

探针**不修改** `backend/` 与 `frontend/` 的任何代码。它的产出是：
① 宿主 ARA 模型的一手样本 ② 一条已验证的 Rust 技术路径 ③ 一份"ARA 不提供什么"的清单。
这三样都是 spec 的输入，而不是产品代码。
