# HiFiShifter ARA v1 Phase 3a Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. 用户已授权本会话按批次自主执行，不重复请求选择执行方式。

**Goal:** 证明宿主 PCM 经每处理器的 ARA region 分配进入 VST3 音频输出，并取得真正倒放的输出结论。

**Architecture:** 模型线程持有宿主 reader，离线准备不可变 PCM 快照；renderer assignment 按 model-ref 键筛选区域，音频线程只读原子发布的快照。先导阶段不运行 ONNX，不以源路径解码替代 ARA。

**Tech Stack:** Rust 1.97、ara2-bridge 0.3.0、本地 vendored plugin 补丁、锁定 VST3 SDK v3.8.0_build_66、ARA SDK、MSVC 14.44、REAPER 7.81。

**Spec:** [Phase 3a 设计](../specs/2026-10-04-ara-plugin-phase3a-design.md)，及 [v1 设计](../specs/2026-10-04-ara-plugin-v1-design.md) §7.1。

## Global Constraints

- worktree `E:\code\HiFiShifter\.worktrees\ara-plugin`；分支 `codex/ara-plugin`，不动主工作区。
- **绝不 push**；**绝不用 `git add -A`**；仅本地逐路径提交。
- 中文文件头、关键函数中文 doc 注释；不改 app、frontend，不修四条既有 `/tmp` 失败。
- SDK 检出不得更改/重克隆；第三方补丁只在 `backend/third-party`，同步补丁说明。
- `process` / `setProcessing` 无锁等待、无分配/释放大对象、无 IO、无 reader、无推理。
- 只接受一进一出 stereo、`kSample32`；未实现的组合返回拒绝，不伪报成功。
- 先导资源上限：30 秒、44100/48000Hz、1/2 通道、512 MiB 总快照预算；超限显式拒绝。
- 所有 cargo 前加载 MSVC，**之后**设置 TEMP/TMP，`--offline --jobs 1`；SDK 环境变量见下。
- REAPER 只在隔离 profile + 一次性工程里采集。启动前确认无其他实例，绝不 `-nonewinst`。
- 本批不关闭 A3/U3/U4。零输出兜底不是成功；最终修音与持续供音仍需 Phase 3b。

## 基线与通用命令

当前基线：Phase 2 提交 `a9d80b06`，插件 19 条通过；采集验证器 6 条通过。
每个执行 cargo 的 PowerShell 都先运行：

```powershell
Set-Location E:\code\HiFiShifter\.worktrees\ara-plugin
$ErrorActionPreference = 'Stop'
. .\tools\msvc-env.ps1
$araBuildTemp = "$PWD\.build-tmp\cl"
New-Item -ItemType Directory -Force $araBuildTemp | Out-Null
$env:TEMP = $araBuildTemp
$env:TMP = $araBuildTemp
$env:ARA_VST3_SDK_DIR = "$PWD\probe\ara\rust-path\.third-party\vst3sdk"
$env:ARA_SDK_DIR = "$PWD\probe\ara\rust-path\.third-party\ARA_SDK"
cargo build --manifest-path backend\Cargo.toml -p hifishifter-plugin --offline --jobs 1
if ($LASTEXITCODE -ne 0) { throw 'build failed' }
cargo test --manifest-path backend\Cargo.toml -p hifishifter-plugin --offline --jobs 1
if ($LASTEXITCODE -ne 0) { throw 'test failed' }
```

## 文件与职责

| 文件 | 职责 |
| --- | --- |
| `backend/hifishifter-plugin/src/audio_abi.rs` | SDK 布局、输出初始化、合法性检查 |
| `backend/hifishifter-plugin/tests/vst3_layout.cpp` | 原生 SDK 布局 oracle，无链接 shim |
| `backend/hifishifter-plugin/src/vst3.rs` | setup/bus/process 与每处理器 lifetime 接线 |
| `backend/third-party/ara2-bridge-plugin/src/extension/mod.rs` | 模型线程 assignment observer |
| `backend/third-party/ara2-bridge-plugin/PATCHED.md` | 本地补丁契约说明 |
| `backend/hifishifter-plugin/src/render/ownership.rs` | renderer 分配、模型键与文档边界 |
| `backend/hifishifter-plugin/src/render/source.rs` | scope 内 PCM 读取与版本撤销 |
| `backend/hifishifter-plugin/src/render/snapshot.rs` | 不可变快照、原子发布、只读播放 |
| `backend/hifishifter-plugin/src/ara/model.rs` | region 键、源版本和模型线程发布 |
| `probe/ara/build_phase3a_probe.lua` | 隔离项目、oracle 素材、REAPER 输出采集 |
| `probe/ara/verify_phase3a_output.ps1` | 比对真实输出和 oracle，拒绝错向/重复输出 |

## Task 12: VST3 音频 ABI 与安全缓冲边界

**Files:** Create `src/audio_abi.rs`、`tests/vst3_layout.cpp`；Modify `src/lib.rs`、`src/vst3.rs`。
路径均相对 `backend/hifishifter-plugin/`。

**Interfaces:**
- Consumes: SDK `ivstaudioprocessor.h` / `ivstprocesscontext.h`。
- Produces: `#[repr(C)] AudioBusBuffers { num_channels: i32, silence_flags: u64, channel_buffers: *mut *mut f32 }`，完整 `ProcessData` / `ProcessContext` / `FrameRate` / `Chord`；已有 `ProcessSetup` 也参加原生校验。
- Produces: `unsafe fn clear_outputs(data: *mut ProcessData) -> Result<(), BufferError>`，`BufferError::{InvalidArgument, UnsupportedFormat}`。零帧/无输出合法；空 inactive channel 不解引用；非法字段/格式不写缓冲。

- [x] **Step 1: 写真正调用 vtable 回调的失败测试**

```rust
// 回调不能留宿主提供的旧样本，更不能越过 numSamples 覆盖尾哨兵。
let mut left = [0.75_f32, 0.75, 123.0];
let mut right = [-0.5_f32, -0.5, 456.0];
// 构造 SDK 同布局 ProcessData：numSamples=2、stereo output、kSample32。
// 调 AUDIO_VTBL.process(audio_ptr(processor), &raw mut data)。
assert_eq!(left, [0.0, 0.0, 123.0]);
assert_eq!(right, [0.0, 0.0, 456.0]);
assert_eq!(bus.silence_flags, 3);
```

另测 null data、负 numSamples、zero-frame flush、empty buses、null inactive plane、
null buffer array、kSample64 拒绝；setup NaN/rate<=0/负块长与非 stereo arrangement 拒绝。
构造在测试工具中，不向生产 Processor 加测试专用方法。

- [x] **Step 2: RED**

运行通用初始化后 `cargo test --manifest-path backend\Cargo.toml -p hifishifter-plugin --lib --offline --jobs 1 audio_boundary_tests`。
现有 `audio_process` 空实现应因旧样本未清零失败；不是编译或环境失败。

- [x] **Step 3: 最小实现**

按头文件逐字段定义 ABI；处理前校验 numSamples>=0、bus count>=0、channel count>=0，
有帧有总线时 outputs 非空。先校验全部总线后写，避免错误时半写。仅 dereference 非空
plane，`write_bytes(plane, 0, frames)`；仅零通道或 stereo 合法，其他布局拒绝；stereo silence mask=3。
`audio_process` 将错误映射到 `K_INVALID_ARGUMENT` / `K_RESULT_FALSE`，无 logger。
仅接安全初始化，不把它当 PCM renderer；Task 14 替换成功路径。

- [x] **Step 4: SDK 原生布局 oracle 与 GREEN**

C++ 包含两份头文件，以 JSON 输出 sizeof/alignof/offsetof。Rust 单测用 `cl.exe`
编译进 `.build-tmp/vst3-layout` 后运行并比较全部字段；缺 cl/SDK 必须失败，不 skip。
原生值不能从 Rust 生成，否则是自证。随后通用完整插件测试、`git diff --check`。

- [x] **Step 5: 记录并本地提交**

实测：7 条边界测试首次运行 6 条失败（旧空实现），修复后通过；独立审查发现输入
形状检查缺失，新增 sentinel 回归 RED/GREEN 后通过。native oracle 逐字段校验：
ProcessSetup=24/align8、AudioBusBuffers=24/align8、ProcessData=80/align8、
ProcessContext=112/align8、Chord=4/align2、FrameRate=8/align4。Task 12 本地 checkpoint
通过复审；完整插件测试 28 条通过，采集验证器 6 条通过。尚未部署/实测发声。

```powershell
git add backend/hifishifter-plugin/src/audio_abi.rs backend/hifishifter-plugin/src/lib.rs backend/hifishifter-plugin/src/vst3.rs backend/hifishifter-plugin/tests/vst3_layout.cpp docs/superpowers/plans/2026-10-04-ara-plugin-v1-phase3a.md probe/ara/EXECUTION-LEDGER.md
git commit -m "feat(ara): validate the VST3 audio ABI and initialize host buffers safely"
```

## Task 13: renderer 区域所有权与绑定释放

**Files:** Modify vendored `extension/mod.rs` / PATCHED.md、`src/vst3.rs` / `src/ara/model.rs`；Create `src/render/ownership.rs`。

**Interfaces:**
- `type AssignmentObserver = Arc<dyn Fn(ExtensionRoles, &[usize]) + Send + Sync>`。
- 新建 `ExtensionBinding::new_with_assignment_observer(generation, known, assigned, supported, observer)`，保持原 `new` 行为兼容。
- `RegionKey=u64`（`CreateContext::realtime_key`）；`DocumentId=u64`（会话内单调递增，不持久化）。
- `RegionOwners::register(key, document_id, slot) -> Result<(), OwnershipError>`、`remove(key)`、`resolve(keys: &[RegionKey]) -> Result<(DocumentId, Vec<usize>), OwnershipError>`。跨文档、未知或已销毁键返回错误，不混音。

- [ ] **Step 1: RED 测试所有权与 teardown**

```rust
let mut owners = RegionOwners::default();
owners.register(101, 1, 0).unwrap();
owners.register(202, 2, 0).unwrap();
assert_eq!(owners.resolve(&[101]).unwrap(), (1, vec![0]));
assert!(owners.resolve(&[101, 202]).is_err());
owners.remove(101);
assert!(owners.resolve(&[101]).is_err());
```

用真实 ExtensionBinding FFI add/remove 测试 observer 更新，两个 playback renderer 各只
持自己分配的键；editor renderer 与 playback 分离；lease/binding 两种析构顺序禁止悬垂。

- [ ] **Step 2: 运行插件单测及 vendored extension 测试，确认未实现 observer/分配解析失败**
- [ ] **Step 3: 实现模型线程通知和 owner**

```rust
// region 创建时登记，销毁时撤销；不从 RawHandle 猜 ModelRef。
let key = context.realtime_key().ok_or(AraError::InvalidState("missing region key"))?;
// observer 在锁释放后调用，发布到自己的 renderer state，不能在 process 锁集合。
// extension owner 由 entry builder 的 Arc 捕获，不使用 Box::leak。
```

重绑定失败时释放未交付扩展；native entry COM 引用存活时保留扩展 storage。
document destroyed 时撤销文档键和快照；不允许“全局最后一个文档”替代归属。

- [ ] **Step 4: GREEN + 失败初始化/entry 仍被 COM 持有时组件释放/双文档实测测试**
- [ ] **Step 5: 显式 stage 本任务源码、补丁说明、ledger；本地 commit `feat(ara): bind renderer assignments to document-owned regions`**

## Task 14: 宿主供源的不可变 PCM 播放快照

**Files:** Create `src/render/{source,snapshot}.rs`；原 `src/render.rs` 转为 `src/render/mod.rs`（原 helper 保留）；Modify `src/ara/model.rs` / `src/vst3.rs`。

**Interfaces:**
- `SourcePcm { sample_rate: u32, planes: Vec<Vec<f32>>, version: u64 }`；`read_source_pcm(host: &HostContentScope<'_, '_>, frames: usize, channels: usize, sample_rate: u32, version: u64) -> Result<SourcePcm, AraError>`（在模型回调当前源 scope 内解析 source_ref）。
- `PlaybackSnapshot { sample_rate: u32, origin_sample: i64, left: Vec<f32>, right: Vec<f32>, revision: u64 }`；`mix_plain_regions(regions: &[AraPlaybackRegion], sources: &HashMap<String, SourcePcm>, sample_rate: u32) -> Result<PlaybackSnapshot, SnapshotError>`（regions 是 Task 13 resolve 后存活区域，sources 是按 persistentID 查询的宿主 PCM 表）。
- `SnapshotPublisher::publish(snapshot) -> Result<(), SnapshotError>` 在非实时线程保留快照；`unsafe fn copy_block(&self, project_sample: i64, outputs: &mut AudioBusBuffers, frames: usize)` 在回调只读原子指针。退役快照预算内保留到 owner drop，不在音频线程释放。

- [ ] **Step 1: 手算 oracle 的 RED 测试**

```rust
// 单声道源 [0.1,0.2,0.3,0.4]，region 从源第 1 帧取 2 帧，放工程第 2 帧。
assert_eq!(snapshot.left, [0.0, 0.0, 0.2, 0.3]);
assert_eq!(snapshot.right, [0.0, 0.0, 0.2, 0.3]);
// seek 到工程第 3 帧读 3 帧，期望 [0.3,0,0]，不能使用累计 cursor。
// 两个 renderer 分配不相交，输出拼合后振幅不能为 [0.4,0.6]。
```

另测负起点、空洞、越界、撤权/版本递增撤销旧快照、预算超限拒绝、禁用源不可读。
双通道保留左右，采样率转换有独立 oracle；非 unit playback_rate 先返回 Unsupported，
不能用重采样冒充保调拉伸。reader 在 scope 返回前 drop。

- [ ] **Step 2: 运行新单测 RED，失败原因是缺 PCM/快照而非文件路径**
- [ ] **Step 3: 在 enable=true / content update 的合法 scope 内按小块读取**

```rust
let source_ref = host.current_audio_source().ok_or(AraError::InvalidState("missing source scope"))?;
let mut reader = host.audio_reader::<f32>(source_ref, channels)?;
// 循环以 4096 帧切片读取，首个 read error 则丢弃未完成结果，不发布半成品。
reader.read(sample_position, &mut channel_slices)?;
```

模型 end_editing、assignment 改变时重新准备已授权 PCM 的普通区域快照。source revoke
立即撤销旧发布；geometry/update/destroy 改变 revision。audio_process 校验 context 的
sample_rate 与 snapshot 一致后按 projectTimeSamples copy；miss 只 atomic 计数。

- [ ] **Step 4: GREEN、实时分配与日志守卫、随机跳转和连续 30 秒块拼接比对**
- [ ] **Step 5: 显式 stage 本任务源码、测试和 ledger；本地 commit `feat(ara): play host-owned PCM through immutable renderer snapshots`**

## Task 15: 隔离 REAPER 最终输出与真实倒放判定

**Files:** Create `probe/ara/build_phase3a_probe.lua`、`verify_phase3a_output.ps1`、`test_phase3a_output.ps1`、`captures/phase3a-FINDINGS.md`；保存 raw log、输出 WAV、JSON、必要截图。

**Interfaces:** 输入普通/裁切/间隙/移动/倒放的宿主导出 WAV 与非对称 oracle；输出 JSON
`{max_abs_error, duplicate_gain, reversed_verified, reverse_output_direction, cold_start_misses}`。
非零 exit 对应证据不匹配，不生成 PASS。

- [ ] **Step 1: verifier RED：正确输出、重复2倍、静音、正向伪倒放、错误 seek、短文件六个样本**

```powershell
# 正确样本最大绝对差 <= 1e-6；重复幅度/静音/错方向/错误位置/缺帧必须 throw。
& .\probe\ara\test_phase3a_output.ps1
```

- [ ] **Step 2: 生成不对称夹具并导出普通/裁切 oracle**

Lua 沿 Task 10/11 的流程插插件；夹具前半/后半用不同幅度与脉冲位置，不能用纯周期
正弦区分倒放。主轨 gain=1、无其他 FX，导出32位float不 normalize。记录 render 参数。
真正倒放用 `reaper.Main_OnCommand(41051,0)`；section reader 必须 `reversed=true`。

- [ ] **Step 3: 通用 build/test 后部署到隔离目录并采集**

```powershell
# 先用 Get-CimInstance 确认现存 REAPER 命令行，仅退出本批隔离实例；新启动前确认无进程。
$araProbeRoot = "$PWD\probe\ara"
Copy-Item backend\target\debug\hifishifter_plugin.dll "$araProbeRoot\vst3\HiFiShifter.vst3" -Force
$env:HIFISHIFTER_ARA_LOG = "$araProbeRoot\captures\phase3a-plugin.log"
Start-Process -FilePath 'D:\Softwares\REAPER (x64)\reaper.exe' -WindowStyle Hidden `
  -ArgumentList @('-cfgfile', "$araProbeRoot\reaper-profile\task10-clean\REAPER.ini", '-new', "$araProbeRoot\build_phase3a_probe.lua") `
  -WorkingDirectory $araProbeRoot
```

需要 UI 验证时用 computer-use skill 激活隔离窗口；先截图再输入。脚本等待 `time_precise`
截止时间，不使用快速 defer 次数充当秒数。系统 VST3 自动扫描的 activation UI 不动
用户配置，复用已隔离扫描缓存。

- [ ] **Step 4: 波形验收和 U1 输出级结论**

普通/裁切/seek 逐样本 maxdiff<=1e-6 且无重复幅度。倒放分两种实测结论：宿主在外部
处理则 oracle 通过；仍正向则 U1 缺口成立，无方向通道前不再推进“全能力 v1”。
不能将映射中的正向样本直接推导为最终倒放失败，也不能靠 UI checkbox 证明输出正确。

- [ ] **Step 5: 关闭隔离 REAPER、保留原始证据、独立代码审查与本地提交**

raw log 被全局 `*.log` 忽略，使用 `git add -f` 仅本任务命名日志；不 stage profile/二进制。
追加每个决策的 Ruling。通过后才写 Phase 3b 逐任务计划，未通过则报告精确阻塞。

## 后续依赖与覆盖检查

Phase 3b 必做：宿主 PCM 注入式 mixdown/vocoder（保持 app 默认路径）、稳定 clip 身份、
source/sequence/modification 完整更新销毁、内容指纹缓存、真实修音 A3、冷启动/缓存 miss/
seek/离线导出供音 U3、离线改源 U4。Phase 4 为命名管道+曲线 state+重开哈希 A4；
Phase 5 为干净机器包验收。它们的实现任务在前一阶段实测后展开，不用未知 API 填假计划。

本批覆盖：设计 §2 ABI -> Task 12；§3 ownership -> Task 13；PCM/快照 -> Task 14；
§4 宿主证据/倒放 -> Task 15。A3/A4、U3/U4 不在本批完成口径内。
