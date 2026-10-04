# Task 2 FINDINGS —— Rust 接触 ARA 的路径

> 写于 2026-10-04。对应 plan 的 `Task 2: 确定 Rust 接触 ARA 的路径`，Step 3。
> 本文件是探针结论；证据全部为**实测**，推断处会显式标注。
> 相关代码：`probe/ara/rust-path/src/{lib,vst3,model}.rs`、`probe/ara/build_task2_probe.lua`。

---

## 1. 结论

**选定路径 A：`ara2-bridge` 0.3.0（含 companion）承担 ARA↔VST3 的桥，外加自写的最小
VST3 模块外壳。** Step 3 的三条判据全部通过 —— R1（Rust 能以可接受成本接触 ARA）成立。

实测达成：一个 Rust 写的 cdylib（`hifishifter_ara_probe.dll`，改名 `HiFiShifterARAProbe.vst3`）
被 REAPER 7.81 加载、识别为 ARA 插件、完成 ARA 绑定，并在日志里打印出它看到的
audioSource / playbackRegion 数量。

## 2. Step 3 三条判据

| 判据 | 结果 | 证据 |
| --- | --- | --- |
| FX 浏览器/扫描能看到该插件 | PASS | 扫描缓存 `HiFiShifterARAProbe.vst3=…,352569748{41534648415272506F62652E50726331,HiFiShifter ARA Probe (HiFiShifter)`；脚本侧 `TrackFX_AddByName -> 0`，FX 名 `VST3: HiFiShifter ARA Probe (HiFiShifter)` |
| 以 ARA 方式激活（非普通 VST 插入） | PASS | REAPER 依次查询 `IPlugInEntryPoint`、`IPlugInEntryPoint2` → 创建 ARA document controller（`apiGeneration=V2Final`）→ `bindToDocumentControllerWithRoles` |
| 日志数量与工程实际一致 | PASS | 1 个源、2 处摆放 → `sources=1 modifications=1 regionSequences=1 playbackRegions=2` |

## 3. 关键日志节选（原始行，未加工）

宿主侧（`captures/task2-capture.log`）：

```
stage0: track created
stage1: items = 2
stage2: calling TrackFX_AddByName("HiFiShifter ARA Probe")
stage2: TrackFX_AddByName -> 0
  fx name = VST3: HiFiShifter ARA Probe (HiFiShifter)
stage3: plugin log shows playback_region at check 4
stage3: track items=2 fx=1
```

插件侧（`captures/task2-plugin.log`，节选）：

```
[0003] document_controller created: apiGeneration=V2Final
[0004] begin_editing
[0005] end_editing: sources=0 modifications=0 regionSequences=0 playbackRegions=0
[vst3] ARA bind: build extension (generation=V2Final, known=0x7, assigned=0x6)
[vst3] IComponent::getControllerClassId -> HFSARAProbe.Ctl1
[vst3] factory createInstance cid=…43746C31 iid=…(IEditController)
[vst3] IEditController::initialize
…
[0007] region_sequence #1: name=task2-ara
[0008] audio_source #1: persistentID=…\fixtures\tone44100.wav sampleRate=44100 sampleCount=88200 channels=1 merits64Bit=true
[0009] audio_source samples_access source=…tone44100.wav enable=true
[0010] audio_modification #1: persistentID=…tone44100.wav name=None
[0011] playback_region #1 created
[0012] playback_region #2 created
[0013] end_editing: sources=1 modifications=1 regionSequences=1 playbackRegions=2
[0014] audio_source samples_access source=…tone44100.wav enable=false
```

REAPER 为一条轨道建了 **3 个处理器实例**（role 分配 `0x6` = editor renderer + editor view，
两次 `0x1` = playback renderer）。这解释了为什么 ARA 里"一个插件实例"与"一个 VST3 组件实例"
不是一对一：**每个处理器实例都必须有自己的 companion 绑定与入口适配器**，共享一个会互相抢绑定。

## 4. 三条必须记住的硬事实（都是实测）

1. **VST3 的 IID 在 Windows 上是 GUID 布局**：`INLINE_UID(l1,l2,l3,l4)` 展开为
   `l1` 小端 + `l2` 拆成两个 u16 各自小端 + `l3`/`l4` 大端。证据：REAPER 请求
   `IPluginFactory2` 时发出的 16 字节是 `50B607004BF20B4CA464EDB9F00B2ABB`，正是
   `{0007B650-F24B-4C0B-A464-EDB9F00B2ABB}` 的 GUID 布局。
   **按"每个字小端"实现会查不到任何接口**，症状是 REAPER 认为该文件有 0 个类
   （缓存里只剩裸时间戳），而不是报错。

2. **`IPluginFactory` 直接继承 `FUnknown`**，不是 `IPluginBase`。若在 vtable 里插入
   `initialize`/`terminate` 两个槽位，宿主的 `countClasses()` 会落到 `getFactoryInfo()` 上，
   于是"类数量 = 0"。

3. **REAPER 要求 ARA 插件提供编辑器控制器。** 只注册 `kVstAudioEffectClass` +
   `kARAMainFactoryClass` 时，REAPER 会走完 initialize 与 ARA bind，随后询问
   `IEditController`、调用 `getControllerClassId`；两者都拿不到就放弃插入、卸载模块，
   紧接着对已失效的控制器指针调用回调 → 进程崩溃（实测 `0xc0000005`，出错模块
   `HiFiShifterARAProbe.vst3_unloaded`，偏移落在 `ara2_bridge_plugin` 的 `begin_editing`）。
   补一个最小 `kVstComponentControllerClass`（无参数、无 GUI、`createView` 返回空）后，
   插入成功、崩溃消失。

## 5. 顺带解决 Task 1 的未决项：`sampleAccessEnabled`

Task 1 留下一个未解释项：两次采集一次 `true` 一次 `false`。本步有了直接证据：

- `[0009] enable=true` —— ARA 绑定后，REAPER **主动**授予样本访问；
- `[0014] enable=false` —— 停用时**撤销**。

即 `false` 是撤销语义，不是拒绝。C++ 测试插件那份 `false` 更可能来自其采集时机与
能力声明（它不声明可分析内容类型），而不是"宿主不给样本"。
**这一条从"未解释"变成"已在宿主回调层面直接观测到授权"**；仍未验证的是持续可用性
（例如播放中反复调用时是否稳定）。

## 6. 残余风险与未覆盖范围（不要当成已解决）

- **没有渲染**：`IAudioProcessor::process()` 是空实现；playback renderer 只登记区域、
  不产出音频。spec §6 的 **R4（提前渲染窗口内总能给出音频）完全未验证**。
- **无参数 / 无 GUI / 无持久化**：`storeAudioSourceContent`、参数编辑通道（spec §5.4）
  一概未碰。
- **协商到 ARA 2.0 Final**，不是 2.3；2.1–2.3 的增量字段未验证。
- **单工程单次观测**：多轨、多实例、工程重开、走带变化都未验证。
- 探针把 ARA 主工厂适配器与扩展绑定**泄漏到进程结束**（探针可接受，产品不可）。
- 崩溃那条路径只是"随插入成功而不再触发"；REAPER 在插件被拒绝时的卸载时序本身
  是个独立问题，产品若要支持"加载失败"路径需要单独处理。

## 7. 复现配方

构建（在 worktree 根目录）：

```powershell
. .\tools\msvc-env.ps1
$tmp = "<worktree>\probe\ara\rust-path\.build-tmp\cl"; New-Item -ItemType Directory -Force $tmp | Out-Null
$env:TEMP = $tmp; $env:TMP = $tmp          # vcvars 之后再设一次
$env:ARA_VST3_SDK_DIR = "<worktree>\probe\ara\rust-path\.third-party\vst3sdk"
$env:ARA_SDK_DIR      = "<worktree>\probe\ara\rust-path\.third-party\ARA_SDK"
cd probe\ara\rust-path; cargo build --jobs 1
```

采集（隔离实例，先确保没有 REAPER 在运行）：

```powershell
Copy-Item target\debug\hifishifter_ara_probe.dll "<worktree>\probe\ara\vst3\HiFiShifterARAProbe.vst3" -Force
$env:HIFISHIFTER_ARA_PROBE_LOG = "<worktree>\probe\ara\captures\task2-plugin.log"
reaper.exe -cfgfile "<worktree>\probe\ara\reaper-profile\REAPER.ini" -new "<worktree>\probe\ara\build_task2_probe.lua"
```
