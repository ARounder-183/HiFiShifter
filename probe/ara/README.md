# probe/ara — ARA2 可行性探针（可丢弃）

对应计划：`docs/superpowers/plans/2026-10-04-ara-bridge-probe.md`
对应设计：`docs/superpowers/specs/2026-10-04-ara-bridge-design.md`

**本目录整体是一次性实验产物。** 探针结束后应整体删除，或至少不得被 `backend/`、
`frontend/` 引用。

## 已完成：Task 1 Step 2 —— SDK 与 VST3 伴随 SDK 就位，测试插件已构建

| 产物 | 路径 |
| --- | --- |
| ARA SDK（含子模块） | `probe/ara/ARA_SDK` |
| VST3 伴随 SDK（v3.7.11_build_10） | `probe/ara/ARA_SDK/vst3sdk` |
| CMake 工程 | `probe/ara/ARA_SDK/ARA_Examples/build-vs2022` |
| **构建出的测试插件** | `probe/ara/ARA_SDK/ARA_Examples/build-vs2022/bin/Release/ARATestPlugIn.vst3` |

- 插件大小 284672 字节，**ARA SDK 版本 2.3.0**。
- 该插件即 SDK README 所说的 Test Plug-In：自带"extensive validation and logging
  capabilities"，正是 Task 1 Step 3 要做 dump 的地方。

### 复现命令

```powershell
cd probe\ara
git clone --depth 1 --recurse-submodules --shallow-submodules `
  https://github.com/Celemony/ARA_SDK.git
cd ARA_SDK
cmake -P install_vst3sdk.cmake          # 拉 VST3 伴随 SDK
cd ARA_Examples
cmake -B build-vs2022 -G "Visual Studio 17 2022" -A x64 -D ARA_SETUP_DEBUGGING=OFF
cmake --build build-vs2022 --config Release --target ARATestPlugInVST3 `
  -- /m:1 /p:TrackFileAccess=false
```

### REAPER SDK（接口签名的唯一依据）

`backend/hifishifter-plugin/src/host/reaper.rs` 是**手写适配**，不是 SDK 文件。
它的函数签名与值键语义必须对着官方头文件核对 —— 尤其是淡化轴这类"区间标注"
（`C_FADE*SHAPE` 标注 v7.80 and earlier、`D_FADE*DIR_NEW` 标注 v7.81 and later）。
头文件不进仓，用这条命令取（落点已在 `.gitignore` 覆盖范围内）：

```powershell
cd probe\ara\rust-path\.third-party
git clone https://github.com/justinfrankel/reaper-sdk.git
# 与 backend/hifishifter-plugin/src/host/REAPER-SDK-NOTICE.md 记录的 pin 对齐：
git -C reaper-sdk checkout c0eafe87863b2bf69c5c822760f1b32a753b211b
```

核对时看 `sdk/reaper_plugin_functions.h`：

| 想确认什么 | 看哪里 |
| --- | --- |
| 淡化轴属于哪个版本区间 | 搜 `D_FADEINDIR` / `C_FADEINSHAPE`（值键说明区，`GetMediaItemInfo_Value` 附近） |
| `GetAppVersion` 的字符串格式 | 搜 `GetAppVersion` 上方的注释 |
| 取 take / 枚举 take | 搜 `GetMediaItemNumTakes` / `GetMediaItemTake` |
| 建轨与插 FX | 搜 `InsertTrackInProject` / `TrackFX_AddByName` |

### 构建这条命令为什么长这样（踩过的坑，照抄别改）

在受限环境里，MSBuild 编译会以
`MSB6003 ... UnauthorizedAccessException ... MSBuildTemp\tmp*.rsp` 失败。三个条件同时满足才通过：

1. **必须先加载 MSVC 环境**（`..\..\tools\msvc-env.ps1`），让 `cl.exe` 在 `PATH` 上 ——
   与仓库自身的 `D8050` 是同一个根因。
2. **`/m:1`（串行）** —— 并行编译是被打断的那一步。
3. **`/p:TrackFileAccess=false`** —— MSBuild 于是完全不创建临时响应文件。

`-D ARA_SETUP_DEBUGGING=OFF` 是刻意的：ON 会让构建把插件复制到
`C:\Program Files\Common Files\VST3`，即写到本工作区之外。保持构建自包含；
需要让 REAPER 发现插件时，改为在 REAPER 里添加该构建目录为 VST 路径，或由人工安装。

## 已应用插桩：ARA 模型图 dump

`instrumentation-reference/` 是**权威副本**（可评审、可复现）；实际编译发生在
git-ignored 的克隆里。要同步，按下面三步照抄。

### 插桩做了什么

在 `ARATestDocumentController` 的两个回调里各插一次调用，把整个 ARA 模型图写成 JSON：

- `willNotifyModelUpdates()` —— 模型更新通知点
- `didEndEditing()` —— 一次编辑会话结束，模型图最稳定的时刻

产出字段：`audioSources`（persistentID / name / sampleRate / sampleCount /
durationSeconds / channelCount / merits64BitSamples / sampleAccessEnabled）、
`musicalContexts` + `regionSequences`、`audioModifications`（含所属 source）、
`playbackRegions`（modification 时间与播放时间两个坐标系、`isTimestretchEnabled`、
`isTimeStretchReflectingTempo`、`hasContentBasedFadeAtHead/Tail`）。

实现内做了 400ms 节流（回调可能被高频触发），失败静默降级，绝不把异常抛回宿主。

### 应用插桩

**1. 复制两个文件**

```powershell
Copy-Item probe\ara\instrumentation-reference\AraProbeDump.* `
          probe\ara\ARA_SDK\ARA_Examples\instrumentation\
```

**2. 打三处补丁**

`probe/ara/ARA_SDK/ARA_Examples/TestPlugIn/ARATestDocumentController.cpp`：

```cpp
// (a) include —— 加在 #include "ExamplesCommon/Utilities/StdUniquePtrUtilities.h" 之后
#include "AraProbeDump.h"

// (b) willNotifyModelUpdates() 开头
AraProbeDumpToFile (this);

// (c) didEndEditing() 的 enableRendererModelGraphAccess (); 之后
AraProbeDumpToFile (this);
```

`probe/ara/ARA_SDK/ARA_Examples/CMakeLists.txt`：

```cmake
# (d) add_library(ARATestPlugInCommon ...) 的源文件列表末尾追加
    "${CMAKE_CURRENT_SOURCE_DIR}/instrumentation/AraProbeDump.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/instrumentation/AraProbeDump.cpp"

# (e) 在 target_include_directories(ARATestPlugInCommon PUBLIC ...) 里追加
    "${CMAKE_CURRENT_SOURCE_DIR}/instrumentation"

# (f) 在 set_target_properties(ARATestPlugInCommon ...) 之后加 —— 必需，理由见下
target_compile_options(ARATestPlugInCommon PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/utf-8>
)
```

### 三个非显然的坑（都是实际踩出来的）

1. **插桩文件必须位于 `ARA_Examples/` 之内**。SDK 的 `ara_group_target_files()`
   假定所有源文件都在工程目录下；放到同级目录会让 CMake 配置直接失败
   （报错是 "is not a prefix of file"，不指向真正原因）。
2. **需要 `/utf-8`**。插桩文件是 UTF-8（含中文注释），而本机 MSVC 默认代码页是 936。
   不加此开关时 MSVC 把 UTF-8 注释读成 CP936，报出与注释毫不相干的语法错误
   （实测 `C2447: '{': missing function header`）。SDK 自身源码是纯 ASCII，
   官方构建不会暴露此问题。
3. **目标按 C++11 编译**：不能用 `std::filesystem`，也不能用 `ARAColor`（属 ARA 2.0
   草案附加项，本编译配置下不可见）。故路径处理改用 Win32
   `GetFullPathNameA` / `GetFileAttributesA`；颜色字段刻意省略（对渲染映射无关）。

### 输出位置

优先取环境变量 `ARA_PROBE_OUT`。未设时依次尝试 `.\\ara-model.json`、
`..\\..\\..\\..\\..\\captures\\ara-model.json`、`captures\\ara-model.json`（取第一个其
父目录存在的）。**推荐显式设环境变量**，避免猜路径。

### 让 REAPER 找到插件

插桩后的插件已复制到 `D:\VST\ARATestPlugIn.vst3\ARATestPlugIn.vst3` ——
`D:\VST` 本就在 `REAPER.ini` 的 `vstpath64` 里，所以无需改配置，只需在 REAPER 里
重新扫描 VST。

## 未完成：需要人手在 REAPER 里操作

Task 1 的 Step 1 / 3 / 4 / 5 需要在 REAPER 图形界面里操作，且要造不同形态的素材，
因此无法由 agent 独立完成：

- Step 1：用已装的 Melodyne 5 确认 ARA 在 REAPER 里确实生效。**已完成并通过。**
- Step 3：挂载插桩后的 Test Plug-In，落盘 `captures/ara-model.json`。
- Step 4：造"不干净"的素材（同源多放、拉伸、倒放、淡化、非 44.1kHz 工程）重采一次。
- Step 5：写 `captures/FINDINGS.md`，重点是**ARA 不提供但渲染需要的字段**清单。

### 命令行没有"运行脚本"开关，但启动钩子能（本机实测，2026-10-09）

**REAPER 7.82 的命令行确实没有"运行 ReaScript"这个开关。** 从 `reaper.exe` 里取出的用法串
（7.82 x64）：

```
-cfgfile file.ini : use full path for alternate resource directory, otherwise uses default path
-saveas -template -fxoffline -profile -project -batchconvert -nulltest -peaktest
-renderproject filename.rpp : render project and exit
-play -nonewinst -newinst -audiocfg -close -ignoreerrors -nosplash -splashlog -noactivate
-resetconfig -new
```

逐个试过且**均不执行脚本**的形式：

| 形式 | 实际行为 |
| --- | --- |
| `reaper.exe -cfgfile X -new script.lua` | 起一个空工程，脚本不跑 |
| `reaper.exe -cfgfile X script.lua` | 把 `script.lua` 当工程加载 → 弹 `Load Error` |
| `reaper.exe -cfgfile X project.rpp script.lua` | 工程正常打开，脚本不跑 |
| `Scripts\__startup.lua` 约定 | 不执行 —— REAPER **没有** `.lua` 的启动约定 |
| `-reascript script.lua` | 未知开关，被忽略 |

**但 `Scripts\__startup.eel` 是有的**（EEL，不是 Lua）。REAPER 启动时会执行资源目录下的
`<resource>\Scripts\__startup.eel`；EEL 里能调 `AddRemoveReaScript` 把任意 `.lua` 注册成
action，再用 `Main_OnCommand` 跑它。于是探针可以在无人值守下跑完 —— 见下一节。
（`.lua` 名字不生效这件事曾让人得出"完全没法自动化"的结论，那个结论只对了一半。）

三个必须知道的细节：

1. **EEL 里的 REAPER API 不带 `reaper.` 前缀**（JSFX 风格）：写 `CountTracks(0)`，不是
   `reaper.CountTracks(0)`。写成后者会**编译失败，而且失败是静默的** —— 钩子只是不执行，
   没有任何提示。
2. **`GetMediaItemInfo_Value` 对不认识的键返回 `0.0`，不报错**。轴名写错一个字母，读数会是
   一串干净的 0，看起来像"宿主不支持"。`fade_axis_capture.lua` 曾把 `C_FADEOUTSHAPE` 写成
   `D_FADEOUTSHAPE`，fade-out 四列整列假 0。
3. `AddRemoveReaScript(add, 0, path, 1)` 的返回值是 action id（实测从 `53000` 起），
   返回 0 表示注册失败，此时 `Main_OnCommand(0, …)` 什么也不做。

### 无头跑探针：`run_probe_headless.ps1`

```powershell
powershell -ExecutionPolicy Bypass -File probe/ara/run_probe_headless.ps1 `
  -Probe "D:\Projects\HiFiShifter\probe\ara\fade_axis_capture.lua" `
  -Out   "D:\Projects\HiFiShifter\.build-tmp\f1\fade_axis.json" `
  -Env   "HIFISHIFTER_FADE_AXIS_OUT=D:\Projects\HiFiShifter\.build-tmp\f1\fade_axis.json"
```

| 参数 | 作用 |
| --- | --- |
| `-Probe <x.lua>` | 要跑的探针。省略则只搭夹具（见 `-SetupEel`） |
| `-Out <x.json>` | 探针写结果的路径；给了 `-Probe` 就必须给 |
| `-SetupEel <x.eel>` | 在注册探针**之前**执行的 EEL 片段，用来搭宿主夹具（建轨、插媒体、挂插件） |
| `-Env NAME=VALUE` | 传给探针的环境变量，可重复（`-File` 模式传不了哈希表） |
| `-Project <x.rpp>` | 用现成工程代替 `-new` |
| `-SettleSeconds <n>` | 哨兵出现后再等 n 秒才杀 REAPER。**搭夹具时必须给**，理由见下 |

探针产出 JSON 时退出码 0；只搭夹具时以钩子日志为完成信号，同样返回 0；没出结果时打印
钩子日志并返回 1。

**为什么搭夹具要 `-SettleSeconds`**：`TrackFX_AddByName` 只是把插件插进去，REAPER 建 ARA
文档、推模型、发播放区域都在**之后**异步发生。夹具一写完哨兵就杀进程的话，插件日志里
只会看到一半流程（实测有时连 `regionSequences=1` 都还没出现），于是"宿主没做某件事"这种
结论就分不清是真没有还是没跑到。

**夹具里的 `__REPO_ROOT__`**：EEL 没有取环境变量的办法（`os.getenv` 是 Lua 的），所以夹具
用这个占位符写仓库根，由脚本替换成正斜杠的绝对路径。

脚本自己会做的隔离与兜底：

- 资源目录指向 `.build-tmp\reaper-probe\`（`-cfgfile`），**不碰 `%APPDATA%\REAPER`**；
- 先把用户的 ini **复制**进临时目录 —— 不复制的话 REAPER 会认为这是全新便携安装，
  弹 "Would you like to scan system VST/CLAP/LV2 paths?" 模态框，启动钩子永远轮不到执行；
- 输出文件所在目录若不存在会先建出来（探针用 `io.open(path,"w")`，父目录不存在时
  会 `assert` 失败并静默中止）；
- 读夹具 EEL 时**显式 `-Encoding UTF8`**。不显式指定的话 Windows PowerShell 5.1 按 ANSI
  （中文系统上是 CP936）读，中文注释里的前导字节在行尾会**吃掉换行**，下一行代码被并进
  注释、整段静默不执行 —— 这正是仓库里 `tools/*.ps1` 要求 BOM 的同一个坑，只是这次踩在
  读夹具这一侧；
- 跑完（或超时、或等满 `-SettleSeconds`）无条件杀掉 reaper 进程。

各 `build_*_probe.lua` 文件头里"用法：`reaper.exe -new build_xxx.lua`"那一行仍然是**错的**
（写它的人没验证过）；但那些脚本现在可以经由 `run_probe_headless.ps1` 无头跑。

## 环境限制（记录在案）

`backend/src-tauri` 的 `cargo test` **在本会话无法取得基线**：`fdk-aac-sys` / `opusic-sys`
经 `cmake` crate 调用 MSBuild，撞上同一个临时文件拒绝，而那条路径没有
`TrackFileAccess` 之类的钩子可关。基线需在普通终端取得。这与本探针的产物无关。

## F-2 探针：宿主到底提不提供速度 / 拍号 / 调号内容 —— **已出结果：不提供**

**结论（2026-10-09，REAPER 7.82/x64）**：REAPER **不会**为 ARA 文档建音乐上下文，
`createMusicalContext` 一次都没被调用。原始证据 `captures/f2-musical-content-7.82.md`。

**怎么跑出来的**（现在是可复现的一条命令，不再需要人手工操作）：

```powershell
powershell -ExecutionPolicy Bypass -File probe/ara/run_probe_headless.ps1 `
  -SetupEel "D:\Projects\HiFiShifter\probe\ara\fixtures\ara_musical_content_setup.eel" `
  -SettleSeconds 20
```

夹具（`fixtures/ara_musical_content_setup.eel`）建一条**有音频**的轨道、把
HiFiShifter 作为 ARA 插件挂上去，并写入两条速度 / 拍号标记（90 BPM 3/4、150 BPM 7/8）
—— 后者是为了排除"工程太简单所以宿主懒得建音乐上下文"这一种解释。夹具自己会把
`tracks=1 media_items=1 tempo_markers=2 track_fx=1` 写进 `.build-tmp/f2-fixture.log`，
用来证明工程真的长这样，而不是"夹具没搭起来"。

**为什么必须 `-SettleSeconds`**：`TrackFX_AddByName` 只是把插件插进去；REAPER 建 ARA
文档、推模型、发播放区域都在**之后**异步发生。不等就杀进程的话，日志里只会看到一半
流程（实测有时连 `regionSequences=1` 都还没出现），那样"宿主没给音乐上下文"就分不清
是真没有还是没跑到。

**代码位置**：`backend/hifishifter-plugin/src/ara/model.rs` 的
`probe_host_musical_content`（在 `create_musical_context` 回调内调用）。

**为什么在这里而不是 Lua 脚本**：要问的是 ARA 内容接口，只有插件侧的
`HostContentScope` 拿得到 —— Lua 脚本走的是 REAPER 自己的 API，看不到这一层。
`HostContentScope` 又是 `!Send`，只能在模型线程的回调内读，所以探针必须内联在
`create_musical_context` 里。

**判决规则**：

| 日志 | 结论 |
|------|------|
| `grade=...` 且 `tempo entries: N>0` | 宿主**提供**速度内容 → 可做速度映射（plan Part 3.6b） |
| `unavailable (...)` | 宿主不提供该内容 → 明确定为【宿主权威·暂不呈现】，写进手册 |
| 完全没有 `[ara][probe]` 行 | 宿主没推音乐上下文 → 同上 ← **本次实测落在这一行** |

**这一条对 plan Part 3 的意义**：不能等 ARA 内容。插件要拿 BPM / 拍号 / 网格，只能走
**REAPER 自己的 API**（`TimeMap_*` / `GetProjectTimeSignature2` 之类，通过已经建立起来的
REAPER host extension —— 日志里的 `[reaper-host] QI succeeded` / `REAPER host extension
available=true` 就是那条通道）。这与"独立 App 自己拥有这些值"并不冲突：宿主侧读一次，
和独立 App 的工程设置是同一种数据。

> 顺带记一笔：这次运行的日志里出现了两条
> `Embedded frontend: Command unavailable in ARA plugin mode: set_timeline_tempo_map`
> —— 那是**已安装的旧构建**的前端在插件里尝试写速度映射，被插件侧明确拒绝。它说明
> 界面上确实有这条路径在尝试，而 Part 3 要决定的就是它该接到哪儿。


## F-3 探针：一个三 take 的 item，宿主发几个 modification —— **已出结果**

**结论（2026-10-09，REAPER 7.82/x64）**：三个 take **各建了一个 audio source**，连非 active
take 的 PCM 都读得到；但播放区域**只有一条**，指向 **active take**。完整结论、对 plan
Part 2 的修正、以及尚未验证的边界都写在 `MULTI-TAKE-FINDINGS.md`；原始证据
`captures/f3-multi-take-7.82.md`。

```powershell
powershell -ExecutionPolicy Bypass -File probe/ara/run_probe_headless.ps1 `
  -SetupEel "D:\Projects\HiFiShifter\probe\ara\fixtures\ara_multi_take_setup.eel" `
  -SettleSeconds 20
```

夹具（`fixtures/ara_multi_take_setup.eel`）建一个 item、三个 take、每个 take 指向不同的音频
文件，并把 active take 设成中间那个 —— 三个不同文件才能把"只发 active"与"全发"区分开，
active 取中间那个才能把"只发 active"与"只发第 0 个"也区分开。自证读数落在
`.build-tmp/f3-fixture.log`（`takes=3` / `curtake=1`）。
