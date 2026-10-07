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

## 环境限制（记录在案）

`backend/src-tauri` 的 `cargo test` **在本会话无法取得基线**：`fdk-aac-sys` / `opusic-sys`
经 `cmake` crate 调用 MSBuild，撞上同一个临时文件拒绝，而那条路径没有
`TrackFileAccess` 之类的钩子可关。基线需在普通终端取得。这与本探针的产物无关。
