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

## 未完成：需要人手在 REAPER 里操作

Task 1 的 Step 1 / 3 / 4 / 5 需要在 REAPER 图形界面里操作，并且要编辑 SDK 示例源码加 dump，
因此无法由 agent 独立完成：

- Step 1：用已装的 Melodyne 5 确认 ARA 在 REAPER 里确实生效。
- Step 3：在 Test Plug-In 的文档控制器里加 dump，落盘 `captures/ara-model.json`。
- Step 4：造"不干净"的素材（同源多放、拉伸、倒放、淡化、非 44.1kHz 工程）重采一次。
- Step 5：写 `captures/FINDINGS.md`，重点是**ARA 不提供但渲染需要的字段**清单。

## 环境限制（记录在案）

`backend/src-tauri` 的 `cargo test` **在本会话无法取得基线**：`fdk-aac-sys` / `opusic-sys`
经 `cmake` crate 调用 MSBuild，撞上同一个临时文件拒绝，而那条路径没有
`TrackFileAccess` 之类的钩子可关。基线需在普通终端取得。这与本探针的产物无关。
